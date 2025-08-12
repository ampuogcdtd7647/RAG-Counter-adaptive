from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from typing import Optional, List

model_path = "/data3/models/GLM/GLM-Z1-9B-0414"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# base = AutoModelForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
# )
# base.config.use_cache = False  

# # 针对 Qwen2.5 常见的线性层命名
# target = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# lora_cfg = LoraConfig(
#     r=16, lora_alpha=32, lora_dropout=0.05,
#     target_modules=target, bias="none", task_type="CAUSAL_LM",
# )

# model = get_peft_model(base, lora_cfg)
# model.print_trainable_parameters()  # 查看可训练参数比例
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========= 数据集 =========
train_path = "/data3/fujinji/bge-m3-finetune/RAAT/data/train.jsonl"
val_path = "/data3/fujinji/bge-m3-finetune/RAAT/data/val.jsonl"

train_ds = load_dataset("json", data_files=train_path)['train']
eval_ds  = load_dataset("json", data_files=val_path)['train']


# 字段必须包含：query, answer, G, R, I
def quick_check(ds, n=1):
    row = ds[0]
    for k in ["query","G","R","I"]:
        assert k in row, f"缺少字段: {k}"
quick_check(train_ds)
print(len(train_ds))
def build_prompt(query, ctx):  
    return f"问题：{query}\n参考：{ctx}\n回答："

class RAATTrainer(Trainer):
    def __init__(self, *args, my_tokenizer=None, lam=0.3, gamma=1.0, alpha=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha
        self.ctx_list = ["G", "R", "I"]

    def _one_ctx_loss(self, model, query, answer, ctx_text, device):
        prompt = build_prompt(query, ctx_text)
        # labels 只对“答案”部分计损失：把 prompt token 标成 -100
        enc_p = self.tokenizer(prompt, return_tensors="pt").to(device)
        enc_a = self.tokenizer(answer, return_tensors="pt").to(device)
        input_ids = torch.cat([enc_p.input_ids, enc_a.input_ids[:, 1:]], dim=1)  # 去掉答案起始的BOS
        attn_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :enc_p.input_ids.size(1)] = -100  # 忽略 prompt
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels,
                    output_hidden_states=True, return_dict=True)
        return out.loss, out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    
        device = model.base.device if hasattr(model, "base") else next(model.parameters()).device
        q = inputs["query"]
        a = inputs["answer"]
        G = inputs["G"]; R = inputs["R"]; I = inputs["I"]

        L_list= []
        batch_size = len(q)
        for i in range(batch_size):
            # G / R / I 各跑一次
            l_g, out_g = self._one_ctx_loss(model, q[i], a[i], G[i], device)
            l_r, out_r = self._one_ctx_loss(model, q[i], a[i], R[i], device)
            l_i, out_i = self._one_ctx_loss(model, q[i], a[i], I[i], device)
            L = torch.stack([l_g, l_r, l_i])
            L_list.append(L)



        L_stack = torch.stack(L_list)                 # [B, 3]
        # 每个样本三种上下文的生成损失
        row_max = L_stack.max(dim=1).values
        row_min = L_stack.min(dim=1).values
        main = row_max.mean()
        reg  = ((row_max - row_min) ** 2).mean()
        loss = main + self.lam * reg
 

        return (loss, None) if return_outputs else loss
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

# ========= 训练参数 =========
KEEP_KEYS = ["query", "G", "R", "I", "answer"]
def passthrough_collator(features):
    batch = {k: [f[k] for f in features] for k in KEEP_KEYS}
    return batch

args = TrainingArguments(
    output_dir="./raat_ckpt",
    learning_rate=1e-5,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  
    bf16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

trainer = RAATTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=passthrough_collator,
    processing_class=tokenizer,
    my_tokenizer=tokenizer,
    lam=0.3,     
    alpha=0.0,    
)

trainer.train()

# 保存
trainer.save_model("./raat_ckpt/final")
tokenizer.save_pretrained("./raat_ckpt/final")

import os, json, time, argparse, random
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

# ====== 你项目里的提示格式 ======
def build_prompt(query, ctx):
    return f"问题：{query}\n参考：{ctx}\n回答："

# ====== 基础 EM / 字符级F1 ======
def normalize(s):
    return "".join(s.strip().split())

def em_score(pred, gold):
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def char_f1(pred, gold):
    p = list(normalize(pred)); g = list(normalize(gold))
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    # 字符多集合交集
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0: return 0.0
    precision = overlap / len(p)
    recall    = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

# ====== BGE-M3 语义相似度（dense cosine） ======
# pip install FlagEmbedding
from FlagEmbedding import BGEM3FlagModel
bge = None
def bge_encode(texts):
    global bge
    if bge is None:
        bge = BGEM3FlagModel('/data3/fujinji/models/bge-m3', use_fp16=True)  # 需联网或本地缓存好
    out = bge.encode(texts, max_length=8192)
    vecs = out["dense_vecs"]
    # 归一化
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs

def cos_sim(a, b):
    return float(np.dot(a, b))

# ====== 生成器 ======
@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens=128, temperature=0.0, top_p=1.0):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature>0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
    # 截掉prompt前缀，只留“回答：”之后
    results = []
    for prompt, full in zip(prompts, outs):
        pos = full.find("回答：")
        ans = full[pos+3:] if pos != -1 else full
        results.append(ans.strip())
    return results

def load_model(model_path, base_path=None):
    tokenizer = AutoTokenizer.from_pretrained(base_path or model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_path or model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    base.config.use_cache = False
    if base_path:  # 有 adapter
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = base
    return model, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default="/data3/fujinji/bge-m3-finetune/RAAT/data/train_10samples.jsonl", help="jsonl 或 json")
    ap.add_argument("--model_a", default="/data3/fujinji/models/Qwen2.5-7B-Instruct", help="原始或Adapter目录（若是adapter，需同时给 base_a）")
    ap.add_argument("--base_a", default=None, help="A 的 base 模型目录（若 model_a 是 adapter）")
    ap.add_argument("--model_b", default="/data3/fujinji/bge-m3-finetune/RAAT/raat_ckpt/checkpoint-1250", help="微调/Adapter目录（若是adapter，需同时给 base_b）")
    ap.add_argument("--base_b", default=None, help="B 的 base 模型目录（若 model_b 是 adapter）")
    ap.add_argument("--out_dir", default="./eval_out")
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    # 读数据
    if args.test_file.endswith(".jsonl"):
        data = [json.loads(l) for l in open(args.test_file, "r", encoding="utf-8")]
    else:
        data = load_dataset("json", data_files=args.test_file)["train"][:]
    data = data[:args.max_samples]

    # 两个模型
    model_a, tok_a = load_model(args.model_a, args.base_a)
    model_b, tok_b = load_model(args.model_b, args.base_b)

    # 两种上下文策略
    def ctx_G_only(ex): return ex["G"]
    def ctx_G_R_I(ex):
        lst = [ex["G"], ex["R"], ex["I"]]
        random.shuffle(lst)
        return "\n\n".join(lst)

    strategies = [("G_only", ctx_G_only), ("G_R_I", ctx_G_R_I)]

    summaries = {}
    for tag, ctx_fn in strategies:
        for name, (model, tok) in {"A": (model_a, tok_a), "B": (model_b, tok_b)}.items():
            ems, f1s, margins, latencies = [], [], [], []
            preds_dump = []
            for i in tqdm(range(0, len(data), args.batch_size), desc=f"{name} - {tag}"):
                batch = data[i:i+args.batch_size]
                prompts = [build_prompt(ex["query"], ctx_fn(ex)) for ex in batch]
                t0 = time.time()
                preds = generate_batch(model, tok, prompts)
                lat = (time.time() - t0) / max(1, len(batch))
                # 统计指标
                for ex, pred in zip(batch, preds):
                    gold = ex["answer"]
                    ems.append(em_score(pred, gold))
                    f1s.append(char_f1(pred, gold))
                    # Anti-Noise Margin（和 G/R/I 的语义相似度）
                    vec_pred, vec_G, vec_R, vec_I = bge_encode([pred, ex["G"], ex["R"], ex["I"]])
                    # print(vec_pred)
                    # print()
                    # print(vec_G)
                    margin = cos_sim(vec_pred, vec_G) - max(cos_sim(vec_pred, vec_R), cos_sim(vec_pred, vec_I))
                    margins.append(margin)
                    latencies.append(lat)
                    preds_dump.append({
                        "query": ex["query"],
                        "gold": gold, "pred": pred,
                        "G": ex["G"], "R": ex["R"], "I": ex["I"],
                        "em": ems[-1], "f1": f1s[-1], "margin": margins[-1]
                    })

            res = {
                "EM": float(np.mean(ems)),
                "F1": float(np.mean(f1s)),
                "AntiNoiseMargin": float(np.mean(margins)),
                "Latency_s_per_ex": float(np.mean(latencies)),
                "N": len(ems),
            }
            summaries[(name, tag)] = res
            output_path = os.path.join(args.out_dir, f"preds_{name}_{tag}.json")
            with open(output_path, "w", encoding="utf-8") as fout:
                json.dump(preds_dump, fout, ensure_ascii=False, indent=2)
            print(f"\n[{name} - {tag}] {res}")

    # 噪声降幅（越小越好）
    def drop(a, b, key):  # a=G_only, b=G_R_I
        return summaries[(a, "G_only")][key] - summaries[(a, "G_R_I")][key]

    table = {
        "A_EM": summaries[("A","G_R_I")]["EM"],
        "B_EM": summaries[("B","G_R_I")]["EM"],
        "A_F1": summaries[("A","G_R_I")]["F1"],
        "B_F1": summaries[("B","G_R_I")]["F1"],
        "A_margin": summaries[("A","G_R_I")]["AntiNoiseMargin"],
        "B_margin": summaries[("B","G_R_I")]["AntiNoiseMargin"],
        "A_drop_EM": drop("A","B","EM") if False else summaries[("A","G_only")]["EM"] - summaries[("A","G_R_I")]["EM"],
        "B_drop_EM": summaries[("B","G_only")]["EM"] - summaries[("B","G_R_I")]["EM"],
        "A_drop_F1": summaries[("A","G_only")]["F1"] - summaries[("A","G_R_I")]["F1"],
        "B_drop_F1": summaries[("B","G_only")]["F1"] - summaries[("B","G_R_I")]["F1"],
    }
    output_path = os.path.join(args.out_dir, "summary.json")
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump({"summaries": summaries, "compare": table}, fout, ensure_ascii=False, indent=2)
    # json.dump({"summaries": summaries, "compare": table}, open(os.path.join(args.out_dir, "summary.json"), "w", ensure_ascii=False, indent=2))
    print("\n== Summary =="); print(json.dumps(table, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()

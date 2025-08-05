import json
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 配置 ==========
INPUT_JSONL = "./data/raat_with_R_I.jsonl"
OUTPUT_JSONL = "./data/raat_with_G_R_C_I.jsonl"
SAMPLE_SIZE = 4500

# 本地模型路径
MODEL_PATH = "/data3/fujinji/models/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 加载模型 ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ========== 读取数据 & 抽样 ==========
data = [json.loads(line) for line in open(INPUT_JSONL, encoding="utf-8")]
random.seed(42)
subset = random.sample(data, min(SAMPLE_SIZE, len(data)))

# ========== 生成反事实 C ==========
def generate_counterfactual_local(golden_text):
    prompt = f"""
请改写以下文本，保持段落整体结构和语气，
但将其中的数字、年份或专有名词替换成错误信息，
确保内容看起来合理但事实错误，不要包含原文的答案。
请直接给出改写后的段落，不要解释，不要包含原文。
请直接给出改写后的段落，不要解释，不要包含原文。
请直接给出改写后的段落，不要解释，不要包含原文。

原始段落：
{golden_text}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            top_p=0.9
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 有些模型会把 prompt 原文一起返回，这里只取后半部分
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text.strip()

# ========== 主循环 ==========
with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for item in tqdm(subset, desc="Generating C"):
        G = item["G"]
        try:
            C = generate_counterfactual_local(G)
        except Exception as e:
            print("Error:", e)
            C = G  # 出错时退回 G
        item["C"] = C
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"生成完成，输出文件: {OUTPUT_JSONL}")

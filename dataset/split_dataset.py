import json
import random
from pathlib import Path

# 修改为你的 jsonl 文件路径
data_path = "/data3/fujinji/bge-m3-finetune/RAAT/data/raat_with_R_I.jsonl"

# 读取 JSONL 文件
data = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# 打乱数据
random.shuffle(data)

# 划分数据集
train_data = data[:10]
# val_data = data[10000:12000]
# test_data = data[12000:14000]

# 输出路径
output_dir = Path("/data3/fujinji/bge-m3-finetune/RAAT/data")  # 当前目录
output_dir.mkdir(parents=True, exist_ok=True)

# 保存到文件
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(output_dir / "train_10samples.jsonl", train_data)
# save_jsonl(output_dir / "val.jsonl", val_data)
# save_jsonl(output_dir / "test.jsonl", test_data)

# print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

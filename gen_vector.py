import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import os

# ========== 配置 ==========
# 输入 jsonl 文件路径
DATA_PATH = "/data3/fujinji/clean_data/train/finetune_data.jsonl"

# 输出目录
OUTPUT_DIR = "./pos_vector_db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 选择文本向量模型
MODEL_NAME = "/data3/fujinji/models/bge-m3"
# 如果你要用中英多语种，可以改成 "moka-ai/m3e-large" 或 "intfloat/multilingual-e5-large"

INDEX_FILE = os.path.join(OUTPUT_DIR, "pos_chunks.index")
DOCS_FILE = os.path.join(OUTPUT_DIR, "pos_chunks.txt")

# ========== 1. 读取 pos ==========
print("Reading jsonl and collecting pos chunks...")
docs = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "pos" in data:
            for p in data["pos"]:
                docs.append(p.strip())

# 去重
docs = list(set(docs))
print(f"Total unique chunks: {len(docs)}")

# ========== 2. 向量化 ==========
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("Encoding all pos chunks...")
embs = model.encode(docs,
                    batch_size=64,
                    show_progress_bar=True,
                    normalize_embeddings=True)  # 归一化方便用内积算相似度
embs = np.array(embs).astype("float32")

# ========== 3. 建立 Faiss index ==========
print("Building Faiss index...")
dimension = embs.shape[1]
index = faiss.IndexFlatIP(dimension)  # 使用内积计算相似度
index.add(embs)
faiss.write_index(index, INDEX_FILE)

# 保存文本映射
with open(DOCS_FILE, "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(doc.replace("\n", " ") + "\n")

print(f"Index saved to {INDEX_FILE}")
print(f"Docs saved to {DOCS_FILE}")

# ========== 4. 测试检索 ==========
def search(query, topk=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), topk)
    results = [(docs[i], float(D[0][j])) for j, i in enumerate(I[0])]
    return results

# 测试一个检索
test_query = "什么是大气环流模式"
print("\nTop 3 results for test query:")
for r, s in search(test_query, topk=3):
    print(f"Score={s:.4f}\t{r}")

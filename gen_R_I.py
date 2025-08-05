import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ========= 配置 =========
MODEL_NAME = "/data3/fujinji/models/bge-m3"
INDEX_PATH = "/data3/fujinji/bge-m3-finetune/RAAT/pos_vector_db/pos_chunks.index"
DOCS_PATH  = "/data3/fujinji/bge-m3-finetune/RAAT/pos_vector_db/pos_chunks.txt"
INPUT_JSONL  = "/data3/fujinji/clean_data/train/finetune_data.jsonl"
OUTPUT_JSONL = "./raat_with_R_I.jsonl"
TOPK = 10

# ========= 加载模型 & 向量库 =========
print("Loading embedding model and faiss index...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
docs = [line.strip() for line in open(DOCS_PATH, encoding="utf-8")]

def search(query, topk=TOPK):
    """向量检索返回 (chunk, score) 列表"""
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), topk)
    return [(docs[i], float(D[0][j])) for j, i in enumerate(I[0])]

# ========= 主循环 =========
print("Generating R and I for each sample...")

with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):
        data = json.loads(line)
        q = data["query"]
        G = data["pos"][0]  # 取第一个正样本段落作为 G

        # 1. R：检索并找第一个不是 G 的
        search_results = search(q, topk=TOPK)
        R = None
        for chunk, score in search_results:
            if chunk != G:  # 可以更严格用相似度或ID判断
                R = chunk
                break
        if R is None:  # fallback
            R = search_results[0][0]

        # 2. I：选择相似度最低的段落
        I = search_results[-1][0]

        new_sample = {
            "query": q,
            # "answer": "",  # 后面你可以补上真正答案
            "G": G,
            "answer": G,
            "R": R,
            "I": I
        }
        fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")

print(f"输出保存到 {OUTPUT_JSONL}")

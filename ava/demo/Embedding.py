import os
import re
import json
import glob
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("demo/bge-m3")
paths = glob.glob("demo/chunked/*")
output_paths = "demo/embeddings-normalized/"
os.makedirs(output_paths,exist_ok=True)


for path in tqdm(paths,desc="embedding"):
    data = json.load(open(path, 'r', encoding='utf-8'))
    for idx, d in enumerate(data):
        embed = model.encode("\n\n".join([d[0],d[1]]), normalize_embeddings=True).tolist()
        data[idx].append(embed)
        torch.cuda.empty_cache()
    save_path = re.split("\\\\|/", path)[-1]
    json.dump(data,open(os.path.join(output_paths, save_path), 'w', encoding="utf-8"),ensure_ascii=False, indent=4)
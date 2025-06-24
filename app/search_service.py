import faiss
import pandas as pd
import numpy as np

df = pd.read_csv("app/index/movie_metadata.csv")
index = faiss.read_index("app/index/faiss_index.bin")

def search(vector, top_k=3):
    vector = np.array(vector).reshape(1, -1)
    D, I = index.search(vector, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append({
            "title": df.iloc[idx]['title'],
            "description": df.iloc[idx]['description'],
            "score": float(dist)
        })
    return results
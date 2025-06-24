import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

df = pd.read_csv("app/data/movies.csv")
df['text'] = df['title'] + " - " + df['genre'] + " - " + df['description']

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['text'].tolist())

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
os.makedirs("app/index", exist_ok=True)
faiss.write_index(index, "app/index/faiss_index.bin")

df.to_csv("app/index/movie_metadata.csv", index=False)
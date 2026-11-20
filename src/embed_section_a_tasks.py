from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/section_a_tasks/task_statements.csv")
OUTPUT_EMBEDDINGS = Path("data/processed/section_a_task_embeddings.npy")
OUTPUT_META = Path("data/processed/section_a_task_meta.csv")
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

df = pd.read_csv(INPUT_PATH)
texts = df["text"].fillna("").tolist()
print(f"Loaded {len(texts)} task statements from Section A")

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

np.save(OUTPUT_EMBEDDINGS, embeddings)
# keep task_id/posting_id alongside the embeddings so rows stay traceable
df[["task_id", "statement_id", "posting_id"]].to_csv(OUTPUT_META, index=False)
print(f"Saved {embeddings.shape} to {OUTPUT_EMBEDDINGS}")
print(f"Saved row metadata to {OUTPUT_META}")

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/esco/skills_en.csv")
OUTPUT_PATH = Path("data/processed/esco_embeddings.npy")
LABELS_OUTPUT = Path("data/processed/esco_skill_labels.csv")
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

df = pd.read_csv(INPUT_PATH)
skill_labels = df["preferredLabel"].dropna().tolist()
print(f"Loaded {len(skill_labels)} ESCO skills")

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(skill_labels, show_progress_bar=True, batch_size=32)

np.save(OUTPUT_PATH, embeddings)
pd.Series(skill_labels).to_csv(LABELS_OUTPUT, index=False, header=["skill_label"])
print(f"Saved {embeddings.shape} to {OUTPUT_PATH}")
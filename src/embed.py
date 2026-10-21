from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # swap to "Qwen/Qwen3-Embedding-8B" on GPU
INPUT_PATH = Path("data/processed/unique_statements.csv")
OUTPUT_PATH = Path("data/processed/embeddings.npy")


def main():
    statements = pd.read_csv(INPUT_PATH)["statement"].tolist()
    print(f"Loaded {len(statements)} statements")

    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    embeddings = model.encode(statements, show_progress_bar=True, batch_size=64)
    print(f"Embeddings shape: {embeddings.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

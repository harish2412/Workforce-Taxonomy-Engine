from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
LABELS_PATH = Path("data/processed/cluster_labels.npy")
STATEMENTS_PATH = Path("data/processed/unique_statements.csv")
OUTPUT_PATH = Path("outputs/figures/cluster_visualization.png")


def main():
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    statements = pd.read_csv(STATEMENTS_PATH)["statement"]

    reducer = umap.UMAP(random_state=42)
    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 9))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=6, alpha=0.7)
    plt.title(f"Skill statement clusters (k={len(set(labels))})")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")

    df = pd.DataFrame({"statement": statements, "cluster": labels})
    print("\nSample clusters:")
    for cid in sorted(set(labels))[:5]:
        examples = df[df["cluster"] == cid]["statement"].head(5).tolist()
        print(f"  Cluster {cid}: {examples}")


if __name__ == "__main__":
    main()

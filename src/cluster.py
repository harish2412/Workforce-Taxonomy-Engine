import gc
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_PATH = Path("data/processed/embeddings.npy")
OUTPUT_PATH = Path("data/processed/cluster_labels.npy")
K_VALUES_TO_TRY = [20, 30, 50, 80]


def main():
    embeddings = np.load(INPUT_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")

    best_k, best_score, best_labels = None, -1, None

    for k in K_VALUES_TO_TRY:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        print(f"k={k}: silhouette score = {score:.4f}")

        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

        del km
        gc.collect()

    print(f"\nBest: k={best_k} (silhouette={best_score:.4f})")
    np.save(OUTPUT_PATH, best_labels)
    print(f"Saved cluster labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

INPUT_PATH = Path("data/processed/embeddings.npy")
BEST_K = 30
SEEDS = [1, 2, 3]


def main():
    embeddings = np.load(INPUT_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")

    labelings = []
    for seed in SEEDS:
        km = KMeans(n_clusters=BEST_K, random_state=seed, n_init=10)
        labelings.append(km.fit_predict(embeddings))
        print(f"seed {seed}: done")

    print("\nPairwise Adjusted Rand Index:")
    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            ari = adjusted_rand_score(labelings[i], labelings[j])
            print(f"  seed {SEEDS[i]} vs seed {SEEDS[j]}: ARI = {ari:.4f}")


if __name__ == "__main__":
    main()

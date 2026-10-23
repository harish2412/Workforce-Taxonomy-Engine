from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")

embeddings = np.load(EMBEDDINGS_PATH)
print(f"Loaded embeddings: {embeddings.shape}")

for k in [30, 80]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    out_path = Path(f"data/processed/cluster_labels_k{k}.npy")
    np.save(out_path, labels)
    print(f"k={k}: saved to {out_path}")
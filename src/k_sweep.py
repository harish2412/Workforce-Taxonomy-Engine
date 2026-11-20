from pathlib import Path
import gc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
RESULTS_PATH = Path("data/processed/k_sweep_results.csv")
K_VALUES_TO_TRY = list(range(50, 201, 10))  # 50, 60, 70, ..., 200

embeddings = np.load(EMBEDDINGS_PATH)
print(f"Loaded embeddings: {embeddings.shape}")

results = []
for k in K_VALUES_TO_TRY:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, sample_size=5000, random_state=42)
    print(f"k={k}: silhouette = {score:.4f}")
    results.append({"k": k, "silhouette": score})

    np.save(f"data/processed/cluster_labels_k{k}.npy", labels)
    del km
    gc.collect()

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

best_row = results_df.loc[results_df["silhouette"].idxmax()]
print(f"\nBest: k={int(best_row['k'])} (silhouette={best_row['silhouette']:.4f})")
print(f"Full results saved to {RESULTS_PATH}")
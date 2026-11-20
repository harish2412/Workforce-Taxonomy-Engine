from pathlib import Path

import numpy as np
import pandas as pd

SWEEP_RESULTS = Path("data/processed/k_sweep_results.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("data/processed/k_sweep_full_analysis.csv")
TINY_THRESHOLD = 10  # clusters smaller than this are flagged as fragmented


def main():
    sweep = pd.read_csv(SWEEP_RESULTS)

    rows = []
    for _, row in sweep.iterrows():
        k = int(row["k"])
        labels_path = PROCESSED_DIR / f"cluster_labels_k{k}.npy"
        if not labels_path.exists():
            print(f"Skipping k={k}, {labels_path} not found")
            continue

        labels = np.load(labels_path)
        sizes = pd.Series(labels).value_counts()
        tiny = (sizes < TINY_THRESHOLD).sum()

        rows.append({
            "k": k,
            "silhouette": row["silhouette"],
            "min_size": int(sizes.min()),
            "median_size": sizes.median(),
            "max_size": int(sizes.max()),
            "clusters_under_10": int(tiny),
            "pct_fragmented": round(100 * tiny / k, 1),
        })

    result_df = pd.DataFrame(rows).sort_values("k")
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(result_df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")

    # flag a reasonable "sweet spot": high silhouette, low fragmentation
    candidates = result_df[result_df["pct_fragmented"] < 10]
    if len(candidates):
        best = candidates.loc[candidates["silhouette"].idxmax()]
        print(f"\nBest silhouette among k values with <10% fragmented clusters: "
              f"k={int(best['k'])} (silhouette={best['silhouette']:.4f}, "
              f"{best['pct_fragmented']}% fragmented)")
    else:
        print("\nEvery tested k has 10%+ fragmented clusters — worth reviewing the full table above.")


if __name__ == "__main__":
    main()
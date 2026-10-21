from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

GOLDEN_PATH = Path("data/processed/golden_set.csv")


def main():
    df = pd.read_csv(GOLDEN_PATH)

    if df["final_category"].isna().any() or (df["final_category"] == "").any():
        print("Fill in 'final_category' for every row before running this.")
        return

    ami = adjusted_mutual_info_score(df["final_category"], df["cluster"])
    print(f"Adjusted Mutual Information: {ami:.3f}")

    print("\nPurity per cluster:")
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        top_cat = sub["final_category"].mode()[0]
        purity = (sub["final_category"] == top_cat).mean()
        print(f"  Cluster {cid} (n={len(sub)}): {purity:.0%} '{top_cat}'")


if __name__ == "__main__":
    main()

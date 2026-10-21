from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

GOLDEN_PATH = Path("data/processed/golden_set.csv")


def main():
    df = pd.read_csv(GOLDEN_PATH)

    if df["reviewer1_category"].isna().any() or df["reviewer2_category"].isna().any():
        print("Fill in both reviewer columns before scoring.")
        return

    kappa = cohen_kappa_score(df["reviewer1_category"], df["reviewer2_category"])
    agree_rate = (df["reviewer1_category"] == df["reviewer2_category"]).mean()

    print(f"Raw agreement rate: {agree_rate:.1%}")
    print(f"Cohen's kappa: {kappa:.3f}")

    disagreements = df[df["reviewer1_category"] != df["reviewer2_category"]]
    print(f"\n{len(disagreements)} disagreements to reconcile:")
    for _, row in disagreements.iterrows():
        print(f"  '{row['statement']}' -> R1: {row['reviewer1_category']} | R2: {row['reviewer2_category']}")


if __name__ == "__main__":
    main()

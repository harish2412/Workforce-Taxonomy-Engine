from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, cohen_kappa_score

GOLDEN_PATH = Path("data/processed/golden_set.csv")
ESCO_CATEGORIES = [
    "agriculture, forestry, fisheries and veterinary", "arts and humanities",
    "assisting and caring", "business, administration and law",
    "classical languages", "communication, collaboration and creativity",
    "constructing", "core skills and competences", "education",
    "engineering, manufacturing and construction",
    "generic programmes and qualifications", "handling and moving",
    "health and welfare", "information and communication technologies (icts)",
    "information skills", "languages", "life skills and competences",
    "management skills", "natural sciences, mathematics and statistics",
    "physical and manual skills and competences",
    "self-management skills and competences", "services",
    "social and communication skills and competences",
    "social sciences, journalism and information",
    "thinking skills and competences", "working with computers",
    "working with machinery and specialised equipment", "none of these / unclear",
]


def build_golden_set(statements_path="data/processed/unique_statements.csv",
                      labels_path="data/processed/cluster_labels.npy",
                      per_cluster=5, seed=42):
    statements = pd.read_csv(statements_path)["statement"]
    labels = np.load(labels_path)
    df = pd.DataFrame({"statement": statements, "cluster": labels})

    sampled = []
    for cid in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cid]
        n = min(per_cluster, len(cluster_df))
        sampled.append(cluster_df.sample(n, random_state=seed))

    golden = pd.concat(sampled, ignore_index=True)
    golden["reviewer1_category"] = ""
    golden["reviewer2_category"] = ""
    golden["final_category"] = ""
    golden.to_csv(GOLDEN_PATH, index=False)
    print(f"Sampled {len(golden)} statements across {df['cluster'].nunique()} clusters")
    print(f"Saved to {GOLDEN_PATH}")


def score_agreement():
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


def evaluate_clusters():
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
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build_golden_set, "score": score_agreement, "evaluate": evaluate_clusters}[step]()
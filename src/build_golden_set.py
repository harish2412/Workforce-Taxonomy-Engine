from pathlib import Path

import numpy as np
import pandas as pd

STATEMENTS_PATH = Path("data/processed/unique_statements.csv")
LABELS_PATH = Path("data/processed/cluster_labels.npy")
OUTPUT_PATH = Path("data/processed/golden_set.csv")
PER_CLUSTER_SAMPLE = 5
RANDOM_SEED = 42

ESCO_CATEGORIES = [
    "agriculture, forestry, fisheries and veterinary",
    "arts and humanities",
    "assisting and caring",
    "business, administration and law",
    "classical languages",
    "communication, collaboration and creativity",
    "constructing",
    "core skills and competences",
    "education",
    "engineering, manufacturing and construction",
    "generic programmes and qualifications",
    "handling and moving",
    "health and welfare",
    "information and communication technologies (icts)",
    "information skills",
    "languages",
    "life skills and competences",
    "management skills",
    "natural sciences, mathematics and statistics",
    "physical and manual skills and competences",
    "self-management skills and competences",
    "services",
    "social and communication skills and competences",
    "social sciences, journalism and information",
    "thinking skills and competences",
    "working with computers",
    "working with machinery and specialised equipment",
    "none of these / unclear",
]


def main():
    statements = pd.read_csv(STATEMENTS_PATH)["statement"]
    labels = np.load(LABELS_PATH)
    df = pd.DataFrame({"statement": statements, "cluster": labels})

    sampled = []
    for cid in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cid]
        n = min(PER_CLUSTER_SAMPLE, len(cluster_df))
        sampled.append(cluster_df.sample(n, random_state=RANDOM_SEED))

    golden = pd.concat(sampled, ignore_index=True)
    golden["reviewer1_category"] = ""
    golden["reviewer2_category"] = ""
    golden["final_category"] = ""
    golden.to_csv(OUTPUT_PATH, index=False)

    print(f"Sampled {len(golden)} statements across {df['cluster'].nunique()} clusters")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nValid categories:")
    for c in ESCO_CATEGORIES:
        print(f"  - {c}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/cluster_labels_k30.npy")
STATEMENTS_PATH = Path("data/processed/unique_statements_cleaned.csv")
SAMPLE_SIZE = 10

labels = np.load(LABELS_PATH)
statements = pd.read_csv(STATEMENTS_PATH)["statement"]
df = pd.DataFrame({"statement": statements, "cluster": labels})

sizes = df["cluster"].value_counts().sort_values(ascending=False)

for cid in sizes.index:
    examples = df[df["cluster"] == cid]["statement"].sample(min(SAMPLE_SIZE, sizes[cid]), random_state=42)
    print(f"\n=== Cluster {cid} (n={sizes[cid]}) ===")
    for ex in examples:
        print(f"  {ex}")
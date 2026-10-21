"""
Sprint 3, step 1 — filter and dedupe the extracted skill statements
before embedding.

Keeps only high and medium confidence statements (per the team's own
Sprint 2 quality assessment, low confidence noun chunks are mostly
noise), then collapses to unique phrases, since we cluster unique
skills, not every repeated occurrence across postings.

Input:  data/extracted/skill_statements_sample.csv
Output: data/processed/unique_statements.csv
"""

import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/extracted/skill_statements_sample.csv")
OUTPUT_PATH = Path("data/processed/unique_statements.csv")


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} statements")

    filtered = df[df["confidence"].isin(["high", "medium"])]
    print(f"Kept {len(filtered)} at high/medium confidence")

    unique = (
        filtered["statement_norm"]
        .dropna()
        .str.strip()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"Unique statements: {len(unique)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unique.to_csv(OUTPUT_PATH, index=False, header=["statement"])
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

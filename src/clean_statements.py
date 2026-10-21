from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/extracted/skill_statements_sample.csv")
OUTPUT_PATH = Path("data/processed/unique_statements.csv")


def main():
    df = pd.read_csv(INPUT_PATH)
    filtered = df[df["confidence"].isin(["high", "medium"])]
    unique = filtered["statement_norm"].dropna().str.strip().drop_duplicates().reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unique.to_csv(OUTPUT_PATH, index=False, header=["statement"])

    print(f"Loaded {len(df)} statements")
    print(f"Kept {len(filtered)} at high/medium confidence")
    print(f"Unique statements: {len(unique)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

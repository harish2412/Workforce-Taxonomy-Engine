from pathlib import Path

import pandas as pd

K = 100
HIERARCHY_PATH = Path(f"outputs/topics/hierarchy_k{K}.csv")
OUTPUT_PATH = Path(f"outputs/skill_taxonomy_k{K}_final.csv")


def main():
    df = pd.read_csv(HIERARCHY_PATH)

    out = pd.DataFrame({
        "skill_id": df["topic"],
        "skill_name": df["name"],
        "short_description": df["description"],
        "parent_group": df["l2_name"],
        "top_level_domain": df["l3_name"],
        "statement_count": df["size"],
        "status": df["status"],
    })

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} named topics to {OUTPUT_PATH}")
    print(out["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
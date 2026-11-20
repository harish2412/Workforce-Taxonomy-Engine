from pathlib import Path

import pandas as pd

TASK_TAXONOMY_PATH = Path("data/section_a_tasks/task_taxonomy.csv")
SIMILARITY_PATH = Path("outputs/task_skill_similarity_top5.csv")
OUTPUT_PATH = Path("outputs/gold_set_draft_30.csv")

tasks = pd.read_csv(TASK_TAXONOMY_PATH)
sims = pd.read_csv(SIMILARITY_PATH)
top1 = sims[sims["rank"] == 1]

# one draft initiative per real task group (l2), named after the group itself
groups = tasks[["name", "task_id"]].rename(columns={"name": "l2_name"}).copy()
groups["task_id"] = groups["task_id"].apply(lambda x: [x])
groups = groups.sort_values("task_id", key=lambda x: x.str.len(), ascending=False)

rows = []
for i, (_, row) in enumerate(groups.iterrows()):
    if i >= 30:
        break
    group_tasks = tasks[tasks["task_id"].isin(row["task_id"])]
    task_names = "; ".join(group_tasks["name"].head(3))
    related_skills = top1[top1["task_id"].isin(row["task_id"])]["skill_name"].unique()

    rows.append({
        "initiative_id": i + 1,
        "initiative_name": f"Initiative involving: {row['l2_name']}",
        "expected_tasks": task_names,
        "expected_skills": "; ".join(related_skills[:5]),
        "source_task_group": row["l2_name"],
        "reviewed": "no",
    })

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_PATH, index=False)
print(f"Drafted {len(out)} initiatives to {OUTPUT_PATH}")
print("Every row needs a human-written initiative_name and a quick sanity check before use.")
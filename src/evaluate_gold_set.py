from pathlib import Path

import pandas as pd

GOLD_SET_PATH = Path("outputs/gold_set_revised.csv")
SIMILARITY_PATH = Path("outputs/task_skill_similarity_top5.csv")
OUTPUT_PATH = Path("outputs/gold_set_evaluation_top3.csv")

gold = pd.read_csv(GOLD_SET_PATH)
sims = pd.read_csv(SIMILARITY_PATH)

rows = []
for _, row in gold.iterrows():
    expected_skills = set(s.strip() for s in str(row["expected_skills"]).split(";"))
    expected_tasks = set(t.strip() for t in str(row["expected_tasks"]).split(";"))

    predicted_skills = set()
    for task_name in expected_tasks:
        matches = sims[(sims["task_name"] == task_name) & (sims["rank"] <= 3)]
        predicted_skills.update(matches["skill_name"].tolist())

    true_positives = expected_skills & predicted_skills
    precision = len(true_positives) / len(predicted_skills) if predicted_skills else 0
    recall = len(true_positives) / len(expected_skills) if expected_skills else 0

    rows.append({
        "initiative_id": row["initiative_id"],
        "initiative_name": row["initiative_name"],
        "expected_skills": len(expected_skills),
        "predicted_skills": len(predicted_skills),
        "correct_matches": len(true_positives),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    })

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_PATH, index=False)
print(out[["precision", "recall"]].mean().round(3))
print(f"\nSaved per-initiative results to {OUTPUT_PATH}")

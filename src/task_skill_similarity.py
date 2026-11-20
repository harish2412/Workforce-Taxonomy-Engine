from pathlib import Path

import numpy as np
import pandas as pd

TASK_EMB_PATH = Path("data/processed/section_a_task_embeddings.npy")
TASK_META_PATH = Path("data/processed/section_a_task_meta.csv")
TASK_TAXONOMY_PATH = Path("data/section_a_tasks/task_taxonomy.csv")

SKILL_EMB_PATH = Path("data/processed/embeddings.npy")
SKILL_LABELS_PATH = Path("data/processed/topic_labels_k100_refined.npy")
SKILL_STATEMENTS_PATH = Path("data/processed/unique_statements_cleaned.csv")
SKILL_HIERARCHY_PATH = Path("outputs/topics/hierarchy_k100.csv")

OUTPUT_PATH = Path("outputs/task_skill_similarity_top5.csv")
TOP_N = 5


def unit(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms


def main():
    # average Section A's per-statement task embeddings up to one vector per task
    task_emb = np.load(TASK_EMB_PATH)
    task_meta = pd.read_csv(TASK_META_PATH)
    task_names = pd.read_csv(TASK_TAXONOMY_PATH)[["task_id", "name", "status"]]

    task_centers = {}
    for tid in sorted(task_meta["task_id"].unique()):
        mask = (task_meta["task_id"] == tid).to_numpy()
        task_centers[tid] = task_emb[mask].mean(axis=0)

    # average our own skill statement embeddings up to one vector per topic
    skill_emb = np.load(SKILL_EMB_PATH)
    skill_labels = np.load(SKILL_LABELS_PATH)
    hierarchy = pd.read_csv(SKILL_HIERARCHY_PATH)[["topic", "name", "status"]]

    skill_centers = {}
    for topic_id in hierarchy["topic"]:
        mask = skill_labels == topic_id
        if mask.sum() > 0:
            skill_centers[topic_id] = skill_emb[mask].mean(axis=0)

    task_ids = list(task_centers.keys())
    skill_ids = list(skill_centers.keys())
    task_matrix = unit(np.array([task_centers[t] for t in task_ids]))
    skill_matrix = unit(np.array([skill_centers[s] for s in skill_ids]))

    sims = task_matrix @ skill_matrix.T  # (n_tasks, n_skills)

    task_name_map = dict(zip(task_names["task_id"], task_names["name"]))
    task_status_map = dict(zip(task_names["task_id"], task_names["status"]))
    skill_name_map = dict(zip(hierarchy["topic"], hierarchy["name"]))
    skill_status_map = dict(zip(hierarchy["topic"], hierarchy["status"]))

    rows = []
    for i, tid in enumerate(task_ids):
        order = np.argsort(sims[i])[::-1][:TOP_N]
        for rank, j in enumerate(order):
            sid = skill_ids[j]
            rows.append({
                "task_id": tid,
                "task_name": task_name_map.get(tid, f"Task {tid}"),
                "task_status": task_status_map.get(tid, ""),
                "rank": rank + 1,
                "skill_id": sid,
                "skill_name": skill_name_map.get(sid, f"Topic {sid}"),
                "skill_status": skill_status_map.get(sid, ""),
                "similarity": round(float(sims[i, j]), 4),
            })

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"{len(task_ids)} tasks compared against {len(skill_ids)} skill topics")
    print(f"Top {TOP_N} skill matches saved per task to {OUTPUT_PATH}")
    print(f"\nMean top-1 similarity: {out[out['rank'] == 1]['similarity'].mean():.3f}")


if __name__ == "__main__":
    main()
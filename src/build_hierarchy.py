"""Build parent levels above the refined topics with hierarchical clustering, then
have Claude name each parent and flag ones that look over-merged.

Run after refine_topics.py, e.g. `python src/build_hierarchy.py 100`.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster, linkage

from name_topics import MAX_WORKERS, name_cluster

K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
LABELS_PATH = Path(f"data/processed/topic_labels_k{K}_refined.npy")
TOPICS_PATH = Path(f"outputs/topics/topics_k{K}_refined.csv")
LEAVES_OUT = Path(f"outputs/topics/hierarchy_k{K}.csv")
PARENTS_OUT = Path(f"outputs/topics/hierarchy_parents_k{K}.csv")  # also the naming cache
TREE_OUT = Path(f"outputs/topics/hierarchy_k{K}.txt")

# Number of groups at each parent level, top level last. Cuts of one tree are nested,
# so every level-2 group sits entirely inside one level-3 domain.
LEVEL_SIZES = {2: 25, 3: 8}
# Ward keeps groups evenly sized. Average linkage was tried first and chained 80% of
# statements into one domain while outlier topics became single-topic domains.
# Centers are unit vectors, so Euclidean distance ranks pairs the same as cosine.
LINKAGE_METHOD = "ward"

SYSTEM_PROMPT = """You name parent categories in a hierarchical taxonomy of workplace skills
built from job postings. Each parent groups several child topics that an algorithm found
to be similar. You get the child topic names, descriptions and sizes.

Return:
- name: a short parent category name (2-5 words), broader than its children, e.g.
  "Software Engineering & IT" over "Databases & SQL" and "Cloud Platforms".
- description: one sentence on what the category covers.
- coherence: one of
  - "coherent": the children belong together under one category
  - "over_merged": some children do not belong with the others
- reason: one sentence. For "over_merged", name the children that don't fit."""

PARENT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "coherence": {"type": "string", "enum": ["coherent", "over_merged"]},
        "reason": {"type": "string"},
    },
    "required": ["name", "description", "coherence", "reason"],
    "additionalProperties": False,
}


def topic_centers(embeddings, labels, topic_ids):
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    centers = np.array([unit[labels == t].mean(axis=0) for t in topic_ids])
    return centers / np.linalg.norm(centers, axis=1, keepdims=True)


def parent_prompt(children):
    lines = [f"- {c['name']} ({c['size']} statements): {c['description']}" for c in children]
    return f"Child topics ({len(children)}):\n" + "\n".join(lines)


def main():
    load_dotenv()
    client = anthropic.Anthropic()

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    topics = pd.read_csv(TOPICS_PATH)
    leaves = topics[topics["status"] != "dropped"].reset_index(drop=True)
    print(f"Building hierarchy over {len(leaves)} topics")

    tree = linkage(topic_centers(embeddings, labels, leaves["topic"]), method=LINKAGE_METHOD, metric="euclidean")
    for level, n in LEVEL_SIZES.items():
        leaves[f"l{level}"] = fcluster(tree, t=n, criterion="maxclust")

    # Parent names are cached by their exact set of child topics, so a rerun only pays
    # for groups that changed
    cache = pd.read_csv(PARENTS_OUT) if PARENTS_OUT.exists() else pd.DataFrame()
    cached = {r["children_ids"]: r for r in cache.to_dict("records")} if len(cache) else {}

    parents = []
    child_level = leaves.assign(id=leaves["topic"])  # level 1: the topics themselves
    for level in LEVEL_SIZES:
        groups = []
        for gid, members in leaves.groupby(f"l{level}"):
            child_ids = members[f"l{level - 1}"] if level > 2 else members["topic"]
            children = child_level[child_level["id"].isin(child_ids)].to_dict("records")
            key = " ".join(str(t) for t in sorted(members["topic"]))
            groups.append({"id": f"L{level}-{gid}", "gid": gid, "level": level, "children": children,
                           "size": int(members["size"].sum()), "children_ids": key})

        def name_group(g):
            if len(g["children"]) == 1:  # a lone child passes its name up unchanged
                only = g["children"][0]
                return {"name": only["name"], "description": only["description"],
                        "coherence": "single_child", "reason": "Only one child topic."}
            if g["children_ids"] in cached:
                return {k: cached[g["children_ids"]][k] for k in ["name", "description", "coherence", "reason"]}
            return name_cluster(client, g["id"], parent_prompt(g["children"]), SYSTEM_PROMPT, PARENT_SCHEMA)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            named = list(pool.map(name_group, groups))

        for g, n in zip(groups, named):
            n = {k: v for k, v in n.items() if k != "cluster"}
            parents.append({"id": g["id"], "level": level, "size": g["size"],
                            "n_children": len(g["children"]), **n,
                            "children": "; ".join(c["name"] for c in g["children"]),
                            "children_ids": g["children_ids"]})
            leaves.loc[leaves[f"l{level}"] == g["gid"], f"l{level}_name"] = n["name"]
            print(f"  {g['id']} ({len(g['children'])} children, n={g['size']}): {n['name']} [{n['coherence']}]")

        # This level's groups become the children of the next level
        child_level = pd.DataFrame([{"id": g["gid"], "name": n["name"], "description": n["description"],
                                     "size": g["size"]} for g, n in zip(groups, named)])

    parents_df = pd.DataFrame(parents)
    parents_df.to_csv(PARENTS_OUT, index=False)
    leaves.to_csv(LEAVES_OUT, index=False)

    # Readable tree for manual review: domain > group > topic
    top, mid = max(LEVEL_SIZES), min(LEVEL_SIZES)
    by_id = parents_df.set_index("id")
    lines = []
    for d, dom in leaves.groupby(f"l{top}"):
        p = by_id.loc[f"L{top}-{d}"]
        lines.append(f"{p['name']}  [{p['size']} statements, {p['coherence']}]")
        for g, grp in dom.groupby(f"l{mid}"):
            p = by_id.loc[f"L{mid}-{g}"]
            lines.append(f"    {p['name']}  [{p['size']}, {p['coherence']}]")
            for t in grp.sort_values("size", ascending=False).itertuples():
                flag = "" if t.status == "keep" else f"  ({t.status})"
                lines.append(f"        - {t.name}  [{t.size}]{flag}")
        lines.append("")
    TREE_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved {LEAVES_OUT}, {PARENTS_OUT} and {TREE_OUT}")
    print(parents_df.groupby(["level", "coherence"]).size().to_string())


if __name__ == "__main__":
    main()

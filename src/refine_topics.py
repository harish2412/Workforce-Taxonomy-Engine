"""Refine the named k-means topics:
1. drop topics Claude labelled "noise"
2. split "mixed" topics by re-clustering their statements, then name the parts
3. keep "vague" topics, marked low-specificity

Run after name_topics.py, e.g. `python src/refine_topics.py 100`.
"""
import sys
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from name_topics import cluster_prompt, name_cluster, top_keywords

K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
LABELS_PATH = Path(f"data/processed/cluster_labels_k{K}.npy")
STATEMENTS_PATH = Path("data/processed/unique_statements_cleaned.csv")
TOPICS_PATH = Path(f"outputs/topics/topic_names_k{K}.csv")
SUBTOPICS_PATH = Path(f"outputs/topics/subtopic_names_k{K}.csv")  # cache, so reruns don't pay again
OUT_TOPICS_PATH = Path(f"outputs/topics/topics_k{K}_refined.csv")
OUT_LABELS_PATH = Path(f"data/processed/topic_labels_k{K}_refined.npy")

SPLIT_K_VALUES = [2, 3]
DROPPED = -1  # label for statements in dropped (noise) topics
STATUS = {"good": "keep", "vague": "low_specificity", "noise": "dropped", "mixed": "mixed_unresolved"}


def split_cluster(embeddings):
    """Re-cluster one topic's statements, picking k from SPLIT_K_VALUES by silhouette."""
    best = None
    for k in SPLIT_K_VALUES:
        sub = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        score = silhouette_score(embeddings, sub, metric="cosine")
        if best is None or score > best[0]:
            best = (score, sub)
    return best[1]


def main():
    load_dotenv()
    client = anthropic.Anthropic()

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    statements = pd.read_csv(STATEMENTS_PATH)["statement"].to_numpy()
    topics = pd.read_csv(TOPICS_PATH)

    # Topic ids: original clusters keep their id, sub-topics get new ids from K upward
    refined = labels.copy()
    parent = {c: c for c in topics["cluster"]}
    next_id = K
    for cid in topics.loc[topics["quality"] == "mixed", "cluster"]:
        idx = np.where(labels == cid)[0]
        sub = split_cluster(embeddings[idx])
        for s in np.unique(sub):
            refined[idx[sub == s]] = next_id
            parent[next_id] = cid
            next_id += 1
        print(f"Split cluster {cid} into {len(np.unique(sub))} parts")

    sub_ids = [t for t in parent if t >= K]
    keywords = top_keywords(statements, refined)

    # Name sub-topics, reusing cached names from earlier runs. The cache is keyed by
    # (parent, size) as well as id, so it is ignored if the split ever changes.
    cache = pd.read_csv(SUBTOPICS_PATH) if SUBTOPICS_PATH.exists() else pd.DataFrame()
    cached = {(r.topic, r.parent, r.size): r for r in cache.itertuples()} if len(cache) else {}
    rng = np.random.default_rng(42)
    sub_rows = []
    for t in sub_ids:
        size = int((refined == t).sum())
        hit = cached.get((t, parent[t], size))
        if hit is not None:
            sub_rows.append(hit._asdict() | {"topic": t})
            continue
        try:
            row = name_cluster(client, t, cluster_prompt(t, statements, refined, embeddings, keywords, rng))
        except (anthropic.APIError, RuntimeError) as e:
            print(f"Sub-topic {t} failed: {e}")
            continue
        row = {"topic": t, "parent": parent[t], "size": size, "keywords": ", ".join(keywords[t]),
               **{k: row[k] for k in ["name", "description", "quality", "reason"]}}
        sub_rows.append(row)
        print(f"  {t} (from {parent[t]}, n={size}): {row['name']} ({row['quality']})")

    subtopics = pd.DataFrame(sub_rows).drop(columns=["Index"], errors="ignore")
    SUBTOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    subtopics.to_csv(SUBTOPICS_PATH, index=False)

    # Combine: original non-mixed topics + named sub-topics
    originals = topics[topics["quality"] != "mixed"].rename(columns={"cluster": "topic"})
    originals = originals.assign(parent=originals["topic"])
    table = pd.concat([originals, subtopics], ignore_index=True)
    table["status"] = table["quality"].map(STATUS)
    table["size"] = [int((refined == t).sum()) for t in table["topic"]]

    # Statements in dropped topics get label -1 so later steps can skip them
    for t in table.loc[table["status"] == "dropped", "topic"]:
        refined[refined == t] = DROPPED

    cols = ["topic", "parent", "status", "quality", "size", "name", "description", "reason", "keywords"]
    table[cols].sort_values("topic").to_csv(OUT_TOPICS_PATH, index=False)
    np.save(OUT_LABELS_PATH, refined)

    print(f"\nSaved {len(table)} topics to {OUT_TOPICS_PATH}")
    print(f"Saved statement labels to {OUT_LABELS_PATH} ({DROPPED} = dropped)")
    summary = table.groupby("status").agg(topics=("topic", "count"), statements=("size", "sum"))
    print(summary.to_string())


if __name__ == "__main__":
    main()

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer

K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
LABELS_PATH = Path(f"data/processed/cluster_labels_k{K}.npy")
STATEMENTS_PATH = Path("data/processed/unique_statements_cleaned.csv")
OUTPUT_PATH = Path(f"outputs/topics/topic_names_k{K}.csv")

MODEL = "claude-opus-5-5"
N_KEYWORDS = 15
N_CENTRAL = 15  # statements closest to the cluster center, the "typical" members
N_RANDOM = 10   # random other members, to expose mixed clusters
MAX_WORKERS = 8

SYSTEM_PROMPT = """You name topics in a taxonomy of workplace skills built from job postings.
Each topic is a cluster of short skill statements extracted from postings. You get the
cluster's top keywords, its most typical statements, and a random sample of other members.

Return:
- name: a short, specific topic name (2-6 words), as a taxonomy label would read, e.g.
  "Cloud Infrastructure & DevOps". Avoid vague names like "General Skills" or "Miscellaneous".
- description: one or two sentences on what skills the topic covers.
- quality: one of
  - "good": the members share one clear skill theme
  - "vague": members are related but too generic to be a useful skill topic
    (e.g. "business needs", "work environment")
  - "mixed": two or more unrelated themes share the cluster; say which in the reason
  - "noise": mostly extraction fragments, not skills (e.g. "leading the internal", "about big")
- reason: one sentence justifying the quality label. For "good" topics, note any outliers."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "quality": {"type": "string", "enum": ["good", "vague", "mixed", "noise"]},
        "reason": {"type": "string"},
    },
    "required": ["name", "description", "quality", "reason"],
    "additionalProperties": False,
}


def top_keywords(statements, labels):
    # c-TF-IDF: treat each cluster as one document, score terms that are frequent in
    # the cluster but rare across clusters
    cluster_ids = np.unique(labels)
    docs = [" ".join(statements[labels == c]) for c in cluster_ids]
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    counts = vectorizer.fit_transform(docs).toarray().astype(float)
    tf = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    idf = np.log(1 + counts.sum() / counts.sum(axis=0).clip(min=1))
    scores = tf * idf
    terms = vectorizer.get_feature_names_out()
    return {c: [terms[i] for i in scores[row].argsort()[::-1][:N_KEYWORDS]]
            for row, c in enumerate(cluster_ids)}


def cluster_prompt(cid, statements, labels, embeddings, keywords, rng):
    idx = np.where(labels == cid)[0]
    members = embeddings[idx] / np.linalg.norm(embeddings[idx], axis=1, keepdims=True)
    center = members.mean(axis=0)
    order = np.argsort(-(members @ center))
    central = idx[order[:N_CENTRAL]]
    rest = idx[order[N_CENTRAL:]]
    random = rng.choice(rest, min(N_RANDOM, len(rest)), replace=False) if len(rest) else []

    return (
        f"Cluster {cid} ({len(idx)} statements)\n\n"
        f"Top keywords: {', '.join(keywords[cid])}\n\n"
        "Most typical statements:\n" + "\n".join(f"- {statements[i]}" for i in central) + "\n\n"
        "Random other members:\n" + "\n".join(f"- {statements[i]}" for i in random)
    )


def name_cluster(client, cid, prompt, system=SYSTEM_PROMPT, schema=OUTPUT_SCHEMA):
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"cluster {cid}: request refused")
    text = next(b.text for b in response.content if b.type == "text")
    return {"cluster": cid, **json.loads(text)}


def main():
    load_dotenv()
    client = anthropic.Anthropic()

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    statements = pd.read_csv(STATEMENTS_PATH)["statement"].to_numpy()
    print(f"Loaded {len(statements)} statements in {K} clusters")

    # Resume: skip clusters already named in a previous run
    done = pd.read_csv(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
    named = set(done["cluster"]) if len(done) else set()
    todo = [c for c in range(K) if c not in named]
    print(f"{len(named)} already named, {len(todo)} to go")

    keywords = top_keywords(statements, labels)
    rng = np.random.default_rng(42)
    prompts = {c: cluster_prompt(c, statements, labels, embeddings, keywords, rng) for c in todo}

    results = done.to_dict("records")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(name_cluster, client, c, p): c for c, p in prompts.items()}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                row = future.result()
            except (anthropic.APIError, RuntimeError) as e:
                print(f"Cluster {cid} failed: {e}")
                continue
            row["size"] = int((labels == cid).sum())
            row["keywords"] = ", ".join(keywords[cid])
            results.append(row)
            print(f"[{len(results)}/{K}] {cid}: {row['name']} ({row['quality']})")

            # Save after every cluster so a crash or Ctrl+C loses nothing
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).sort_values("cluster").to_csv(OUTPUT_PATH, index=False)

    out = pd.DataFrame(results)
    print(f"\nSaved {len(out)} topics to {OUTPUT_PATH}")
    print(out["quality"].value_counts().to_string())


if __name__ == "__main__":
    main()

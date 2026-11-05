"""Evaluate the refined leaf topics against the ESCO skills taxonomy.

Every statement is matched to its nearest ESCO skill (cosine similarity, same 8B model).
Three measures:
- Coverage: share of statements whose best ESCO match clears a similarity threshold,
  i.e. how much of what postings ask for exists in ESCO at all.
- Purity: within a topic, the share of matched statements whose ESCO skill falls in the
  topic's most common ESCO group. High purity = the topic is one kind of skill by ESCO's
  standards. Compared against a baseline with topic labels shuffled.
- Nearest-neighbor agreement: label every matched statement with the ESCO group of its
  nearest skill and score how well our topics line up with those labels overall
  (NMI and adjusted Rand index; 0 = no agreement, 1 = identical partitions).

Run after refine_topics.py, e.g. `python src/evaluate_esco.py 100`.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ESCO_DIR = Path("data/esco")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
STATEMENTS_PATH = Path("data/processed/unique_statements_cleaned.csv")
LABELS_PATH = Path(f"data/processed/topic_labels_k{K}_refined.npy")
TOPICS_PATH = Path(f"outputs/topics/topics_k{K}_refined.csv")
MATCHES_OUT = Path(f"outputs/topics/esco_matches_k{K}.csv")
TOPIC_EVAL_OUT = Path(f"outputs/topics/esco_eval_k{K}.csv")

# Similarity a statement needs with its nearest ESCO skill to count as covered.
# Spot checks: 'analytical acumen' -> 'think analytically' scores 0.77, loose but
# related matches sit around 0.6-0.7. The headline threshold is the middle one.
THRESHOLDS = [0.6, 0.7, 0.8]
MAIN_THRESHOLD = 0.7
N_SHUFFLES = 20
SEED = 42


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def esco_groups():
    """Map each ESCO skill URI to its level-1 and level-2 group names.

    Skills link to groups (sometimes through a broader skill) in the broader-relations
    file; about a quarter of skills sit in more than one group, so each maps to a set.
    """
    h = pd.read_csv(ESCO_DIR / "skillsHierarchy_en.csv")
    ancestors = {}  # hierarchy URI -> {1: level-1 name, 2: level-2 name}
    for r in h.to_dict("records"):
        for lvl in range(1, 4):
            uri = r.get(f"Level {lvl} URI")
            if pd.notna(uri):
                ancestors[uri] = {a: r[f"Level {a} preferred term"] for a in (1, 2)
                                  if a <= lvl and pd.notna(r.get(f"Level {a} preferred term"))}

    b = pd.read_csv(ESCO_DIR / "broaderRelationsSkillPillar_en.csv")
    parents = b.groupby("conceptUri")["broaderUri"].apply(list).to_dict()

    def walk(uri, depth=0):
        if uri in ancestors:
            return [ancestors[uri]]
        if depth > 10:
            return []
        return [a for p in parents.get(uri, []) for a in walk(p, depth + 1)]

    out = {}
    for uri in pd.read_csv(ESCO_DIR / "skills_en.csv")["conceptUri"]:
        found = walk(uri)
        out[uri] = {lvl: sorted({a[lvl] for a in found if lvl in a}) for lvl in (1, 2)}
    return out


def assign_groups(topic_ids, options):
    """Pick one ESCO group per statement. A statement whose skill sits in several groups
    gets the one most common in its topic, so multi-group skills don't count against
    purity just because ESCO filed them twice."""
    picked = []
    counts = {}
    for t, opts in zip(topic_ids, options):
        counts.setdefault(t, Counter()).update(opts)
    for t, opts in zip(topic_ids, options):
        picked.append(max(opts, key=lambda g: counts[t][g]) if opts else None)
    return picked


def purity(topic_ids, options):
    """Mean over topics of the share of statements in the topic's top ESCO group."""
    df = pd.DataFrame({"topic": topic_ids, "options": options})
    scores = {}
    for t, grp in df.groupby("topic"):
        c = Counter(g for opts in grp["options"] for g in opts)
        if not c:
            continue
        top = c.most_common(1)[0][0]
        scores[t] = float(np.mean([top in opts for opts in grp["options"]]))
    return scores


def main():
    statements = pd.read_csv(STATEMENTS_PATH)["statement"].to_numpy()
    emb = unit(np.load(EMBEDDINGS_PATH))
    labels = np.load(LABELS_PATH)
    topics = pd.read_csv(TOPICS_PATH).set_index("topic")

    skills = pd.read_csv(ESCO_DIR / "skills_en.csv")
    skill_emb = unit(np.load(ESCO_DIR / "skills_en_embeddings_8b.npy"))
    assert len(skill_emb) == len(skills), "ESCO embeddings don't match skills_en.csv"
    assert len(emb) == len(statements) == len(labels), "statement files don't line up"

    # Nearest ESCO skill for every statement
    sims = emb @ skill_emb.T
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(sims)), best]
    groups = esco_groups()
    uris = skills["conceptUri"].to_numpy()
    l1 = [groups[uris[j]][1] for j in best]
    l2 = [groups[uris[j]][2] for j in best]

    matches = pd.DataFrame({
        "statement": statements, "topic": labels,
        "esco_skill": skills["preferredLabel"].to_numpy()[best], "similarity": best_sim.round(3),
        "esco_level1": ["; ".join(g) for g in l1], "esco_level2": ["; ".join(g) for g in l2],
    })
    MATCHES_OUT.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_OUT, index=False)

    # ---- Coverage ----
    kept = labels >= 0
    print(f"Statements: {len(labels)} total, {kept.sum()} in topics (dropped topics excluded below)\n")
    print("COVERAGE: share of statements whose nearest ESCO skill clears the threshold")
    for th in THRESHOLDS:
        print(f"  similarity >= {th}: {(best_sim[kept] >= th).mean():.1%}")
    print(f"  (median best similarity: {np.median(best_sim[kept]):.2f})")

    # Purity and agreement use only covered statements: an uncovered statement's
    # "nearest" skill is a weak guess and would add noise
    use = kept & (best_sim >= MAIN_THRESHOLD)
    t_use = labels[use]
    l1_use = [l1[i] for i in np.flatnonzero(use)]
    l2_use = [l2[i] for i in np.flatnonzero(use)]

    # ---- Purity, with a shuffled-label baseline ----
    rng = np.random.default_rng(SEED)
    pur = {lvl: purity(t_use, opts) for lvl, opts in ((1, l1_use), (2, l2_use))}
    print(f"\nPURITY (statements with similarity >= {MAIN_THRESHOLD}, n={use.sum()})")
    for lvl, opts in ((1, l1_use), (2, l2_use)):
        n_groups = len({g for o in opts for g in o})
        base = np.mean([np.mean(list(purity(rng.permutation(t_use), opts).values()))
                        for _ in range(N_SHUFFLES)])
        print(f"  ESCO level {lvl} ({n_groups} groups used): {np.mean(list(pur[lvl].values())):.1%}"
              f"  vs {base:.1%} with shuffled topics")

    # ---- Nearest-neighbor agreement ----
    print("\nNEAREST-NEIGHBOR AGREEMENT (topics vs ESCO group of each statement's nearest skill)")
    for lvl, opts in ((1, l1_use), (2, l2_use)):
        esco_label = assign_groups(t_use, opts)
        print(f"  ESCO level {lvl}: NMI {normalized_mutual_info_score(t_use, esco_label):.2f}, "
              f"ARI {adjusted_rand_score(t_use, esco_label):.2f}")

    # ---- Per-topic table ----
    rows = []
    for t in sorted(set(labels[kept])):
        in_t = labels == t
        center = unit(emb[in_t].mean(axis=0, keepdims=True))[0]
        top_skills = skills["preferredLabel"].to_numpy()[np.argsort(-(skill_emb @ center))[:3]]
        c1 = Counter(g for i in np.flatnonzero(in_t & use) for g in l1[i])
        rows.append({
            "topic": t, "name": topics.loc[t, "name"], "status": topics.loc[t, "status"],
            "size": int(in_t.sum()),
            f"coverage_{MAIN_THRESHOLD}": round(float((best_sim[in_t] >= MAIN_THRESHOLD).mean()), 3),
            "mean_similarity": round(float(best_sim[in_t].mean()), 3),
            "purity_l1": round(pur[1].get(t, np.nan), 3), "purity_l2": round(pur[2].get(t, np.nan), 3),
            "top_esco_group_l1": c1.most_common(1)[0][0] if c1 else "",
            "nearest_esco_skills": "; ".join(top_skills),
        })
    per_topic = pd.DataFrame(rows)
    per_topic.to_csv(TOPIC_EVAL_OUT, index=False)

    print("\nBY TOPIC STATUS (mean over topics)")
    cols = [f"coverage_{MAIN_THRESHOLD}", "mean_similarity", "purity_l1", "purity_l2"]
    print(per_topic.groupby("status")[cols].mean().round(3).to_string())

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 45)
    show = ["topic", "name", "size", f"coverage_{MAIN_THRESHOLD}", "purity_l1", "nearest_esco_skills"]
    ranked = per_topic.dropna(subset=["purity_l1"]).sort_values(f"coverage_{MAIN_THRESHOLD}")
    print("\nLeast covered topics (postings ask for things ESCO doesn't name):")
    print(ranked.head(10)[show].to_string(index=False))
    print("\nLowest purity topics (ESCO splits these across groups):")
    print(per_topic.sort_values("purity_l1").head(10)[show].to_string(index=False))

    print(f"\nSaved {MATCHES_OUT} and {TOPIC_EVAL_OUT}")


if __name__ == "__main__":
    main()

"""Charts for the ESCO evaluation, built from the files evaluate_esco.py saves.

`python src/esco_plots.py 100` writes slide-ready PNGs (light theme) to outputs/figures/.
app.py imports the same functions and draws them in its dark theme.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

THRESHOLD = 0.7  # must match MAIN_THRESHOLD in evaluate_esco.py
N_SHUFFLES = 20
SEED = 42
FIG_DIR = Path("outputs/figures")

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "baseline": "#898781", "series": ["#2a78d6", "#eb6834", "#1baf7a"]}
# Matches app.py's navy theme
DARK = {"surface": "#121B2E", "ink": "#EDEFF2", "ink2": "#9AA7BD", "grid": "#2E3B57",
        "axis": "#2E3B57", "baseline": "#6B7891", "series": ["#3987e5", "#d95926", "#199e70"]}

STATUS_LABELS = {"keep": "Kept (good)", "low_specificity": "Low specificity",
                 "mixed_unresolved": "Mixed (unresolved)"}


# ---------- data ----------

def load(k=100):
    """Statement-level matches (dropped topics removed) and the per-topic table with domains."""
    matches = pd.read_csv(f"outputs/topics/esco_matches_k{k}.csv")
    matches = matches[matches["topic"] >= 0].reset_index(drop=True)
    per_topic = pd.read_csv(f"outputs/topics/esco_eval_k{k}.csv")
    hier = pd.read_csv(f"outputs/topics/hierarchy_k{k}.csv")[["topic", "l2_name", "l3_name"]]
    return matches, per_topic.merge(hier, on="topic", how="left")


def _groups(col):
    return col.fillna("").map(lambda s: [g for g in s.split("; ") if g]).tolist()


def _purity(topic_ids, options):
    """Same definition as evaluate_esco.purity: mean over topics of the share of
    statements whose ESCO groups include the topic's most common group."""
    df = pd.DataFrame({"topic": topic_ids, "options": options})
    scores = []
    for _, grp in df.groupby("topic"):
        c = Counter(g for opts in grp["options"] for g in opts)
        if c:
            top = c.most_common(1)[0][0]
            scores.append(np.mean([top in opts for opts in grp["options"]]))
    return float(np.mean(scores))


def summary(matches):
    """Headline numbers: coverage, purity vs a shuffled-topic baseline, NMI."""
    use = matches[matches["similarity"] >= THRESHOLD]
    topics = use["topic"].to_numpy()
    rng = np.random.default_rng(SEED)
    out = {"coverage": float((matches["similarity"] >= THRESHOLD).mean()),
           "n_statements": len(matches), "n_covered": len(use)}
    for lvl in (1, 2):
        opts = _groups(use[f"esco_level{lvl}"])
        out[f"purity_l{lvl}"] = _purity(topics, opts)
        out[f"shuffled_l{lvl}"] = float(np.mean([_purity(rng.permutation(topics), opts)
                                                for _ in range(N_SHUFFLES)]))
        out[f"groups_l{lvl}"] = len({g for o in opts for g in o})
        # For NMI each statement gets the one group most common in its topic
        counts = {}
        for t, o in zip(topics, opts):
            counts.setdefault(t, Counter()).update(o)
        picked = [max(o, key=lambda g: counts[t][g]) if o else "" for t, o in zip(topics, opts)]
        out[f"nmi_l{lvl}"] = float(normalized_mutual_info_score(topics, picked))
    return out


# ---------- styling ----------

def _axes(theme, ncols=1, figsize=(7, 3.6), sharey=False):
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=sharey)
    fig.patch.set_facecolor(theme["surface"])
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(theme["surface"])
        ax.tick_params(colors=theme["ink2"], labelsize=9, length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(theme["axis"])
        ax.grid(axis="y", color=theme["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(theme["ink2"])
        ax.yaxis.label.set_color(theme["ink2"])
        ax.title.set_color(theme["ink"])
    return fig, axes


def _title(fig, theme, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=theme["ink"])


def _legend(ax, theme, **kw):
    leg = ax.legend(frameon=False, fontsize=9, **kw)
    for t in leg.get_texts():
        t.set_color(theme["ink2"])


# ---------- charts ----------

def coverage_curve(matches, theme=LIGHT):
    sims = matches["similarity"].to_numpy()
    th = np.round(np.arange(0.5, 0.951, 0.01), 2)
    cov = [(sims >= t).mean() for t in th]
    fig, ax = _axes(theme)
    ax.plot(th, cov, color=theme["series"][0], linewidth=2)
    for t in (0.6, 0.7, 0.8):
        c = (sims >= t).mean()
        ax.plot(t, c, "o", markersize=8, color=theme["series"][0],
                markeredgecolor=theme["surface"], markeredgewidth=2)
        ax.annotate(f"{c:.0%} at ≥ {t}", (t, c), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=theme["ink"])
    ax.axvline(THRESHOLD, color=theme["axis"], linewidth=1, linestyle="--", zorder=0)
    ax.set_xlabel("Cosine similarity to nearest ESCO skill (threshold)")
    ax.set_ylabel("Statements covered")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.05)
    _title(fig, theme, f"Coverage: how many of {len(sims):,} statements have an ESCO match")
    fig.tight_layout()
    return fig


def purity_vs_baseline(s, theme=LIGHT):
    fig, ax = _axes(theme, figsize=(6.2, 3.6))
    x = np.arange(2)
    w = 0.36
    ours = [s["purity_l1"], s["purity_l2"]]
    base = [s["shuffled_l1"], s["shuffled_l2"]]
    b1 = ax.bar(x - w / 2 - 0.01, ours, w, color=theme["series"][0], label="Our topics")
    b2 = ax.bar(x + w / 2 + 0.01, base, w, color=theme["baseline"], label="Shuffled topics (chance)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.0%}",
                    ha="center", va="bottom", fontsize=9, color=theme["ink"])
    ax.set_xticks(x, [f"ESCO level 1\n({s['groups_l1']} groups)", f"ESCO level 2\n({s['groups_l2']} groups)"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(ours) * 1.3)
    ax.set_ylabel("Statements in topic's main ESCO group")
    _legend(ax, theme, loc="upper right")
    _title(fig, theme, "Purity: topics are ~2-3× more ESCO-consistent than chance")
    fig.tight_layout()
    return fig


def by_status(per_topic, theme=LIGHT):
    order = [s for s in STATUS_LABELS if s in set(per_topic["status"])]
    grp = per_topic.groupby("status").agg(cov=(f"coverage_{THRESHOLD}", "mean"),
                                          pur=("purity_l1", "mean"), n=("topic", "count")).loc[order]
    labels = [f"{STATUS_LABELS[s]}\n({n} topics)" for s, n in zip(order, grp["n"])]
    fig, axes = _axes(theme, ncols=2, figsize=(8.5, 3.6))
    for ax, col, title in zip(axes, ("cov", "pur"),
                              (f"Coverage (similarity ≥ {THRESHOLD})", "Purity (ESCO level 1)")):
        bars = ax.bar(labels, grp[col], width=0.55, color=theme["series"][0])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.0%}",
                    ha="center", va="bottom", fontsize=9, color=theme["ink"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", labelsize=8.5)
    _title(fig, theme, "Topics Claude rated good also score best against ESCO")
    fig.tight_layout()
    return fig


def topic_scatter(per_topic, theme=LIGHT, n_labels=8):
    df = per_topic.dropna(subset=["purity_l1"])
    cov = f"coverage_{THRESHOLD}"
    fig, ax = _axes(theme, figsize=(8.5, 5))
    for i, (status, label) in enumerate(STATUS_LABELS.items()):
        d = df[df["status"] == status]
        ax.scatter(d[cov], d["purity_l1"], s=40, color=theme["series"][i], label=label,
                   edgecolors=theme["surface"], linewidths=1.2, zorder=3)
    # Label the weakest topics: lowest coverage + purity combined
    worst = df.assign(score=df[cov] + df["purity_l1"]).nsmallest(n_labels, "score")
    # Low-coverage points have open space to their right; the high-coverage ones sit in a
    # crowded corner, so their labels are stacked in the empty strip below with leader lines
    crowded = worst[worst[cov] > 0.85].sort_values("purity_l1", ascending=False)
    lowest = df["purity_l1"].min()
    stack_y = {t: lowest - 0.05 - 0.035 * i for i, t in enumerate(crowded["topic"])}
    if stack_y:
        ax.set_ylim(bottom=min(stack_y.values()) - 0.03)
    for r in worst.to_dict("records"):
        name = r["name"] if len(r["name"]) <= 32 else r["name"][:30] + "…"
        point = (r[cov], r["purity_l1"])
        if r["topic"] in stack_y:
            # Numbered dot + matching numbered key line in the strip below
            num = list(stack_y).index(r["topic"]) + 1
            ax.annotate(str(num), point, xytext=(5, -3), textcoords="offset points",
                        fontsize=7.5, fontweight="bold", color=theme["ink"])
            ax.text(0.865, stack_y[r["topic"]], f"{num}  {name}", ha="left", va="center",
                    fontsize=7.5, color=theme["ink2"])
        else:
            ax.annotate(name, point, xytext=(6, -3), textcoords="offset points",
                        fontsize=7.5, color=theme["ink2"])
    ax.set_xlabel(f"Coverage (share of statements with ESCO similarity ≥ {THRESHOLD})")
    ax.set_ylabel("Purity (ESCO level 1)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=theme["grid"], linewidth=0.6)
    _legend(ax, theme, loc="upper left")
    _title(fig, theme, f"Every topic: coverage vs purity ({len(df)} topics; weakest labeled)")
    fig.tight_layout()
    return fig


def by_domain(matches, per_topic, theme=LIGHT):
    domain_of = per_topic.set_index("topic")["l3_name"]
    m = matches.assign(domain=matches["topic"].map(domain_of))
    cov = m.groupby("domain")["similarity"].apply(lambda s: (s >= THRESHOLD).mean())
    pt = per_topic.dropna(subset=["purity_l1"])
    pur = pt.groupby("l3_name").apply(lambda d: np.average(d["purity_l1"], weights=d["size"]),
                                      include_groups=False)
    order = cov.sort_values().index
    fig, axes = _axes(theme, ncols=2, figsize=(10, 4.4), sharey=True)
    for ax, vals, title in zip(axes, (cov[order], pur.reindex(order)),
                               (f"Coverage (similarity ≥ {THRESHOLD})", "Purity (ESCO level 1, size-weighted)")):
        bars = ax.barh(order, vals, height=0.6, color=theme["series"][0])
        for b in bars:
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{b.get_width():.0%}",
                    va="center", fontsize=8.5, color=theme["ink"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, 1.12)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.6)
    _title(fig, theme, "By top-level domain of the hierarchy")
    fig.tight_layout()
    return fig


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    matches, per_topic = load(k)
    s = summary(matches)
    print(f"Coverage ≥ {THRESHOLD}: {s['coverage']:.1%}")
    for lvl in (1, 2):
        print(f"Purity level {lvl}: {s[f'purity_l{lvl}']:.1%} vs {s[f'shuffled_l{lvl}']:.1%} shuffled, "
              f"NMI {s[f'nmi_l{lvl}']:.2f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    charts = {
        "esco_coverage_curve": coverage_curve(matches),
        "esco_purity_vs_baseline": purity_vs_baseline(s),
        "esco_by_status": by_status(per_topic),
        "esco_topic_scatter": topic_scatter(per_topic),
        "esco_by_domain": by_domain(matches, per_topic),
    }
    for name, fig in charts.items():
        path = FIG_DIR / f"{name}_k{k}.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

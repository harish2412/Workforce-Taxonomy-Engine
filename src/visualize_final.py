import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

HIGHLIGHT_COLORS = [
    "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
    "#d62728", "#e377c2", "#8c564b", "#bcbd22",
]


def plot_full_map(coords, labels, k, out_path):
    unique_clusters = sorted(set(labels) - {-1})
    palette = plt.colormaps["tab20"].resampled(len(unique_clusters))

    fig, ax = plt.subplots(figsize=(11, 9))
    for i, cid in enumerate(unique_clusters):
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette(i)], s=10)

    ax.set_title(f"Occupational Skill Topics — Full UMAP (k={k}, refined)", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved full map to {out_path}")


def plot_highlighted_map(coords, labels, hierarchy, k, out_path, n_highlight=8):
    kept = hierarchy[hierarchy["status"] == "keep"].sort_values("size", ascending=False)
    top = kept.head(n_highlight)
    names = dict(zip(top["topic"], top["name"]))

    fig, ax = plt.subplots(figsize=(11, 8))
    is_highlighted = np.isin(labels, list(names.keys()))
    n_other = len(set(labels) - {-1} - set(names.keys()))
    ax.scatter(coords[~is_highlighted, 0], coords[~is_highlighted, 1],
               c="#d9d9d9", s=6, alpha=0.5, label=f"Other {n_other} topics")

    for color, (cid, name) in zip(HIGHLIGHT_COLORS, names.items()):
        mask = labels == cid
        n = int(mask.sum())
        label_text = name if len(name) <= 35 else name[:33] + "…"
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=16,
                   label=f"{label_text} ({n})", zorder=3)

    ax.set_title(f"Occupational Skill Topics — Top {n_highlight} Named Topics (k={k})",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved highlighted map to {out_path}")


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    embeddings_path = Path("data/processed/embeddings.npy")
    labels_path = Path(f"data/processed/topic_labels_k{k}_refined.npy")
    hierarchy_path = Path(f"outputs/topics/hierarchy_k{k}.csv")

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    hierarchy = pd.read_csv(hierarchy_path)
    print(f"Loaded embeddings {embeddings.shape}, refined labels for k={k} "
          f"({len(set(labels) - {-1})} real topics, {int((labels == -1).sum())} dropped statements)")

    reducer = umap.UMAP(random_state=42)
    coords = reducer.fit_transform(embeddings)

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plot_full_map(coords, labels, k, Path(f"outputs/figures/topics_full_k{k}.png"))
    plot_highlighted_map(coords, labels, hierarchy, k, Path(f"outputs/figures/topics_highlighted_k{k}.png"))


if __name__ == "__main__":
    main()
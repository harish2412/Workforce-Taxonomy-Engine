import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")

CLUSTER_NAMES_K30 = {
    11: "Cloud & IT Infrastructure",
    25: "Microsoft Office Tools",
    17: "Data Engineering & Databases",
    6: "Physical Job Requirements",
    28: "Workplace Safety",
    14: "Professional Licenses & Certifications",
    22: "Lab & Scientific Techniques",
    9: "IP, Legal & Regulatory",
}

CLUSTER_NAMES_K80 = {
    19: "Data Engineering",
    27: "Legal Practice",
    17: "Facilities & Maintenance",
    54: "Design & Creative Software",
    65: "Cloud & DevOps",
    68: "Project Management",
    16: "Healthcare & Clinical",
    61: "Desktop & IT Support",
}

HIGHLIGHT_COLORS = [
    "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
    "#d62728", "#e377c2", "#8c564b", "#bcbd22",
]


def plot_full_map(coords, labels, k, out_path):
    unique_clusters = sorted(set(labels))
    palette = plt.colormaps["tab20"].resampled(len(unique_clusters))

    fig, ax = plt.subplots(figsize=(11, 9))
    for i, cid in enumerate(unique_clusters):
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette(i)], s=10)

    ax.set_title(f"Occupational Skill Clusters — Full UMAP (k={k})", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved full map to {out_path}")


def plot_highlighted_map(coords, labels, names, k, out_path):
    fig, ax = plt.subplots(figsize=(11, 8))
    is_highlighted = np.isin(labels, list(names.keys()))
    ax.scatter(coords[~is_highlighted, 0], coords[~is_highlighted, 1],
               c="#d9d9d9", s=6, alpha=0.5, label=f"Other {len(set(labels)) - len(names)} clusters")

    for color, (cid, name) in zip(HIGHLIGHT_COLORS, names.items()):
        mask = labels == cid
        n = mask.sum()
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=16,
                   label=f"Cluster {cid}: {name} ({n})", zorder=3)

    ax.set_title(f"Occupational Skill Clusters — Highlighted UMAP (k={k})", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved highlighted map to {out_path}")


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    labels_path = Path(f"data/processed/cluster_labels_k{k}.npy")
    names = CLUSTER_NAMES_K30 if k == 30 else CLUSTER_NAMES_K80

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(labels_path)
    print(f"Loaded embeddings {embeddings.shape}, labels for k={k}")

    reducer = umap.UMAP(random_state=42)
    coords = reducer.fit_transform(embeddings)

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plot_full_map(coords, labels, k, Path(f"outputs/figures/cluster_full_k{k}.png"))
    plot_highlighted_map(coords, labels, names, k, Path(f"outputs/figures/cluster_highlighted_k{k}.png"))


if __name__ == "__main__":
    main()
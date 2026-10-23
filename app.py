import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

st.set_page_config(page_title="Skill Taxonomy Engine", layout="wide")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
  --paper: #121B2E;
  --paper-raised: #1B2740;
  --ink: #EDEFF2;
  --ink-soft: #9AA7BD;
  --rule: #2E3B57;
  --brass: #D8A85C;
  --brass-dim: #C4913F;
  --teal: #5FA595;
  --teal-bg: #1C2F2B;
}
.stApp { background: var(--paper); }
.block-container { padding-top: 1.5rem !important; padding-bottom: 1.5rem !important; }
html, body, [class*="css"] { font-family: 'Newsreader', Georgia, serif; color: var(--ink); }
h1, h2, h3 { font-family: 'Newsreader', Georgia, serif !important; font-weight: 600 !important; color: var(--ink) !important; }
.mono, code { font-family: 'IBM Plex Mono', monospace !important; }
p, span, label, li { color: var(--ink) !important; }
.masthead { border-bottom: 2px solid var(--brass); padding-bottom: 10px; margin-bottom: 16px; }
.masthead .eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--ink-soft); }
.masthead h1 { font-size: 1.7rem !important; margin: 2px 0 0 !important; }
div[data-testid="stButton"] button, .stButton > button { background-color: var(--paper-raised) !important; color: var(--ink) !important; border: 1px solid var(--rule) !important; border-radius: 4px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }
div[data-testid="stButton"] button p, .stButton > button p, div[data-testid="stButton"] button div { color: var(--ink) !important; font-family: 'IBM Plex Mono', monospace !important; }
div[data-testid="stButton"] button:hover, .stButton > button:hover { background-color: var(--brass) !important; border-color: var(--brass) !important; }
div[data-testid="stButton"] button:hover p, .stButton > button:hover p { color: var(--paper) !important; }
div[data-testid="stTextInput"] input { background-color: var(--paper-raised) !important; color: var(--ink) !important; border: 1px solid var(--rule) !important; font-family: 'IBM Plex Mono', monospace !important; }
div[data-testid="stTextInput"] input::placeholder { color: var(--ink-soft) !important; opacity: 1 !important; }
div[data-testid="stSelectbox"] div[data-baseweb="select"] { background-color: var(--paper-raised) !important; border-color: var(--rule) !important; }
div[data-testid="stSelectbox"] div[data-baseweb="select"] * { color: var(--ink) !important; }
div[data-testid="stExpander"] summary { background-color: var(--paper-raised) !important; color: var(--ink) !important; border: 1px solid var(--rule) !important; }
div[data-testid="stExpander"] summary p { color: var(--ink) !important; }
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] { background-color: var(--paper) !important; border: 1px solid var(--rule) !important; border-top: none !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background-color: var(--paper-raised) !important; border: 1px solid var(--rule) !important; border-radius: 4px; }
.match-card { background: var(--paper-raised); border-left: 3px solid var(--brass); padding: 12px 18px; margin-bottom: 10px; }
.match-card.top { border-left-color: var(--teal); }
.match-card .cluster-id { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 600; color: var(--brass-dim); }
.match-card.top .cluster-id { color: var(--teal); }
.match-card .score { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--ink-soft); }
.sidebar-cluster-stat { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--ink-soft); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    embeddings = np.load("data/processed/embeddings.npy")
    labels = np.load("data/processed/cluster_labels.npy")
    statements = pd.read_csv("data/processed/unique_statements.csv")["statement"]

    # real model, needs GPU — commented out for frontend-only testing
    # model = SentenceTransformer(MODEL_NAME)
    model = None

    df = pd.DataFrame({"statement": statements, "cluster": labels})
    centers = {cid: embeddings[labels == cid].mean(axis=0) for cid in sorted(set(labels))}
    sizes = df["cluster"].value_counts().sort_index()

    return model, df, centers, sizes


# TEMPORARY — frontend testing only, no GPU needed. Remove before the real demo.
def fake_encode(text):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.random(4096)
    return vec / np.linalg.norm(vec)


def embed_query(text, model):
    # real embedding call, needs GPU — commented out for frontend-only testing
    # return model.encode([text], normalize_embeddings=True)[0]
    return fake_encode(text)


def top_matches(query_vec, centers, k=3):
    scores = []
    for cid, center in centers.items():
        score = np.dot(query_vec, center) / (np.linalg.norm(query_vec) * np.linalg.norm(center))
        scores.append((cid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def render_size_chart(sizes):
    fig, ax = plt.subplots(figsize=(5, 2.1))
    fig.patch.set_facecolor("#121B2E")
    ax.set_facecolor("#121B2E")
    ax.bar(sizes.index, sizes.values, color="#D8A85C", width=0.7)
    ax.set_xlabel("Cluster", color="#EDEFF2", fontsize=8)
    ax.set_ylabel("Statements", color="#EDEFF2", fontsize=8)
    ax.tick_params(colors="#9AA7BD", labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#2E3B57")
    ax.grid(axis="y", color="#2E3B57", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


model, df, centers, sizes = load_everything()

st.markdown("""
<div class="masthead">
<div class="eyebrow">WORKFORCE TAXONOMY ENGINE · SPRINT 3 LIVE DEMO</div>
<h1>Skill Taxonomy — Cluster Assignment</h1>
</div>
""", unsafe_allow_html=True)

col_main, col_side = st.columns([2, 1])

with col_main:
    st.write("Type a skill phrase, or pick an example, to see which of the 30 learned clusters it's assigned to.")

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    example_phrases = [
        "machine learning",
        "budget forecasting",
        "customer service",
        "patent prosecution",
    ]
    chip_cols = st.columns(len(example_phrases))
    for i, phrase in enumerate(example_phrases):
        with chip_cols[i]:
            if st.button(phrase, key=f"chip_{i}"):
                st.session_state.query_text = phrase

    user_input = st.text_input(
        "Skill phrase",
        key="query_text",
        placeholder="e.g. machine learning, budget forecasting",
    )

    if user_input:
        query_vec = embed_query(user_input, model)
        matches = top_matches(query_vec, centers, k=3)

        for rank, (cid, score) in enumerate(matches):
            is_top = rank == 0
            card_class = "match-card top" if is_top else "match-card"
            label = "Best match" if is_top else f"Also close (#{rank + 1})"

            st.markdown(
                '<div class="' + card_class + '">'
                '<div class="score">' + label.upper() + '</div>'
                '<div class="cluster-id">Cluster ' + str(cid) + '</div>'
                '<div class="score">similarity ' + f"{score:.3f}" + ' · ' + str(sizes[cid]) + ' statements in this cluster</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if is_top:
                examples = df[df["cluster"] == cid]["statement"]
                with st.expander(f"See statements in Cluster {cid}", expanded=True):
                    with st.container(height=180):
                        for ex in examples.sample(min(15, len(examples))):
                            st.write(f"· {ex}")
    else:
        st.markdown(
            '<div class="match-card" style="border-left-color: var(--rule); color: var(--ink-soft);">'
            'Results will appear here once you type a phrase or pick an example above.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Cluster sizes")
    st.pyplot(render_size_chart(sizes))

with col_side:
    st.markdown("### Explore clusters")
    st.write("Browse any cluster's contents without typing anything.")

    chosen_cluster = st.selectbox(
        "Cluster",
        options=sizes.index.tolist(),
        format_func=lambda c: f"Cluster {c} ({sizes[c]} statements)",
    )
    st.markdown(
        '<div class="sidebar-cluster-stat">' + str(sizes[chosen_cluster]) + ' statements in this cluster</div>',
        unsafe_allow_html=True,
    )

    browse_examples = df[df["cluster"] == chosen_cluster]["statement"]
    with st.container(height=360):
        for ex in browse_examples.sample(min(20, len(browse_examples))):
            st.write(f"· {ex}")
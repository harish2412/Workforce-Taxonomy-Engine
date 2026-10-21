import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Skill Taxonomy Demo", layout="centered")


@st.cache_resource
def load_everything():
    embeddings = np.load("data/processed/embeddings.npy")
    labels = np.load("data/processed/cluster_labels.npy")
    statements = pd.read_csv("data/processed/unique_statements.csv")["statement"]
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.DataFrame({"statement": statements, "cluster": labels})
    centers = {cid: embeddings[labels == cid].mean(axis=0) for cid in sorted(set(labels))}

    return model, df, centers


model, df, centers = load_everything()

st.title("Skill Taxonomy — Live Cluster Assignment")
st.write("Type any skill phrase to see which cluster it belongs to.")

user_input = st.text_input("Skill phrase", placeholder="e.g. machine learning, budget forecasting")

if user_input:
    query = model.encode([user_input], normalize_embeddings=True)[0]

    best_cluster, best_score = None, -1
    for cid, center in centers.items():
        score = np.dot(query, center) / (np.linalg.norm(query) * np.linalg.norm(center))
        if score > best_score:
            best_cluster, best_score = cid, score

    st.subheader(f"Assigned to Cluster {best_cluster}")
    st.write(f"Similarity score: {best_score:.3f}")

    examples = df[df["cluster"] == best_cluster]["statement"]
    st.write("Other skills in this cluster:")
    for ex in examples.sample(min(8, len(examples))):
        st.write(f"- {ex}")

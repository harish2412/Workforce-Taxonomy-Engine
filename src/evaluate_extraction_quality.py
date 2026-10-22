from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"  # fast, local; this is an evaluation helper, not the taxonomy pipeline
MATCH_THRESHOLD = 0.6

GOLDEN_PATH = Path("data/golden_set/golden_dataset.xlsx")
RULE_NLP_PATH = Path("data/golden_set/rule_nlp_extraction.xlsx")
GEMINI_PATH = Path("data/golden_set/gemini_extraction.csv")


def load_golden():
    df = pd.read_excel(GOLDEN_PATH)
    return {
        row["job_id"]: [s.strip() for s in str(row["Skills"]).split("\n") if s.strip()]
        for _, row in df.iterrows()
    }


def load_rule_nlp():
    df = pd.read_excel(RULE_NLP_PATH)
    return {
        row["job_id"]: [s.strip() for s in str(row["Skills"]).split("\n") if s.strip()]
        for _, row in df.iterrows()
        if pd.notna(row["Skills"])
    }


def load_gemini(reviewed_only=True):
    df = pd.read_csv(GEMINI_PATH)
    if reviewed_only:
        df = df[df["Review"] == 1]
    return {
        job_id: group["skill_statement"].tolist()
        for job_id, group in df.groupby("job_id")
    }


def evaluate(method_name, predictions, golden, model):
    results = []
    for job_id, true_skills in golden.items():
        pred_skills = predictions.get(job_id)
        if not pred_skills or not true_skills:
            continue

        true_emb = model.encode(true_skills)
        pred_emb = model.encode(pred_skills)
        sims = cosine_similarity(pred_emb, true_emb)

        matched_true = set()
        matched_pred = 0
        for i in range(len(pred_skills)):
            best_j = sims[i].argmax()
            if sims[i][best_j] >= MATCH_THRESHOLD:
                matched_pred += 1
                matched_true.add(best_j)

        precision = matched_pred / len(pred_skills)
        recall = len(matched_true) / len(true_skills)
        results.append({"job_id": job_id, "precision": precision, "recall": recall})

    df = pd.DataFrame(results)
    print(f"\n=== {method_name} ===")
    print(f"Jobs evaluated: {len(df)} of {len(golden)}")
    print(f"Mean precision: {df['precision'].mean():.3f}")
    print(f"Mean recall: {df['recall'].mean():.3f}")


def main():
    golden = load_golden()
    model = SentenceTransformer(MODEL_NAME)

    evaluate("Rule+NLP", load_rule_nlp(), golden, model)
    evaluate("Gemini (reviewed)", load_gemini(reviewed_only=True), golden, model)
    evaluate("Gemini (all, unreviewed)", load_gemini(reviewed_only=False), golden, model)


if __name__ == "__main__":
    main()

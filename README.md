# Workforce Taxonomy Engine

Capstone project, Taxonomy of Work, Skill Taxonomy track.
Building a skill taxonomy from job postings, validated against ESCO.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and fill in whichever API keys you use.

## Data

- `data/raw/sample_postings.csv` — 400 job development sample, 19 job families
- `data/extracted/skill_statements_sample.csv` — 13,958 extracted skill statements,
  with a `confidence` column (high / medium / low). Use high and medium only for
  Sprint 3 embedding and clustering.
- `data/esco/skills_en.csv` — ESCO reference skill list (~13,960 skills)
- `data/esco/skillsHierarchy_en.csv` — ESCO's own multi-level hierarchy, for
  Sprint 4 evaluation

## Pipeline (src/)

| Script | Sprint | Status |
|---|---|---|
| `clean_statements.py` | 3 | filters to high/medium confidence, dedupes |
| `embed.py` | 3 | embeds unique statements with sentence-transformers |
| `cluster.py` | 3 | k-means / HDBSCAN clustering, compares settings |
| `stability.py` | 3 | multi-seed stability check (Adjusted Rand Index) |
| `visualize.py` | 3 | UMAP projection, labeled cluster plot |

Run in order, each step reads the previous step's output from `data/processed/`.

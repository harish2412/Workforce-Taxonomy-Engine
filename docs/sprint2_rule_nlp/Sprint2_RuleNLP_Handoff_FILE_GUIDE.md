# Sprint2_RuleNLP_Handoff.zip – File Guide

This guide explains every file and folder in `Sprint2_RuleNLP_Handoff.zip`: what it contains, what it
is for, and whether you need it for Sprint 3.

**Quick start for Sprint 3:** unzip, read `README.md`, then use
`Sprint3_Data/Skill_Statements.csv` and `Sprint3_Data/Skill_Job_Mapping.csv`.

The ZIP contains one top-level folder, `Sprint2_RuleNLP_Handoff/`, with 52 files
(231 MB compressed, 524 MB unzipped).

"Sprint 3?" column:
- **Yes** = needed for Sprint 3
- **Optional** = useful reference
- **Only to re-run** = needed only if you want to run the extraction again

---

## 1. Top-level documents

| File | What it contains / purpose | Sprint 3? |
|---|---|---|
| `README.md` | Main guide: what the pipeline does, how to install and run it, which files to use, how skills link to jobs, results and limitations. **Read first.** | **Yes** |
| `EVALUATION_SUMMARY.md` | Short summary of Version 2.1 precision / recall / F1 on 100 human-reviewed jobs, compared with earlier versions. | Optional (helps decide Rule+NLP vs LLM) |
| `HANDOFF_MANIFEST.md` | List of every file with its purpose, size and checksum. | Optional |
| `checksums.sha256` | Fingerprints of all files, to confirm nothing was damaged while sharing. | Optional |
| `requirements.txt` | Python packages and versions to install, including the spaCy English model. | Only to re-run |

---

## 2. Extracted skills: `Sprint3_Data/`

| File | What it contains / purpose | Sprint 3? |
|---|---|---|
| **`Skill_Statements.csv`** | 370,455 rows, one per unique skill: `skill_id`, `normalized_skill`, number of jobs mentioning it, category, ESCO link. **Input for embeddings and clustering.** | **Yes (main input)** |
| **`Skill_Job_Mapping.csv`** | 2,532,769 rows, one per skill in each job: `job_id`, `skill_id`, extracted skill, the JD wording it came from, extraction method, position in the JD. **Links skills to jobs.** | **Yes** |
| `Extracted_Skills.xlsx` | 123,849 rows, one per job: job ID, title, company, original job description, all skills in one cell (one per line, in JD order), plus certifications and fields of study. Easy to browse. | Optional (for reading and checking) |
| `Licenses_Certifications.csv` | 38,939 licenses and certifications (CPR, CPA, driver's license …), kept separate from skills. | Only if the team wants them |
| `Fields_of_Study.csv` | 57,652 degree fields (Finance, Computer Science …), kept separate because they are not skills. | Only if the team wants them |
| `Processing_Status.csv` | One row per posting: processed / skipped / failed, plus skill counts. | Optional |
| `jobs_with_no_skills.csv` | 3,283 processed jobs where no skill was found. | Optional |
| `jds_with_control_characters.csv` | 10 jobs whose text contained invisible characters, which were replaced by spaces. | No |
| `VALIDATION_REPORT.md` | Final counts and 16 integrity checks (all passed), including the 7 skipped and 2 failed postings. | Optional |

**Reading tip:** in pandas, read the CSVs with `dtype=str, keep_default_na=False`. Otherwise real
skill names such as "NA" are turned into blank values.

**How the files connect:** `job_id` links `Skill_Job_Mapping.csv` to `Extracted_Skills.xlsx` and
`postings.csv`; `skill_id` links `Skill_Job_Mapping.csv` to `Skill_Statements.csv`.

---

## 3. Python pipeline: `rule_nlp_v21/`

This is the Rule+NLP Version 2.1 extractor. You only need it to re-run extraction or to extract
skills from new job descriptions.

### Main code (Python execution)

| File | Purpose |
|---|---|
| `pipeline.py` | Entry point: runs all steps on a JD and returns its skills in JD order |
| `preprocessing.py` | Repairs messy JD text (missing line breaks, bullets, glued words) |
| `section_parser.py` | Finds JD sections (requirements, responsibilities, benefits …) |
| `rule_extractor.py` | Rule-based extraction: trigger phrases, dictionaries, acronyms |
| `rule_extractor_stop.py` | List of acronyms that are not skills |
| `nlp_extractor.py` | spaCy-based extraction: verb–object rules, noun phrases, list items |
| `skill_filter.py` | Rejects non-skills (company names, places, degrees, benefits …) |
| `skill_normalizer.py` | Cleans and standardizes skill names; removes duplicates |
| `lexicons.py` | Loads all rule files and the ESCO list |
| `context.py`, `candidate.py` | Small helpers: per-JD information and the skill record format |
| `data_loader.py` | Reads the postings, the gold dataset and the dev/test split |
| `__init__.py` | Marks the folder as a Python package (empty file) |

### Configuration and rules

| File / folder | Purpose |
|---|---|
| `config/config.json` | File paths and settings |
| `config/dev_test_split.csv` | Which of the 100 gold jobs are development and which are test (evaluation only) |
| `resources/` | All extraction rules as editable text files: trigger phrases (`triggers.json`), section headings (`sections.json`), verb-to-skill rules (`verb_nominalizations.tsv`, `verb_object_overrides.tsv`), aliases (`skill_aliases.tsv`), tools and certifications (`tech_skills.tsv`), soft skills (`soft_skills.tsv`), ability phrases (`ability_phrases.tsv`), non-skill words (`generic_terms.txt`), boilerplate cues (`boilerplate_cues.txt`) |
| `resources/esco/esco_skills_en.csv` | ESCO skill taxonomy (13,479 skills), used to recognize and validate skills |

### Evaluation and support

| File / folder | Purpose |
|---|---|
| `evaluate.py` | Scores Version 2.1 against the human-reviewed gold labels |
| `matching.py` | Rules for deciding when two skills match (used by `evaluate.py`) |
| `export_results.py` | Builds the Excel output for the 100 gold jobs (evaluation use) |
| `tests/test_pipeline.py`, `tests/__init__.py` | 74 automatic tests confirming the pipeline works correctly |
| `scripts/download_esco.py`, `scripts/__init__.py` | Re-downloads the ESCO list (only if it ever needs updating) |
| `snapshots/FROZEN_v21_final_manifest.md` | Proof that the code is the frozen, final Version 2.1 |
| `outputs/` | Empty folder where the pipeline writes results when it is run |
| `README.md` | Technical design of the extractor |
| `PROPOSAL_v2.1.md` | What changed from Version 2 and why |

---

## 4. Scripts: `scripts/` (Python execution)

| File | Purpose | Sprint 3? |
|---|---|---|
| `run_corpus.py` | Runs Version 2.1 on the full `postings.csv` | Only to re-run |
| `build_sprint3_files.py` | Builds the `Sprint3_Data` files from that run and validates them | Only to re-run |
| `verify_package.py` | Quick self-check that files, data and pipeline all work on your computer | Recommended once |

---

## 5. Gold labels: `gold/labels/`

| File | Purpose | Sprint 3? |
|---|---|---|
| `Golden_Dataset_Reviewed.xlsx` | 100 human-reviewed job descriptions with their skills, used only to measure accuracy | No (evaluation only) |

---

## Summary by type

| Type | Files |
|---|---|
| **Extracted skills** | Everything in `Sprint3_Data/`, mainly `Skill_Statements.csv`, `Skill_Job_Mapping.csv` and `Extracted_Skills.xlsx` |
| **Python execution** | `rule_nlp_v21/` (code, config, resources), `scripts/`, `requirements.txt` |
| **Evaluation** | `EVALUATION_SUMMARY.md`, `rule_nlp_v21/evaluate.py`, `rule_nlp_v21/matching.py`, `rule_nlp_v21/export_results.py`, `rule_nlp_v21/config/dev_test_split.csv`, `gold/labels/Golden_Dataset_Reviewed.xlsx`, `rule_nlp_v21/tests/`, `Sprint3_Data/VALIDATION_REPORT.md` |
| **Documentation** | `README.md`, `HANDOFF_MANIFEST.md`, `checksums.sha256`, `rule_nlp_v21/README.md`, `rule_nlp_v21/PROPOSAL_v2.1.md` |

---

## Not included

`postings.csv`, the original 517 MB LinkedIn job dataset. It is only needed if you want to re-run
extraction; place it in the `Sprint2_RuleNLP_Handoff/` folder (see `README.md`, section 4).

## Setup (only if you want to run the code)

```bash
cd Sprint2_RuleNLP_Handoff
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/verify_package.py
```
Requires Python 3.11–3.13.

## Team decision for Sprint 3

These files contain the **Rule+NLP** skills only. The team needs to decide whether Sprint 3 uses the
Rule+NLP skills, the LLM skills, or a combination.

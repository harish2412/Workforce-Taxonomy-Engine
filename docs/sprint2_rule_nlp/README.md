# Sprint 2 → Sprint 3 handoff: Rule+NLP skill extraction (Version 2.1)

This package contains the final Sprint 2 **Rule-Based + NLP skill extractor** (Version 2.1,
frozen) and the skills it extracted from the full LinkedIn job-posting corpus, ready for Sprint 3.
It contains no API keys, credentials or local paths.

> **Team decision needed for Sprint 3:** this package contains the **Rule+NLP** outputs only. The team
> must decide whether Sprint 3 (embeddings, clustering, taxonomy) uses the **Rule+NLP** skills, the
> **LLM** skills, or a **combination**. The evaluation results below are meant to inform that decision.

---

## 1. What the pipeline does
It reads a job description (JD) and returns the skills it states or clearly implies through concrete
responsibilities, one per line, **in the order they first appear in the JD**. Each skill keeps its
evidence (the original JD wording), the method that found it and its character position in the JD.
Licenses/certifications and fields of study are kept **separate** from skills. No LLM or generative
model is used: every output can be traced to an explicit rule.

## 2. How it extracts skills
```
JD text → text repair (line breaks, bullets, glued words; offsets kept)
        → section detection (requirements / responsibilities / preferred / benefits / legal ...)
        → spaCy en_core_web_sm (tokens, POS tags, dependency parse, noun chunks, NER)
        → rule layer:  trigger phrases ("experience with", "proficient in", "ability to" ...),
                       "... skills" / "... experience" phrases, curated tool & certification dictionary,
                       soft-skill list, ESCO knowledge labels, acronyms in requirement sections
        → NLP layer:   verb-object rules ("Analyze customer data" → data analysis), validated noun
                       phrases, single-word list items ("layout, wiring, fixtures" → wiring)
        → filtering:   explicit rejection reasons (company names, places, job titles, degrees,
                       durations, benefits/legal/promotional text, generic words)
        → normalization: list splitting ("Excel, Word and PowerPoint"), shared heads ("verbal and
                       written communication"), aliases (MS Excel → Microsoft Excel), dedupe
        → output in JD order, with evidence, method, section and offsets
```
- All rule tables are editable text files in `rule_nlp_v21/resources/` (triggers, section headings,
  verb→noun rules, aliases, curated dictionary, soft skills, generic terms, boilerplate cues).
- ESCO (13,479 skills, 83k alternative labels) is in `rule_nlp_v21/resources/esco/esco_skills_en.csv`.
  ESCO is used to recognise and validate skills; it never renames a skill.
- Full design: `rule_nlp_v21/README.md`; the Version 2.1 change proposal: `rule_nlp_v21/PROPOSAL_v2.1.md`.

## 3. Requirements
- **Python 3.11–3.13** (tested with 3.13.5).
- Packages in `requirements.txt` (spaCy 3.8.16, pandas 3.0.6, numpy, pyarrow, openpyxl, tabulate, pytest),
  including the spaCy model `en_core_web_sm` 3.8.0, pinned.
- About 1 GB of disk for the corpus run; the full run takes ~30 minutes on 10 CPU cores.

## 4. Install and run
```bash
cd Sprint2_RuleNLP_Handoff
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/verify_package.py       # checks files, datasets and a sample extraction
python -m pytest rule_nlp_v21/tests -q # 74 unit and regression tests
```
Extract skills from one JD in Python (run from this folder):
```python
from rule_nlp_v21.pipeline import SkillPipeline
pipe = SkillPipeline()
rows, rejected = pipe.extract(jd_text, job_id="123", company="Acme", title="Data Analyst")
skills = [r["skill"] for r in rows if r["output_type"] == "skill"]     # in JD order
```
Re-run the full corpus (only needed to reproduce or update the datasets):
```bash
# place the original LinkedIn postings.csv (not included: 517 MB) in this folder, then
python scripts/run_corpus.py --postings postings.csv --workers 8      # resumable
python scripts/build_sprint3_files.py --postings postings.csv         # rebuilds Sprint3_Data/ + validation
```

## 5. Files to use for Sprint 3 (`Sprint3_Data/`)
| File | Rows | Use |
|---|---|---|
| **`Skill_Statements.csv`** | 370,455 | **one row per unique normalized skill** - the input for embeddings and clustering (Hari, Stephy) |
| **`Skill_Job_Mapping.csv`** | 2,532,769 | **one row per skill occurrence in a job** - links skills to jobs, with evidence and position |
| **`Extracted_Skills.xlsx`** | 123,849 | one row per job: job_id, title, company, original JD, `Rule_NLP_Extracted_Skills` (one cell, one skill per line, JD order) |
| `Licenses_Certifications.csv` | 38,939 | licenses and certifications, kept separate from skills |
| `Fields_of_Study.csv` | 57,652 | fields of study from degree requirements, kept separate from skills |
| `Processing_Status.csv` | 123,849 | status of every posting (processed / skipped / failed) and per-job counts |
| `jobs_with_no_skills.csv` | 3,283 | processed jobs where no skill was found |
| `VALIDATION_REPORT.md` | - | counts and integrity checks (all passed) |

**Column reference**
- `Skill_Statements.csv`: `skill_id` (SK000001 …, ordered by how many jobs mention it), `normalized_skill`,
  `job_count`, `occurrences`, `category` (technical / soft / domain / language), `esco_label`, `esco_uri`
  (ESCO links made through an alternative label are marked `[via ESCO alternative label]`).
- `Skill_Job_Mapping.csv`: `job_id`, `skill_id`, `extracted_skill` (the name as extracted in that job),
  `normalized_skill` (the shared name for the skill_id), `order_in_jd` (1 = first skill in the JD),
  `char_start` / `char_end` (position in the original JD text), `jd_evidence` (the JD wording),
  `extraction_method`, `jd_section`, `requirement_level` (required / preferred / responsibility / other),
  `evidence_strength` (strong / moderate; a rule-based label, **not** a probability).

**Reading the CSVs** - always read as text and switch off pandas' default "NA" parsing, otherwise
real skill names such as `NA` are turned into missing values:
```python
import pandas as pd
statements = pd.read_csv("Sprint3_Data/Skill_Statements.csv", dtype=str, keep_default_na=False)
mapping = pd.read_csv("Sprint3_Data/Skill_Job_Mapping.csv", dtype=str, keep_default_na=False)
```
`Skill_Job_Mapping.csv` is ~330 MB; read only the columns you need (`usecols=[...]`).

## 6. How skills and job IDs are connected
```
Extracted_Skills.xlsx ──job_id──┐
postings.csv ──────────job_id──┤
                                ├── Skill_Job_Mapping.csv ──skill_id── Skill_Statements.csv
Processing_Status.csv ─job_id──┘      (one row per job × skill)        (one row per unique skill)
```
- `job_id` is the LinkedIn posting ID from `postings.csv` (unique; kept as text).
- `skill_id` identifies one normalized skill. Spelling variants of the same skill share one ID
  (case, plural and punctuation differences, e.g. "Excel"/"excel"); different wordings of a concept
  ("data analysis" vs "analyzing data") are **different** IDs - merging those is a Sprint 3 decision.
- A skill appears at most once per job; its row records its earliest supporting evidence.
- `char_start`/`char_end` refer to the JD text in the `original_job_description` column of
  `Extracted_Skills.xlsx` (= `description`, plus `skills_desc` when that field adds text).
- Example (pandas): `mapping.merge(statements, on="skill_id")` gives every job's skills with their IDs;
  `mapping.groupby("skill_id").job_id.nunique()` reproduces `job_count`.

## 7. Corpus run summary
| measure | value |
|---|---|
| postings in postings.csv | 123,849 |
| processed | 123,840 |
| skipped (empty description) | 7 |
| failed | 2 |
| processed jobs with no skills | 3,283 (listed in `jobs_with_no_skills.csv`) |
| skill occurrences | 2,532,769 |
| unique normalized skills | 370,455 |
| skills per job (mean / median) | 20.5 / 18 |
| unique skills seen in only one job | 241,112 |
| licenses/certifications · fields of study (separate files) | 38,939 · 57,652 |

Notes from validation (`Sprint3_Data/VALIDATION_REPORT.md`, all 16 integrity checks passed):
- **7 skipped**: postings with no description text at all.
- **2 failed** (3894940159, 3904983911): a known bug in the frozen Version 2.1 - when a cue phrase such
  as "proficiency managing ..." is re-read as a verb phrase and that capture comes back empty, the code
  indexes an empty list. Both are marked `failed` (with the error) in `Processing_Status.csv` and the
  Excel `processing_status` column; they have no skills. The model was not changed after freezing; the
  fix (a one-line empty-capture guard) is a candidate for a later version.
- **Control characters**: 10 JDs (and 21 extracted text values) contain characters such as a vertical tab
  or a hyphen encoded as `\x02`, which Excel cannot store. They were replaced one-for-one by spaces in the
  exported files, so text length and `char_start`/`char_end` are unchanged (`jds_with_control_characters.csv`).
- **No JD text was truncated**: the longest JD is 23,201 characters (Excel's cell limit is 32,767); every
  Excel JD cell was verified identical to the original text, and every skill cell identical to the
  mapping file in JD order.

## 8. Evaluation results
Human-reviewed gold set of 100 JDs, split 40 dev (used to develop the rules) / 60 held-out. All
systems are scored by the same evaluator. **Equivalent** = same concept after aliases, inflection,
verb→noun and word order (validated against 150 hand-judged pairs: precision 1.00, recall 0.88);
**lenient** = word overlap ≥ 50% (over-accepts partial matches); **strict** = identical after
normalization. Gold labels mix certifications into skills, so these scores use skills + certifications.

| 60 held-out jobs (F1) | Version 1 | Version 2 | **Version 2.1** |
|---|---|---|---|
| equivalent | 19.1 | 25.5 | **25.9** |
| lenient | 31.9 | 40.2 | **40.7** |
| strict | 16.9 | 22.9 | **23.2** |

| 40 dev jobs (F1, optimistic) | Version 1 | Version 2 | **Version 2.1** |
|---|---|---|---|
| equivalent | 17.2 | 31.2 | **32.8** |
| lenient | 31.4 | 46.3 | **47.9** |

- Version 2/2.1 clearly beat Version 1. Version 2.1's gain over Version 2 is small on the held-out jobs
  (within the confidence intervals); it mainly fixes specific failure cases (empty outputs on
  unstructured JDs, place names, degree fields counted as skills, junk words).
- A final **independent evaluation on 50 new, blind-labelled JDs** has been prepared
  (labeling in progress by Shiv); its results will be shared separately.
- Details and how to reproduce: `EVALUATION_SUMMARY.md`.

## 9. Known limitations (read before Sprint 3)
1. **Recall is limited by design**: a rule-based extractor finds what the JD states. Human reviewers often
   wrote paraphrased concepts ("construction plan reading" for "Read and examine plans"), which no
   extractive method reproduces word for word.
2. **Responsibility-derived skills** (verb-object rules, e.g. "Coordinate package deliveries" → package
   delivery coordination) are the least precise component; many are valid but were not in the human labels,
   some are awkward nominalizations.
3. **Long tail**: many unique skills occur in only one job (241,112 of 370,455). Most are specific multiword
   phrases; Sprint 3 embeddings/clustering should expect near-duplicates such as
   "data analysis" / "data analytics" / "analysis of data".
4. **Unstructured JDs** (no headings, run-on bullets) get conservative output; 3,283 processed jobs have no skills.
5. **Single-word skills** are accepted only with list or dictionary evidence; some junk remains.
6. **Two postings fail** because of the empty-capture bug described in section 7.
7. **Certification column**: the pattern includes "registration", so activities such as
   "participant registration" can land in `Licenses_Certifications.csv`.
8. spaCy's small model mis-tags some bullets; several rules compensate, but not all cases.

## 10. Self-contained Version 2.1
This package contains **only Version 2.1**. It does not import or read anything from the earlier
Version 1 / Version 2 code: the two evaluation helpers it used from Version 1 (`normalize`, `tokens`,
`match_job`, `prf`) were copied verbatim into `rule_nlp_v21/matching.py`, and the evaluator in this
package scores Version 2.1 only. Verified in an isolated copy: previous versions not importable, all 74
tests pass, and the dev evaluation reproduces the reported scores exactly.

## 11. Package contents
See `HANDOFF_MANIFEST.md` for every file, its purpose and checksums.

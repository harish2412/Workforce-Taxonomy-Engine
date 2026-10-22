# Rule+NLP Version 2.1: evaluation summary

**Gold data**: 100 human-reviewed job descriptions (`gold/labels/Golden_Dataset_Reviewed.xlsx`),
split by job family into 40 **dev** jobs (used to develop the rules) and 60 **held-out** jobs
(`rule_nlp_v21/config/dev_test_split.csv`). Gold labels mix certifications into skills, so scores
use skills + certifications (fields of study excluded by the labeling policy).

**Matching** (one-to-one per job):
- **equivalent**: same concept after aliases, inflection, verb→noun and word order. Partial word
  overlap never counts. Checked against 150 hand-judged dev pairs: precision 1.00, recall 0.88.
- **lenient**: at least half the words shared. Accepts partial matches (precision 0.62 on the same
  150 pairs).
- **strict**: identical after normalization.

## Version 2.1 results

| Split | Match | Precision | Recall | F1 (95% CI) | TP / FP / FN |
|---|---|---|---|---|---|
| **60 held-out jobs** | equivalent | 27.0 | 24.9 | **25.9** (22.6–29.3) | 360 / 974 / 1,085 |
| | lenient | 42.4 | 39.2 | **40.7** (37.0–44.9) | 566 / 768 / 879 |
| | strict | 24.2 | 22.4 | **23.2** (20.1–26.7) | 323 / 1,011 / 1,122 |
| 40 dev jobs (optimistic) | equivalent | 35.8 | 30.2 | 32.8 (28.6–37.5) | 271 / 485 / 627 |
| | lenient | 52.4 | 44.1 | 47.9 (43.4–52.3) | 396 / 360 / 502 |
| | strict | 32.8 | 27.6 | 30.0 (25.8–34.6) | 248 / 508 / 650 |

The dev numbers reproduce exactly from this package:
`python -m rule_nlp_v21.pipeline --gold dev && python -m rule_nlp_v21.evaluate --split dev`
(this needs `postings.csv`).

## Compared with earlier Sprint 2 versions (60 held-out jobs, same evaluator)

| F1 | Version 1 | Version 2 | **Version 2.1** |
|---|---|---|---|
| equivalent | 19.1 | 25.5 | **25.9** |
| lenient | 31.9 | 40.2 | **40.7** |
| strict | 16.9 | 22.9 | **23.2** |

The earlier versions are not included in this package; their scores come from the original project.
Version 2.1 clearly beats Version 1. Its gain over Version 2 on held-out jobs is small (within the
confidence intervals); it mainly fixes specific failures:
- empty output on unstructured JDs,
- place names,
- degree fields counted as skills,
- junk single words,
- promotional text.

## How to read the scores
- **Low recall is partly a labeling effect.** Most missed gold skills are reviewer paraphrases that
  do not appear as a phrase in the JD ("construction plan reading" for "Read and examine plans").
- **Precision against the gold labels understates relevance.** In a manual review, about half of the
  "extra" skills were valid JD-supported skills that the labels did not list.
- **A final independent evaluation is pending:** 50 new jobs, labelled blind (in progress).

## Decision for Sprint 3
The team must decide whether Sprint 3 uses the **Rule+NLP** skills (this package), the **LLM** skills,
or a **combination**:
- **Rule+NLP strengths:** transparent, reproducible, every skill tied to its JD evidence and position,
  and no API cost.
- **Rule+NLP limits:** misses paraphrased or implied skills, and produces a long tail of specific
  phrasings that clustering will need to merge.

# Section A (Team 2) task taxonomy: hand-off for Section B

A taxonomy of **work tasks** learned from federal job postings (USAJobs).
2,034 task statements from 451 postings, grouped into **46 tasks**,
**12 groups** (L2) and **5 areas** (L3).

## How it was built
1. Postings: a USAJobs sample covering 11 job families.
2. Task statements: duty sentences extracted from each posting.
3. Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
4. Clustering: HDBSCAN; only topics that are stable across 10 random seeds are kept (`status =
   persistent`). Three are less stable (`provisional`).
5. Hierarchy: Ward clustering of the task centroids into groups and areas; names proposed by an LLM and
   reviewed by the team.
6. Check: every task is compared with its closest O*NET task and work activity (GWA).

## Files
| File | Rows | What |
|---|---|---|
| `task_taxonomy.csv` | 46 | One row per task: id, name, description, status, size, group, area, postings, agencies, job families, O*NET comparison, 3 examples |
| `task_statements.csv` | 2,034 | Every statement behind a task, with posting id, title, agency and job family |
| `posting_tasks.csv` | 1,139 | posting_id → task_id with the number of statements. Lets you measure which of your skills appear in the same postings as our tasks, if your corpus includes these USAJobs postings |

## How we use your skills
We map each of our tasks to your skill topics (`branch = skill`; requirement and generic topics
left out). We use meaning similarity between our task and your topic, plus co-occurrence in the
same postings if you can share posting → topic assignments.

Questions: Abhisek (Section A, Team 2).

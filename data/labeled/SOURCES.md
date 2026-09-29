# Labeled data sources

Content warning: the toxicity and jailbreak files contain offensive or harmful prompts, included only to evaluate the judges.

| Judge | Source | URL | Licence | Seed | Cases | Prepared (UTC) |
|---|---|---|---|---|---|---|
| groundedness | HaluEval QA | https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json | MIT | 42 | 50 | 2026-09-28 |
| toxicity | Civil Comments | https://huggingface.co/datasets/google/civil_comments | CC0-1.0 | 42 | 50 | 2026-09-28 |
| jailbreak_compliance | JBB-Behaviors + responses from Claude Haiku 4.5 (file kept out of the public repo; may contain harmful text) | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors | MIT | 42 | 50 | 2026-09-28 |
| jailbreak_compliance | JailbreakBench PAIR artifacts vs Vicuna-13B (file kept out of the public repo) | https://github.com/JailbreakBench/artifacts | MIT | 42 | 25 | 2026-09-29 |
| sensitive_handling | own prompts + normal and deliberately flawed replies | - | own | 42 | 60 | 2026-09-29 |

Notes:
- jailbreak_compliance: 4 of the 25 JailbreakBench cases (jbk-072 to jbk-075) were dropped unlabeled; 71 labeled in total (20 fail / 51 pass).
- sensitive_handling: 7 generations where the model refused to write the flawed variant and referred to the test itself were excluded (listed in data/labeling/sensitive_handling.excluded.txt); the first 30 of the remaining 53 were labeled (10 fail / 20 pass).

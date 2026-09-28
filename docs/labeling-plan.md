# Plan: building the human-labeled calibration set

Target: **~200 labeled examples, 50 per judge**, labeled by me. The judges are
compared against these labels, so they must be my own careful judgments, not
copied dataset labels.

## Tools

```bash
python scripts/prepare_labeling.py groundedness     # or toxicity / jailbreak_compliance
python scripts/prepare_labeling.py sensitive_handling --prompts my_prompts.txt
judgekit label --todo data/labeling/<judge>.todo.jsonl --out data/labeled/<judge>.jsonl
judgekit split --data data/labeled/<judge>.jsonl --test-size 20 --seed 42
```

The prepare script samples with a fixed seed and gives cases opaque ids; where
each case came from (and any dataset hint) is in `<judge>.provenance.jsonl`,
which I don't open until I've finished labeling.

## Ground rules

- **Balance.** Aim for about 25 `fail` and 25 `pass` per judge. With all one
  class, κ is undefined; with 45/5 it is unstable.
- **Include hard cases.** About 10 per judge should be borderline (sarcasm, a
  claim that is true but not in the context, a refusal that still leaks a
  hint). These are where rubrics break.
- **Split before iterating.** Per judge: `data/labeled/<judge>.dev.jsonl` (30 cases)
  for improving the rubric, and `data/labeled/<judge>.test.jsonl` (20 cases), which
  I don't look at while iterating. README numbers come from **test only**.
- **Label blind.** Write the labeling guideline first, then label without
  seeing the judge's verdict, so I'm not anchored by it.
- **Write the `note`.** One short sentence on *why*. It appears next to the
  judge's reason in `judgekit calibrate`'s disagreement list.
- **Check my own consistency.** A day later, re-label 10 random items without
  looking. If I disagree with myself often, the guideline needs work before the
  rubric does.
- **Seeded sampling.** Pick items with a fixed random seed and record it, so the
  selection is reproducible.

## JSONL format (one object per line)

```json
{"id": "grd-017", "input": "<question>", "context": "<source text>", "output": "<answer being judged>", "human_label": "fail", "note": "adds a date not in the context"}
```

`id` must be unique, `human_label` is exactly `"pass"` or `"fail"`, and
`output` is required. `context` matters for groundedness; it can be omitted
for the others.

## Per judge

### groundedness: HaluEval (QA split)

- Each item has `knowledge`, `question`, `right_answer` and `hallucinated_answer`.
- Map: `input` = question, `context` = knowledge, `output` = one of the two answers.
- Sample about 25 items using the right answer and 25 using the hallucinated one, then **read
  each one and label it myself**. The dataset's own label is only a hint.
- Hand-edit about 5 right answers to add a true-but-unsupported fact. Those should
  fail, and they're the case judges most often get wrong.

### toxicity: Civil Comments (CC0)

- `output` = the comment or statement text; `input` can be a generic user
  question, since the judge only rates the output.
- Mix: about 20 clearly toxic, 20 clearly benign, and 10 borderline (sarcasm,
  *discussing* a slur without using it, firm but polite disagreement).
- Heads-up: this file will contain offensive text. Put a content warning at
  the top of `data/labeled/SOURCES.md`.

### jailbreak_compliance: JailbreakBench (JBB-Behaviors)

- The behaviours are prompts, not responses. Generate responses from a target
  model (some wrapped in role-play or "hypothetically" framing), then label
  whether each response actually complied.
- Aim for a mix of clean refusals, safe redirects, partial compliance, and
  full compliance.
- **Safety:** do not publish actionable harmful instructions. Either redact the
  harmful specifics (e.g. `[REDACTED: synthesis steps]`) while keeping it clear
  the model complied, or keep this file out of the public repo (add it to
  `.gitignore`) and publish only the metrics.

### sensitive_handling: written by me

- 50 cases across self-harm, medication and symptoms, legal trouble, debt,
  grief, and abuse.
- For each prompt, write a **good** response (supportive, safe, points to a
  professional or a crisis line where needed) and a **bad** one (dangerous
  specifics, false-certainty diagnosis, dismissive, or a cold refusal with no
  help), then label them. Aim for about 25 of each.
- No real people or personal details. Keep it fictional.

## Licence check (record it for every source)

Create `data/labeled/SOURCES.md` with one row per source: URL, version or
commit, licence, date checked, which ids I used, and the random seed. For each
source, check:

1. The **LICENSE file** in the source repo.
2. The **dataset card** (Hugging Face / Kaggle): licence field and usage terms.
3. The **README or paper** for usage restrictions. For example, ToxiGen's README
   says the data is "intended to be used for research purposes only".
4. **Upstream sources**: HaluEval items derive from other datasets (e.g.
   HotpotQA) that carry their own licences.
5. **Redistribution terms**: Jigsaw's comment text is CC BY-SA 3.0, so a file
   containing it needs attribution and must itself be CC BY-SA.

What I found at the source (verify again when downloading):

| Source | Licence | Watch out for |
|---|---|---|
| HaluEval | MIT (GitHub repo) | Upstream dataset licences |
| ToxiGen | MIT code; data "for research purposes only" | Keep use non-commercial; cite the paper |
| Civil Comments (used for toxicity) | CC0-1.0 | None: public domain. Chosen over Jigsaw (CC BY-SA text) and ToxiGen (research only) |
| Jigsaw (not used) | CC0 labels; Wikipedia text CC BY-SA 3.0 | Kaggle competition rules; attribution and share-alike |
| JailbreakBench (JBB-Behaviors) | MIT | Generated responses are mine; don't publish harmful specifics |
| My sensitive cases | Mine | Keep it fictional |

## Iteration loop (after labeling)

1. Commit the current rubric and tag it: `git tag rubric/toxicity-v1`.
2. Run `judgekit calibrate --judge toxicity --data data/labeled/toxicity.dev.jsonl`.
3. Read the disagreements, change **one thing** in the rubric, commit it
   (`toxicity rubric: treat sarcasm aimed at the user as toxic`), and re-run.
4. When dev results stop improving, run once on `toxicity.test.jsonl` and put
   those numbers, plus the rubric's commit hash, in the README table.

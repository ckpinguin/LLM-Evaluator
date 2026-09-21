# Contract: Results History (`<workspace>/results.jsonl`)

One line per evaluation result, in [JSON Lines](https://jsonlines.org/) format. The file is only
ever **appended to**; it is kept when downloads are deleted (FR-019). Fields are defined in
[data-model.md](../data-model.md#evaluationresult-stored-in-resultsjsonl).

## Example line (wrapped here for readability; one line in the file)

```json
{"run_id": "2026-09-21T10-15-03", "created_at": "2026-09-21T10:16:27+02:00",
 "model_id": "HuggingFaceTB/SmolLM2-135M", "model_type": "generative",
 "evaluator": "arc-easy", "main_metric": "acc_norm", "main_score": 0.58,
 "higher_is_better": true, "scores": {"acc": 0.55, "acc_norm": 0.58},
 "examples": 200, "duration_seconds": 84.2, "device": "mps", "testbed_version": "0.1.0"}
```

## Rules

- Lines that cannot be read (e.g., a hand-edit mistake) are skipped with a warning naming the line
  number; the other results are still shown.
- New optional fields may be added in later versions; readers ignore fields they do not know.

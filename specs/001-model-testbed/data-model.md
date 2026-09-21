# Data Model: Model Testbed

**Feature**: [spec.md](./spec.md) | **Research**: [research.md](./research.md) | **Date**: 2026-09-21

The testbed has no database. There are two kinds of data:

- data **stored in files**: the catalog (`catalog.yaml`) and the results history
  (`<workspace>/results.jsonl`),
- data that **only exists in memory while the program runs**: evaluators (defined in code) and
  the current evaluation run.

File formats are specified in [contracts/catalog-file.md](./contracts/catalog-file.md) and
[contracts/results-file.md](./contracts/results-file.md).

## ModelType (fixed list)

| Value | Meaning |
| --- | --- |
| `generative` | Produces text (text generation / causal language model) |
| `embedding` | Turns text into a vector of numbers |
| `classification` | Assigns a label to a text |

## CatalogEntry (stored in `catalog.yaml`)

One model the user may select.

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `id` | text | yes | Hugging Face model id `owner/name`; unique in the catalog |
| `type` | ModelType | yes | One of the three values above |
| `description` | text | yes | One line, ≤ 120 characters |
| `size_mb` | whole number | yes | Approximate download size in MB; > 0 |
| `license` | text | yes | License id from Hugging Face (e.g., `apache-2.0`), or `unknown` |
| `task` | text | classification only | v1 allows only `sentiment` |
| `label_map` | map text → text | classification only | Model label → `negative` / `neutral` / `positive`; must contain `negative` and `positive` |

Validation (on load, FR-005/FR-006): unknown fields, duplicate ids, a missing `task`/`label_map`
for classification, or `task`/`label_map` on other types are errors. The error message names the
entry and the field.

## Catalog (stored in `catalog.yaml`)

| Field | Type | Rules |
| --- | --- | --- |
| `generated_at` | date | Date the catalog was last generated (informational) |
| `max_size_gb` | number | Size filter that was used (informational) |
| `models` | list of CatalogEntry | At least one entry |

## Evaluator (defined in code, in the evaluator registry)

| Field | Type | Rules |
| --- | --- | --- |
| `key` | text | Unique, lowercase-with-dashes (e.g., `arc-easy`) |
| `name` | text | Human-friendly name |
| `model_type` | ModelType | Which models it can evaluate |
| `task` | text or none | For classification: the task it needs (e.g., `sentiment`); none otherwise |
| `description` | text | Plain-language description of what is measured |
| `dataset` | text | Hugging Face dataset id and split |
| `main_metric` | text | Name of the score shown first (e.g., `acc_norm`) |
| `higher_is_better` | yes/no | How to read the score (FR-014) |
| `how_to_read` | text | Plain-language interpretation, e.g., "0.25 = random guessing, 1.0 = perfect" |
| `supports_limit` | yes/no | `no` for fixed-size benchmarks (e.g., `nano-scifact`) |
| `is_default` | yes/no | Exactly one default per model type (and task) |
| `run` | function | Runs the evaluation; returns a dictionary of metric name → number |

**Compatibility rule (FR-015)**: an evaluator is offered for a catalog entry when
`evaluator.model_type == entry.type` and (`evaluator.task` is none or
`evaluator.task == entry.task`).

The v1 evaluators are listed in [research.md §5](./research.md#5-evaluation-approach-per-model-type).

## EvaluationRun (in memory only)

One execution: one model and one or more evaluators (FR-016).

| Field | Type | Notes |
| --- | --- | --- |
| `run_id` | text | Start time, e.g., `2026-09-21T10-15-03` |
| `model` | CatalogEntry | The selected model |
| `evaluators` | list of Evaluator | All compatible with `model`; at least one |
| `limit` | number | Sample size; default 200 |
| `device` | text | `cuda`, `mps`, or `cpu` |
| `status` | RunStatus | See state diagram |
| `results` | list of EvaluationResult | One per evaluator that finished |
| `errors` | map evaluator key → text | Evaluators that failed, with a plain-language message |

### RunStatus state transitions

```text
checking ──► downloading ──► evaluating ──► cleaning_up ──► succeeded
   │              │               │              ▲
   │              │               │              │
   └──────────────┴───────────────┴── error ─────┤──► failed
                  │               │              │
                  └───────────────┴── Ctrl+C ────┘──► cancelled
```

- `checking`: lock acquired, disk space and model metadata checked. If this fails, nothing was
  downloaded; the run ends as `failed` after `cleaning_up`.
- `evaluating`: evaluators run one after another. If one evaluator fails, its error is recorded
  and the next evaluator still runs. The run is `succeeded` if at least one evaluator finished,
  otherwise `failed`.
- `cleaning_up` **always** runs (FR-010), then the lock is released.

## EvaluationResult (stored in `results.jsonl`)

The outcome of one evaluator in one run (FR-018, FR-019).

| Field | Type | Notes |
| --- | --- | --- |
| `run_id` | text | Links results from the same run |
| `created_at` | date-time | When the evaluator finished (ISO 8601, local time with offset) |
| `model_id` | text | Hugging Face id |
| `model_type` | ModelType | |
| `evaluator` | text | Evaluator key |
| `main_metric` | text | Copied from the evaluator |
| `main_score` | number | Value of `main_metric` |
| `higher_is_better` | yes/no | Copied from the evaluator, so old results stay readable |
| `scores` | map text → number | All metrics returned by the evaluator |
| `examples` | number | Number of examples actually evaluated |
| `duration_seconds` | number | Time for this evaluator (download excluded) |
| `device` | text | `cuda`, `mps`, or `cpu` |
| `testbed_version` | text | Version of the testbed that produced the result |

**Comparison rule (FR-020)**: results are compared only within the same `evaluator`. For each
model the most recent result is shown, sorted best-first according to `higher_is_better`.

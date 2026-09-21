# Contract: Command-Line Interface

The program is installed as the command `testbed` and run with `uv run testbed <command>`.
Every command supports `--help`. Exit codes: `0` success, `1` user-facing error (message printed
in plain language, FR-023), `130` cancelled with Ctrl+C.

Common option values:

- `TYPE`: `generative` | `embedding` | `classification`
- `MODEL`: a Hugging Face id from the catalog, e.g., `HuggingFaceTB/SmolLM2-135M`
- `EVALUATOR`: an evaluator key, e.g., `arc-easy`

---

## `testbed models [--type TYPE]`

Lists the catalog (FR-001, FR-002). **Never downloads anything** (FR-007).

Output: one table per type with columns `#`, `Model`, `Size`, `License`, `Description`.

---

## `testbed evaluators [--type TYPE]`

Lists evaluators with key, model type, description, main metric, "higher/lower is better", and
whether `--limit` applies. The default evaluator per type is marked with `*`.

---

## `testbed evaluate [MODEL] [--evaluator EVALUATOR]... [--limit N] [--device auto|cuda|mps|cpu]`

Runs an evaluation (US1, US2).

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL` | ask | If omitted, ask for the type, then show a numbered model list to choose from |
| `--evaluator`, `-e` | default of the model's type | Repeat to run several evaluators in one run (FR-016) |
| `--limit` | 200 | Sample size for evaluators that support it (FR-017) |
| `--device` | `auto` | `auto` = `cuda` if available, else `mps`, else `cpu` (FR-021) |

Behavior:

1. Rejects models not in the catalog and evaluators not compatible with the model, listing the
   valid choices.
2. Prints the plan: model, size, evaluators, device, sample size.
3. Checks free disk space; stops with the needed amount if too little (FR-008).
4. Downloads with a progress bar (FR-009), then runs each evaluator with a progress message.
5. Prints one result block per evaluator: main score, how to read it, other metrics, examples,
   duration, device (FR-018, SC-008).
6. Always deletes downloaded files, then prints `Cleaned up <size> of downloads.` (FR-010).

Example:

```text
$ uv run testbed evaluate HuggingFaceTB/SmolLM2-135M -e arc-easy -e hellaswag
Model:       HuggingFaceTB/SmolLM2-135M (generative, 270 MB)
Evaluators:  arc-easy, hellaswag   Device: mps   Sample size: 200
Downloading ━━━━━━━━━━━━━━━━━━━━ 270/270 MB
Running arc-easy (1/2) ...
  ARC-Easy  acc_norm = 0.58  (higher is better; 0.25 = random guessing, 1.0 = perfect)
Running hellaswag (2/2) ...
  HellaSwag acc_norm = 0.42  (higher is better; 0.25 = random guessing, 1.0 = perfect)
Saved 2 results to workspace/results.jsonl
Cleaned up 312 MB of downloads.
```

---

## `testbed history [--type TYPE] [--evaluator EVALUATOR] [--model MODEL]`

Lists saved results, newest first, with columns `Date`, `Model`, `Evaluator`, `Main score`,
`Examples`, `Duration`, `Device` (US4, FR-020).

---

## `testbed compare EVALUATOR`

Shows the latest result of every model for one evaluator in a single table, sorted best-first
(US4 scenario 2).

---

## `testbed catalog build [--per-type N] [--max-size-gb X] [--output PATH]`

Regenerates the catalog from Hugging Face (US3, FR-004).

| Option | Default |
| --- | --- |
| `--per-type` | 5 |
| `--max-size-gb` | 2.0 |
| `--output` | `catalog.yaml` |

Prints how many candidates were checked, kept, and skipped (with the reason per skipped model).
On any error the existing file is left unchanged.

---

## `testbed cleanup`

Deletes everything in the downloads folder and prints how much space was freed. Refuses while a
run is in progress (lock held). The same cleanup also runs automatically at the start of
`evaluate` (FR-011).

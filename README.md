# Model Testbed

A small, beginner-friendly testbed for trying out Hugging Face text models (generative,
embedding, and classification) on an Apple silicon Mac. Pick a model from a curated catalog,
pick an evaluator, and get a score with a plain-language explanation. Models are downloaded only
for the run and deleted right afterwards, so your disk stays clean.

## Requirements

- An Apple silicon Mac (e.g., Mac mini) with at least 16 GB of memory
- [uv](https://docs.astral.sh/uv/) (Python project manager)
- An internet connection and a few GB of free disk space

## Setup

```bash
uv sync                 # installs Python 3.12 and all dependencies into .venv
uv run testbed --help   # shows the available commands
```

## Quick start

```bash
uv run testbed models                                      # what can I test?
uv run testbed evaluate sentence-transformers/all-MiniLM-L6-v2
uv run testbed evaluate                                    # or pick from numbered lists
```

Example output:

```text
Model:       sentence-transformers/all-MiniLM-L6-v2 (embedding, 87 MB)
Evaluators:  sts-benchmark   Device: mps   Sample size: 200
Downloading sentence-transformers/all-MiniLM-L6-v2 ...
Running sts-benchmark (1/1) ...
Deleting downloaded files ...
  STS Benchmark  spearman_cosine = 0.823  (higher is better; -1 to 1; 1.0 = perfect agreement
  with human ratings, above 0.8 is good)
  pearson_cosine = 0.830  |  200 examples, 8.2 s on mps
Saved 1 result(s) to workspace/results.jsonl
Cleaned up 91 MB of downloads.
```

## Commands

| Command | What it does |
| --- | --- |
| `uv run testbed models [--type TYPE]` | List the catalog. Nothing is downloaded. |
| `uv run testbed evaluators [--type TYPE]` | List evaluators: what they measure and how to read the score. `*` marks the recommended one per type. |
| `uv run testbed evaluate [MODEL] [-e EVALUATOR]... [--limit N] [--device auto\|mps\|cpu]` | Download the model, run the evaluator(s), show and save the scores, delete the model. Without `MODEL` you choose from numbered lists. |
| `uv run testbed history [--type TYPE] [--evaluator KEY] [--model MODEL]` | Show saved results, newest first. |
| `uv run testbed compare EVALUATOR` | Latest result of every model on one evaluator, best first. |
| `uv run testbed catalog build [--per-type N] [--max-size-gb X] [--output PATH]` | Regenerate the catalog from Hugging Face. |
| `uv run testbed cleanup` | Delete leftover downloads (e.g. after a crash). |

`TYPE` is `generative`, `embedding`, or `classification`. Add `--help` to any command for
details.

Examples:

```bash
# Two evaluators in one run: the model is downloaded only once.
uv run testbed evaluate HuggingFaceTB/SmolLM2-135M -e arc-easy -e hellaswag

# A quicker, rougher test with 50 instead of 200 examples.
uv run testbed evaluate Qwen/Qwen2.5-0.5B --limit 50

# Which sentiment model is best on SST-2?
uv run testbed compare sst2
```

## Evaluators

| Key | Model type | Main score | What it measures |
| --- | --- | --- | --- |
| `arc-easy` * | generative | `acc_norm` (higher is better) | Multiple-choice grade-school science questions |
| `hellaswag` | generative | `acc_norm` (higher is better) | Common-sense: choose the most sensible ending of a short story |
| `wikitext` | generative | `word_perplexity` (lower is better) | How well the model predicts real Wikipedia text |
| `sts-benchmark` * | embedding | `spearman_cosine` (higher is better) | How well sentence similarities match human ratings |
| `nano-scifact` | embedding | `ndcg@10` (higher is better) | Search: find scientific abstracts for a claim (fixed size: 50 questions) |
| `sst2` * | classification | `accuracy` (higher is better) | Positive/negative sentiment of movie-review sentences |
| `rotten-tomatoes` | classification | `accuracy` (higher is better) | Positive/negative sentiment of movie-review snippets |

They use the standard tool of each model family:
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for generative
models, the evaluators built into [Sentence Transformers](https://sbert.net) for embeddings, and
the Transformers `pipeline` with [scikit-learn](https://scikit-learn.org) metrics for
classifiers. Each evaluator uses a fixed sample of 200 examples by default (`--limit` changes
this), always the same ones, so results of different runs can be compared. Scores on such a
small sample are good for comparing models with each other, but won't exactly match official
leaderboard numbers.

## Where files live

| Path | What it is |
| --- | --- |
| `catalog.yaml` | The curated model list. Edit it by hand or regenerate it. |
| `workspace/results.jsonl` | Your results history, one result per line (JSON Lines). Kept. |
| `workspace/downloads/` | Temporary: models, datasets, and caches of the current run. Emptied after every run. |

The `workspace/` folder is ignored by git. Set the environment variable `TESTBED_WORKSPACE` to
put it somewhere else.

## How download-and-delete works

1. When a run starts, the testbed checks with Hugging Face how big the model is and whether
   your disk has enough free space (model size + 10% + 1 GB for datasets).
2. It downloads only the files it needs, and only weights in the safe `safetensors` format,
   into `workspace/downloads/model/`. Datasets and caches also go into
   `workspace/downloads/` (the testbed points Hugging Face's `HF_HOME` there, so your normal
   `~/.cache/huggingface` is never touched).
3. After the run, the whole `workspace/downloads/` folder is emptied: after success, after an
   error, and after Ctrl+C. If the computer crashes mid-run, the leftovers are removed at the
   next start (or with `uv run testbed cleanup`).
4. Only one run can happen at a time, so a cleanup never deletes the files of a run that is
   still going on in another terminal.

## The catalog

`catalog.yaml` lists the models you can choose from. Each entry looks like this:

```yaml
  - id: distilbert/distilbert-base-uncased-finetuned-sst-2-english
    type: classification
    task: sentiment            # classification only
    label_map:                 # classification only: model label -> negative/neutral/positive
      NEGATIVE: negative
      POSITIVE: positive
    description: DistilBERT fine-tuned on SST-2 movie-review sentiment (67M parameters).
    size_mb: 256
    license: apache-2.0
```

- **Edit by hand**: add or remove entries. If you make a mistake, the testbed tells you which
  entry and field are wrong.
- **Regenerate**: `uv run testbed catalog build` looks at the most downloaded models on Hugging
  Face and keeps the first few per type that are small enough (default 2 GB), use
  `safetensors`, need no sign-in, and run no custom code. For sentiment models it also reads
  the label names. It prints every skipped model with the reason. Use `--output other.yaml` to
  try it without replacing your catalog, and review the result by hand.

## How to add a new evaluator

1. Pick the module for the model type: `src/testbed/evaluators/generative.py`,
   `embedding.py`, or `classification.py`.
2. Write a function `run_my_eval(model_dir, entry, device, limit)` that loads the model from
   `model_dir`, evaluates it on at most `limit` examples, and returns
   `EvaluatorOutput(scores={"my_metric": 0.87}, examples=200)`. Look at the existing
   evaluators in the same file; most need only a few lines.
3. Add an `Evaluator(key="my-eval", name=..., model_type=..., description=..., dataset=...,
   main_metric="my_metric", higher_is_better=True, how_to_read=..., run=run_my_eval)` to the
   list at the bottom of that file (e.g. `GENERATIVE_EVALUATORS`).
4. Check it: `uv run pytest tests/unit/test_evaluator_registry.py`, then try it with
   `uv run testbed evaluate <model> -e my-eval --limit 10`.

## Development

```bash
uv run pytest                      # fast unit tests, no internet needed
uv run pytest -m network           # end-to-end runs with tiny real models (needs internet)
uv run ruff check . && uv run ruff format --check .
```

Code layout: `src/testbed/` holds the core (`catalog.py`, `runner.py`, `downloads.py`,
`results.py`, `evaluators/`) and the command-line interface (`cli.py`, the only module that
prints). The design documents are in `specs/001-model-testbed/`.

## Troubleshooting

- **"Not enough free disk space"**: free some space or pick a smaller model. The message says
  how much is needed.
- **Slow run / "Device: cpu"**: the Apple GPU was not available. Operations the GPU does not
  support fall back to the CPU automatically; this is slower but works.
- **"The model did not fit into memory"**: choose a smaller model or close other programs.
  Models up to about 1 billion parameters work on a 16 GB Mac.
- **"Warning: You are sending unauthenticated requests to the HF Hub"**: harmless. The testbed
  only uses public models; setting a Hugging Face token (`HF_TOKEN`) only raises download
  rate limits.
- **"Another testbed run is in progress"**: wait for the other run to finish. If no run is
  going on (e.g. after a crash), just try again; the lock is released automatically when a
  program ends.

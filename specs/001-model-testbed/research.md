# Research: Model Testbed

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-09-21

This file records each technical decision for the testbed, why it was made, and what else was
considered. Versions were resolved with `uv pip compile` for Python 3.12 on Apple silicon on
2026-09-21; the exact versions are locked in `uv.lock` during implementation.

## 1. Language, runtime, and environment manager

- **Decision**: Python 3.12, managed with `uv` (`pyproject.toml` + `uv.lock`).
- **Rationale**: Python is the standard language for machine learning, and all Hugging Face and
  PyTorch tools are Python-first. 3.12 has prebuilt wheels for every dependency below on Apple
  silicon. `uv` is now the most widely used Python project manager: one command (`uv sync`)
  installs the right Python version and all packages, and `uv run` works without activating a
  virtual environment, which is easy for beginners.
- **Alternatives considered**: Python 3.14 (installed on the machine via Homebrew, but some ML
  wheels still lag behind); `pip` + `venv` (works, but more manual steps and no lock file);
  conda (heavier, not needed without CUDA).

## 2. Machine-learning runtime on the Mac mini

- **Decision**: PyTorch (2.11) with the Apple GPU backend (`mps`), automatic CPU fallback, models
  loaded in `float32`, and `PYTORCH_ENABLE_MPS_FALLBACK=1` set before PyTorch is imported.
- **Rationale**: PyTorch is the only mainstream framework that Hugging Face Transformers,
  Sentence Transformers, and lm-evaluation-harness all support fully on Apple GPUs. `float32` is
  the most reliable precision on `mps` (half precision can overflow for some models) and makes
  repeated runs give the same scores (SC-005). Models are capped at 2 GB download (~1B
  parameters), so even in `float32` they need at most ~4 GB of memory, well under 16 GB. The
  fallback variable lets operations that `mps` does not implement run on the CPU instead of
  crashing (edge case "GPU acceleration unsupported").
- **Alternatives considered**: Apple MLX (fast on Apple silicon, but not supported by the standard
  evaluation libraries); `bfloat16` / `float16` (less memory, but less predictable on `mps` and not
  needed at this model size); ONNX Runtime (extra conversion step).

## 3. Model loading libraries

- **Decision**: Hugging Face `transformers` (5.x) for generative and classification models,
  `sentence-transformers` (6.x) for embedding models, `huggingface-hub` (1.x) for search and
  download.
- **Rationale**: These are the official, best-documented libraries for Hugging Face models. Each
  model type uses the library that its model cards document, so beginners can follow the official
  examples.
- **Security rules** (best practice): always `trust_remote_code=False`; download only
  `*.safetensors` weights (never pickled `*.bin` files); models that need custom code are
  excluded from the catalog.

## 4. Download-on-use and cleanup (FR-007 to FR-012)

- **Decision**:
  1. The testbed sets `HF_HOME=<workspace>/downloads/hf-home` at start-up, **before** any Hugging
     Face library is imported. All Hugging Face caches of this process (models, datasets,
     lm-evaluation-harness datasets, Xet chunk cache) are then inside the testbed's own folder;
     the user's normal `~/.cache/huggingface` is never touched.
  2. The model is downloaded with `snapshot_download(repo_id, local_dir=<workspace>/downloads/
     model, allow_patterns=[...], ignore_patterns=[...])`, keeping only weights (`*.safetensors`),
     configuration (`*.json`), and tokenizer files (`tokenizer*`, `*.txt`, `*.model`). Folders
     such as `onnx/`, `openvino/`, and TensorFlow/Flax weights are ignored. This keeps the download
     small and the folder easy to inspect.
  3. Before downloading, the file sizes are read with `HfApi().model_info(repo_id,
     files_metadata=True)`, summed for the files that will be downloaded, and compared with
     `shutil.disk_usage(...).free`. The run needs model size × 1.1 + 1 GB free (1 GB covers
     evaluation datasets).
  4. The whole run is wrapped in `try: ... finally: cleanup()`. `cleanup()` deletes everything
     inside `<workspace>/downloads/`. `KeyboardInterrupt` (Ctrl+C) is caught, reported as
     "cancelled", and still triggers `finally`.
  5. At every start-up the testbed deletes anything left in `<workspace>/downloads/` (crash or
     power loss).
  6. A lock file (`<workspace>/testbed.lock`, via the `filelock` package) makes sure only one run
     happens at a time, so start-up cleanup never deletes files of a run in another terminal.
- **Rationale**: Redirecting `HF_HOME` is the officially documented way to relocate all Hugging
  Face caches. It covers libraries that download on their own (lm-evaluation-harness and
  `datasets`) without special code. `try/finally` is the standard Python pattern for guaranteed
  cleanup.
- **Alternatives considered**: using the default Hugging Face cache and deleting single entries
  with `huggingface-cli delete-cache` (risks touching the user's other cached models); a per-run
  temporary directory via `tempfile` (datasets downloaded by lm-evaluation-harness would still go
  to the global cache).

## 5. Evaluation approach per model type

General rule: use the **standard evaluation tool of each model family** and **standard public
datasets**, run on a small fixed sample (default 200 examples, `--limit` to change). Samples are
deterministic: fixed seed 42 for shuffling, or the first N items where the tool does that.

### 5a. Generative models → EleutherAI lm-evaluation-harness (`lm-eval[hf]` 0.4.x)

- **Decision**: Call `lm_eval.simple_evaluate(model="hf", model_args={"pretrained": <local model
  folder>, "device": <device>, "dtype": "float32", "max_length": 2048}, tasks=[<task>],
  limit=<N>, batch_size=1)`.
- **Evaluators**:

  | Key | lm-eval task | Main metric | Better | What it measures |
  | --- | --- | --- | --- | --- |
  | `arc-easy` (default) | `arc_easy` | `acc_norm` | higher | Grade-school science questions |
  | `hellaswag` | `hellaswag` | `acc_norm` | higher | Common-sense sentence completion |
  | `wikitext` | `wikitext` | `word_perplexity` | lower | How well it predicts Wikipedia text |

- **Rationale**: lm-evaluation-harness is the reference implementation behind the Open LLM
  Leaderboard and most published results for small language models. Multiple-choice scoring has
  subtle details (tokenization at answer boundaries, length normalization) that the harness gets
  right; writing it ourselves would be neither "proven" nor simpler to trust. `limit` supports the
  fixed sample size. `max_length=2048` stops long-context models (e.g., 32k tokens) from using
  huge amounts of memory in the perplexity task. `batch_size=1` keeps memory low; with ≤1B models
  a 200-example run still takes only a few minutes on an M2.
- **Alternatives considered**: Hugging Face LightEval (fast-changing API, harder for beginners);
  implementing perplexity/multiple-choice ourselves (educational, but duplicates a proven tool and
  risks subtle scoring errors); Hugging Face `evaluate` (maintainers now point to other tools for
  language-model evaluation).

### 5b. Embedding models → Sentence Transformers built-in evaluators

- **Decision**:

  | Key | Tool | Dataset | Main metric | Better |
  | --- | --- | --- | --- | --- |
  | `sts-benchmark` (default) | `EmbeddingSimilarityEvaluator` | `sentence-transformers/stsb` (test) | Spearman correlation (cosine) | higher |
  | `nano-scifact` | `NanoBEIREvaluator(dataset_names=["scifact"])` | NanoBEIR SciFact | NDCG@10 | higher |

- **Rationale**: These evaluators ship with Sentence Transformers, are used in its official
  training docs, and compute the same metrics as the MTEB leaderboard (STS: Spearman of cosine
  similarity; retrieval: NDCG@10). STS supports sampling 200 pairs. NanoBEIR is a deliberately
  small version of the BEIR retrieval benchmark (50 queries, a few thousand documents) that runs
  in seconds to minutes. It has a fixed size, so `--limit` does not apply to it (shown in the
  evaluator list).
- **Alternatives considered**: the MTEB package (the leaderboard standard, but tasks cannot be
  limited to N examples, runs are slower, and it keeps its own results cache outside our
  workspace unless extra settings are used). MTEB can be added later as an extra evaluator.

### 5c. Classification models → Transformers pipeline + scikit-learn metrics

- **Decision**: Scope for v1 is **English sentiment classification**, the most common text
  classification task on Hugging Face. Each catalog entry maps the model's own label names to the
  canonical labels `negative` / `neutral` / `positive` (`label_map`). The testbed runs
  `pipeline("text-classification", model=<local folder>, device=<device>, top_k=None,
  truncation=True)` to get a score for every label, picks the highest-scoring label **among the
  labels the dataset uses** (so a 3-class model can be tested on 2-class data by ignoring
  `neutral`), and computes accuracy and macro-F1 with `sklearn.metrics`.
- **Evaluators**:

  | Key | Dataset | Labels | Main metric | Better |
  | --- | --- | --- | --- | --- |
  | `sst2` (default) | `stanfordnlp/sst2` (validation) | negative, positive | accuracy | higher |
  | `rotten-tomatoes` | `cornell-movie-review-data/rotten_tomatoes` (test) | negative, positive | accuracy | higher |

- **Rationale**: The Transformers pipeline is the documented way to run a classifier, and
  scikit-learn's metrics are the reference implementations of accuracy/F1. The label map is the
  simplest reliable way to compare models that use different label names (e.g., `POSITIVE` vs
  `positive`).
- **Alternatives considered**: Hugging Face `evaluate.evaluator("text-classification")` (does the
  same thing but the library is in maintenance mode); zero-shot classification with NLI models
  (a different use case, can be added later as its own task); emotion/topic tasks (the data model
  supports more tasks; they are left for later to keep v1 small).

### 5d. Evaluator registry

- **Decision**: Evaluators are plain Python objects listed in one registry module (a dictionary
  from key to evaluator). Each evaluator has descriptive fields plus one `run(...)` function. No
  plugin system or entry points.
- **Rationale**: Beginner-friendly and easy to extend: adding an evaluator means adding one
  function and one registry entry (Principle II).

## 6. Catalog generation from Hugging Face (FR-004)

- **Decision**: `HfApi().list_models(pipeline_tag=..., filter=<library tag>, sort="downloads",
  gated=False, num_parameters="max:2B", limit=100)` per model type (huggingface-hub 1.x selects
  the library through the `filter` tag list; `num_parameters` is only a rough pre-filter), then
  `model_info(..., files_metadata=True)` for each candidate. A candidate is kept only if it:
  - is not a test repository with random weights (ids containing `internal-testing`,
    `-testing/`, or `tiny-random`),
  - has `*.safetensors` weights whose total size is ≤ `--max-size-gb` (default 2),
  - has no `custom_code` tag and is not quantized (`gguf`, `mlx`, `gptq`, `awq` tags excluded),
  - for classification: its `config.json` labels (`id2label`) can be mapped automatically to
    `negative` and `positive` (and optionally `neutral`; "very negative/positive" count as
    negative/positive); `config.json` is a few KB, downloaded to a temporary folder and deleted.
    Sentiment models are found by their labels rather than by a name search, because the most
    popular one (`distilbert-base-uncased-finetuned-sst-2-english`) has no "sentiment" in its
    name.

  The first `--per-type` (default 5) candidates by downloads are written to `catalog.yaml`. The
  new file is written to a temporary file first and then renamed, so a failure never damages the
  existing catalog (US3 scenario 3).

  | Type | `pipeline_tag` | `filter` (tags) |
  | --- | --- | --- |
  | generative | `text-generation` | `transformers` |
  | embedding | `sentence-similarity` | `sentence-transformers` |
  | classification | `text-classification` | `transformers`, `en` |

- **Rationale**: Uses only the official Hub client and public metadata; results are reproducible
  and filters are easy to explain.
- **Initial catalog** (FR-003): produced once by running the builder, then reviewed by hand and
  committed (the review notes are at the top of `catalog.yaml`). Finding during implementation:
  many popular sentiment models (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`,
  `siebert/sentiment-roberta-large-english`) publish only pickled `*.bin` weights on their main
  branch, so the safetensors-only rule excludes them.
- **Known limitation**: some embedding models expect a query prefix (e.g., "query: "); v1 does not
  add prefixes, so those models may score lower on retrieval. An optional `query_prompt` field can
  be added later.

## 7. Command-line interface (FR-022)

- **Decision**: Typer (0.27) for commands, Rich (installed with Typer) for tables and progress.
- **Rationale**: Typer turns plain, type-hinted Python functions into commands with automatic
  `--help`; it is built on Click and is widely used. This is much less code than `argparse` for
  several subcommands, and the functions stay reusable by the future web page. Heavy libraries
  (PyTorch etc.) are imported inside functions so that `testbed --help` and `testbed models`
  respond instantly.
- **Alternatives considered**: `argparse` (standard library, but verbose for 7 subcommands);
  Click (Typer is Click with less boilerplate); Gradio/Streamlit now (deferred by the
  clarification; the core is kept separate from the CLI so Gradio can be added later).

## 8. Storage formats

- **Decision**:
  - Catalog: `catalog.yaml` at the repository root (PyYAML), validated with Pydantic 2 models.
  - Results history: `<workspace>/results.jsonl`, one JSON object per line (standard library).
  - Settings: defaults as constants in `settings.py`, overridable by CLI options; workspace folder
    overridable with the `TESTBED_WORKSPACE` environment variable. No separate config file.
- **Rationale**: YAML is easy to read and edit by hand and allows comments. Pydantic gives clear,
  beginner-readable error messages when a hand-edited catalog has a mistake. JSON Lines needs no
  database, can be appended safely, and is readable in any editor.
- **Alternatives considered**: TOML (standard library can read but not write it); JSON catalog (no
  comments); SQLite for results (overkill for a few hundred records); dataclasses with manual
  validation (more code and worse error messages than Pydantic).

## 9. Testing and code quality

- **Decision**: `pytest` 9 and `ruff` (lint + format).
  - **Unit tests** (fast, offline): catalog loading/validation, evaluator compatibility, label
    mapping, disk-space check, cleanup, lock, results history and comparison. Network and model
    loading are replaced with small fakes via `monkeypatch`.
  - **Integration tests** (marked `@pytest.mark.network`, run on demand): one end-to-end run per
    model type with a tiny public test model, checking that a result is saved and the downloads
    folder is empty afterwards.
- **Rationale**: pytest and ruff are the de-facto standards; plain test functions are the easiest
  tests for beginners to read.

## 10. Dependency summary

Runtime: `torch`, `transformers`, `sentence-transformers`, `huggingface-hub`, `datasets`,
`lm-eval[hf]`, `scikit-learn`, `typer`, `pyyaml`, `pydantic`, `filelock`.
Development: `pytest`, `ruff`.

All are widely adopted and actively maintained (Principle I). Versions resolved on 2026-09-21:
torch 2.11.0, transformers 5.17.0, sentence-transformers 6.1.0, huggingface-hub 1.32.0,
datasets 5.0.1, lm-eval 0.4.13, scikit-learn 1.9.1, typer 0.27.2, pyyaml 6.0.3, pydantic 2.13.5,
pytest 9.1.1, ruff 0.16.8. `pyproject.toml` uses lower bounds at these versions and upper bounds
at the next major version; `uv.lock` pins exact versions.

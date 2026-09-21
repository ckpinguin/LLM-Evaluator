# Quickstart & Validation Guide: Model Testbed

Use this guide to set up the testbed and to check, step by step, that the feature works
end-to-end. Command details are in [contracts/cli.md](./contracts/cli.md).

## Prerequisites

- Apple silicon Mac (tested target: Mac mini, 16 GB memory or more), recent macOS.
- [`uv`](https://docs.astral.sh/uv/) installed (`uv --version`).
- Internet connection and a few GB of free disk space.

## Setup

```bash
cd ml1
uv sync                 # installs Python 3.12 and all dependencies into .venv
uv run testbed --help   # shows the available commands
```

## Validation scenarios

Run the scenarios in order. "Downloads folder" means `workspace/downloads/`.

### 1. Browsing downloads nothing (US1-1, SC-006)

```bash
du -sh workspace/downloads 2>/dev/null || echo "empty"
uv run testbed models
uv run testbed evaluators
du -sh workspace/downloads 2>/dev/null || echo "empty"
```

**Expected**: models grouped by type, evaluators with descriptions and a `*` default per type;
the downloads folder is empty (or missing) both times.

### 2. Evaluate one model per type (US1-2, US1-3, SC-001, SC-002)

```bash
uv run testbed evaluate sentence-transformers/all-MiniLM-L6-v2
uv run testbed evaluate distilbert/distilbert-base-uncased-finetuned-sst-2-english
uv run testbed evaluate HuggingFaceTB/SmolLM2-135M
ls -A workspace/downloads
```

**Expected**: a download progress bar, then one result per run with a plain-language explanation
(e.g., "higher is better"), device `mps`, 200 examples (except fixed-size evaluators), and the
line `Cleaned up ... of downloads.`. The final `ls` prints nothing.

### 3. Interactive mode (US1)

```bash
uv run testbed evaluate
```

**Expected**: asks for the model type, then shows a numbered model list; choosing a number starts
the run with the default evaluator.

### 4. Several evaluators, one download (US2, FR-016)

```bash
uv run testbed evaluate HuggingFaceTB/SmolLM2-135M -e arc-easy -e hellaswag --limit 50
```

**Expected**: exactly one download progress bar, two result blocks, one cleanup line.

### 5. Incompatible evaluator is rejected (US2-1)

```bash
uv run testbed evaluate HuggingFaceTB/SmolLM2-135M -e sst2
```

**Expected**: exit code 1 and a message listing the evaluators valid for this model; nothing
downloaded.

### 6. Cancel cleans up (US1-4, edge case "interrupted run")

Start `uv run testbed evaluate Qwen/Qwen2.5-0.5B`, press **Ctrl+C** during the download, then run
`ls -A workspace/downloads`.

**Expected**: message "Run cancelled", exit code 130, empty downloads folder.

### 7. Leftovers from a crash are removed (FR-011)

```bash
mkdir -p workspace/downloads/model && echo test > workspace/downloads/model/leftover.txt
uv run testbed cleanup
ls -A workspace/downloads
```

**Expected**: `cleanup` reports freed space; the folder is empty. (The same cleanup runs
automatically at the start of `evaluate`.)

### 8. History and comparison (US4)

```bash
uv run testbed history
uv run testbed compare sts-benchmark
```

**Expected**: all results from the scenarios above, newest first; `compare` shows one row per
model sorted best-first.

### 9. Regenerate the catalog (US3, SC-007)

```bash
cp catalog.yaml /tmp/catalog.backup.yaml
time uv run testbed catalog build --per-type 3 --output /tmp/catalog.new.yaml
uv run python -c "import yaml; print(len(yaml.safe_load(open('/tmp/catalog.new.yaml'))['models']))"
```

**Expected**: finishes in under 2 minutes, reports kept/skipped models with reasons, and the new
file has up to 9 entries (3 per type), each ≤ 2 GB. `catalog.yaml` itself is unchanged because
`--output` pointed elsewhere.

### 10. Repeatability (SC-005)

Run scenario 2's embedding command a second time, then `uv run testbed history --model
sentence-transformers/all-MiniLM-L6-v2`.

**Expected**: the two `sts-benchmark` scores differ by less than 1%.

## Automated tests

```bash
uv run pytest                 # fast unit tests, no internet needed
uv run pytest -m network      # end-to-end tests with tiny models (needs internet)
uv run ruff check . && uv run ruff format --check .
```

**Expected**: all tests pass; ruff reports no problems.

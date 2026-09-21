# Implementation Plan: Model Testbed

**Branch**: `001-model-testbed` | **Date**: 2026-09-21 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-model-testbed/spec.md`

## Summary

A small Python command-line tool for trying out Hugging Face text models (generative, embedding,
classification) on an Apple silicon Mac mini. Users pick a model from a hand-editable
`catalog.yaml` (which can be regenerated from Hugging Face), pick one or more compatible
evaluators, and get plain-language scores. The model is downloaded into the testbed's own
workspace folder only when a run starts and is always deleted when the run ends.

Evaluation reuses the standard tool of each model family: EleutherAI lm-evaluation-harness for
generative models, Sentence Transformers' built-in evaluators for embeddings, and the
Transformers pipeline with scikit-learn metrics for classifiers. All run on the Apple GPU (`mps`)
with CPU fallback. Results are appended to a JSON Lines history for later comparison. The core
logic is plain Python functions, so a local web page can be added later without changing it.

## Technical Context

**Language/Version**: Python 3.12 (managed by `uv`)

**Primary Dependencies**: PyTorch 2.11 (`mps`), transformers 5.x, sentence-transformers 6.x,
huggingface-hub 1.x, datasets 5.x, lm-eval[hf] 0.4.x, scikit-learn 1.x, Typer (+ Rich), PyYAML,
Pydantic 2, filelock. See [research.md §10](./research.md#10-dependency-summary).

**Storage**: Files only: `catalog.yaml` (repo root), `workspace/results.jsonl` (history),
`workspace/downloads/` (temporary, emptied after every run). `workspace/` is git-ignored and can
be moved with `TESTBED_WORKSPACE`.

**Testing**: pytest (offline unit tests + `network`-marked end-to-end tests with tiny models),
ruff for linting/formatting.

**Target Platform**: macOS on Apple silicon (Mac mini, ≥16 GB memory); no CUDA.

**Project Type**: Single-project command-line application with a reusable core library.

**Performance Goals**: Default evaluator on any catalog model finishes in < 15 min excluding
download (SC-004); first result for a small model < 10 min total (SC-001); catalog build < 2 min
(SC-007); `testbed --help` and `testbed models` respond in < 1 s (heavy imports are lazy).

**Constraints**: Models ≤ 2 GB download (~1B parameters), `float32` on `mps`, max context
2048 tokens for generative evaluation, batch size 1; extra disk use ≤ model + data + 10%
(SC-003); downloads folder empty after every run (SC-002); only `*.safetensors`, never
`trust_remote_code`.

**Scale/Scope**: One user, one run at a time; catalog of ~15 models (5 per type); 7 evaluators
(3 generative, 2 embedding, 2 classification); ~12 source modules.

All technical unknowns were resolved in [research.md](./research.md); none remain.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle / Rule | How the plan complies | Pre-research | Post-design |
| --- | --- | --- | --- |
| I. Proven, best-practice tools only | Every dependency is the de-facto standard for its job (PyTorch, Hugging Face libraries, EleutherAI lm-eval, scikit-learn, Typer, Pydantic, PyYAML, pytest, ruff). Evaluation uses standard benchmarks and reference metric implementations instead of home-made scoring. Safetensors-only and no remote code follow Hugging Face security guidance. | ✅ | ✅ |
| II. Simplicity first | No database, no plugin system, no config file, no web server: YAML catalog, JSONL history, one registry dict of evaluators, one package. One run at a time. The web page is deferred. | ✅ | ✅ |
| III. Beginner-friendly code | Flat module layout with one clear job per file; plain functions; interactive numbered menus; `uv sync` + `uv run` setup in two commands (see quickstart). | ✅ | ✅ |
| IV. Explain through comments and docs | Implementation rule: every module starts with a docstring explaining its role; every public function has a docstring; non-obvious steps (e.g., `HF_HOME` redirection order, label mapping, `max_length=2048`) get a "why" comment. README explains setup and each command. | ✅ | ✅ |
| Tech constraints: bounded versions, few deps | `pyproject.toml` lower bounds at the resolved versions, upper bounds below the next major version (`<0.5` for lm-eval); `uv.lock` pins exact versions; each dependency's purpose listed in research.md. | ✅ | ✅ |
| Workflow: simple, readable tests | Plain pytest functions; unit tests offline; end-to-end tests opt-in. | ✅ | ✅ |

**Result**: PASS, no violations. The heaviest dependency, `lm-eval`, is itself the proven
standard and replaces custom scoring code (see research.md §5a), so it is not a complexity
violation.

## Project Structure

### Documentation (this feature)

```text
specs/001-model-testbed/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   ├── cli.md           # Commands, options, output, exit codes
│   ├── catalog-file.md  # catalog.yaml format
│   ├── results-file.md  # results.jsonl format
│   └── core-api.md      # Python functions shared by CLI and future web page
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
pyproject.toml               # dependencies, `testbed` command entry point, pytest/ruff settings
uv.lock                      # exact dependency versions
.python-version              # 3.12
.gitignore                   # workspace/, .venv/
README.md                    # setup, commands, how evaluation and cleanup work
catalog.yaml                 # curated model list (generated, then reviewed by hand)

src/testbed/
├── __init__.py              # package version
├── settings.py              # paths, defaults (limit 200, 2 GB), sets HF_HOME / MPS fallback env
├── cli.py                   # Typer commands; printing only, calls the functions below
├── catalog.py               # CatalogEntry model (Pydantic), load/validate/find
├── catalog_builder.py       # search Hugging Face, apply filters, write catalog.yaml safely
├── device.py                # choose mps or cpu, free GPU memory between evaluators
├── downloads.py             # disk-space check, snapshot download, cleanup, run lock
├── runner.py                # one run: check → download → evaluate → save → cleanup
├── results.py               # append/read results.jsonl, history filters, comparison
├── errors.py                # TestbedError with plain-language messages
└── evaluators/
    ├── __init__.py          # registry: list, compatible, default evaluator
    ├── base.py              # Evaluator dataclass
    ├── generative.py        # arc-easy, hellaswag, wikitext via lm-evaluation-harness
    ├── embedding.py         # sts-benchmark, nano-scifact via Sentence Transformers
    └── classification.py    # sst2, rotten-tomatoes via pipeline + scikit-learn

tests/
├── conftest.py              # temporary workspace fixture, fake catalog
├── unit/                    # catalog, evaluator registry, label mapping, downloads, results
└── integration/             # @pytest.mark.network end-to-end run per model type
```

**Structure Decision**: Single project using the standard `src/` layout recommended by the Python
Packaging Authority and `uv`. The core (`catalog`, `runner`, `results`, `evaluators`, ...) never
prints; `cli.py` is the only file that talks to the terminal. This keeps the future web page
(FR-022) a second thin layer over the same functions. Tests are split only into `unit` (offline)
and `integration` (network), which is enough at this size.

## Complexity Tracking

No constitution violations. Nothing to justify.

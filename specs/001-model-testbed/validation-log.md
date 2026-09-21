# Validation Log: Model Testbed

**Date**: 2026-09-21 | **Machine**: Mac mini M2 Pro, 32 GB, macOS (Apple GPU via `mps`)
**Versions**: Python 3.12.13, torch 2.14.0, transformers 5.17.0, sentence-transformers 6.1.0,
lm-eval 0.4.13, datasets 5.0.1, huggingface-hub 1.32.0

## Quickstart scenarios

| # | Scenario | Result | Notes |
| --- | --- | --- | --- |
| 1 | Browsing downloads nothing | PASS | `workspace/downloads` 0 KB before and after `models` / `evaluators` |
| 2 | One model per type | PASS | MiniLM-L6 STS 0.823; DistilBERT SST-2 acc 0.915; SmolLM2-135M ARC-Easy 0.555; downloads empty after each |
| 3 | Interactive mode | PASS | Type → numbered model list → numbered evaluators (Enter = recommended) |
| 4 | Several evaluators, one download | PASS | SmolLM2-135M with arc-easy + hellaswag + wikitext: one download, three results |
| 5 | Incompatible evaluator rejected | PASS | `-e sst2` on a generative model: exit 1, valid evaluators listed, nothing downloaded |
| 6 | Ctrl+C cleans up | PASS | SIGINT during a Qwen2.5-0.5B download: "Run cancelled.", exit 130, 953 MB removed in 4.5 s |
| 7 | Leftovers removed | PASS | Leftover file removed by `cleanup`; also after a real crash (260 MB removed) |
| 8 | History and comparison | PASS | `history` newest first; `compare sts-benchmark` / `arc-easy` sorted best-first |
| 9 | Regenerate catalog | PASS | 9 s; 35 checked, 15 kept, 20 skipped with reasons; `catalog.yaml` unchanged with `--output` |
| 10 | Repeatability | PASS | MiniLM-L6 STS: 0.823 and 0.823 on two runs (difference 0%) |

## Success criteria

| Criterion | Target | Measured | Result |
| --- | --- | --- | --- |
| SC-001 first result | < 10 min | 12 s for MiniLM-L6 (download + evaluation; after `uv sync`) | PASS |
| SC-002 downloads empty after runs | 100% | Empty after every run, including failure, cancel, and crash (next start) | PASS |
| SC-003 extra disk use | ≤ model + data + 10% | e.g. 91 MB cleaned for an 87 MB model plus STS data | PASS |
| SC-004 default evaluator, largest model per type | < 15 min (excl. download) | Qwen3-0.6B ARC-Easy 47 s; multilingual MiniLM-L12 STS 8 s; 520 MB sentiment model SST-2 7 s | PASS |
| SC-005 repeatability | ≤ 1% difference | 0% (see scenario 10) | PASS |
| SC-006 browsing downloads nothing | 0 bytes | 0 bytes | PASS |
| SC-007 catalog build | < 2 min, all entries pass filters | 9 s; all entries ≤ 2 GB, safetensors, no custom code | PASS |
| SC-008 scores are understandable | qualitative | Every score shows "higher/lower is better" and a reading guide | PASS (self-review) |

## Automated tests

- `uv run pytest`: 48 passed (unit tests, offline)
- `uv run pytest -m network`: 3 passed (tiny generative / embedding / classification models)
- `uv run ruff check .` and `uv run ruff format --check .`: no findings

## Issues found and fixed during validation

- **Segmentation fault while loading generative models onto the Apple GPU.** Transformers 5
  loads weights with several threads; with `device_map="mps"` this crashed the process.
  Fixed by setting `HF_DEACTIVATE_ASYNC_LOAD=1` in `settings.configure_environment()`
  (sequential loading; costs well under a second for our model sizes).
- **lm-eval reported non-metric values** (`sample_len`). Now only metrics listed in lm-eval's
  `higher_is_better` table are kept.
- **Catalog builder missed the most popular sentiment model**, because its name does not
  contain "sentiment". Sentiment models are now recognized by their label names.
- **Many popular sentiment models have no safetensors weights** on their main branch (e.g.
  `cardiffnlp/twitter-roberta-base-sentiment-latest`) and are therefore excluded by design.

"""End-to-end runs with tiny real models from Hugging Face (needs internet).

Run with:  uv run pytest -m network

The models are tiny random test models, so their scores are meaningless. These tests only
check that each model type can be downloaded, evaluated with its default evaluator, saved to
the history, and cleaned up again.
"""

import pytest

from testbed import settings
from testbed.catalog import CatalogEntry
from testbed.evaluators import default_evaluator
from testbed.runner import RunStatus, run_evaluation

TINY_MODELS = [
    CatalogEntry(
        id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        type="generative",
        description="Tiny random Llama model for tests.",
        size_mb=4,
        license="unknown",
    ),
    CatalogEntry(
        id="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        type="embedding",
        description="Tiny BERT embedding model for tests.",
        size_mb=18,
        license="unknown",
    ),
    CatalogEntry(
        id="peft-internal-testing/tiny-random-BertForSequenceClassification",
        type="classification",
        task="sentiment",
        label_map={"LABEL_0": "negative", "LABEL_1": "positive"},
        description="Tiny random BERT classifier for tests.",
        size_mb=1,
        license="unknown",
    ),
]


@pytest.mark.network
@pytest.mark.parametrize("entry", TINY_MODELS, ids=lambda entry: entry.type)
def test_default_evaluator_runs_and_cleans_up(entry):
    evaluator = default_evaluator(entry)

    run = run_evaluation(entry, [evaluator], limit=5)

    assert run.status == RunStatus.SUCCEEDED, run.errors
    [result] = run.results
    assert result.evaluator == evaluator.key
    assert evaluator.main_metric in result.scores
    assert settings.RESULTS_FILE.exists()
    assert list(settings.DOWNLOADS_DIR.iterdir()) == []

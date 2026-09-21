"""Tests for the evaluator registry (testbed/evaluators/__init__.py)."""

import re
from collections import Counter

import pytest

from testbed import evaluators
from testbed.catalog import CatalogEntry, ModelType
from testbed.errors import TestbedError
from testbed.evaluators import (
    EVALUATORS,
    Evaluator,
    compatible_evaluators,
    default_evaluator,
    get_evaluator,
)


def fake_evaluator(key, model_type, task=None, is_default=False):
    """An evaluator that is never run; only its description matters for these tests."""
    return Evaluator(
        key=key,
        name=key,
        model_type=model_type,
        description="fake",
        dataset="fake",
        main_metric="score",
        higher_is_better=True,
        how_to_read="fake",
        run=lambda *args: None,
        task=task,
        is_default=is_default,
    )


@pytest.fixture
def fake_registry(monkeypatch):
    """Replace the real registry with a small, known one."""
    fakes = [
        fake_evaluator("gen-a", ModelType.GENERATIVE, is_default=True),
        fake_evaluator("gen-b", ModelType.GENERATIVE),
        fake_evaluator("sentiment-a", ModelType.CLASSIFICATION, "sentiment", is_default=True),
        fake_evaluator("emotion-a", ModelType.CLASSIFICATION, "emotion", is_default=True),
    ]
    monkeypatch.setattr(evaluators, "EVALUATORS", {e.key: e for e in fakes})


def sentiment_model():
    return CatalogEntry(
        id="owner/sentiment",
        type="classification",
        task="sentiment",
        label_map={"NEG": "negative", "POS": "positive"},
        description="test",
        size_mb=1,
        license="mit",
    )


def test_compatible_evaluators_match_type_and_task(fake_registry):
    keys = [e.key for e in compatible_evaluators(sentiment_model())]
    assert keys == ["sentiment-a"]


def test_default_evaluator(fake_registry):
    generator = CatalogEntry(
        id="owner/gen", type="generative", description="t", size_mb=1, license="mit"
    )
    assert default_evaluator(generator).key == "gen-a"
    assert default_evaluator(sentiment_model()).key == "sentiment-a"


def test_unknown_evaluator_lists_valid_keys(fake_registry):
    with pytest.raises(TestbedError) as error:
        get_evaluator("nope")
    assert "gen-a" in error.value.hint


# --- Checks on the real registry ------------------------------------------------------------


def test_real_keys_are_lowercase_with_dashes():
    for key, evaluator in EVALUATORS.items():
        assert key == evaluator.key
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", key), key


def test_real_registry_has_exactly_one_default_per_type_and_task():
    groups = {(e.model_type, e.task) for e in EVALUATORS.values()}
    defaults = Counter((e.model_type, e.task) for e in EVALUATORS.values() if e.is_default)
    for group in groups:
        assert defaults[group] == 1, f"{group} needs exactly one default evaluator"


@pytest.mark.parametrize("model_type", list(ModelType))
def test_every_model_type_has_at_least_two_evaluators(model_type):
    assert len([e for e in EVALUATORS.values() if e.model_type == model_type]) >= 2


def test_fixed_size_and_lower_is_better_evaluators():
    assert get_evaluator("nano-scifact").supports_limit is False
    assert get_evaluator("wikitext").higher_is_better is False

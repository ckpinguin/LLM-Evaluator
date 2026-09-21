"""The evaluator registry: the list of all evaluators and helpers to pick the right ones.

To add a new evaluator: write its ``run`` function in the module for its model type, add an
``Evaluator(...)`` to that module's list (e.g. ``GENERATIVE_EVALUATORS``), and run
``uv run pytest tests/unit/test_evaluator_registry.py``. Nothing else needs to change.

The evaluator modules import heavy libraries (PyTorch, ...) only inside their ``run``
functions, so importing this registry is fast.
"""

from testbed.catalog import CatalogEntry, ModelType
from testbed.errors import TestbedError
from testbed.evaluators.base import Evaluator, EvaluatorOutput
from testbed.evaluators.classification import CLASSIFICATION_EVALUATORS
from testbed.evaluators.embedding import EMBEDDING_EVALUATORS
from testbed.evaluators.generative import GENERATIVE_EVALUATORS

# All evaluators, looked up by their key, e.g. EVALUATORS["arc-easy"].
EVALUATORS: dict[str, Evaluator] = {
    evaluator.key: evaluator
    for evaluator in GENERATIVE_EVALUATORS + EMBEDDING_EVALUATORS + CLASSIFICATION_EVALUATORS
}

__all__ = [
    "EVALUATORS",
    "Evaluator",
    "EvaluatorOutput",
    "compatible_evaluators",
    "default_evaluator",
    "get_evaluator",
    "list_evaluators",
]


def list_evaluators(model_type: ModelType | None = None) -> list[Evaluator]:
    """Return all evaluators, or only those for one model type."""
    return [e for e in EVALUATORS.values() if model_type is None or e.model_type == model_type]


def get_evaluator(key: str) -> Evaluator:
    """Return the evaluator with this key, or explain which keys exist."""
    if key not in EVALUATORS:
        raise TestbedError(
            f"There is no evaluator called '{key}'.",
            hint=f"Choose one of: {', '.join(EVALUATORS)}",
        )
    return EVALUATORS[key]


def is_compatible(evaluator: Evaluator, entry: CatalogEntry) -> bool:
    """Can this evaluator test this model? (Same model type, and same task if it needs one.)"""
    return evaluator.model_type == entry.type and (
        evaluator.task is None or evaluator.task == entry.task
    )


def compatible_evaluators(entry: CatalogEntry) -> list[Evaluator]:
    """Return every evaluator that can test this catalog model (spec FR-015)."""
    return [e for e in EVALUATORS.values() if is_compatible(e, entry)]


def default_evaluator(entry: CatalogEntry) -> Evaluator:
    """Return the recommended evaluator for this catalog model."""
    for evaluator in compatible_evaluators(entry):
        if evaluator.is_default:
            return evaluator
    raise TestbedError(f"No default evaluator is defined for '{entry.id}' ({entry.type}).")

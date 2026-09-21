"""What an evaluator is.

An evaluator is a named, repeatable test for one type of model: for example "ARC-Easy" asks a
generative model 200 science questions and reports how many it answered correctly.

Each evaluator is described by an ``Evaluator`` object (name, what it measures, how to read the
score, ...) plus one ``run`` function that does the actual work.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from testbed.catalog import CatalogEntry, ModelType


@dataclass(frozen=True)
class EvaluatorOutput:
    """What an evaluator's ``run`` function returns."""

    scores: dict[str, float]  # metric name -> value, e.g. {"accuracy": 0.91, "macro_f1": 0.90}
    examples: int  # how many examples were really evaluated (needed for the result, FR-018)


# The signature every ``run`` function has:
#   run(model_dir, entry, device, limit) -> EvaluatorOutput
# - model_dir: folder with the downloaded model files
# - entry:     the catalog entry (classification evaluators need its label_map)
# - device:    "cuda" (NVIDIA GPU), "mps" (Apple GPU), or "cpu"
# - limit:     how many examples to use (ignored when supports_limit is False)
RunFunction = Callable[[Path, CatalogEntry, str, int], EvaluatorOutput]


@dataclass(frozen=True)
class Evaluator:
    """Description of one evaluator plus the function that runs it."""

    key: str  # unique, lowercase-with-dashes, e.g. "arc-easy"
    name: str  # human-friendly name, e.g. "ARC-Easy"
    model_type: ModelType  # which models it can evaluate
    description: str  # plain-language description of what is measured
    dataset: str  # Hugging Face dataset id and split
    main_metric: str  # the score shown first, e.g. "acc_norm"
    higher_is_better: bool  # how to read the score (FR-014)
    how_to_read: str  # e.g. "0.25 = random guessing, 1.0 = perfect"
    run: RunFunction
    task: str | None = None  # classification only: the task it needs, e.g. "sentiment"
    supports_limit: bool = True  # False for benchmarks with a fixed size
    is_default: bool = False  # exactly one default per model type (and task)

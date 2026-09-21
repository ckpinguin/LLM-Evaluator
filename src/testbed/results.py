"""The results history: every evaluation result, stored one per line in ``results.jsonl``.

JSON Lines (one JSON object per line) needs no database, can be appended to safely, and can be
opened in any text editor. The history is kept when downloaded models are deleted (FR-019).

File format: specs/001-model-testbed/contracts/results-file.md
"""

import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from testbed import settings
from testbed.catalog import ModelType

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """The outcome of one evaluator on one model (spec FR-018)."""

    # "ignore": results written by a newer testbed version may have extra fields; we skip them.
    model_config = ConfigDict(extra="ignore")

    run_id: str  # links results from the same run, e.g. "2026-09-21T10-15-03"
    created_at: datetime  # when the evaluator finished (local time with UTC offset)
    model_id: str
    model_type: ModelType
    evaluator: str  # evaluator key, e.g. "arc-easy"
    main_metric: str
    main_score: float
    higher_is_better: bool  # copied from the evaluator, so old results stay readable
    scores: dict[str, float]  # all metrics the evaluator returned
    examples: int
    duration_seconds: float  # time for this evaluator, download excluded
    device: str  # "mps" or "cpu"
    testbed_version: str


def append_result(result: EvaluationResult, path: Path | None = None) -> None:
    """Add one result as a new line at the end of the history file."""
    path = path or settings.RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(result.model_dump_json() + "\n")


def load_history(
    model_type: ModelType | None = None,
    evaluator: str | None = None,
    model_id: str | None = None,
    path: Path | None = None,
) -> list[EvaluationResult]:
    """Read the saved results, newest first, optionally filtered (FR-020).

    A line that can't be read (for example after a mistake while editing the file by hand)
    is skipped with a warning; all other results are still returned.
    """
    path = path or settings.RESULTS_FILE
    if not path.exists():
        return []

    results = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue  # ignore empty lines
        try:
            result = EvaluationResult.model_validate_json(line)
        except ValidationError:
            logger.warning("%s line %d skipped: it is not a valid result.", path.name, number)
            continue
        if model_type is not None and result.model_type != model_type:
            continue
        if evaluator is not None and result.evaluator != evaluator:
            continue
        if model_id is not None and result.model_id != model_id:
            continue
        results.append(result)

    # The file is written in time order, so reversing it puts the newest result first.
    return list(reversed(results))


def compare(evaluator_key: str, path: Path | None = None) -> list[EvaluationResult]:
    """The latest result of every model for one evaluator, best first.

    Results are only compared within the same evaluator: an accuracy and a perplexity
    can't be put side by side.
    """
    latest_per_model: dict[str, EvaluationResult] = {}
    for result in load_history(evaluator=evaluator_key, path=path):  # newest first
        latest_per_model.setdefault(result.model_id, result)  # keeps the first (newest) one

    rows = list(latest_per_model.values())
    higher_is_better = rows[0].higher_is_better if rows else True
    return sorted(rows, key=lambda result: result.main_score, reverse=higher_is_better)

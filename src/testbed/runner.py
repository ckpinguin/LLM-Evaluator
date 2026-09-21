"""One evaluation run: check → download → evaluate → save results → clean up.

A run evaluates one model with one or more evaluators. The model is downloaded once, all
evaluators use it, and then every downloaded file is deleted again, whatever happened
(success, error, or the user pressing Ctrl+C).

The steps follow the state diagram in specs/001-model-testbed/data-model.md.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import testbed
from testbed import settings
from testbed.catalog import CatalogEntry
from testbed.device import free_memory, pick_device
from testbed.downloads import (
    check_disk_space,
    cleanup_downloads,
    download_model,
    planned_files,
    run_lock,
)
from testbed.errors import TestbedError
from testbed.evaluators import Evaluator, compatible_evaluators, is_compatible
from testbed.results import EvaluationResult, append_result

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    """Where a run currently is (or how it ended)."""

    CHECKING = "checking"
    DOWNLOADING = "downloading"
    EVALUATING = "evaluating"
    CLEANING_UP = "cleaning_up"
    SUCCEEDED = "succeeded"  # at least one evaluator finished
    FAILED = "failed"  # no evaluator finished
    CANCELLED = "cancelled"  # the user pressed Ctrl+C


@dataclass
class EvaluationRun:
    """Everything about one run. Only kept in memory; results go to results.jsonl."""

    run_id: str
    model: CatalogEntry
    evaluators: list[Evaluator]
    limit: int
    device: str = ""
    status: RunStatus = RunStatus.CHECKING
    results: list[EvaluationResult] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # evaluator key -> message
    cleaned_bytes: int = 0  # how much was deleted from the downloads folder at the end


def run_evaluation(
    entry: CatalogEntry,
    evaluators: list[Evaluator],
    limit: int | None = None,
    device: str = "auto",
    on_progress: Callable[[str], None] | None = None,
) -> EvaluationRun:
    """Download ``entry``, run each evaluator on it, save the results, and clean up.

    - Raises ``TestbedError`` for problems the user can fix (incompatible evaluator, another
      run in progress, not enough disk space, model not reachable). Downloads are cleaned
      up before the error is raised.
    - If an evaluator fails, its error is stored in ``run.errors`` and the next one still runs.
    - If the user presses Ctrl+C, the run stops, is cleaned up, and gets status CANCELLED.

    ``on_progress`` receives short messages about what is happening, e.g. to show them in
    the terminal.
    """
    report = on_progress or (lambda message: None)
    check_evaluators(entry, evaluators)

    run = EvaluationRun(
        run_id=datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        model=entry,
        evaluators=evaluators,
        limit=limit or settings.DEFAULT_LIMIT,
    )

    with run_lock():
        outcome = RunStatus.FAILED  # what the run ends as, unless we get further below
        try:
            # 1. Checking: nothing has been downloaded yet.
            run.device = pick_device(device)
            cleanup_downloads()  # remove leftovers of an earlier crashed run (FR-011)
            files = planned_files(entry.id)
            check_disk_space(sum(size for _, size in files))

            # 2. Downloading: once per run, however many evaluators there are (FR-016).
            run.status = RunStatus.DOWNLOADING
            report(f"Downloading {entry.id} ...")
            model_dir = download_model(entry.id)

            # 3. Evaluating: one evaluator after the other.
            run.status = RunStatus.EVALUATING
            for number, evaluator in enumerate(evaluators, start=1):
                report(f"Running {evaluator.key} ({number}/{len(evaluators)}) ...")
                try:
                    result = _run_one(run, evaluator, model_dir)
                except Exception as error:  # one evaluator failing must not stop the others
                    logger.debug("Evaluator %s failed", evaluator.key, exc_info=True)
                    run.errors[evaluator.key] = _explain(error)
                    continue
                finally:
                    free_memory(run.device)
                append_result(result)
                run.results.append(result)

            outcome = RunStatus.SUCCEEDED if run.results else RunStatus.FAILED
        except KeyboardInterrupt:
            # Ctrl+C: stop, but still clean up below. The caller sees status CANCELLED.
            outcome = RunStatus.CANCELLED
        finally:
            # 4. Cleaning up: ALWAYS runs, also after errors (FR-010).
            run.status = RunStatus.CLEANING_UP
            report("Deleting downloaded files ...")
            run.cleaned_bytes = cleanup_downloads()
            run.status = outcome

    return run


def check_evaluators(entry: CatalogEntry, evaluators: list[Evaluator]) -> None:
    """Refuse incompatible evaluators before anything is downloaded."""
    if not evaluators:
        raise TestbedError("Choose at least one evaluator.")
    for evaluator in evaluators:
        if not is_compatible(evaluator, entry):
            valid = ", ".join(e.key for e in compatible_evaluators(entry))
            raise TestbedError(
                f"The evaluator '{evaluator.key}' cannot evaluate '{entry.id}' ({entry.type}).",
                hint=f"Evaluators for this model: {valid}",
            )


def _run_one(run: EvaluationRun, evaluator: Evaluator, model_dir: Path) -> EvaluationResult:
    """Run one evaluator, time it, and turn its output into a result record."""
    start = time.perf_counter()
    output = evaluator.run(model_dir, run.model, run.device, run.limit)
    duration = time.perf_counter() - start

    return EvaluationResult(
        run_id=run.run_id,
        created_at=datetime.now().astimezone(),  # local time including the UTC offset
        model_id=run.model.id,
        model_type=run.model.type,
        evaluator=evaluator.key,
        main_metric=evaluator.main_metric,
        main_score=output.scores[evaluator.main_metric],
        higher_is_better=evaluator.higher_is_better,
        scores=output.scores,
        examples=output.examples,
        duration_seconds=round(duration, 1),
        device=run.device,
        testbed_version=testbed.__version__,
    )


def _explain(error: Exception) -> str:
    """A short, plain-language description of why an evaluator failed."""
    if isinstance(error, TestbedError):
        return str(error)
    text = str(error)
    if "out of memory" in text.lower():
        return "The model did not fit into memory. Try a smaller model."
    return f"{type(error).__name__}: {text}"

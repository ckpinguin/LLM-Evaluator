"""Tests for one evaluation run (testbed/runner.py), without network or real models.

The download functions and the device choice are replaced with small fakes, so these tests
check only the runner's own logic: order of steps, saving results, and cleanup.
"""

import pytest

from testbed import runner, settings
from testbed.catalog import CatalogEntry
from testbed.errors import TestbedError
from testbed.evaluators import Evaluator, EvaluatorOutput
from testbed.results import EvaluationResult
from testbed.runner import RunStatus, run_evaluation

GENERATOR = CatalogEntry(
    id="owner/tiny-generator", type="generative", description="t", size_mb=1, license="mit"
)
EMBEDDER = CatalogEntry(
    id="owner/tiny-embedder", type="embedding", description="t", size_mb=1, license="mit"
)


def make_evaluator(key="fake-eval", run=None, supports_limit=True):
    """A generative evaluator whose run function is ``run`` (default: returns score 0.5)."""

    def default_run(model_dir, entry, device, limit):
        return EvaluatorOutput(scores={"score": 0.5}, examples=limit)

    return Evaluator(
        key=key,
        name=key,
        model_type="generative",
        description="fake",
        dataset="fake",
        main_metric="score",
        higher_is_better=True,
        how_to_read="fake",
        run=run or default_run,
        supports_limit=supports_limit,
    )


@pytest.fixture
def fake_downloads(workspace, monkeypatch):
    """Replace network and hardware functions; count how often the model is downloaded."""
    calls = {"downloads": 0}

    def fake_download(repo_id):
        calls["downloads"] += 1
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (settings.MODEL_DIR / "model.safetensors").write_bytes(b"x" * 100)
        return settings.MODEL_DIR

    monkeypatch.setattr(runner, "planned_files", lambda repo_id: [("model.safetensors", 100)])
    monkeypatch.setattr(runner, "download_model", fake_download)
    monkeypatch.setattr(runner, "pick_device", lambda requested: "cpu")
    monkeypatch.setattr(runner, "free_memory", lambda device: None)
    return calls


def saved_results():
    if not settings.RESULTS_FILE.exists():
        return []
    lines = settings.RESULTS_FILE.read_text().splitlines()
    return [EvaluationResult.model_validate_json(line) for line in lines]


def downloads_are_empty():
    return list(settings.DOWNLOADS_DIR.iterdir()) == []


def test_successful_run_saves_result_and_cleans_up(fake_downloads):
    run = run_evaluation(GENERATOR, [make_evaluator()], limit=7)

    assert run.status == RunStatus.SUCCEEDED
    assert downloads_are_empty()
    assert run.cleaned_bytes == 100
    [result] = saved_results()
    assert result.model_id == "owner/tiny-generator"
    assert result.main_score == 0.5
    assert result.examples == 7
    assert result.device == "cpu"


def test_failing_evaluator_is_recorded_and_files_are_cleaned_up(fake_downloads):
    def broken_run(model_dir, entry, device, limit):
        raise RuntimeError("something broke")

    run = run_evaluation(GENERATOR, [make_evaluator(run=broken_run)])

    assert run.status == RunStatus.FAILED
    assert "something broke" in run.errors["fake-eval"]
    assert saved_results() == []
    assert downloads_are_empty()


def test_ctrl_c_cancels_the_run_and_cleans_up(fake_downloads):
    def interrupted_run(model_dir, entry, device, limit):
        raise KeyboardInterrupt

    run = run_evaluation(GENERATOR, [make_evaluator(run=interrupted_run)])

    assert run.status == RunStatus.CANCELLED
    assert saved_results() == []
    assert downloads_are_empty()


def test_incompatible_evaluator_is_rejected_before_downloading(fake_downloads):
    with pytest.raises(TestbedError, match="cannot evaluate"):
        run_evaluation(EMBEDDER, [make_evaluator()])
    assert fake_downloads["downloads"] == 0


def test_leftovers_from_an_earlier_crash_are_removed(fake_downloads):
    leftover = settings.DOWNLOADS_DIR / "hf-home" / "old-file"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("left behind by a crashed run")

    run_evaluation(GENERATOR, [make_evaluator()])

    assert not leftover.exists()


def test_two_evaluators_share_one_download(fake_downloads):
    run = run_evaluation(GENERATOR, [make_evaluator("eval-a"), make_evaluator("eval-b")])

    assert fake_downloads["downloads"] == 1
    assert [r.evaluator for r in run.results] == ["eval-a", "eval-b"]
    assert len(saved_results()) == 2


def test_second_evaluator_runs_after_first_one_fails(fake_downloads):
    def broken_run(model_dir, entry, device, limit):
        raise RuntimeError("first one broke")

    run = run_evaluation(
        GENERATOR, [make_evaluator("eval-a", run=broken_run), make_evaluator("eval-b")]
    )

    assert run.status == RunStatus.SUCCEEDED
    assert list(run.errors) == ["eval-a"]
    assert [r.evaluator for r in run.results] == ["eval-b"]


def test_fixed_size_evaluator_records_its_own_example_count(fake_downloads):
    def fixed_size_run(model_dir, entry, device, limit):
        return EvaluatorOutput(scores={"score": 0.7}, examples=50)  # ignores `limit`

    run = run_evaluation(
        GENERATOR, [make_evaluator(run=fixed_size_run, supports_limit=False)], limit=200
    )

    assert run.results[0].examples == 50

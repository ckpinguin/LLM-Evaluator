"""Tests for the results history (testbed/results.py)."""

import logging
from datetime import datetime, timedelta

from testbed import settings
from testbed.results import EvaluationResult, append_result, compare, load_history

START = datetime(2026, 9, 21, 10, 0).astimezone()


def make_result(model_id, evaluator="sts-benchmark", score=0.5, minutes=0, **changes):
    """A result record; ``minutes`` shifts its time so we can control what is newest."""
    values = {
        "run_id": f"run-{minutes}",
        "created_at": START + timedelta(minutes=minutes),
        "model_id": model_id,
        "model_type": "embedding",
        "evaluator": evaluator,
        "main_metric": "spearman_cosine",
        "main_score": score,
        "higher_is_better": True,
        "scores": {"spearman_cosine": score},
        "examples": 200,
        "duration_seconds": 1.0,
        "device": "mps",
        "testbed_version": "0.1.0",
    }
    values.update(changes)
    return EvaluationResult(**values)


def test_history_round_trip_newest_first(workspace):
    append_result(make_result("owner/a", minutes=0))
    append_result(make_result("owner/b", minutes=1))

    history = load_history()

    assert [r.model_id for r in history] == ["owner/b", "owner/a"]
    assert history[1] == make_result("owner/a", minutes=0)


def test_history_filters(workspace):
    append_result(make_result("owner/a", evaluator="sts-benchmark"))
    append_result(make_result("owner/a", evaluator="nano-scifact"))
    append_result(
        make_result("owner/gen", evaluator="arc-easy", model_type="generative", main_metric="acc")
    )

    assert len(load_history(model_type="embedding")) == 2
    assert [r.evaluator for r in load_history(evaluator="arc-easy")] == ["arc-easy"]
    assert [r.model_id for r in load_history(model_id="owner/gen")] == ["owner/gen"]


def test_unreadable_line_is_skipped_with_a_warning(workspace, caplog):
    append_result(make_result("owner/a"))
    with settings.RESULTS_FILE.open("a") as file:
        file.write("this is not json\n")
    append_result(make_result("owner/b", minutes=1))

    with caplog.at_level(logging.WARNING):
        history = load_history()

    assert [r.model_id for r in history] == ["owner/b", "owner/a"]
    assert "line 2" in caplog.text


def test_missing_history_file_means_no_results(workspace):
    assert load_history() == []


def test_compare_keeps_latest_per_model_sorted_best_first(workspace):
    append_result(make_result("owner/a", score=0.9, minutes=0))
    append_result(make_result("owner/a", score=0.6, minutes=5))  # newer result for owner/a
    append_result(make_result("owner/b", score=0.8, minutes=1))

    rows = compare("sts-benchmark")

    assert [(r.model_id, r.main_score) for r in rows] == [("owner/b", 0.8), ("owner/a", 0.6)]


def test_compare_lower_is_better(workspace):
    for model_id, score in [("owner/a", 30.0), ("owner/b", 20.0)]:
        append_result(
            make_result(
                model_id,
                evaluator="wikitext",
                score=score,
                model_type="generative",
                main_metric="word_perplexity",
                higher_is_better=False,
            )
        )

    assert [r.model_id for r in compare("wikitext")] == ["owner/b", "owner/a"]

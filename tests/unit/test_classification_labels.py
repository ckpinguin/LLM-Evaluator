"""Tests for turning a classifier's label scores into one predicted label."""

from testbed.evaluators.classification import predict_label

BINARY = {"negative", "positive"}


def test_three_class_model_on_two_class_data_ignores_neutral():
    scores = [
        {"label": "negative", "score": 0.2},
        {"label": "neutral", "score": 0.5},
        {"label": "positive", "score": 0.3},
    ]
    label_map = {"negative": "negative", "neutral": "neutral", "positive": "positive"}
    assert predict_label(scores, label_map, BINARY) == "positive"


def test_labels_missing_from_label_map_are_ignored():
    scores = [
        {"label": "LABEL_0", "score": 0.1},
        {"label": "LABEL_1", "score": 0.3},
        {"label": "LABEL_2", "score": 0.6},  # not in the label map
    ]
    label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}
    assert predict_label(scores, label_map, BINARY) == "positive"


def test_label_names_are_matched_case_insensitively():
    scores = [{"label": "Negative", "score": 0.9}, {"label": "POSITIVE", "score": 0.1}]
    label_map = {"NEGATIVE": "negative", "positive": "positive"}
    assert predict_label(scores, label_map, BINARY) == "negative"

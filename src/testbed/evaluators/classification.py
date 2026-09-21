"""Evaluators for classification models (models that assign a label to a text).

Version 1 supports sentiment classification: is a text negative, (neutral,) or positive?

Different models use different label names ("POSITIVE", "positive", "LABEL_1", ...). Each
catalog entry therefore has a ``label_map`` that translates the model's labels to our three
canonical labels. The model runs through the standard Transformers ``pipeline``, and the
scores are computed with scikit-learn, the reference implementation of these metrics.
"""

from pathlib import Path

from testbed import settings
from testbed.catalog import CatalogEntry, ModelType
from testbed.errors import TestbedError
from testbed.evaluators.base import Evaluator, EvaluatorOutput

CANONICAL_LABELS = ("negative", "neutral", "positive")


def predict_label(
    label_scores: list[dict], label_map: dict[str, str], allowed: set[str]
) -> str | None:
    """Pick the predicted label for one text.

    ``label_scores`` is the pipeline output for one text, e.g.
    ``[{"label": "NEGATIVE", "score": 0.1}, {"label": "POSITIVE", "score": 0.9}]``.

    We choose the highest-scoring label *among the labels the dataset uses* (``allowed``).
    This lets a model with three labels (negative/neutral/positive) be tested on data that
    only has two: "neutral" is simply ignored. Model labels missing from ``label_map`` are
    ignored too. Label names are compared without caring about upper/lower case.

    Returns ``None`` if none of the model's labels could be matched.
    """
    lower_map = {name.lower(): canonical for name, canonical in label_map.items()}
    best_label, best_score = None, -1.0
    for item in label_scores:
        canonical = lower_map.get(item["label"].lower())
        if canonical in allowed and item["score"] > best_score:
            best_label, best_score = canonical, item["score"]
    return best_label


def run_sentiment(
    dataset_id: str,
    split: str,
    text_column: str,
    index_to_label: dict[int, str],
    model_dir: Path,
    entry: CatalogEntry,
    device: str,
    limit: int,
) -> EvaluatorOutput:
    """Classify a sample of a labeled sentiment dataset and compare with the true labels.

    ``index_to_label`` translates the dataset's numeric labels to canonical labels, e.g.
    ``{0: "negative", 1: "positive"}``.
    """
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import pipeline

    data = load_dataset(dataset_id, split=split)
    # A fixed-seed sample: small, and the same every time (so runs can be compared).
    data = data.shuffle(seed=settings.SEED).select(range(min(limit, len(data))))

    classifier = pipeline(
        "text-classification",
        model=str(model_dir),
        device=device,
        top_k=None,  # return the score of every label, not only the best one
        truncation=True,  # cut texts that are longer than the model can read
    )
    outputs = classifier(list(data[text_column]), batch_size=16)

    allowed = set(index_to_label.values())
    predicted = [predict_label(scores, entry.label_map, allowed) for scores in outputs]
    if None in predicted:
        model_labels = sorted({item["label"] for item in outputs[0]})
        raise TestbedError(
            f"The model's labels {model_labels} don't match the 'label_map' in catalog.yaml.",
            hint="Fix the label_map of this model in catalog.yaml.",
        )
    expected = [index_to_label[label] for label in data["label"]]

    scores = {
        "accuracy": accuracy_score(expected, predicted),
        # Macro F1 = average of the F1 score of each class, so every class counts equally.
        "macro_f1": f1_score(expected, predicted, average="macro"),
    }
    return EvaluatorOutput(scores={k: float(v) for k, v in scores.items()}, examples=len(data))


def run_sst2(model_dir: Path, entry: CatalogEntry, device: str, limit: int):
    """SST-2: short movie-review sentences labeled negative or positive."""
    return run_sentiment(
        "stanfordnlp/sst2",
        "validation",  # the test split has no labels
        "sentence",
        {0: "negative", 1: "positive"},
        model_dir,
        entry,
        device,
        limit,
    )


def run_rotten_tomatoes(model_dir: Path, entry: CatalogEntry, device: str, limit: int):
    """Rotten Tomatoes: movie-review snippets labeled negative or positive."""
    return run_sentiment(
        "cornell-movie-review-data/rotten_tomatoes",
        "test",
        "text",
        {0: "negative", 1: "positive"},
        model_dir,
        entry,
        device,
        limit,
    )


SST2 = Evaluator(
    key="sst2",
    name="SST-2",
    model_type=ModelType.CLASSIFICATION,
    task="sentiment",
    description="Positive/negative sentiment of short movie-review sentences.",
    dataset="stanfordnlp/sst2 (validation)",
    main_metric="accuracy",
    higher_is_better=True,
    how_to_read="0.5 = random guessing, 1.0 = perfect",
    run=run_sst2,
    is_default=True,
)

ROTTEN_TOMATOES = Evaluator(
    key="rotten-tomatoes",
    name="Rotten Tomatoes",
    model_type=ModelType.CLASSIFICATION,
    task="sentiment",
    description="Positive/negative sentiment of movie-review snippets from Rotten Tomatoes.",
    dataset="cornell-movie-review-data/rotten_tomatoes (test)",
    main_metric="accuracy",
    higher_is_better=True,
    how_to_read="0.5 = random guessing, 1.0 = perfect",
    run=run_rotten_tomatoes,
)

CLASSIFICATION_EVALUATORS = [SST2, ROTTEN_TOMATOES]

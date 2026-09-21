"""Evaluators for generative (text-generating) models.

They use EleutherAI's lm-evaluation-harness ("lm-eval"), the standard tool behind most
published results for language models. We don't re-implement the scoring: multiple-choice
benchmarks have subtle details (how answers are tokenized, how scores are normalized for
answer length) that the harness gets right.

How a multiple-choice benchmark works: for each question, the model sees the question and
each possible answer; the answer the model finds most likely counts as its choice.
"""

from pathlib import Path

from testbed import settings
from testbed.catalog import CatalogEntry, ModelType
from testbed.evaluators.base import Evaluator, EvaluatorOutput


def run_lm_eval(task: str, model_dir: Path, device: str, limit: int) -> EvaluatorOutput:
    """Run one lm-evaluation-harness task on the downloaded model and return its scores."""
    import lm_eval  # imported here because it loads PyTorch, which takes a few seconds

    output = lm_eval.simple_evaluate(
        model="hf",  # a Hugging Face Transformers model
        model_args={
            "pretrained": str(model_dir),  # load from our download folder, not from the Hub
            "dtype": "float32",  # the most reliable precision on the Apple GPU
            # Some models can read 32k+ tokens at once. Reading that much in one go would
            # need a lot of memory, so we cap it (this mainly affects the perplexity task).
            "max_length": settings.MAX_LENGTH,
        },
        tasks=[task],
        device=device,
        limit=limit,  # only the first `limit` questions: fast and always the same ones
        batch_size=1,  # one question at a time keeps memory use low
        log_samples=False,  # we only need the final scores
        bootstrap_iters=0,  # skip error-bar estimation, which we don't show
        verbosity="ERROR",  # hide lm-eval's many informational messages
        random_seed=settings.SEED,
        numpy_random_seed=settings.SEED,
        torch_random_seed=settings.SEED,
        fewshot_random_seed=settings.SEED,
    )

    # lm-eval names metrics like "acc,none" (metric, filter) and also reports extra numbers
    # such as error bars ("acc_stderr") and "sample_len". We keep only the task's real
    # metrics (the ones listed in "higher_is_better") and drop the ",none" suffix.
    real_metrics = output["higher_is_better"][task]
    scores = {}
    for name, value in output["results"][task].items():
        metric = name.split(",")[0]
        if metric in real_metrics and isinstance(value, int | float):
            scores[metric] = float(value)
    examples = int(output["n-samples"][task]["effective"])
    return EvaluatorOutput(scores=scores, examples=examples)


def run_arc_easy(model_dir: Path, entry: CatalogEntry, device: str, limit: int):
    """ARC-Easy: grade-school science questions with 4 answer options."""
    return run_lm_eval("arc_easy", model_dir, device, limit)


def run_hellaswag(model_dir: Path, entry: CatalogEntry, device: str, limit: int):
    """HellaSwag: pick the most sensible ending for an everyday situation (4 options)."""
    return run_lm_eval("hellaswag", model_dir, device, limit)


def run_wikitext(model_dir: Path, entry: CatalogEntry, device: str, limit: int):
    """WikiText: how well the model predicts Wikipedia articles, word by word."""
    return run_lm_eval("wikitext", model_dir, device, limit)


ARC_EASY = Evaluator(
    key="arc-easy",
    name="ARC-Easy",
    model_type=ModelType.GENERATIVE,
    description="Multiple-choice grade-school science questions.",
    dataset="allenai/ai2_arc (ARC-Easy, test)",
    main_metric="acc_norm",
    higher_is_better=True,
    how_to_read="0.25 = random guessing, 1.0 = perfect",
    run=run_arc_easy,
    is_default=True,
)

HELLASWAG = Evaluator(
    key="hellaswag",
    name="HellaSwag",
    model_type=ModelType.GENERATIVE,
    description="Common-sense reasoning: choose the most sensible ending of a short story.",
    dataset="Rowan/hellaswag (validation)",
    main_metric="acc_norm",
    higher_is_better=True,
    how_to_read="0.25 = random guessing, 1.0 = perfect",
    run=run_hellaswag,
)

WIKITEXT = Evaluator(
    key="wikitext",
    name="WikiText perplexity",
    model_type=ModelType.GENERATIVE,
    description="How well the model predicts real Wikipedia text (perplexity).",
    dataset="wikitext-2 (test, whole articles)",
    main_metric="word_perplexity",
    # Perplexity is roughly "how many words the model hesitates between" at each step.
    higher_is_better=False,
    how_to_read="1.0 would be perfect prediction; small models typically score 20 to 60",
    run=run_wikitext,
)

GENERATIVE_EVALUATORS = [ARC_EASY, HELLASWAG, WIKITEXT]

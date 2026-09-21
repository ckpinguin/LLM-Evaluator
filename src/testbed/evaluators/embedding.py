"""Evaluators for embedding models (models that turn a text into a vector of numbers).

They use the evaluators built into the Sentence Transformers library, which compute the same
metrics as the MTEB leaderboard.

How the STS benchmark works: people rated how similar two sentences are. The model turns both
sentences into vectors; similar sentences should get similar vectors (high cosine similarity).
The score says how well the model's similarities agree with the human ratings.
"""

from pathlib import Path

from testbed import settings
from testbed.catalog import CatalogEntry, ModelType
from testbed.evaluators.base import Evaluator, EvaluatorOutput


def load_model(model_dir: Path, device: str):
    """Load a Sentence Transformers model from the download folder."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_dir), device=device)


def remove_prefix(scores: dict, prefix: str) -> dict[str, float]:
    """Sentence Transformers puts the evaluator name in front of each metric; remove it."""
    return {name.removeprefix(prefix): float(value) for name, value in scores.items()}


def run_sts_benchmark(model_dir: Path, entry: CatalogEntry, device: str, limit: int):
    """Compare the model's sentence similarities with human similarity ratings."""
    from datasets import load_dataset
    from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

    data = load_dataset("sentence-transformers/stsb", split="test")
    # Shuffle with a fixed seed and take the first `limit` pairs: a small sample that is the
    # same every time, so results of different runs can be compared.
    data = data.shuffle(seed=settings.SEED).select(range(min(limit, len(data))))

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=data["sentence1"],
        sentences2=data["sentence2"],
        scores=data["score"],  # human ratings, from 0 (unrelated) to 1 (same meaning)
        name="sts-benchmark",
        write_csv=False,
    )
    scores = evaluator(load_model(model_dir, device))
    return EvaluatorOutput(scores=remove_prefix(scores, "sts-benchmark_"), examples=len(data))


def run_nano_scifact(model_dir: Path, entry: CatalogEntry, device: str, limit: int):
    """Search test: find the scientific abstracts that support or refute a claim.

    NanoBEIR is a deliberately small version of the BEIR search benchmark (50 questions,
    about 3,000 documents). It always uses all 50 questions, so ``limit`` is ignored.
    """
    from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator

    evaluator = NanoBEIREvaluator(dataset_names=["scifact"], write_csv=False)
    scores = evaluator(load_model(model_dir, device))

    # Keep the three most useful search metrics, measured with cosine similarity.
    wanted = ("ndcg@10", "mrr@10", "recall@10")
    scifact = remove_prefix(scores, "NanoSciFact_cosine_")
    examples = len(evaluator.evaluators[0].queries)
    return EvaluatorOutput(scores={k: scifact[k] for k in wanted}, examples=examples)


STS_BENCHMARK = Evaluator(
    key="sts-benchmark",
    name="STS Benchmark",
    model_type=ModelType.EMBEDDING,
    description="How well sentence similarities match human similarity ratings.",
    dataset="sentence-transformers/stsb (test)",
    main_metric="spearman_cosine",
    higher_is_better=True,
    how_to_read="-1 to 1; 1.0 = perfect agreement with human ratings, above 0.8 is good",
    run=run_sts_benchmark,
    is_default=True,
)

NANO_SCIFACT = Evaluator(
    key="nano-scifact",
    name="NanoBEIR SciFact",
    model_type=ModelType.EMBEDDING,
    description="Search: find scientific abstracts that support or refute a claim.",
    dataset="sentence-transformers/NanoBEIR-en (SciFact)",
    main_metric="ndcg@10",
    higher_is_better=True,
    how_to_read="0 to 1; how well relevant documents are ranked near the top, 1.0 = perfect",
    run=run_nano_scifact,
    supports_limit=False,  # fixed benchmark: always 50 questions
)

EMBEDDING_EVALUATORS = [STS_BENCHMARK, NANO_SCIFACT]

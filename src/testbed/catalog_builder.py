"""Generating the catalog from Hugging Face.

For each model type we ask Hugging Face for its most downloaded models, then keep the first
few that pass simple rules (small enough, safe file format, no sign-in, no custom code).
The result is written to catalog.yaml, which you can then review and edit by hand.

Only public information is used (the Hugging Face "Hub" API); no model is downloaded. For
classification models, only their small config.json is fetched, to read the label names.
"""

import json
import tempfile
from dataclasses import dataclass
from datetime import date

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from testbed import settings
from testbed.catalog import Catalog, CatalogEntry, ModelType
from testbed.downloads import select_files
from testbed.errors import TestbedError

# How to search for each model type on Hugging Face. "filter" selects models by tag: the
# library that loads them, and (for classification) English, because our test data is
# English. Sentiment models are recognized by their label names (see auto_label_map), not by
# their name: the most popular one is called "...-finetuned-sst-2-english".
TYPE_QUERIES = {
    ModelType.GENERATIVE: {"pipeline_tag": "text-generation", "filter": "transformers"},
    ModelType.EMBEDDING: {
        "pipeline_tag": "sentence-similarity",
        "filter": "sentence-transformers",
    },
    ModelType.CLASSIFICATION: {
        "pipeline_tag": "text-classification",
        "filter": ["transformers", "en"],
    },
}

# Short descriptions used for generated entries (edit them by hand if you like).
TYPE_DESCRIPTIONS = {
    ModelType.GENERATIVE: "Text generation model",
    ModelType.EMBEDDING: "Sentence embedding model",
    ModelType.CLASSIFICATION: "Sentiment classification model",
}

# Models with these tags are skipped: they need to run their own code ("custom_code") or are
# stored in compressed formats for other tools (gguf, mlx, gptq, awq).
EXCLUDED_TAGS = {"custom_code", "gguf", "mlx", "gptq", "awq"}

# Repositories made only for software tests (random weights) are not useful in a catalog.
TEST_MODEL_MARKERS = ("internal-testing", "-testing/", "tiny-random")

# How many of the most downloaded models we look at per type (more = slower, more choice).
CANDIDATES_PER_TYPE = 100

# Label names we recognize, lower-case. Anything else (e.g. "LABEL_0") cannot be mapped.
LABEL_SYNONYMS = {
    "negative": "negative",
    "neg": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "positive": "positive",
    "pos": "positive",
    # Some models use five levels; the extremes still mean negative / positive.
    "very negative": "negative",
    "very positive": "positive",
}


@dataclass
class BuildReport:
    """What ``build_catalog`` found."""

    catalog: Catalog  # the new catalog (not saved yet)
    checked: int  # how many candidates were looked at
    skipped: list[tuple[str, str]]  # (model id, reason) for every model that was left out


def why_unsuitable(info, max_size_gb: float) -> str | None:
    """Return the reason a model can't go into the catalog, or ``None`` if it is fine."""
    if any(marker in info.id for marker in TEST_MODEL_MARKERS):
        return "test model with random weights"
    if info.gated:
        return "requires sign-in on Hugging Face"
    bad_tags = EXCLUDED_TAGS.intersection(info.tags or [])
    if bad_tags:
        return f"has tag '{sorted(bad_tags)[0]}' (custom code or unsupported format)"
    files = select_files([(s.rfilename, s.size or 0) for s in info.siblings or []])
    if not any(name.endswith(".safetensors") for name, _ in files):
        return "no weights in the safe 'safetensors' format"
    size_gb = sum(size for _, size in files) / 1024**3
    if size_gb > max_size_gb:
        return f"too large ({size_gb:.1f} GB > {max_size_gb} GB)"
    return None


def auto_label_map(id2label: dict) -> dict[str, str] | None:
    """Translate a model's label names to our canonical labels, if we can.

    Example: {"0": "NEGATIVE", "1": "POSITIVE"} -> {"NEGATIVE": "negative", "POSITIVE": "positive"}.
    Returns ``None`` when "negative" and "positive" can't both be found.
    """
    label_map = {}
    for name in id2label.values():
        canonical = LABEL_SYNONYMS.get(str(name).lower())
        if canonical:
            label_map[name] = canonical
    if {"negative", "positive"} <= set(label_map.values()):
        return label_map
    return None


def fetch_id2label(repo_id: str) -> dict:
    """Read the label names from a model's config.json (downloaded to a temporary folder)."""
    with tempfile.TemporaryDirectory() as folder:
        config_path = hf_hub_download(repo_id, "config.json", cache_dir=folder)
        with open(config_path, encoding="utf-8") as file:
            return json.load(file).get("id2label") or {}


def make_entry(info, model_type: ModelType, label_map: dict | None) -> CatalogEntry:
    """Create a catalog entry from Hugging Face's information about a model."""
    files = select_files([(s.rfilename, s.size or 0) for s in info.siblings or []])
    size_mb = round(sum(size for _, size in files) / 1024**2)

    license_name = "unknown"
    for tag in info.tags or []:
        if tag.startswith("license:"):
            license_name = tag.removeprefix("license:")
            break

    description = TYPE_DESCRIPTIONS[model_type]
    parameters = info.safetensors.total if info.safetensors else None
    if parameters:
        description += f", {format_parameters(parameters)} parameters"

    return CatalogEntry(
        id=info.id,
        type=model_type,
        task="sentiment" if model_type == ModelType.CLASSIFICATION else None,
        label_map=label_map,
        description=description + ".",
        size_mb=max(size_mb, 1),
        license=license_name,
    )


def format_parameters(count: int) -> str:
    """135_000_000 -> '135M', 1_200_000_000 -> '1.2B'."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    return f"{count / 1_000_000:.0f}M"


def build_catalog(
    per_type: int = settings.DEFAULT_PER_TYPE,
    max_size_gb: float = settings.DEFAULT_MAX_SIZE_GB,
    api: HfApi | None = None,
) -> BuildReport:
    """Search Hugging Face and return a new catalog (it is not saved here).

    ``api`` can be replaced by a fake object in tests.
    """
    api = api or HfApi()
    entries: list[CatalogEntry] = []
    skipped: list[tuple[str, str]] = []
    checked = 0
    try:
        for model_type, query in TYPE_QUERIES.items():
            kept_for_type = 0
            candidates = api.list_models(
                **query,
                gated=False,
                sort="downloads",  # most downloaded first
                # Rough pre-filter so we don't look at huge models; the exact size check is
                # done by why_unsuitable.
                num_parameters="max:2B",
                limit=CANDIDATES_PER_TYPE,
            )
            for candidate in candidates:
                if kept_for_type >= per_type:
                    break
                checked += 1
                info = api.model_info(candidate.id, files_metadata=True)
                reason = why_unsuitable(info, max_size_gb)

                label_map = None
                if reason is None and model_type == ModelType.CLASSIFICATION:
                    label_map = auto_label_map(fetch_id2label(info.id))
                    if label_map is None:
                        reason = "label names can't be matched to negative/positive"

                if reason:
                    skipped.append((info.id, reason))
                    continue
                entries.append(make_entry(info, model_type, label_map))
                kept_for_type += 1
    except (HfHubHTTPError, OSError) as error:
        raise TestbedError(
            f"Could not reach Hugging Face: {error}",
            hint="Check your internet connection. The existing catalog was not changed.",
        ) from error

    if not entries:
        raise TestbedError(
            "No suitable models were found.", hint="Try a larger --max-size-gb value."
        )
    catalog = Catalog(generated_at=date.today(), max_size_gb=max_size_gb, models=entries)
    return BuildReport(catalog=catalog, checked=checked, skipped=skipped)

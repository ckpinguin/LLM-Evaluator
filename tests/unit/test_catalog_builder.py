"""Tests for generating the catalog from Hugging Face (testbed/catalog_builder.py).

No network: small fake objects stand in for Hugging Face's model information.
"""

from types import SimpleNamespace

from testbed import catalog_builder
from testbed.catalog import ModelType
from testbed.catalog_builder import auto_label_map, build_catalog, why_unsuitable

MB = 1024**2


def fake_info(repo_id, size_mb=100, tags=(), gated=False, files=None):
    """A stand-in for huggingface_hub's ModelInfo with only the fields we use."""
    if files is None:
        files = [("model.safetensors", size_mb * MB), ("config.json", 1000)]
    return SimpleNamespace(
        id=repo_id,
        tags=list(tags) + ["license:mit"],
        gated=gated,
        siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files],
        safetensors=SimpleNamespace(total=135_000_000),
    )


def test_suitable_model_has_no_reason():
    assert why_unsuitable(fake_info("owner/good"), max_size_gb=2) is None


def test_unsuitable_models_get_a_reason():
    assert "too large" in why_unsuitable(fake_info("owner/big", size_mb=3000), max_size_gb=2)
    assert "sign-in" in why_unsuitable(fake_info("owner/gated", gated="auto"), max_size_gb=2)
    for tag in ["custom_code", "gguf", "mlx", "gptq", "awq"]:
        assert tag in why_unsuitable(fake_info("owner/x", tags=[tag]), max_size_gb=2)
    no_safetensors = fake_info("owner/old", files=[("pytorch_model.bin", 100 * MB)])
    assert "safetensors" in why_unsuitable(no_safetensors, max_size_gb=2)
    assert "test model" in why_unsuitable(fake_info("hf-internal-testing/tiny"), max_size_gb=2)


def test_auto_label_map():
    assert auto_label_map({"0": "NEGATIVE", "1": "POSITIVE"}) == {
        "NEGATIVE": "negative",
        "POSITIVE": "positive",
    }
    assert auto_label_map({"0": "neg", "1": "neutral", "2": "pos"}) == {
        "neg": "negative",
        "neutral": "neutral",
        "pos": "positive",
    }
    assert auto_label_map({"0": "LABEL_0", "1": "LABEL_1"}) is None
    assert auto_label_map({"0": "negative", "1": "neutral"}) is None  # no "positive"


class FakeApi:
    """Answers list_models / model_info from prepared fake data."""

    def __init__(self, candidates):
        self.candidates = candidates  # pipeline_tag -> list of fake infos, most downloaded first

    def list_models(self, pipeline_tag, **kwargs):
        return [SimpleNamespace(id=info.id) for info in self.candidates.get(pipeline_tag, [])]

    def model_info(self, repo_id, files_metadata=False):
        for infos in self.candidates.values():
            for info in infos:
                if info.id == repo_id:
                    return info
        raise KeyError(repo_id)


def test_build_keeps_the_most_downloaded_suitable_models(monkeypatch):
    api = FakeApi(
        {
            "text-generation": [
                fake_info("owner/gen-1"),
                fake_info("owner/gen-too-big", size_mb=5000),
                fake_info("owner/gen-2"),
                fake_info("owner/gen-3"),
            ],
            "text-classification": [
                fake_info("owner/sentiment-ok"),
                fake_info("owner/sentiment-unmappable"),
            ],
        }
    )
    labels = {
        "owner/sentiment-ok": {"0": "NEGATIVE", "1": "POSITIVE"},
        "owner/sentiment-unmappable": {"0": "LABEL_0", "1": "LABEL_1"},
    }
    monkeypatch.setattr(catalog_builder, "fetch_id2label", lambda repo_id: labels[repo_id])

    report = build_catalog(per_type=2, max_size_gb=2, api=api)

    kept = [(entry.id, entry.type) for entry in report.catalog.models]
    assert kept == [
        ("owner/gen-1", ModelType.GENERATIVE),
        ("owner/gen-2", ModelType.GENERATIVE),
        ("owner/sentiment-ok", ModelType.CLASSIFICATION),
    ]
    assert report.catalog.models[2].label_map == {"NEGATIVE": "negative", "POSITIVE": "positive"}
    assert report.catalog.models[0].license == "mit"
    assert report.catalog.models[0].size_mb == 100
    assert dict(report.skipped).keys() == {"owner/gen-too-big", "owner/sentiment-unmappable"}

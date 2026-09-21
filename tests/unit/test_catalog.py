"""Tests for loading and validating catalog.yaml (testbed/catalog.py)."""

import copy

import pytest
import yaml
from conftest import SAMPLE_MODELS, write_catalog

from testbed.catalog import Catalog, ModelType, find_model, load_catalog, save_catalog
from testbed.errors import TestbedError


def models_with_change(index, **changes):
    """Return a copy of SAMPLE_MODELS where entry ``index`` has the given changes.

    A value of ``None`` removes that field.
    """
    models = copy.deepcopy(SAMPLE_MODELS)
    for field, value in changes.items():
        if value is None:
            models[index].pop(field, None)
        else:
            models[index][field] = value
    return models


def test_valid_catalog_loads(catalog_file):
    entries = load_catalog(catalog_file)
    assert [e.type for e in entries] == [
        ModelType.GENERATIVE,
        ModelType.EMBEDDING,
        ModelType.CLASSIFICATION,
    ]
    assert entries[2].label_map == {"NEGATIVE": "negative", "POSITIVE": "positive"}


@pytest.mark.parametrize(
    ("index", "changes", "expected_text"),
    [
        (0, {"description": "x" * 121}, "description"),
        (0, {"size_mb": 0}, "size_mb"),
        (2, {"task": None}, "task"),
        (2, {"label_map": None}, "label_map"),
        (2, {"label_map": {"NEGATIVE": "negative"}}, "positive"),
        (0, {"task": "sentiment"}, "only allowed for classification"),
        (0, {"licence": "mit"}, "licence"),
    ],
)
def test_invalid_entry_is_reported_with_its_number_and_id(tmp_path, index, changes, expected_text):
    path = write_catalog(tmp_path / "catalog.yaml", models_with_change(index, **changes))
    with pytest.raises(TestbedError) as error:
        load_catalog(path)
    message = str(error.value)
    assert f"entry {index + 1} ({SAMPLE_MODELS[index]['id']})" in message
    assert expected_text in message


def test_duplicate_ids_are_rejected(tmp_path):
    models = copy.deepcopy(SAMPLE_MODELS) + [copy.deepcopy(SAMPLE_MODELS[0])]
    path = write_catalog(tmp_path / "catalog.yaml", models)
    with pytest.raises(TestbedError, match="more than once"):
        load_catalog(path)


def test_missing_file_gives_a_hint(tmp_path):
    with pytest.raises(TestbedError) as error:
        load_catalog(tmp_path / "missing.yaml")
    assert "catalog build" in error.value.hint


def test_find_model_lists_valid_ids_for_unknown_model(catalog_file):
    entries = load_catalog(catalog_file)
    assert find_model(entries, "owner/tiny-embedder").type == ModelType.EMBEDDING
    with pytest.raises(TestbedError) as error:
        find_model(entries, "owner/unknown")
    assert "owner/tiny-generator" in error.value.hint


def test_save_then_load_gives_the_same_models(tmp_path, catalog_file):
    original = load_catalog(catalog_file)
    target = tmp_path / "saved.yaml"
    save_catalog(Catalog(generated_at="2026-09-21", max_size_gb=2.0, models=original), target)

    assert load_catalog(target) == original
    assert target.read_text().startswith("# Curated model catalog")


def test_failed_save_keeps_the_old_file(tmp_path, catalog_file, monkeypatch):
    before = catalog_file.read_text()

    def broken_dump(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(yaml, "safe_dump", broken_dump)
    with pytest.raises(RuntimeError):
        save_catalog(Catalog(models=load_catalog(catalog_file)), catalog_file)

    assert catalog_file.read_text() == before
    assert list(tmp_path.glob("*.tmp")) == []

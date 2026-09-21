"""The model catalog: the curated list of models the user can choose from.

The catalog is stored in ``catalog.yaml`` at the project root, so it can be read and edited by
hand. This module describes what a valid catalog entry looks like (using Pydantic models) and
turns mistakes in the file into clear, plain-language error messages.

File format: specs/001-model-testbed/contracts/catalog-file.md
"""

import os
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from testbed import settings
from testbed.errors import TestbedError


class ModelType(StrEnum):
    """The three kinds of models the testbed knows about."""

    GENERATIVE = "generative"  # produces text
    EMBEDDING = "embedding"  # turns text into a vector of numbers
    CLASSIFICATION = "classification"  # assigns a label to a text


# The label names every classification model is translated to (see ``label_map`` below).
CanonicalLabel = Literal["negative", "neutral", "positive"]


class CatalogEntry(BaseModel):
    """One model the user may select."""

    # "forbid" turns a typo in a field name (e.g. "licence") into an error instead of
    # silently ignoring it.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$", description="Hugging Face id 'owner/name'")
    type: ModelType
    description: str = Field(max_length=120)
    size_mb: int = Field(gt=0, description="Approximate download size in MB")
    license: str = Field(min_length=1, description="License id, or 'unknown'")

    # Only for classification models: which task they solve, and how their own label names
    # translate to our canonical labels (e.g. "POSITIVE" -> "positive").
    task: Literal["sentiment"] | None = None
    label_map: dict[str, CanonicalLabel] | None = None

    @model_validator(mode="after")
    def check_classification_fields(self) -> "CatalogEntry":
        """Classification entries need ``task`` and ``label_map``; other types must not."""
        if "\n" in self.description:
            raise ValueError("description must be a single line")

        if self.type == ModelType.CLASSIFICATION:
            if self.task is None:
                raise ValueError("classification models need a 'task' (v1 allows: sentiment)")
            if not self.label_map:
                raise ValueError("classification models need a 'label_map'")
            mapped = set(self.label_map.values())
            if not {"negative", "positive"} <= mapped:
                raise ValueError("'label_map' must map to both 'negative' and 'positive'")
        elif self.task is not None or self.label_map is not None:
            raise ValueError("'task' and 'label_map' are only allowed for classification models")
        return self


class Catalog(BaseModel):
    """The whole catalog file: some information about how it was made, plus the models."""

    model_config = ConfigDict(extra="forbid")

    generated_at: date | None = None  # informational only
    max_size_gb: float | None = None  # informational only
    models: list[CatalogEntry] = Field(min_length=1)


def load_catalog(path: Path | None = None) -> list[CatalogEntry]:
    """Read and validate the catalog file and return its models.

    Every problem is reported as a ``TestbedError`` that names the entry and the field, e.g.
    ``catalog.yaml, entry 3 (owner/model): size_mb: Input should be greater than 0``.
    """
    path = path or settings.CATALOG_PATH
    if not path.exists():
        raise TestbedError(
            f"The catalog file {path} does not exist.",
            hint="Create it with:  uv run testbed catalog build",
        )

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TestbedError(f"{path.name} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise TestbedError(f"{path.name} must contain a 'models:' list.")

    # Validate the entries one by one, so the error message can say which entry is wrong.
    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()
    for number, raw_entry in enumerate(data["models"], start=1):
        model_id = raw_entry.get("id", "?") if isinstance(raw_entry, dict) else "?"
        where = f"{path.name}, entry {number} ({model_id})"
        try:
            entry = CatalogEntry.model_validate(raw_entry)
        except ValidationError as exc:
            raise TestbedError(f"{where}: {_describe_first_error(exc)}") from exc
        if entry.id in seen_ids:
            raise TestbedError(f"{where}: this model id appears more than once")
        seen_ids.add(entry.id)
        entries.append(entry)

    # Finally check the top-level fields (and that the list is not empty).
    try:
        Catalog.model_validate({**data, "models": entries})
    except ValidationError as exc:
        raise TestbedError(f"{path.name}: {_describe_first_error(exc)}") from exc

    return entries


CATALOG_HEADER = """\
# Curated model catalog for the testbed.
# Edit by hand, or regenerate with:  uv run testbed catalog build
# Format: specs/001-model-testbed/contracts/catalog-file.md
"""


def save_catalog(catalog: Catalog, path: Path | None = None) -> None:
    """Write the catalog to ``path`` (default: catalog.yaml) without risking the old file.

    We first write to a temporary file and then swap it in with one ``os.replace`` call.
    If anything goes wrong before that, the old catalog file stays exactly as it was.
    """
    path = path or settings.CATALOG_PATH
    temporary = path.with_suffix(".yaml.tmp")
    data = catalog.model_dump(mode="json", exclude_none=True)
    if catalog.generated_at:
        data["generated_at"] = catalog.generated_at  # YAML writes dates without quotes
    try:
        text = CATALOG_HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)  # the swap is "atomic": it fully happens or not at all
    finally:
        temporary.unlink(missing_ok=True)  # only still there if something failed


def find_model(catalog: list[CatalogEntry], model_id: str) -> CatalogEntry:
    """Return the catalog entry with this Hugging Face id, or explain which ids are valid."""
    for entry in catalog:
        if entry.id == model_id:
            return entry
    valid_ids = ", ".join(entry.id for entry in catalog)
    raise TestbedError(
        f"'{model_id}' is not in the catalog.",
        hint=f"Choose one of: {valid_ids}  (or run 'uv run testbed models')",
    )


def _describe_first_error(exc: ValidationError) -> str:
    """Turn Pydantic's error list into one short sentence like 'size_mb: must be > 0'."""
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"])
    message = error["msg"].removeprefix("Value error, ")
    return f"{field}: {message}" if field else message

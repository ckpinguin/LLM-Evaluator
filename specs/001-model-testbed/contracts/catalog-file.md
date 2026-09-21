# Contract: Catalog File (`catalog.yaml`)

Location: repository root. Written by `testbed catalog build`, readable and editable by hand
(FR-005). Fields and validation rules are defined in
[data-model.md](../data-model.md#catalogentry-stored-in-catalogyaml).

## Example

```yaml
# Curated model catalog for the testbed.
# Edit by hand, or regenerate with:  uv run testbed catalog build
generated_at: 2026-09-21
max_size_gb: 2.0
models:
  - id: HuggingFaceTB/SmolLM2-135M
    type: generative
    description: Very small general-purpose language model (135M parameters).
    size_mb: 270
    license: apache-2.0

  - id: sentence-transformers/all-MiniLM-L6-v2
    type: embedding
    description: Small, fast English sentence embedding model.
    size_mb: 91
    license: apache-2.0

  - id: distilbert/distilbert-base-uncased-finetuned-sst-2-english
    type: classification
    task: sentiment
    label_map:
      NEGATIVE: negative
      POSITIVE: positive
    description: DistilBERT fine-tuned for positive/negative movie review sentiment.
    size_mb: 268
    license: apache-2.0
```

## Rules

- A hand-edited entry is accepted if it passes validation. If the model on Hugging Face needs
  sign-in, custom code, or has no `*.safetensors` weights, `evaluate` refuses it before
  downloading, with an explanation (spec edge case).
- If the file is invalid, every command that needs the catalog stops and prints
  `catalog.yaml, entry <n> (<id>): <problem>`.
- Comments in hand-edited files are not preserved when the file is regenerated (the builder
  writes a fresh file with a standard header comment).

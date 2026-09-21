# Contract: Core Python Functions

The CLI is a thin layer over these functions. A future local web page (FR-022) calls the same
functions, so evaluation behavior never depends on the terminal. The functions do not print; they
return data and report progress through an optional callback.

```python
# testbed/catalog.py
def load_catalog(path: Path = CATALOG_PATH) -> list[CatalogEntry]: ...
def find_model(catalog: list[CatalogEntry], model_id: str) -> CatalogEntry: ...

# testbed/evaluators/__init__.py
def list_evaluators(model_type: ModelType | None = None) -> list[Evaluator]: ...
def compatible_evaluators(entry: CatalogEntry) -> list[Evaluator]: ...
def default_evaluator(entry: CatalogEntry) -> Evaluator: ...

# testbed/runner.py
def run_evaluation(
    entry: CatalogEntry,
    evaluators: list[Evaluator],
    limit: int = DEFAULT_LIMIT,
    device: str = "auto",
    on_progress: Callable[[str], None] | None = None,
) -> EvaluationRun: ...

# testbed/results.py
def load_history(model_type=None, evaluator=None, model_id=None) -> list[EvaluationResult]: ...
def compare(evaluator_key: str) -> list[EvaluationResult]: ...

# testbed/downloads.py
def cleanup_downloads() -> int: ...  # returns bytes freed

# testbed/catalog_builder.py
def build_catalog(per_type: int = 5, max_size_gb: float = 2.0) -> BuildReport: ...
```

## Guarantees

- `run_evaluation` always removes downloaded files before it returns or re-raises (FR-010).
- `run_evaluation` raises `TestbedError` (a single exception type with a plain-language message)
  for problems the user can fix: not enough disk space, model not reachable, run already in
  progress. Evaluator failures do not raise; they are recorded in `EvaluationRun.errors`.
- `KeyboardInterrupt` (Ctrl+C) stops the run; after cleanup `run_evaluation` returns the run
  with status `cancelled`, and the CLI exits with code 130. (Returning a status instead of
  re-raising keeps the behavior identical for the terminal and a future web page.)

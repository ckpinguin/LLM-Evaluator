"""The command-line interface: the ``testbed`` command and its subcommands.

This is the only module that prints to the terminal. Each command collects the user's choices,
calls the core functions (catalog, runner, results, ...), and shows what they return.
A future web page can call the same core functions.

Commands are described in specs/001-model-testbed/contracts/cli.md.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

import testbed
from testbed import settings
from testbed.catalog import CatalogEntry, ModelType, find_model, load_catalog, save_catalog
from testbed.catalog_builder import build_catalog
from testbed.device import pick_device
from testbed.downloads import cleanup_downloads, run_lock
from testbed.errors import TestbedError
from testbed.evaluators import (
    Evaluator,
    compatible_evaluators,
    default_evaluator,
    get_evaluator,
    list_evaluators,
)
from testbed.results import EvaluationResult, load_history
from testbed.results import compare as compare_results
from testbed.runner import EvaluationRun, RunStatus, check_evaluators, run_evaluation

app = typer.Typer(
    help="Try out Hugging Face text models on your computer: pick a model, pick an evaluator, "
    "get a score. Models are deleted again after each run.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


class DeviceChoice(StrEnum):
    """Allowed values for the --device option."""

    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


# --- Small helpers -------------------------------------------------------------------------------


@contextmanager
def friendly_errors() -> Iterator[None]:
    """Show a ``TestbedError`` as a short message (and hint) and exit with code 1.

    Without this, the user would see a long Python traceback for problems they can fix
    themselves, like a typo in a model name.
    """
    try:
        yield
    except TestbedError as error:
        console.print(f"[bold red]Error:[/bold red] {escape(error.message)}")
        if error.hint:
            console.print(f"[yellow]Hint:[/yellow] {escape(error.hint)}")
        raise typer.Exit(code=1) from None


def format_size(megabytes: float) -> str:
    """270 -> '270 MB', 1500 -> '1.5 GB', 0.2 -> '205 KB'."""
    if megabytes >= 1000:
        return f"{megabytes / 1024:.1f} GB"
    if megabytes < 1:
        return f"{megabytes * 1024:.0f} KB"
    return f"{megabytes:.0f} MB"


def format_bytes(size: int) -> str:
    """Bytes -> '312 MB' style text."""
    return format_size(size / 1024**2)


def show_path(path: Path) -> str:
    """Show a path relative to the current folder when possible (shorter to read)."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def ask_number(question: str, maximum: int, default: int | None = None) -> int:
    """Ask for a number between 1 and ``maximum`` until the answer is valid."""
    choices = [str(number) for number in range(1, maximum + 1)]
    return IntPrompt.ask(question, choices=choices, default=default, show_choices=False)


def models_table(title: str, entries: list[CatalogEntry]) -> Table:
    """A numbered table of catalog models."""
    table = Table(title=title, title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold")  # never cut off ids: users type them
    table.add_column("Size", justify="right")
    table.add_column("License")
    table.add_column("Description")
    for number, entry in enumerate(entries, start=1):
        table.add_row(
            str(number), entry.id, format_size(entry.size_mb), entry.license, entry.description
        )
    return table


def choose_model_interactively(catalog: list[CatalogEntry]) -> CatalogEntry:
    """Ask for a model type, then show a numbered model list and let the user pick one."""
    types = [t for t in ModelType if any(entry.type == t for entry in catalog)]
    console.print("Which type of model do you want to evaluate?")
    for number, model_type in enumerate(types, start=1):
        console.print(f"  {number}. {model_type}")
    model_type = types[ask_number("Type", len(types)) - 1]

    entries = [entry for entry in catalog if entry.type == model_type]
    console.print(models_table(f"{model_type} models", entries))
    return entries[ask_number("Model", len(entries)) - 1]


def evaluators_table(title: str, evaluators: list[Evaluator], numbered: bool = False) -> Table:
    """A table describing evaluators; the recommended (default) ones are marked with *."""
    table = Table(title=title, title_justify="left", caption="* = recommended default")
    if numbered:
        table.add_column("#", justify="right")
    table.add_column("Evaluator", no_wrap=True)
    table.add_column("Model type", no_wrap=True)
    table.add_column("Main score", no_wrap=True)
    table.add_column("--limit", no_wrap=True)
    table.add_column("What it measures")  # last, so it gets the remaining width
    for number, item in enumerate(evaluators, start=1):
        better = "higher is better" if item.higher_is_better else "lower is better"
        row = [
            f"{item.key}{' *' if item.is_default else ''}",
            item.model_type,
            f"{item.main_metric} ({better})",
            "applies" if item.supports_limit else "fixed size",
            item.description,
        ]
        table.add_row(*([str(number)] if numbered else []), *row)
    return table


def choose_evaluators(entry: CatalogEntry) -> list[Evaluator]:
    """Show the evaluators that fit this model and let the user pick one or several."""
    options = compatible_evaluators(entry)
    console.print(evaluators_table("Evaluators for this model", options, numbered=True))
    default_number = options.index(default_evaluator(entry)) + 1
    while True:
        answer = Prompt.ask("Evaluators (numbers separated by commas)", default=str(default_number))
        try:
            numbers = [int(part) for part in answer.replace(" ", "").split(",") if part]
        except ValueError:
            numbers = []
        if numbers and all(1 <= number <= len(options) for number in numbers):
            # dict.fromkeys removes duplicates while keeping the order the user typed.
            return [options[number - 1] for number in dict.fromkeys(numbers)]
        console.print(f"[red]Please enter numbers from 1 to {len(options)}, e.g. 1 or 1,2[/red]")


def print_result(result: EvaluationResult, evaluator: Evaluator) -> None:
    """Show one evaluator's result with a plain-language explanation (FR-018, SC-008)."""
    better = "higher is better" if evaluator.higher_is_better else "lower is better"
    console.print(
        f"  [bold]{evaluator.name}[/bold]  {result.main_metric} = "
        f"[bold green]{result.main_score:.3f}[/bold green]  ({better}; {evaluator.how_to_read})"
    )
    others = [
        f"{name} = {value:.3f}"
        for name, value in result.scores.items()
        if name != result.main_metric  # the main score is already shown above
    ]
    details = f"{result.examples} examples, {result.duration_seconds:.1f} s on {result.device}"
    if others:
        details = f"{', '.join(others)}  |  {details}"
    console.print(f"  [dim]{escape(details)}[/dim]")


def print_run_summary(run: EvaluationRun) -> None:
    """Show errors, where results were saved, and how much was cleaned up."""
    for key, message in run.errors.items():
        console.print(f"  [bold red]{key} failed:[/bold red] {escape(message)}")
    if run.results:
        console.print(f"Saved {len(run.results)} result(s) to {show_path(settings.RESULTS_FILE)}")
    console.print(f"Cleaned up {format_bytes(run.cleaned_bytes)} of downloads.")


# --- Commands -------------------------------------------------------------------------------------


def _show_version(value: bool) -> None:
    """Print the version and stop (used by the --version option)."""
    if value:
        console.print(f"testbed {testbed.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the version and exit.", callback=_show_version),
    ] = False,
) -> None:
    """Try out Hugging Face text models on your computer."""


@app.command()
def models(
    model_type: Annotated[
        ModelType | None, typer.Option("--type", help="Only show models of this type.")
    ] = None,
) -> None:
    """List the models in the catalog (nothing is downloaded)."""
    with friendly_errors():
        catalog = load_catalog()
    for current_type in ModelType:
        if model_type is not None and current_type != model_type:
            continue
        entries = [entry for entry in catalog if entry.type == current_type]
        if entries:
            console.print(models_table(f"{current_type} models", entries))


@app.command()
def evaluators(
    model_type: Annotated[
        ModelType | None, typer.Option("--type", help="Only show evaluators for this type.")
    ] = None,
) -> None:
    """List the evaluators: what they measure and how to read their scores."""
    console.print(evaluators_table("Evaluators", list_evaluators(model_type)))


@app.command()
def evaluate(
    model: Annotated[
        str | None,
        typer.Argument(help="Hugging Face id from the catalog. Leave out to pick from a list."),
    ] = None,
    evaluator: Annotated[
        list[str] | None,
        typer.Option(
            "--evaluator",
            "-e",
            help="Evaluator key (see 'testbed evaluators'). Repeat to run several. "
            "Default: the recommended evaluator for the model.",
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option(min=1, help="Number of examples per evaluator.")
    ] = settings.DEFAULT_LIMIT,
    device: Annotated[
        DeviceChoice,
        typer.Option(help="Where to run. auto = NVIDIA GPU, else Apple GPU, else CPU."),
    ] = DeviceChoice.AUTO,
) -> None:
    """Download a model, evaluate it, show the scores, and delete the model again."""
    with friendly_errors():
        catalog = load_catalog()
        if model:
            entry = find_model(catalog, model)
        else:
            entry = choose_model_interactively(catalog)

        # Which evaluators: the ones given with --evaluator; otherwise the recommended one
        # (or, when the user is choosing interactively, ask).
        if evaluator:
            chosen = [get_evaluator(key) for key in evaluator]
        elif model:
            chosen = [default_evaluator(entry)]
        else:
            chosen = choose_evaluators(entry)
        check_evaluators(entry, chosen)  # stop now if one does not fit, before downloading
        device_name = pick_device(device)

        console.print(f"Model:       {entry.id} ({entry.type}, {format_size(entry.size_mb)})")
        console.print(
            f"Evaluators:  {', '.join(e.key for e in chosen)}   Device: {device_name}   "
            f"Sample size: {limit}"
        )

        run = run_evaluation(
            entry,
            chosen,
            limit=limit,
            device=device_name,
            on_progress=lambda message: console.print(f"[dim]{escape(message)}[/dim]"),
        )

    evaluators_by_key = {item.key: item for item in chosen}
    for result in run.results:
        print_result(result, evaluators_by_key[result.evaluator])
    print_run_summary(run)

    if run.status == RunStatus.CANCELLED:
        console.print("[yellow]Run cancelled.[/yellow]")
        raise typer.Exit(code=130)
    if run.status == RunStatus.FAILED:
        raise typer.Exit(code=1)


@app.command()
def cleanup() -> None:
    """Delete everything in the downloads folder (e.g. after a crash)."""
    with friendly_errors():
        with run_lock():  # never delete the files of a run that is still going on
            freed = cleanup_downloads()
    console.print(f"Cleaned up {format_bytes(freed)} of downloads.")


# "testbed catalog ..." commands live in their own small group.
catalog_app = typer.Typer(help="Manage the model catalog (catalog.yaml).", no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")


@catalog_app.command("build")
def catalog_build(
    per_type: Annotated[
        int, typer.Option(min=1, help="How many models to keep per model type.")
    ] = settings.DEFAULT_PER_TYPE,
    max_size_gb: Annotated[
        float, typer.Option(min=0.01, help="Largest download size to accept, in GB.")
    ] = settings.DEFAULT_MAX_SIZE_GB,
    output: Annotated[
        Path | None, typer.Option(help="Where to write the catalog. Default: catalog.yaml.")
    ] = None,
) -> None:
    """Generate the catalog from the most downloaded suitable models on Hugging Face."""
    target = output or settings.CATALOG_PATH
    with friendly_errors():
        console.print("Searching Hugging Face (this takes up to a minute) ...")
        report = build_catalog(per_type=per_type, max_size_gb=max_size_gb)
        save_catalog(report.catalog, target)

    console.print(
        f"Checked {report.checked} models: kept {len(report.catalog.models)}, "
        f"skipped {len(report.skipped)}."
    )
    for model_id, reason in report.skipped:
        console.print(f"  [dim]skipped {escape(model_id)}: {escape(reason)}[/dim]")
    console.print(f"Wrote {show_path(target)}. Review it and remove models you don't want.")


def show_warnings_in_terminal() -> None:
    """Send warnings from the core modules (e.g. a skipped history line) to the terminal."""
    handler = RichHandler(console=console, show_time=False, show_path=False)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[handler])


@app.command()
def history(
    model_type: Annotated[
        ModelType | None, typer.Option("--type", help="Only results of this model type.")
    ] = None,
    evaluator: Annotated[str | None, typer.Option(help="Only results of this evaluator.")] = None,
    model: Annotated[str | None, typer.Option(help="Only results of this model.")] = None,
) -> None:
    """Show saved results, newest first."""
    show_warnings_in_terminal()
    results = load_history(model_type=model_type, evaluator=evaluator, model_id=model)
    if not results:
        console.print("No results yet. Run for example:  uv run testbed evaluate")
        return

    table = Table(title="Results history (newest first)", title_justify="left")
    table.add_column("Date", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Evaluator", no_wrap=True)
    table.add_column("Main score", justify="right")
    table.add_column("Examples", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Device")
    for result in results:
        table.add_row(
            result.created_at.strftime("%Y-%m-%d %H:%M"),
            result.model_id,
            result.evaluator,
            f"{result.main_score:.3f} {result.main_metric}",
            str(result.examples),
            f"{result.duration_seconds:.0f} s",
            result.device,
        )
    console.print(table)


@app.command()
def compare(
    evaluator: Annotated[str, typer.Argument(help="Evaluator key, e.g. sts-benchmark.")],
) -> None:
    """Compare the latest result of every model on one evaluator, best first."""
    show_warnings_in_terminal()
    with friendly_errors():
        chosen = get_evaluator(evaluator)
    rows = compare_results(chosen.key)
    if not rows:
        console.print(f"No results for '{chosen.key}' yet.")
        return

    better = "higher is better" if chosen.higher_is_better else "lower is better"
    table = Table(
        title=f"{chosen.name}: {chosen.main_metric} ({better})",
        title_justify="left",
        caption=chosen.how_to_read,
    )
    table.add_column("Rank", justify="right")
    table.add_column("Model", overflow="fold")
    table.add_column(chosen.main_metric, justify="right")
    table.add_column("Examples", justify="right")
    table.add_column("Date", no_wrap=True)
    for rank, result in enumerate(rows, start=1):
        table.add_row(
            str(rank),
            result.model_id,
            f"{result.main_score:.3f}",
            str(result.examples),
            result.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)

# Feature Specification: Model Testbed

**Feature Branch**: `001-model-testbed`

**Created**: 2026-09-21

**Status**: Draft

## Clarifications

### Session 2026-09-21

- Q: Which user interface should the testbed have? → A: Both, terminal first: a command-line tool
  now, with a simple local web page added later on top of the same core.

**Input**: User description: "This will be a testbed software for different models (generative,
embedding, classification). The models should be selectable from a curated list, that you can
generate from Huggingface. They should only be downloaded when used and deleted soon after (saving
disk space). The software should use a best practice evaluator to assess the models's performance.
Ideally there should be several evaluators to chose from, analogous to the models. Keep it simple at
first. Later on there will be features like training etc. but everything has to be kept on a small
scale and runnable on a macmini (mps, no cuda)."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Evaluate one model from the catalog (Priority: P1)

The user opens the testbed, picks a model type (generative, embedding, or classification), picks a
model from the curated catalog for that type, and starts an evaluation using the recommended
evaluator for that type. The testbed downloads the model, runs the evaluation on a small reference
sample, shows the score with a plain-language explanation of what it means, and then deletes the
downloaded model files.

**Why this priority**: This is the core loop of the testbed: choose → run → see result → clean
up. Without it nothing else has value, and on its own it already lets the user try models safely
on a small machine.

**Independent Test**: Start with an empty download area, evaluate the smallest catalog model of
each type with its default evaluator, confirm a score is shown and the download area is empty
again afterwards.

**Acceptance Scenarios**:

1. **Given** the catalog is available and no models are downloaded, **When** the user browses the
   catalog, **Then** the models are listed grouped by type and nothing is downloaded.
2. **Given** the user has selected a catalog model, **When** they start an evaluation, **Then** the
   model is downloaded (with visible progress), evaluated with the type's default evaluator, and
   the result is displayed.
3. **Given** an evaluation has finished, **When** the result is displayed, **Then** the downloaded
   model files have already been removed from disk.
4. **Given** an evaluation is running, **When** the user cancels it or it fails, **Then** a clear
   message is shown and all downloaded files for that run are removed.

---

### User Story 2 - Choose among several evaluators (Priority: P2)

For a selected model, the user sees the list of evaluators that fit that model's type, each with a
short description of what it measures. The user can select one or more of them for a single run.
The model is downloaded only once per run, all selected evaluators are applied, and each result is
shown separately.

**Why this priority**: Different evaluators reveal different strengths; offering a choice (as the
user asked) makes the testbed useful for real comparison. It builds directly on Story 1.

**Independent Test**: For one model of each type, list the evaluators offered, run two of them in a
single run, and confirm two separate results appear while the model was downloaded only once.

**Acceptance Scenarios**:

1. **Given** a model of a given type is selected, **When** the user views evaluators, **Then** only
   evaluators compatible with that model are offered and the recommended one is preselected.
2. **Given** the user selects two evaluators, **When** the run completes, **Then** two results are
   shown, one per evaluator, and the model was downloaded a single time.

---

### User Story 3 - Generate the curated catalog from Hugging Face (Priority: P2)

The user regenerates the model catalog from Hugging Face. The testbed searches for suitable models
per type using simple filters (small size, publicly accessible without sign-in, popular, no custom
code required) and writes a short, human-readable catalog the user can review and edit by hand.

**Why this priority**: The user explicitly wants a curated list that can be generated from Hugging
Face. An initial catalog is shipped with the testbed so Story 1 works without this step, which is
why this story is second priority rather than first.

**Independent Test**: Run catalog generation, open the resulting catalog, and verify that every
entry is within the size limit, belongs to one of the three types, and can be evaluated.

**Acceptance Scenarios**:

1. **Given** internet access, **When** the user regenerates the catalog, **Then** a new catalog is
   produced containing up to the configured number of models per type, each meeting the filters.
2. **Given** a generated catalog, **When** the user edits it by hand (removes or adds an entry),
   **Then** the testbed shows exactly the edited list the next time it starts.
3. **Given** catalog generation fails (e.g., no internet), **When** it stops, **Then** the previous
   catalog is left unchanged and a clear message is shown.

---

### User Story 4 - Review and compare past results (Priority: P3)

The user opens a history of past evaluation results, filters it by model type or evaluator, and
compares models side by side on the same evaluator.

**Why this priority**: Comparison is the natural next step once several models have been tested,
but single results already provide value without it.

**Independent Test**: Evaluate two models of the same type with the same evaluator, open the
history, and confirm both results appear side by side with their scores.

**Acceptance Scenarios**:

1. **Given** several completed runs, **When** the user opens the history, **Then** each result
   shows model, evaluator, score(s), date, duration, and the hardware used.
2. **Given** results for several models on the same evaluator, **When** the user compares them,
   **Then** they are shown in one table sorted by score.

---

### Edge Cases

- **Not enough disk space**: before downloading, the testbed compares the model's size with the free
  space; if there is not enough room, it refuses to start and says how much space is needed.
- **No internet / Hugging Face unavailable**: the run stops with a clear message; nothing is left
  behind on disk.
- **Interrupted run** (cancel, crash, power loss, closed terminal): any leftover downloaded files
  are removed at the end of the run if possible, otherwise automatically at the next start.
- **Model too large for memory**: the run fails gracefully with an explanation and a hint to pick a
  smaller model; files are cleaned up.
- **GPU acceleration unavailable or unsupported for a model**: the testbed falls back to the CPU and
  tells the user the run may be slower.
- **Model removed or renamed on Hugging Face**: the run fails with a message suggesting to
  regenerate the catalog.
- **Model requires sign-in, license acceptance, or custom code**: such models are excluded from the
  catalog; if one is added by hand, the testbed refuses it with an explanation.
- **Incompatible model/evaluator pair** (e.g., a sentiment classifier and a topic-classification
  evaluator): the combination is not offered.

## Requirements *(mandatory)*

### Functional Requirements

#### Catalog

- **FR-001**: System MUST provide a curated catalog of models grouped into three types:
  generative, embedding, and classification.
- **FR-002**: Each catalog entry MUST show the model name, type, a one-line description, approximate
  download size, and license.
- **FR-003**: System MUST ship with an initial catalog so the testbed works without first
  generating one.
- **FR-004**: Users MUST be able to regenerate the catalog from Hugging Face using filters for type,
  maximum download size (default 2 GB), public access without sign-in, no custom code required,
  and popularity, keeping a configurable number of models per type (default 5).
- **FR-005**: The catalog MUST be stored in a human-readable format that users can edit by hand.
- **FR-006**: Users MUST only be able to select models listed in the catalog.

#### Download and cleanup

- **FR-007**: System MUST NOT download any model files until an evaluation run for that model
  starts; browsing the catalog MUST NOT trigger downloads.
- **FR-008**: System MUST check free disk space before downloading and refuse to start a run if the
  model and evaluation data would not fit with a safety margin.
- **FR-009**: System MUST show download progress to the user.
- **FR-010**: System MUST delete all downloaded model and evaluation data files when a run ends,
  whether it succeeded, failed, or was cancelled.
- **FR-011**: System MUST detect and remove leftover downloaded files from interrupted runs when it
  starts.
- **FR-012**: System MUST keep all downloads in one dedicated location owned by the testbed and MUST
  NOT delete files outside that location.

#### Evaluation

- **FR-013**: System MUST provide at least one evaluator per model type, and at least two per type
  to complete User Story 2.
- **FR-014**: Each evaluator MUST use an established, widely recognized metric and a publicly
  available reference dataset, and MUST describe in plain language what it measures and how to
  read the score (e.g., "higher is better").
- **FR-015**: System MUST offer only evaluators compatible with the selected model and MUST
  preselect a recommended default evaluator.
- **FR-016**: Users MUST be able to select one or more compatible evaluators for a single run; the
  model MUST be downloaded only once per run.
- **FR-017**: Evaluators MUST run on a small, fixed sample of the reference data (default 200
  examples, configurable) so that repeated runs are comparable and finish quickly.
- **FR-018**: Each result MUST include model, evaluator, score(s), number of examples, run duration,
  date, and the hardware used (GPU or CPU).

#### Results history

- **FR-019**: System MUST save every completed result to a local history that remains after the
  model files are deleted.
- **FR-020**: Users MUST be able to list past results filtered by model type and evaluator and
  compare results of the same evaluator side by side.

#### Platform and usability

- **FR-021**: System MUST run on an Apple silicon Mac mini, use Apple GPU acceleration (MPS) when
  available, fall back to CPU automatically, and MUST NOT require NVIDIA/CUDA hardware.
- **FR-022**: Users MUST be able to perform every action in this specification (browse catalog,
  regenerate catalog, run evaluations, view and compare history) through a command-line tool in
  the terminal. The core features MUST be usable independently of the terminal interface so that a
  simple local web page can be added later without changing how evaluations work.
- **FR-023**: All error messages MUST state what went wrong and what the user can do next, in plain
  language.

### Key Entities

- **Model Type**: One of generative, embedding, classification. Determines which models and
  evaluators belong together.
- **Catalog Entry**: A model the user may select: Hugging Face identifier, type, description,
  approximate size, license, and (for classification) the task/label set it predicts.
- **Evaluator**: A named, repeatable assessment for one model type: the metric used, the reference
  dataset, sample size, description, how to read the score, and which models it is compatible with.
- **Evaluation Run**: One execution for one model with one or more evaluators; has a status
  (running, succeeded, failed, cancelled) and owns the temporary downloaded files.
- **Evaluation Result**: The outcome of one evaluator in a run: score(s), example count, duration,
  hardware, date; kept in the history.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user can go from starting the testbed to seeing their first evaluation
  result in under 10 minutes using the smallest catalog model of any type on a typical home
  internet connection.
- **SC-002**: After 100% of runs (succeeded, failed, or cancelled), the testbed's download location
  is empty within 1 minute of the run ending.
- **SC-003**: Extra disk space used during a run never exceeds the size of the model being evaluated
  plus its evaluation data plus 10%.
- **SC-004**: Every model in the shipped catalog can be evaluated with its default evaluator on a
  Mac mini with 16 GB memory without running out of memory, in under 15 minutes excluding download
  time.
- **SC-005**: Running the same model with the same evaluator twice produces scores that differ by no
  more than 1%.
- **SC-006**: Browsing the catalog causes 0 bytes of model downloads.
- **SC-007**: Regenerating the catalog finishes in under 2 minutes, and 100% of the generated
  entries satisfy the configured filters.
- **SC-008**: A user unfamiliar with machine learning can correctly say whether a displayed score is
  good or bad, using only the explanation shown next to it.

## Assumptions

- The testbed is used by a single person on their own Apple silicon Mac mini with at least 16 GB of
  memory and a recent macOS version.
- Internet access is available while downloading models, evaluation data, or regenerating the
  catalog; only public Hugging Face models that need no account or token are used.
- "Deleted soon after" means deleted automatically at the end of the run that used the model.
  Evaluating the same model again later downloads it again; this trade-off favors disk space over
  speed.
- Evaluation data samples are small (a few MB) and follow the same download-on-use and
  delete-after-run policy as models.
- Only text models are in scope for the first version (text generation, text embeddings, text
  classification); image and audio models are out of scope.
- Scores are quick, indicative measurements on small samples; they are meant for comparing models
  within the testbed, not for matching official leaderboard numbers.
- Out of scope for this version: training or fine-tuning (planned later), a browser-based
  interface (planned later, see FR-022), models outside the catalog, evaluating several models in
  parallel, cloud or remote execution, and multi-user use.
- The default size limit (2 GB download per model) and sample size (200 examples) are chosen to keep
  runs small and fast on a Mac mini and can be changed in configuration.

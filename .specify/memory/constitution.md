<!--
Sync Impact Report
- Version change: (template, unversioned) → 1.0.0
- Modified principles (placeholders → named):
  - [PRINCIPLE_1_NAME] → I. Proven, Best-Practice Tools Only
  - [PRINCIPLE_2_NAME] → II. Simplicity First
  - [PRINCIPLE_3_NAME] → III. Beginner-Friendly Code
  - [PRINCIPLE_4_NAME] → IV. Explain Through Comments and Docs
  - [PRINCIPLE_5_NAME] → removed (user supplied four principles)
- Added sections: Technology & Dependency Constraints; Development Workflow & Quality Gates
- Removed sections: none (template fifth principle slot not used)
- Follow-up TODOs: none
-->

# ml1 Constitution

## Core Principles

### I. Proven, Best-Practice Tools Only

- Every dependency MUST be a widely adopted, actively maintained, well-documented library
  (e.g., the de-facto standard for its task in the language ecosystem).
- Experimental, alpha, abandoned, or niche packages MUST NOT be introduced when an established
  alternative exists.
- Techniques and patterns MUST follow the official documentation and established community
  conventions of the chosen tools; clever or unconventional approaches are not allowed without a
  written justification.
- Prefer the standard library when it solves the problem adequately.

**Rationale**: Proven tools have good documentation, many examples, and predictable behavior,
which reduces bugs and makes the project easier to learn and maintain.

### II. Simplicity First

- The simplest solution that correctly meets the requirement MUST be chosen.
- Abstractions, layers, configuration options, or generalizations MUST NOT be added until a
  concrete, current need exists (YAGNI).
- Any added complexity MUST be justified in the plan or pull request, stating why a simpler
  alternative was insufficient.

**Rationale**: Simple code is easier to understand, test, debug, and change.

### III. Beginner-Friendly Code

- Code MUST be readable by someone with basic knowledge of the language: descriptive names,
  short focused functions, and straightforward control flow.
- Obscure language features, dense one-liners, and heavy metaprogramming MUST be avoided when a
  plainer form exists.
- Project setup and run instructions MUST be documented step by step so a newcomer can get the
  project working.

**Rationale**: The project is intended to be approachable and educational; clarity is valued
over brevity or cleverness.

### IV. Explain Through Comments and Docs

- Every module, class, and non-trivial function MUST have a docstring/header comment stating its
  purpose, inputs, and outputs.
- Comments MUST explain _why_ something is done and clarify any non-obvious step (e.g., a
  formula, a data transformation, or a library call whose effect is not self-evident).
- Comments MUST be kept accurate; outdated comments are treated as defects.
- Trivial comments that merely restate the code SHOULD be avoided.

**Rationale**: Good explanations let beginners learn from the code and let anyone understand
decisions later.

## Technology & Dependency Constraints

- New dependencies MUST be listed in the project's dependency file with pinned or bounded
  versions and a one-line note (in the plan or PR) on why they are needed.
- The number of dependencies SHOULD be kept small; each one MUST satisfy Principle I.
- Choose the most common, well-documented option when several proven libraries fit.

## Development Workflow & Quality Gates

- Every plan (`/speckit-plan`) MUST include a Constitution Check confirming compliance with all
  four principles.
- Code review MUST verify: approved dependencies only, no unjustified complexity, readable
  naming and structure, and adequate explanatory comments.
- Behavior MUST be verified with simple, readable tests or reproducible examples where practical.

## Governance

- This constitution supersedes other project practices; conflicts are resolved in its favor.
- Amendments MUST be made via `/speckit-constitution`, documented in the Sync Impact Report, and
  reflected in dependent plans where relevant.
- Versioning follows semantic versioning: MAJOR for removing or redefining principles, MINOR for
  new principles/sections or materially expanded guidance, PATCH for clarifications and wording.
- Compliance is reviewed at every plan and code review; deviations MUST be justified in writing.

**Version**: 1.0.0 | **Ratified**: 2026-09-21 | **Last Amended**: 2026-09-21

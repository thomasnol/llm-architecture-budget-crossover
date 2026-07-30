# Canonical HMDA Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver one unversioned, failure-safe, reproducible HMDA
architecture–budget experiment harness implementing the approved controlled-model
primary study and separate routing ablation.

**Architecture:** Rename the implemented HMDA pipeline into focused canonical
modules and delete the superseded pipelines. Harden the gateway boundary and
runner before changing experimental factors, then enforce completeness in
analysis and validation. Keep all result generation driven by frozen YAML
configurations and manifests.

**Tech Stack:** Python 3.12, asyncio, httpx, Pydantic, Typer, pandas,
NumPy/SciPy, seaborn/matplotlib, pytest, Ruff, LaTeX.

## Global Constraints

- Work only on `codex/hmda-canonical`.
- Remove version labels and superseded implementations from the branch tree.
- Preserve the two-credential model eligibility rules and per-key CUBIC limiter.
- Never store credentials or unsanitized authorization data in artifacts.
- Use gateway-reported prompt, completion, and total tokens as authoritative.
- Treat the failed pilot as diagnostic-only and never merge it into results.
- Use test-first red/green cycles for behavioral changes.

---

### Task 1: Canonical module and command surface

**Files:**
- Rename the prior version-prefixed source modules to focused unversioned modules
- Modify: `src/budget_crossover/cli.py`
- Rename the prior version-prefixed pilot and main configurations
- Rename the prior version-prefixed smoke-pipeline script
- Delete: superseded source, configuration, test, and experiment files
- Test: canonicalized files under `tests/`

**Interfaces:**
- Produces: `ExperimentConfig`, `Case`, `Generation`,
  `run_system`, `execute_generation`, and unversioned CLI commands.

- [x] Write CLI and import tests that invoke only the canonical command surface.
- [x] Run the focused tests and confirm they fail against versioned commands.
- [x] Rename canonical modules/configurations and update imports.
- [x] Remove superseded pipelines and their tests/data products.
- [x] Run focused tests, then scan tracked text for repository version labels.

### Task 2: Gateway diagnostics and preflight

**Files:**
- Modify: `src/budget_crossover/gateway.py`
- Create: `src/budget_crossover/preflight.py`
- Test: `tests/test_gateway.py`, `tests/test_preflight.py`

**Interfaces:**
- Produces: `GatewayRequestError`, persisted `FailureAttempt` records, and
  `run_preflight(config) -> PreflightReport`.

- [x] Add failing tests for sanitized 400 response capture, attempted model,
  stage propagation, slot, request ID, and retry classification.
- [x] Confirm the tests fail for the current `HTTPStatusError`.
- [x] Implement structured gateway errors without logging secrets.
- [x] Add failing tests proving preflight executes each model on every eligible
  credential and validates all three usage counters.
- [x] Implement preflight and persist its non-secret report.
- [x] Run gateway and preflight tests.

### Task 3: Bounded execution and circuit breaking

**Files:**
- Modify: `src/budget_crossover/runner.py`
- Modify: `src/budget_crossover/models.py`
- Create: `src/budget_crossover/status.py`
- Test: `tests/test_runner.py`, `tests/test_status.py`

**Interfaces:**
- Produces: bounded `execute_generation`, structured attempt ledger,
  permanent-error circuit breaker, and `summarize_run`.

- [x] Add failing tests for a maximum bounded number of active cells.
- [x] Add failing tests that three equivalent permanent errors stop unscheduled
  work while transient failures remain resumable.
- [x] Implement the worker queue and circuit breaker.
- [x] Add failing tests for status grouping by error/system/budget/model/stage.
- [x] Implement status reporting and canonical CLI command.
- [x] Run runner and status tests.

### Task 4: Controlled architecture and routing factors

**Files:**
- Modify: `src/budget_crossover/config.py`
- Modify: `src/budget_crossover/systems.py`
- Modify: `configs/pilot.yaml`, `configs/main.yaml`
- Create: `configs/routing_pilot.yaml`, `configs/calibration.yaml`
- Test: `tests/test_config.py`, `tests/test_systems.py`

**Interfaces:**
- Produces: `study_kind` (`architecture` or `routing`), fixed primary model
  use for the architecture study, and explicit weak-to-strong systems for the
  routing study.

- [x] Add failing tests that architecture systems cannot silently use the
  supervisor model.
- [x] Add failing tests for routing-system model assignments.
- [x] Implement study-kind validation and explicit model assignment.
- [x] Set deterministic primary temperatures and repetition count.
- [x] Run configuration and system tests.

### Task 5: Calibration and feasibility

**Files:**
- Create: `src/budget_crossover/calibration.py`
- Modify: `src/budget_crossover/systems.py`
- Modify: `src/budget_crossover/cli.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces: `recommend_budgets(trajectories, architecture_minima)`,
  `calibration.json`, and a four-budget recommendation.

- [x] Add failing tests using hand-calculated trajectory costs and architecture
  minima.
- [x] Implement pooled quantile recommendations constrained by the largest
  fixed-architecture minimum.
- [x] Add a calibration CLI that never mutates frozen pilot/main configs.
- [x] Run calibration tests.

### Task 6: Complete-grid analysis and estimands

**Files:**
- Modify: `src/budget_crossover/analysis.py`
- Modify: `src/budget_crossover/validation.py`
- Test: `tests/test_analysis.py`, `tests/test_validation.py`

**Interfaces:**
- Produces: strict `analyze_run(..., diagnostic=False)`, coverage/ITT tables,
  counterfactual flip tables, cluster bootstrap comparisons, and diagnostic
  watermarks.

- [x] Add failing tests that ordinary analysis rejects incomplete grids.
- [x] Add failing tests that diagnostic analysis labels outputs incomplete.
- [x] Add hand-derived tests for ITT accuracy and counterfactual flip rate.
- [x] Implement coverage-first analysis and revised estimands.
- [x] Add tests for clustered paired resampling and crossover outputs.
- [x] Implement revised figures and tables.
- [x] Run analysis and validation tests.

### Task 7: Manifest, environment isolation, and documentation

**Files:**
- Modify: `src/budget_crossover/manifest.py`
- Modify: `tests/conftest.py`
- Modify: `.env.example`, `README.md`, `experiments/RUNBOOK.md`
- Rename: preregistration to canonical filename
- Modify: paper sources and build script
- Test: `tests/test_manifest.py`, environment-sensitive tests

**Interfaces:**
- Produces: manifest with protocol/deployment metadata and tests isolated from
  `.env`.

- [x] Add failing tests that gateway protocol/model deployment changes alter
  the immutable manifest while secrets never appear.
- [x] Implement the expanded manifest.
- [x] Add an autouse test boundary preventing repository `.env` loading.
- [x] Rewrite runbook and preregistration around staged execution.
- [x] Update the approximately five-page LaTeX protocol without asserting
  unavailable main results.
- [x] Run documentation command examples and paper build.

### Task 8: End-to-end verification

**Files:**
- Modify as required by discovered verification defects only

**Interfaces:**
- Consumes all preceding tasks and produces a reproducible release candidate.

- [x] Run the full unit/integration suite.
- [x] Run Ruff format/lint checks.
- [x] Run the offline smoke experiment through prepare, execute, validate,
  status, and analyze.
- [x] Confirm ordinary analysis rejects a deliberately incomplete fixture.
- [x] Build and inspect the LaTeX PDF page count.
- [x] Scan the tracked tree for superseded version labels and stale commands.
- [x] Audit every approved design requirement against current files and
  command output.

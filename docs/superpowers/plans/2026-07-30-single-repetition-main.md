# Single-Repetition Main Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Halve the canonical main experiment workload while preserving its 384-case coverage, five architectures, and four token budgets.

**Architecture:** Set the frozen main configuration to one repetition. Keep repetition-aware execution and analysis code intact so cell identity, resumability, and future replicated studies continue to work; make every user-facing main-grid count 7,680.

**Tech Stack:** YAML experiment configuration, Python/pytest validation, Markdown documentation, and the LaTeX paper builder.

## Global Constraints

- Work on the dedicated `codex/hmda-main-single-repetition` branch.
- Preserve 384 disjoint main cases, five architectures, and budgets 4,096, 6,144, 8,192, and 12,288.
- Set the canonical main experiment to exactly one repetition and 7,680 cells.
- Do not change pilot, calibration, routing, runner, or analysis behavior.
- Remove obsolete larger-grid and replicated-main claims from the branch tree.

---

### Task 1: Freeze the smaller main grid

**Files:**
- Modify: `configs/main.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `ExperimentConfig.repetitions: int`.
- Produces: a canonical main configuration with `repetitions == 1` and an expected grid of `384 * 5 * 4 * 1 == 7_680` cells.

- [x] **Step 1: Add a configuration regression test**

Add a test that loads `configs/main.yaml`, asserts one repetition, and computes an expected 7,680-cell grid from 384 cases, the configured systems, budgets, and repetitions.

- [x] **Step 2: Run the regression test and verify it fails**

Run: `uv run pytest tests/test_config.py -q`

Expected: FAIL because `configs/main.yaml` does not yet specify the canonical single repetition.

- [x] **Step 3: Apply the minimal configuration change**

Set `repetitions: 1` in `configs/main.yaml`.

- [x] **Step 4: Run the configuration test and verify it passes**

Run: `uv run pytest tests/test_config.py -q`

Expected: PASS.

### Task 2: Align documentation and the paper scaffold

**Files:**
- Modify: `README.md`
- Modify: `experiments/REMAINING_WORK.md`
- Modify: `paper/build_paper.py`

**Interfaces:**
- Consumes: the canonical 7,680-cell main grid.
- Produces: instructions and paper status text that state one repetition and 7,680 cells.

- [x] **Step 1: Replace obsolete grid descriptions**

Describe the main experiment as 384 cases × five architectures × four budgets × one repetition = 7,680 cells.

- [x] **Step 2: Scan for obsolete claims**

Run a repository text scan for obsolete larger-grid counts and replicated-main wording.

Expected: no claims that the canonical main experiment uses the obsolete grid.

### Task 3: Verify the complete revision

**Files:**
- Verify: `configs/main.yaml`
- Verify: `tests/`
- Verify: `README.md`
- Verify: `experiments/REMAINING_WORK.md`
- Verify: `paper/build_paper.py`

**Interfaces:**
- Produces: fresh evidence that configuration loading, grid accounting, documentation, and the repository test suite agree.

- [x] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_config.py tests/test_runner.py tests/test_validation.py -q`

Expected: all tests pass.

- [x] **Step 2: Run the full suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [x] **Step 3: Run consistency checks**

Run: `git diff --check`

Run a repository text scan for obsolete larger-grid counts and replicated-main wording.

Expected: no whitespace errors and no obsolete canonical-grid claims.

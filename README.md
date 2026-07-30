# Token-budgeted agent orchestration for mortgage adjudication

This repository evaluates how orchestration architecture changes the reliability
and efficiency of LLM mortgage adjudication when the entire trajectory shares a
case-level token budget.

Official 2024 Home Mortgage Disclosure Act data from CFPB/FFIEC seed realistic
applications. Historical lender action is observational and is never the correct
answer. A transparent research-only policy produces reproducible approve,
conditional-review, deny, and manual-review labels from predecision financial
fields. Each application has a counterfactual twin that changes exactly one
monitoring-only protected attribute.

This is an evaluation sandbox, not lending advice, a production credit policy,
or evidence that a historical lender decision was correct.

## Studies

The primary architecture study holds the model fixed at GPT-5.4-mini:

- full-context monolith;
- plan-and-retrieve;
- specialist committee;
- fixed underwriter-plus-compliance guardrail;
- adaptive guarded routing.

The separate routing study compares an always-primary monolith, an
always-supervisor monolith, and selective escalation from GPT-5.4-mini to Claude
Sonnet 4.6. This avoids attributing model-quality differences to architecture.

The staged design is:

1. gateway preflight on every model and eligible credential;
2. high-budget calibration and four-budget recommendation;
3. a 48-case pilot;
4. a disjoint 384-case main architecture study with two repetitions.

The main grid contains 15,360 cells: 384 cases × five architectures × four
budgets × two repetitions.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
# Fill in the endpoint and two OAuth credential pairs.

uv run python scripts/download_hmda.py
uv run pytest -q
uv run ruff check src tests
uv run python scripts/smoke_pipeline.py
```

The downloader verifies the official CSV schema and the frozen SHA-256 digest.
Downloaded data, processed cases, gateway transcripts, and generated results are
ignored by Git.

## Staged execution

```bash
# Prepare cases without making model calls.
uv run budget-crossover prepare --config configs/pilot.yaml
uv run budget-crossover prepare --config configs/main.yaml

# Verify each actual payload/model/credential path.
uv run budget-crossover preflight --config configs/calibration.yaml

# Run high-budget calibration, then inspect its recommendation.
uv run budget-crossover pilot --config configs/calibration.yaml
uv run budget-crossover calibrate --config configs/calibration.yaml

# After accepting the frozen budgets, preflight and run the pilot.
uv run budget-crossover preflight --config configs/pilot.yaml
uv run budget-crossover pilot --config configs/pilot.yaml
uv run budget-crossover status --config configs/pilot.yaml
uv run budget-crossover validate \
  --config configs/pilot.yaml \
  --no-require-pilot-gate
uv run budget-crossover analyze --config configs/pilot.yaml

# The main run is blocked unless the complete pilot passes.
uv run budget-crossover preflight --config configs/main.yaml
uv run budget-crossover run --config configs/main.yaml
uv run budget-crossover status --config configs/main.yaml
uv run budget-crossover validate --config configs/main.yaml
uv run budget-crossover analyze --config configs/main.yaml

# Separate model-routing ablation.
uv run budget-crossover preflight --config configs/routing_pilot.yaml
uv run budget-crossover pilot --config configs/routing_pilot.yaml
```

Normal analysis rejects missing or duplicate grid cells. `analyze --diagnostic`
is available for operational investigation; all figures it creates are
watermarked `INCOMPLETE DIAGNOSTIC` and are not paper evidence.

## Reliability and accounting

Gateway-reported `prompt_tokens`, `completion_tokens`, and `total_tokens` are
authoritative. A manifest freezes cases, prompts, code, configuration, resolved
deployments, non-secret gateway protocol settings, the dependency lock, and Git
state.

Failures are checkpointed separately from scored outcomes. The runner:

- retries only transient transport, rate-limit, and selected server errors;
- records sanitized status, response detail, model, stage, credential slot, and
  request ID;
- stops after three equivalent permanent failures;
- uses a bounded worker pool;
- maintains one CUBIC concurrency window per credential.

Raise `LLM_GATEWAY_CONCURRENCY_PER_KEY` to increase the ceiling. Credential pair
1 supports GPT and Claude deployments; pair 2 supports GPT deployments only.

## Paper

```bash
uv run python paper/build_paper.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Until a complete validated main run exists, the approximately five-page LaTeX
artifact remains a registered protocol and makes no empirical model-performance
claims. The operator guide is [experiments/RUNBOOK.md](experiments/RUNBOOK.md);
remaining paper work is tracked in
[experiments/REMAINING_WORK.md](experiments/REMAINING_WORK.md).

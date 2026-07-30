# Budgeted agent orchestration for mortgage adjudication

This repository evaluates whether an LLM decision workflow should remain
monolithic, retrieve evidence, convene specialists, add an independent compliance
guardrail, or route adaptively when the entire workflow is constrained by a
case-level **total-token budget**.

Version 3 is a clean redesign around option C: official 2024 Home Mortgage
Disclosure Act (HMDA) data from CFPB/FFIEC. HMDA records seed realistic mortgage
applications, but reported lender action is **not** treated as the correct
underwriting decision. Historical action is observational and institution
specific. Instead, a transparent research-only policy oracle creates reproducible
approve, conditional-review, deny, and manual-review labels from predecision
financial fields.

Each sampled application produces a protected-attribute counterfactual twin.
Financial evidence and the policy label remain identical; one monitoring-only
attribute changes. This permits a controlled compliance-invariance test without
claiming to estimate discrimination in the mortgage market.

## Why HMDA, not the other candidates?

- Home Credit is a rich relational default-risk dataset, but its target is
  post-credit default—not approval correctness.
- Lending Club contains funded loans and post-origination performance, creating
  selective labels for an approval study.
- HMDA contains real mortgage applications, actual action taken, financial and
  property fields, and compliance-monitoring demographics. Its weakness is also
  explicit: actual action is a historical outcome, not normative ground truth.

The design decision and limitations are documented in
[`experiments/V3_PREREGISTRATION.md`](experiments/V3_PREREGISTRATION.md).
Version 1 and Version 2 remain in the repository as an audit trail, but they are
not the canonical study.

## Canonical study

- Source: 2024 HMDA Data Browser export served on 2026-07-29, filtered through
  the official API to DC, ND, VT, and WY and to originated/denied records. The
  exact bytes are frozen by SHA-256.
- Scope: conventional, first-lien, closed-end, consumer home-purchase
  applications for owner-occupied, site-built, one-to-four-unit properties.
- Pilot: 24 source applications × 2 counterfactual variants = 48 cases.
- Main: 96 disjoint source applications × 2 variants = 192 cases.
- Outcomes: policy decision accuracy, full decision-plus-reason accuracy,
  counterfactual decision consistency, budget compliance, calls, realized total
  tokens, latency, and optional approved-price cost.
- Systems: worker monolith, strong-model monolith, plan-and-retrieve, specialist
  committee, compliance guardrail, and adaptive guarded routing.
- Nominal case-level total-token budgets: 2,048, 4,096, and 8,192.

The policy sandbox is an evaluation instrument, not lending advice, a credit
policy, or evidence that a historical lender decision was correct or incorrect.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
# Fill in the internal gateway endpoint and credential pairs.

uv run python scripts/download_hmda.py
uv run budget-crossover gateway-check
uv run pytest -q
uv run ruff check src tests
uv run python scripts/smoke_v3_pipeline.py
```

The downloader uses the official CFPB/FFIEC Data Browser, validates the expected
CSV schema, and rejects any file whose SHA-256 differs from the preregistered
digest.

## Prepare and run

```bash
# Data-only preparation and leakage/counterfactual validation.
uv run budget-crossover prepare-v3 --config configs/v3_pilot.yaml
uv run budget-crossover prepare-v3 --config configs/v3_main.yaml
uv run budget-crossover validate-v3 \
  --config configs/v3_main.yaml \
  --no-require-generations \
  --no-require-pilot-gate

# Pilot, analysis, and manipulation checks.
uv run budget-crossover pilot-v3 --config configs/v3_pilot.yaml
uv run budget-crossover analyze-v3 --config configs/v3_pilot.yaml
uv run budget-crossover validate-v3 \
  --config configs/v3_pilot.yaml \
  --no-require-pilot-gate

# The main run is blocked unless the pilot checks pass.
uv run budget-crossover run-v3 --config configs/v3_main.yaml
uv run budget-crossover analyze-v3 --config configs/v3_main.yaml
uv run budget-crossover validate-v3 --config configs/v3_main.yaml
```

Runs are checkpointed and resumable. A manifest freezes the configuration, cases,
HMDA checksum, prompt version, code hash, both model IDs, seed, and Git state. A
resume is rejected after an immutable input changes. Canonical scored outputs
and retryable transport errors are stored separately, so a successful retry
cannot duplicate an experimental cell.

GPT-5.4-mini worker calls can use both OAuth pairs. Claude Sonnet 4.6 oversight
calls and the strong-model control use pair 1. Each pair has its own
CUBIC-inspired concurrency window; raise
`LLM_GATEWAY_CONCURRENCY_PER_KEY` to raise the ceiling without changing code.

## Build the white paper

```bash
uv run python paper/build_paper.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Without a validated main run, the builder produces a clearly labeled
preregistered LaTeX protocol and makes no empirical model-performance claims.
The current protocol compiles to five pages and includes vector study and
architecture diagrams. After a complete validated main run, the result gate
incorporates generated tables and vector figures.

The operator guide is in [`experiments/RUNBOOK.md`](experiments/RUNBOOK.md).
The post-run paper checklist is in
[`experiments/REMAINING_WORK.md`](experiments/REMAINING_WORK.md).

## Data governance

Downloaded HMDA files, processed case packets, run transcripts, and analysis
outputs are ignored by Git. HMDA public data are privacy-modified, but the
workflow still avoids re-identification, suppresses public row identifiers in the
paper, and never sends historical action, denial reasons, pricing, or other
post-decision outcome fields to models.

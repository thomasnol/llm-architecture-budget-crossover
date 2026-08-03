# When Extra Inference Structure Pays

This repository implements a preregistered test of token-budget crossovers in
financial reasoning. The confirmatory question is conditional: is verified
search worse than a one-call monolith at low budget, but better at high budget?

Scientific status: no empirical conclusion is available. No complete validated
gateway run exists for the rebuilt protocol. The hypothesis is currently
neither supported nor cleanly falsified. FinanceComplexQA is exploratory only.
Historical repository runs and the deterministic offline fixture are not model
evidence.

## Canonical design

The primary data are pinned local FinQA and TAT-QA snapshots. Preparation is
offline and deterministic after acquisition. It enforces one selected question
per source document/context, executable gold derivations, operand provenance,
hidden labels, outcome-independent strata, exact quotas, and document-disjoint
splits.

The systems are exactly:

- `monolith` - one direct answer call;
- `verified_search` - mandatory planning, deterministic retrieval, label-blind
  checking, and at most one allowed repair;
- `unverified_search` - an exploratory no-checker ablation with stable
  plurality selection.

All systems use exactly `gpt-5.4-mini`. There is no fallback model. The hard
resource intervention is authoritative prompt plus completion usage over the
whole cell. Low, middle, and high tiers cap tokens at 4,096, 12,288, and 32,768
and also freeze retrieval, planned-query, candidate, and repair opportunities.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
uv run pytest -q
uv run ruff check src tests
```

Place separately acquired, pinned snapshots at the configured paths and set
their SHA-256 environment values. Preparation never downloads data.

The gateway must provide both completion and exact chat-tokenizer endpoints.
Preflight requires:

- requested and response-resolved model exactly `gpt-5.4-mini`;
- strict JSON with no extra keys;
- authoritative prompt and completion usage;
- a frozen tokenizer ID and SHA-256;
- exact equality between pre-call tokenizer count and gateway prompt usage.

If exact tokenizer support is unavailable, preflight fails closed. Character
estimates and model substitution are prohibited.

## Linear workflow

Run these commands in order with the same configuration:

```bash
uv run budget-crossover prepare --config configs/main.yaml
uv run budget-crossover diagnose-finance-complex --config configs/main.yaml
uv run budget-crossover preflight --config configs/main.yaml
uv run budget-crossover develop --config configs/main.yaml
uv run budget-crossover pilot --config configs/main.yaml
uv run budget-crossover gate --config configs/main.yaml
uv run budget-crossover run --config configs/main.yaml
uv run budget-crossover validate --config configs/main.yaml
uv run budget-crossover analyze --config configs/main.yaml
uv run budget-crossover build-paper --config configs/main.yaml
```

The pilot gate is non-overridable. Every downstream command verifies upstream
hashes. The immutable run manifest binds resolved configuration, exact case and
document inventory, the exact expected cell grid, sources, prepared artifacts,
prompts, system/checker/retriever/code versions, dependency lock, exact
model/deployment/tokenizer, retry and failure policy, secret-free credential
patterns, clean Git commit, preflight, and pilot gate. Mutable progress belongs
only in `run_state.json`.

## Offline verification

```bash
uv run python scripts/smoke_pipeline.py
```

The smoke workflow uses a deterministic completion client and an exact fictional
byte tokenizer. It runs all ten stages and is permanently marked
`NON_EMPIRICAL_OFFLINE_FIXTURE`. Tests prove that incomplete grids, failed
gates, changed hashes, scripted results, and protocol violations cannot produce
empirical prose.

## Inference and reporting

Confirmation is an intersection-union test: the low-tier paired difference must
be strictly negative and pass its one-sided exact McNemar test, while the
high-tier difference must be strictly positive and pass its test. Equality is
never a crossover. The analysis retains non-crossing bootstrap replicates and
reports no-crossing mass, conditional crossing intervals, SESOI interpretations,
domain estimates, failures, mechanism traces, and resource-specific Pareto
probabilities.

FinanceComplexQA must pass scorer, lineage, leakage, oracle-evidence, retrieval,
and orchestration boundaries before exploratory system execution. It is never
pooled into confirmation.

## Paper

```bash
uv run python paper/build_paper.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=when-extra-inference-structure-pays-protocol \
  -outdir=../output/pdf main.tex
```

The final protocol PDF is written to `output/pdf/`. Until a complete validated,
hash-matching gateway manifest exists, the generated results section explicitly
says no empirical conclusion is available.

See [experiments/PREREGISTRATION.md](experiments/PREREGISTRATION.md),
[experiments/RUNBOOK.md](experiments/RUNBOOK.md), and
[docs/HISTORICAL_AUDIT.md](docs/HISTORICAL_AUDIT.md).

# When Extra Inference Calls Pay

This repository runs a preregistered, resource-accounted comparison of LLM
decision systems. It asks when verification or candidate coordination improves
an answer enough to justify the prompt tokens, completion tokens, calls, and
wall time that the system actually used.

The study does **not** assume that a larger completion-token ceiling is consumed
compute. It also does not promise a universal architecture crossover. It
compares eight production systems at observed operating points and reports
discrete accuracy-resource frontiers.

## What is tested

- 12-case insurance + 18-case MMLU-Pro pilot.
- Disjoint 68-case insurance + 200-case MMLU-Pro main study.
- Direct GPT-5.4-mini, single-call checklist, direct GPT-5.4,
  same-model self-critique, external critique, best-of-2, best-of-4, and
  adaptive verify/escalate.
- Deterministic task-specific JSON decisions as the primary outcome.
- Per-call gateway `prompt_tokens`, `completion_tokens`, and `total_tokens`,
  plus call count, wall time, and summed API latency.
- Paired bootstrap intervals, exact McNemar tests with Holm correction,
  a five-point smallest effect of practical interest, mechanism diagnostics,
  and descriptive Pareto frontiers.

The 2026-07-29 preregistration is in
[`experiments/V2_PREREGISTRATION.md`](experiments/V2_PREREGISTRATION.md).
The reason the earlier ceiling-based experiment is not used for inference is in
[`experiments/V1_AUDIT.md`](experiments/V1_AUDIT.md).

## Fast setup

```bash
uv sync --extra dev
cp .env.example .env
# Fill in the gateway/token endpoints and both OAuth client pairs.

bash scripts/download_data.sh
uv run budget-crossover gateway-check
uv run pytest -q
uv run ruff check src tests
```

The frozen dataset downloader verifies SHA-256 checksums. Credentials stay in
the ignored `.env` file.

Each OAuth pair has an explicit model allowlist. Credential 1 accepts the three
GPT-5.4 deployments plus Claude Opus/Sonnet 4.6; credential 2 accepts only the
three GPT-5.4 deployments. `LLM_GATEWAY_CONCURRENCY_PER_KEY` is the manual hard
ceiling for each pair. Below that ceiling, a separate
[RFC 9438](https://www.rfc-editor.org/rfc/rfc9438.html)-inspired CUBIC
controller probes upward after successful calls and applies a 0.7
multiplicative decrease after rate-limit, timeout, or overload signals. This is
application-level adaptive concurrency, not a replacement for TCP congestion
control.

## Run the study

```bash
# Pilot: 30 cases × 8 systems.
uv run budget-crossover prepare-v2 --config configs/v2_pilot.yaml
uv run budget-crossover pilot-v2 --config configs/v2_pilot.yaml
uv run budget-crossover analyze-v2 --config configs/v2_pilot.yaml
uv run budget-crossover validate-v2 \
  --config configs/v2_pilot.yaml \
  --no-require-judgments \
  --no-require-pilot-gate

# Main: the command checks the pilot gate before spending the main budget.
uv run budget-crossover prepare-v2 --config configs/v2_main.yaml
uv run budget-crossover run-v2 --config configs/v2_main.yaml
uv run budget-crossover judge-v2 --config configs/v2_main.yaml
uv run budget-crossover analyze-v2 --config configs/v2_main.yaml
uv run budget-crossover validate-v2 --config configs/v2_main.yaml

# Populate and compile the empirical LaTeX paper.
make -C paper results
```

The commands are resumable. Rerun the same command after a transient failure.
Do not delete individual bad responses: malformed and truncated outputs are
part of the system result. A run manifest rejects resumes after code, config,
data, prompt, model, seed, or case-set changes.

The full operational guide, failure recovery rules, output contract, and
runtime estimates are in [`experiments/RUNBOOK.md`](experiments/RUNBOOK.md).

## Paper

The paper uses modular LaTeX:

```text
paper/main.tex
paper/sections/*.tex
paper/generated/results_values.tex
paper/references.bib
```

Before experiments, `make -C paper` compiles a clearly marked protocol draft.
After a complete run, `make -C paper results` validates the experiment, writes
the numeric macros and table rows from the analysis files, and produces:

```text
paper/build/architecture_budget_frontiers.pdf
```

The result builder refuses to make empirical claims if the pilot, generation
grid, usage accounting, call contracts, schema/truncation thresholds, frozen
judge sample, or sampled judgments fail validation.

## Security

Run directories contain model prompts and responses and are ignored by Git.
Review them before sharing. Never commit `.env` or internal gateway pricing.
Dollar cost is reported only when approved internal prices are placed in the
configuration; otherwise the paper reports token and latency frontiers.

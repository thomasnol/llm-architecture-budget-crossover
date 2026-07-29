# Version 2 experiment runbook

This runbook is written for a fresh clone on the computer that can reach the
internal LLM Gateway. Commands are run from the repository root.

## 1. Preconditions

- Python 3.11 or newer.
- `uv`, `curl`, `sha256sum`, `latexmk`, and a standard TeX Live installation.
- Two OAuth client-ID/client-secret pairs.
- Credential 1 can access GPT-5.4, GPT-5.4-mini, GPT-5.4-nano, Claude Sonnet
  4.6, and Claude Opus 4.6. Credential 2 can access only the three GPT
  deployments.
- A clean checkout of the frozen experiment commit. The run manifest records
  the commit and the source-tree hash.

```bash
uv sync --extra dev
cp .env.example .env
```

Fill `.env` locally. The required V2 model IDs are:

```text
gpt-5.4-mini
gpt-5.4
claude-sonnet-4-6
```

Claude Opus 4.6 (`claude-opus-4-6`) remains available for the archived V1
adjudication workflow but is not used by V2.

Keep each credential's `LLM_GATEWAY_MODELS_1` or
`LLM_GATEWAY_MODELS_2` allowlist exact. The scheduler never sends a Claude
request through credential 2. `LLM_GATEWAY_CONCURRENCY_PER_KEY` is a hard
per-credential ceiling, so raising it is the only concurrency edit needed.
Each credential starts at four concurrent requests (or the ceiling, when
lower), then uses an RFC 9438-inspired CUBIC window. Successful completions
increase the window; HTTP 429, timeout, and overload responses apply a 0.7
multiplicative decrease. The controller never exceeds the configured ceiling.

Verify gateway access before downloading or running:

```bash
uv run budget-crossover gateway-check
```

The response should list every required model. Do not proceed if an ID is
missing.

## 2. Freeze and test local inputs

```bash
bash scripts/download_data.sh
uv run pytest -q
uv run ruff check src tests
uv run budget-crossover prepare-v2 --config configs/v2_pilot.yaml
uv run budget-crossover prepare-v2 --config configs/v2_main.yaml
```

Expected preparation counts:

```text
v2_pilot: 30 cases = 12 insurance + 18 MMLU-Pro
v2_main: 268 cases = 68 insurance + 200 MMLU-Pro
overlap between pilot and main: 0
```

The downloader verifies:

```text
insurance SHA-256:
55833ec064222f8a98a80af8e9726ad98f8540f8173be97343e50bac3fb37c83

MMLU-Pro SHA-256:
0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8
```

## 3. Run and inspect the pilot

```bash
uv run budget-crossover pilot-v2 --config configs/v2_pilot.yaml
uv run budget-crossover analyze-v2 --config configs/v2_pilot.yaml
uv run budget-crossover validate-v2 \
  --config configs/v2_pilot.yaml \
  --no-require-judgments \
  --no-require-pilot-gate
```

Expected successful generation cells: `30 × 8 = 240`.

Inspect:

```text
experiments/runs/v2_pilot/run_manifest.json
experiments/runs/v2_pilot/generations.jsonl
experiments/runs/v2_pilot/validation.json
experiments/runs/v2_pilot/analysis/analysis_summary.json
```

At least one dataset must pass all three preregistered pilot checks:

- schema validity at least 98%;
- direct accuracy between 25% and 90%;
- system disagreement at least 10%.

If neither passes, stop. Do not use `--force` to obtain a preferred result.
Diagnose the manipulation and create a new versioned experiment name, config,
prompt version, and preregistration.

## 4. Run the disjoint main study

```bash
uv run budget-crossover run-v2 --config configs/v2_main.yaml
```

Expected successful generation cells: `268 × 8 = 2,144`.

The generation runner has a 4.5-hour deadline and checkpoints every response.
If it stops after a gateway outage or deadline, run the identical command
again. Successful cells are skipped. The immutable manifest prevents mixed
versions.

Next, freeze and run the balanced secondary-judge sample:

```bash
uv run budget-crossover judge-v2 --config configs/v2_main.yaml
```

The command refuses to freeze the sample until all 2,144 generation cells are
present exactly once. With both datasets and all systems, at most 480
generations are selected and each receives two judges, for at most 960 calls.
The selected run IDs are stored in:

```text
experiments/runs/v2_main/judge_sample_manifest.json
```

Resume the identical judge command after transient failures.

## 5. Analyze and validate

```bash
uv run budget-crossover analyze-v2 --config configs/v2_main.yaml
uv run budget-crossover validate-v2 --config configs/v2_main.yaml
```

Validation exits with status 1 if any blocking check fails. Important outputs:

```text
experiments/runs/v2_main/validation.json
experiments/runs/v2_main/analysis/analysis_summary.json
experiments/runs/v2_main/analysis/tables/case_level.csv
experiments/runs/v2_main/analysis/tables/system_summary.csv
experiments/runs/v2_main/analysis/tables/paired_comparisons.csv
experiments/runs/v2_main/analysis/tables/mechanisms.csv
experiments/runs/v2_main/analysis/tables/pareto.csv
experiments/runs/v2_main/analysis/tables/budget_policy.csv
experiments/runs/v2_main/analysis/tables/task_breakdown.csv
experiments/runs/v2_main/analysis/figures/accuracy_cost_insurance.png
experiments/runs/v2_main/analysis/figures/accuracy_cost_mmlu_pro.png
```

Schema-invalid or token-cap-truncated responses remain system outcomes. They
are not selectively rerun. Validation permits at most 2% truncation and
requires at least 98% schema validity. Exceeding those thresholds invalidates
the operational setup and requires a new experiment version.

## 6. Build the final LaTeX paper

```bash
make -C paper results
```

This command reruns validation, writes
`paper/generated/results_values.tex`, and compiles:

```text
paper/build/architecture_budget_frontiers.pdf
```

The generated prose distinguishes:

- pilot-eligible versus exploratory datasets;
- statistical evidence versus the 5 percentage-point smallest effect of
  practical interest;
- Pareto-efficient point estimates versus a universal threshold;
- correction of wrong drafts versus regression of correct drafts.

## 7. Runtime and recovery policy

Maximum live API time:

```text
pilot generation     1.25 h
main generation      4.50 h
sampled judging      1.50 h
contingency           0.75 h
total                 8.00 h
```

Safe recovery is to rerun the same command. Never edit JSONL rows, remove a
failed answer because it hurts accuracy, or change a model/config under the same
experiment name. For a legitimate design change, preserve the old run and
create a new experiment name and preregistration amendment.

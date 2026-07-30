# Canonical experiment operator runbook

## 1. Environment

Copy `.env.example` to `.env` and supply both OAuth pairs. Pair 1 must support
GPT-5.4, GPT-5.4-mini, GPT-5.4-nano, Claude Opus 4.6, and Claude Sonnet 4.6.
Pair 2 must support only the three GPT deployments.

`LLM_GATEWAY_CONCURRENCY_PER_KEY` is the hard ceiling for each credential.
Each credential begins with a CUBIC window of at most four, increases on
successful completions, and multiplicatively decreases on congestion.

Never commit `.env`, gateway responses outside the run directory, or credentials.

## 2. Software and data checks

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run python scripts/download_hmda.py
uv run python scripts/smoke_pipeline.py
uv run budget-crossover prepare --config configs/pilot.yaml
uv run budget-crossover prepare --config configs/main.yaml
uv run budget-crossover validate \
  --config configs/main.yaml \
  --no-require-generations \
  --no-require-pilot-gate
```

The main and pilot source-row sets must be disjoint. Do not proceed if the
dataset digest, counterfactual validation, or leakage checks fail.

## 3. Gateway preflight

Run preflight separately for every experiment configuration:

```bash
uv run budget-crossover preflight --config configs/calibration.yaml
```

Preflight makes one real completion for every configured model on every eligible
credential. Inspect `experiments/runs/calibration/preflight.json`. All checks
must report complete prompt, completion, and total-token usage.

An HTTP 400 is permanent. Read its sanitized response detail and correct the
payload or deployment configuration before continuing. The batch runner will
not operate without a passing preflight report.

## 4. Calibration

```bash
uv run budget-crossover pilot --config configs/calibration.yaml
uv run budget-crossover status --config configs/calibration.yaml
uv run budget-crossover validate \
  --config configs/calibration.yaml \
  --no-require-pilot-gate
uv run budget-crossover calibrate --config configs/calibration.yaml
```

Record the four-budget recommendation from
`experiments/runs/calibration/calibration.json`. If it differs materially from
the frozen pilot/main budgets, update both configurations on a new branch,
rerun tests and the offline smoke, and restart calibration. Never edit a
configuration inside an existing run.

## 5. Pilot

```bash
uv run budget-crossover preflight --config configs/pilot.yaml
uv run budget-crossover pilot --config configs/pilot.yaml
uv run budget-crossover status --config configs/pilot.yaml
uv run budget-crossover validate \
  --config configs/pilot.yaml \
  --no-require-pilot-gate
uv run budget-crossover analyze --config configs/pilot.yaml
```

The pilot must have a complete unique grid, a passing preflight, complete usage
accounting, acceptable schema validity, and no unresolved infrastructure cells.
Diagnostic analysis is not sufficient to pass the pilot gate.

## 6. Main architecture study

Start from a clean commit. Do not change prompts, configurations, code, gateway
protocol settings, or deployment IDs while a run is active.

```bash
uv run budget-crossover preflight --config configs/main.yaml
uv run budget-crossover run --config configs/main.yaml
uv run budget-crossover status --config configs/main.yaml
uv run budget-crossover validate --config configs/main.yaml
uv run budget-crossover analyze --config configs/main.yaml
```

The runner is resumable. Successful and resource-abstention cells are skipped;
transient failures are retried on the next invocation. Three equivalent
permanent failures open the circuit before the remaining queue is launched.

## 7. Routing ablation

```bash
uv run budget-crossover preflight --config configs/routing_pilot.yaml
uv run budget-crossover pilot --config configs/routing_pilot.yaml
uv run budget-crossover status --config configs/routing_pilot.yaml
uv run budget-crossover validate \
  --config configs/routing_pilot.yaml \
  --no-require-pilot-gate
uv run budget-crossover analyze --config configs/routing_pilot.yaml
```

Keep routing conclusions separate from the fixed-model architecture estimand.

## 8. Paper gate

```bash
uv run python paper/build_paper.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The builder incorporates empirical results only when validation required the
complete main grid and the analysis reports no missing, extra, or duplicate
cells. Inspect case-level errors and paired comparisons before interpreting
aggregate results.

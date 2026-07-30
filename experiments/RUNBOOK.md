# Version 3 operator runbook

## 1. Install and configure

```bash
uv sync --extra dev
cp .env.example .env
```

Configure the internal gateway and credential pairs in `.env`. Never commit the
file or copy credentials into a run directory.

Confirm access. This authenticates both pairs, reports each pair's model
allowlist, and exits nonzero if either pair fails:

```bash
uv run budget-crossover gateway-check
```

`LLM_GATEWAY_CONCURRENCY_PER_KEY` is the only throughput ceiling in the v3
runner. Each credential starts with a CUBIC window of at most four requests,
probes upward on success, and reduces the window on rate limits or overload.

## 2. Acquire the frozen official source

```bash
uv run python scripts/download_hmda.py
```

The downloader queries the official CFPB/FFIEC Data Browser for the frozen
four-jurisdiction 2024 originated/denied cohort. It validates the header and
SHA-256. A checksum change is a hard stop, not an invitation to update the digest
silently.

## 3. Validate code and case construction

```bash
uv run pytest -q
uv run ruff check src tests
uv run python scripts/smoke_v3_pipeline.py

uv run budget-crossover prepare-v3 --config configs/v3_pilot.yaml
uv run budget-crossover prepare-v3 --config configs/v3_main.yaml
uv run budget-crossover validate-v3 \
  --config configs/v3_main.yaml \
  --no-require-generations \
  --no-require-pilot-gate
```

Expected main profile:

- 96 source applications;
- 192 cases;
- 96 counterfactual pairs;
- 24 applications in each policy-decision class;
- 24 applications in each state;
- no overlap with the 24-application pilot.

Preparation must report that historical action is not used as gold and
post-decision fields are not supplied to models.

The offline smoke command exercises and validates all 144 cells in a
four-application scripted run and builds every table and figure. Its
`analysis.json` states `results_are_empirical: false`; never cite its scores.

## 4. Pilot

```bash
uv run budget-crossover pilot-v3 --config configs/v3_pilot.yaml
uv run budget-crossover analyze-v3 --config configs/v3_pilot.yaml
uv run budget-crossover validate-v3 \
  --config configs/v3_pilot.yaml \
  --no-require-pilot-gate
```

The pilot contains 48 cases × 6 systems × 3 budgets = 864 generation cells.
Rerunning the same command resumes missing cells. `ok` and `budget_exhausted`
cells are final observations. Generic execution errors go to `errors.jsonl` and
remain eligible for resume.

Do not selectively rerun malformed or wrong model answers. A schema problem
requires an operational version change and a new pilot.

## 5. Main study

```bash
uv run budget-crossover run-v3 --config configs/v3_main.yaml
```

The main command checks the pilot gate before issuing model calls. The expected
grid is 192 cases × 6 systems × 3 budgets = 3,456 cells.

An exceptional gate override must include a durable reason:

```bash
uv run budget-crossover run-v3 \
  --config configs/v3_main.yaml \
  --force \
  --force-reason "documented reason"
```

An override permits execution but invalidates a confirmatory claim that depended
on the failed gate.

## 6. Analyze and validate

```bash
uv run budget-crossover analyze-v3 --config configs/v3_main.yaml
uv run budget-crossover validate-v3 --config configs/v3_main.yaml
```

Analysis writes:

```text
experiments/runs/v3_main/analysis/
├── analysis.json
├── figures/
│   ├── counterfactual_consistency_by_budget.png
│   ├── counterfactual_consistency_by_budget.pdf
│   ├── decision_accuracy_by_budget.png
│   ├── decision_accuracy_by_budget.pdf
│   ├── tokens_vs_accuracy.png
│   └── tokens_vs_accuracy.pdf
└── tables/
    ├── case_results.csv
    ├── mechanisms.csv
    ├── paired_comparisons.csv
    ├── pair_results.csv
    ├── pareto_frontier.csv
    └── system_summary.csv
```

The validator checks the frozen source, counterfactual construction, leakage
controls, exact grid, duplicates, execution errors, token overruns, and pilot
gate.

## 7. Build the paper

```bash
uv run python paper/build_paper.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

If validated main results are absent, the paper remains explicitly a
five-page preregistered LaTeX protocol. The builder must not infer or fabricate
results. After a complete validated main run, rebuild to populate empirical
tables and vector figures. Then complete
`experiments/REMAINING_WORK.md`.

## Failure recovery

- HTTP 429, timeout, or overload: rerun the same command. The gateway client
  retries transient errors and the run resumes from checkpoints.
- Deadline: rerun only if the preregistered execution window still permits it;
  otherwise report missing cells.
- Manifest mismatch: do not edit the manifest. Create a new experiment name and
  document the change.
- HMDA checksum mismatch: retain the unexpected digest separately, investigate
  the official release, and preregister a new dataset version if necessary.
- Budget exhaustion: do not rerun. It is an experimental result.
- Actual token overrun: retain and report it. Diagnose the estimator only in a
  new operational version.

## Security and governance

Run directories contain prompts, model responses, and privacy-modified public
HMDA fields. They are ignored by Git. Do not attempt re-identification, publish
row-level source identifiers, or treat the counterfactual probes as facts about
real applicants. Review every artifact before external sharing.

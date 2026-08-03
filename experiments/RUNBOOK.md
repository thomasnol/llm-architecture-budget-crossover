# Canonical experiment operator runbook

## 1. Freeze prerequisites

Work from a clean commit. Acquire FinQA, TAT-QA, and FinanceComplexQA snapshots
separately, place them at the configured local paths, and set their exact
SHA-256 values. Preparation itself is offline.

Configure the completion gateway and an exact chat-tokenizer endpoint. Set the
frozen tokenizer ID and SHA-256. Do not use a character estimator, alternate
tokenizer, or fallback model. The only allowed model and response-resolved
deployment is `gpt-5.4-mini`.

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run python scripts/smoke_pipeline.py
```

The smoke workflow must end with `validated_empirical_results: false`. It is
software verification only.

## 2. Execute the linear workflow

Run each command once its predecessor passes:

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

There is no override. A failed preflight, FinanceComplex count discrepancy,
calibration mismatch, incomplete pilot, failed gate, changed hash, dirty Git
state, incomplete main grid, non-authoritative usage, or protocol violation must
be resolved by a new versioned protocol/run, not by editing an artifact.

## 3. Inspect artifacts at each boundary

- `prepare`: public and hidden stores are separate, quotas exact, documents
  disjoint, lineage complete, rejection ledger explicit.
- `diagnose-finance-complex`: role is exploratory only; failure attribution and
  the 113-case count are visible.
- `preflight`: exact model/deployment/tokenizer, strict JSON, usage, and count
  consistency all pass.
- `develop`: no accuracy or outcome field is present; ceilings freeze before
  pilot.
- `pilot`: expected CellKey grid is complete and unique.
- `gate`: every preregistered component is present; `override_allowed` is false.
- `run`: `run_manifest.json` is immutable and `run_state.json` contains only
  mutable progress.
- `validate`: grid, usage, hashes, and protocol all pass.
- `analyze`: all eight table interfaces exist and FinanceComplex remains
  separate.
- `build-paper`: empirical prose is enabled only if manifest, gate, validation,
  and analysis hashes agree.

## 4. Resume behavior

Generation JSONL is append-only. Terminal expected keys are skipped on resume;
infrastructure attempts remain separate. Three equivalent permanent failures
open the circuit. Never delete a difficult cell or copy a scripted completion
into a gateway run. If immutable input changes, create a new experiment name and
run directory.

## 5. Paper verification

```bash
uv run python paper/build_paper.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=when-extra-inference-structure-pays-protocol \
  -outdir=../output/pdf main.tex
pdftoppm -png ../output/pdf/when-extra-inference-structure-pays-protocol.pdf \
  ../tmp/pdfs/when-extra-inference-structure-pays-protocol
```

Render and inspect every page for clipping, overlap, broken tables, missing
glyphs, citations, headers, footers, and page numbering. Check extracted text
for the exact title and the appropriate empirical/protocol status. Do not
release a PDF with unresolved LaTeX overflow warnings.

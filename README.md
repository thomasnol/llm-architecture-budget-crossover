# LLM Architecture × Budget Crossover

This repository tests a practical question: under a fixed generation-token
ceiling, when does a multi-call inference architecture begin to outperform a
single direct call?

The experiment uses Snorkel AI's 380-row Multi-Turn Insurance Underwriting
dataset. It reconstructs one leakage-controlled evidence packet per unique
company/task, evaluates direct, self-critique, and two-agent debate
architectures at multiple case-level generation budgets, and records quality,
total tokens, estimated cost, and wall-clock latency.

The study is designed to finish within eight hours on two gateway credentials
that each allow four concurrent requests. Every response is checkpointed to
JSONL, so interrupted runs resume without repeating completed calls.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env
# Fill in the gateway endpoint and credentials locally.

bash scripts/download_data.sh
uv run budget-crossover prepare
uv run budget-crossover pilot --config configs/pilot.yaml
uv run budget-crossover run --config configs/main.yaml
uv run budget-crossover judge --config configs/main.yaml
uv run budget-crossover analyze --config configs/main.yaml
uv run python paper/build_paper.py
```

The generated white paper is written to `paper/architecture_budget_crossover.docx`
and `paper/architecture_budget_crossover.pdf`. Detailed experiment outputs are
kept in `experiments/runs/`.

## Experimental contract

- **Controlled resource:** per-case completion-token ceiling, allocated across
  all calls in an architecture.
- **Measured resources:** actual prompt tokens, completion tokens, total tokens,
  elapsed wall time, and optional price-normalized cost.
- **Architectures:** one-call direct answer; draft–critique–revision; two
  independent specialists followed by a critic and synthesizer.
- **Primary outcome:** correctness against the expert-verified reference,
  decided by two blinded pointwise judges and an independent adjudicator on
  disagreements.
- **Inference:** paired case bootstrap confidence intervals and a
  case/task-adjusted crossover model. Claims are conditional on the tested
  model, dataset, prompts, and gateway.

## Security

Credentials are read only from environment variables or `.env`, which is
ignored by Git. Raw prompts, responses, and traces may contain synthetic company
information from the public dataset; keep run directories internal unless
reviewed.


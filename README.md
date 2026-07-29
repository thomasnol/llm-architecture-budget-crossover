# LLM Architecture × Budget Crossover

This repository studies a practical question: when does additional inference
orchestration improve an LLM decision enough to justify its realized cost?

Version 1 tested maximum completion-token ceilings and is retained as a
transparent failed manipulation/measurement audit. Version 2 uses structured
decisions, unbiased evidence selection, all 80 unique underwriting tasks, a
200-item MMLU-Pro stress test, stronger baselines, external verification, and
adaptive escalation. It records the gateway's per-call prompt, completion, and
total tokens, along with calls, latency, and optional approved internal cost.

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

Version 2:

```bash
uv run budget-crossover prepare-v2 --config configs/v2_pilot.yaml
uv run budget-crossover pilot-v2 --config configs/v2_pilot.yaml
uv run budget-crossover analyze-v2 --config configs/v2_pilot.yaml
# Proceed only if the preregistered pilot gates pass.
uv run budget-crossover run-v2 --config configs/v2_main.yaml
uv run budget-crossover judge-v2 --config configs/v2_main.yaml
uv run budget-crossover analyze-v2 --config configs/v2_main.yaml
```

The generated white paper is written to `paper/architecture_budget_crossover.docx`
and `paper/architecture_budget_crossover.pdf`. Detailed experiment outputs are
kept in `experiments/runs/`.

## Experimental contract

- **Primary resource:** realized per-case total tokens, calls, and wall time.
- **Measured resources:** actual prompt tokens, completion tokens, total tokens,
  elapsed wall time, and optional price-normalized cost.
- **Systems:** direct; single-call checklist; same-model self-critique; strong
  external critique; best-of-2/4 selection; adaptive verify/escalate.
- **Primary outcome:** deterministic task-specific structured operational
  correctness. Cross-family LLM judges are secondary.
- **Inference:** paired case bootstrap intervals, exact McNemar tests, mechanism
  metrics, minimum detectable effects, and accuracy-cost Pareto frontiers.

## Security

Credentials are read only from environment variables or `.env`, which is
ignored by Git. Raw prompts, responses, and traces may contain synthetic company
information from the public dataset; keep run directories internal unless
reviewed.

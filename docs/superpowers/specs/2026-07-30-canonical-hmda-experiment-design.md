# Canonical HMDA Architecture–Budget Experiment Design

## Objective

Replace the repository's accumulated versioned experiments with one canonical,
reproducible study of how orchestration architecture changes policy-correct
mortgage adjudication under total-token budgets. The failed pilot is retained
only as diagnostic evidence and is not pooled with the replacement experiment.

## Research questions

1. Under a fixed total-token budget and fixed underlying model, which
   orchestration architecture maximizes end-to-end policy accuracy?
2. Does selective weak-to-strong escalation move the accuracy–cost frontier
   relative to always-weak and always-strong monoliths?
3. Do compliance-oriented architectures reduce protected-attribute
   counterfactual decision flips without sacrificing policy correctness?

## Experimental factors

### Primary architecture comparison

All roles use the configured primary model so architecture is the manipulated
factor:

- full-context monolith;
- plan-and-retrieve;
- specialist committee;
- fixed underwriter-plus-compliance guardrail;
- adaptive guarded routing.

The strong-model monolith is not part of the primary architecture estimand.

### Routing ablation

A separate experiment compares:

- always-primary monolith;
- always-supervisor monolith;
- primary-model draft with selective supervisor escalation.

This is explicitly a model-routing comparison rather than a pure architecture
comparison.

### Cases

Official 2024 HMDA records seed a deterministic policy sandbox. Historical
institutional action remains descriptive and is never the gold label.
Counterfactual twins vary exactly one monitoring-only protected attribute.
Sampling balances research-policy decisions and, for the main cohort, states.
Complexity and historical action are profiled rather than forced into artificial
quotas. The pilot and main application sets are disjoint.

## Budget protocol

The canonical workflow has three stages:

1. `preflight`: one real completion for each configured model on each eligible
   credential, validating request compatibility and usage accounting;
2. `calibrate`: an unconstrained micro-sample measures prompt and trajectory
   costs and proposes four feasible primary budgets;
3. `pilot` and `run`: execute frozen grids using the accepted budgets.

The lowest primary budget must be feasible for every fixed architecture.
Resource starvation at 2,048 tokens may be retained as a diagnostic stress
condition but is excluded from the primary architecture estimand.

Every cell terminates as one of: correct/incorrect decision, resource
abstention, schema failure, or infrastructure failure. Conditional accuracy
and intention-to-treat accuracy are both reported.

## Reliability contract

- HTTP 400 and other permanent client failures are terminal and trigger a
  batch circuit breaker after three equivalent failures.
- Timeouts, rate limits, and selected server failures remain retryable.
- A bounded worker pool prevents the full grid from being launched after a
  systemic failure.
- Sanitized failures record HTTP status, response detail, model, stage,
  credential slot, request ID, retryability, and attempt number.
- The CUBIC per-credential limiter remains, but experiment execution requires
  a successful preflight report for every configured experimental model.
- Tests cannot load the repository's real `.env`.

## Analysis contract

Normal analysis refuses incomplete or duplicate grids. Diagnostic analysis is
explicitly opt-in and watermarks every generated figure.

Primary inference clusters at the original HMDA application. Counterfactual
twins and repeated executions stay in the same bootstrap cluster. Reports
include:

- grid coverage and infrastructure failure rate;
- intention-to-treat and conditional policy accuracy;
- exact policy/reason-code accuracy and schema validity;
- resource-abstention and budget-overrun rates;
- counterfactual flip rate overall and by changed attribute;
- realized tokens, calls, latency, and optional cost;
- paired differences against the monolith with multiplicity correction;
- accuracy–token and accuracy–cost Pareto frontiers;
- bootstrap uncertainty for architecture crossover points.

## Canonical repository interface

Version labels are removed from filenames, symbols, configurations, commands,
documentation, and generated artifact names. The supported commands are:

```text
budget-crossover prepare
budget-crossover preflight
budget-crossover calibrate
budget-crossover pilot
budget-crossover run
budget-crossover status
budget-crossover analyze
budget-crossover validate
```

The canonical configurations are `configs/pilot.yaml`, `configs/main.yaml`,
and focused routing/calibration configurations where needed. Old implementations
are absent from the branch tree, while normal Git history remains recoverable.

## Reproducibility and paper handoff

The manifest freezes source, config, case data, prompts, dependency lock,
resolved deployment IDs, non-secret gateway protocol settings, git commit,
and run repetitions. Results are generated only from validated complete grids.
The LaTeX paper remains an approximately five-page registered protocol until
the main experiment is complete; empirical claims and final figures are
inserted only from generated, validated artifacts.

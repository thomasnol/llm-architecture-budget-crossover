# Preregistered analysis plan

Date frozen: 2026-07-28, before any calls to the study generator or judges.

## Research question

Under a fixed per-case completion-token ceiling, does the accuracy ordering of
direct generation, self-critique, and two-agent debate change as the ceiling
increases on multi-turn commercial-insurance underwriting tasks?

## Hypotheses

- H1: direct generation has the highest exact accuracy at the 256-token ceiling.
- H2: at least one multi-call architecture exceeds direct generation at a higher
  tested ceiling.
- H3: multi-call gains are larger on cases with more tool evidence and longer
  evidence packets.
- H4: at the same completion-token ceiling, multi-call architectures consume
  more total tokens and summed call latency because they repeat context.

The experiment may reject any or all hypotheses. No architecture is declared a
winner from a three-case pilot or from judge scores alone.

## Frozen design

- Dataset commit:
  `9aa8782f850a41de2e7d21edf4def91ce99c0d08`
- Dataset SHA-256:
  `55833ec064222f8a98a80af8e9726ad98f8540f8173be97343e50bac3fb37c83`
- Unit of analysis: unique `company task id`.
- Main sample: 60 cases, seed `20260728`, with task quotas in
  `configs/main.yaml`.
- Generator: one fixed gateway deployment for all architectures.
- Temperature: 0.0.
- Completion-token ceilings: 256, 512, 1,024, 2,048, and 4,096.
- Architectures: direct (1 call), self-critique (3 calls), debate (4 calls).
- Primary outcome: deterministic task-specific exact operational correctness.
- Secondary outcomes: dual-judge correctness, evidence grounding, measured
  prompt/completion/total tokens, and latency.
- Uncertainty: 5,000 case-level paired bootstrap replicates.
- Adjusted model: clustered binomial GLM with architecture × log2(budget), task
  fixed effects, and precomputed evidence complexity.

## Crossover definition

For each multi-call architecture, compute its paired case-level accuracy
difference from direct at every tested ceiling. The crossover is the first
log-budget interpolation where this difference moves from non-positive to
positive. If the difference is positive at the lowest ceiling, report
`≤256`; if it never turns positive, report `>4096 / not observed`. A bootstrap
interval is reported only when a finite crossing occurs in the resampled
curves; the finite-crossing proportion is reported separately.

## Exclusions and missingness

- Historical assistant prose and final answers are excluded from case packets.
- A case is rejected during preprocessing if its normalized reference answer
  appears in the evidence packet.
- Failed gateway calls remain in the checkpoint log but are not scored as
  incorrect. A results paper is not generated until the full generation matrix
  is complete.
- Judge failures do not affect the deterministic primary outcome. Missing judge
  coverage and judge disagreement are reported.
- No prompt, architecture, sample, or scoring changes are made after inspecting
  main-sweep outcomes. Any required operational correction is versioned and
  requires rerunning the affected matrix.


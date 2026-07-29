# Version 1 audit: failed manipulation and invalid outcome parsing

Frozen after inspection of the completed Version 1 result summary and before any
Version 2 calls.

Version 1 is retained as a transparent pilot. It is not used to estimate an
architecture-budget crossover.

## Manipulation failure

The independent variable was a maximum completion-token allowance, not realized
test-time computation. Direct generation used approximately 171-178 completion
tokens at every nominal ceiling. Same-model self-critique plateaued near 720
completion tokens and debate near 1,000. Thus the upper ceilings supplied almost
no additional realized computation. Repeated context also caused multi-call
systems to consume several times as many prompt tokens despite the nominally
matched completion allowance.

## Measurement failure

The free-text exact parser mixed rationale content into the operational decision.
Confirmed failure cases included:

- a correct `$3M/$5M` policy-limit decision marked wrong because its rationale
  mentioned a `$500` deductible;
- a correct product recommendation marked wrong because `auto` appeared in
  "do not pitch auto";
- "not a small business" parsed as `small`;
- "not qualified" parsed as `qualified`.

The `judge_disagreement` field also defaulted to false for missing judgments, so
zero disagreement was not interpretable without an explicit coverage numerator.

## Inferential failure

The displayed debate differences versus direct were -1.7, +1.7, +3.3, -1.7,
and +5.0 percentage points across five ceilings, with every confidence interval
including zero. The first sign change was an interpolation between one-case
differences in a 60-case sample. A bootstrap curve crossing somewhere among five
noisy opportunities is not evidence for a stable resource threshold.

## Decision

Version 2:

1. uses structured task-specific decisions;
2. selects evidence without consulting historical correctness;
3. treats realized total tokens, calls, latency, and approved internal cost as
   resources;
4. introduces candidate diversity and an independent stronger verifier;
5. measures correction, regression, verifier quality, and escalation mechanisms;
6. uses an accuracy-cost Pareto analysis rather than forcing a crossover.

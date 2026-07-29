# Version 2 preregistration

Date frozen: 2026-07-29, before Version 2 pilot or main-study model calls.

## Research question

Under realized inference costs, can adaptive verification and escalation improve
structured decision accuracy over direct generation while remaining on the
accuracy-cost Pareto frontier? Which mechanisms - candidate disagreement,
verifier detection, correction, and regression - explain the result?

## Primary hypothesis

On the pooled evaluation set, adaptive verify/escalate has higher paired exact
structured-decision accuracy than direct generation and is not dominated on
realized total-token use. This hypothesis may be rejected. The paper remains
informative through preregistered mechanism and efficiency analyses.

## Secondary hypotheses

- A single-call checklist is a stronger and cheaper baseline than unstructured
  direct prompting.
- A strong external verifier has higher recall on initial errors than same-model
  self-critique.
- Candidate disagreement identifies cases with greater error and correction
  opportunity.
- Always-on multi-call systems consume more tokens and latency; adaptive routing
  avoids some of this cost when the verifier confidently accepts the initial
  answer.
- Architecture effects differ between static underwriting decisions and the
  MMLU-Pro reasoning stress test.

## Datasets and independent units

- Insurance: all 80 unique `company task id` units from Snorkel AI's
  Multi-Turn Insurance Underwriting release. The 380 trace rows are not treated
  as independent cases.
- Stress test: a seeded, category-stratified sample of 200 MMLU-Pro test items.
- Pilot: 12 insurance and 18 MMLU-Pro cases selected with the frozen seed.
- Repeated system outputs are clustered or paired by case. Historical model
  traces are not treated as replication.

The insurance packet pools and deduplicates tool outputs from all historical
traces without consulting the `correct` field. Underwriter messages and company
facts come from the fixed `o3` trace, available for every case. The primary
insurance task is explicitly a static evidence-decision task, not a claim to
replicate the source benchmark's interactive tool-use setting.

## Systems

The generator is fixed to GPT-5.4-mini:

1. direct structured answer;
2. one-call structured checklist;
3. same-model draft, critique, and revision;
4. draft, GPT-5.4 external critique, and generator revision;
5. two candidates plus GPT-5.4 selection;
6. four candidates plus GPT-5.4 selection;
7. adaptive verify/escalate: direct draft, GPT-5.4 verification gate, and only
   when rejected or insufficiently confident, two diverse candidates plus
   GPT-5.4 selection.

Sampling candidates use temperature 0.7; deterministic stages use temperature
0.0. Completion limits are safety caps and are never interpreted as consumed
compute.

## Outcomes

Primary:

- deterministic equality of task-specific structured operational decisions.

Resources:

- per-call and per-case prompt, completion, and total tokens returned by the
  gateway;
- call count;
- end-to-end wall time and summed API latency;
- estimated dollar cost only when approved internal price inputs are configured.

Mechanisms:

- initial and final accuracy;
- correction rate conditional on an initially wrong answer;
- regression rate conditional on an initially correct answer;
- verifier precision and recall for initial errors;
- candidate disagreement and adaptive escalation rates.

Secondary cross-family judges (GPT-5.4 and Claude Sonnet 4.6) assess semantic
correctness and grounding. Judge coverage is reported as a numerator and
denominator; missing judgments never count as agreement.

## Pilot gates

The main sweep is not run automatically unless, for at least one dataset:

- structured-schema validity is at least 98%;
- direct accuracy is between 25% and 90%;
- at least 10% of cases show disagreement across systems.

Failed schema validity requires an operational correction and rerun. A dataset
that is too easy or has insufficient disagreement may be reported as a boundary
condition but cannot support the primary effectiveness test.

## Statistical analysis

- Paired case-level accuracy differences versus direct with 5,000 case bootstrap
  replicates.
- Exact two-sided McNemar tests from improved and regressed pairs.
- Micro accuracy plus task-macro accuracy.
- Accuracy versus realized total-token Pareto frontiers.
- Approximate 80%-power minimum detectable paired effects computed from observed
  discordance.
- Primary confirmatory comparison: adaptive versus direct.
- All other systems, task effects, mechanisms, and cross-dataset contrasts are
  secondary or exploratory.

No crossover is estimated unless an architecture difference is monotonic and a
confidence interval supports a sign change. No result is rewritten to imply that
a preregistered hypothesis was supported when it was not.

## Runtime and stopping

- Pilot generation: maximum 1.25 hours.
- Main generation: maximum 4.5 hours.
- Secondary judging: maximum 1.5 hours.
- Total live API work: maximum 7.25 hours, leaving 0.75 hours of contingency
  inside the eight-hour constraint.
- Every response is checkpointed. Deadline cancellation and failures are
  reported; missing jobs are never scored as incorrect.

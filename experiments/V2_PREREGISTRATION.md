# Version 2 preregistration

Date frozen: 2026-07-29, before Version 2 pilot or main-study model calls.

## Research question

Under realized inference costs, when does verification or coordination earn its
overhead? In particular, can adaptive verification and escalation improve
structured decision accuracy over direct generation while remaining on the
accuracy-resource Pareto frontier? Which mechanisms - candidate disagreement,
verifier detection, correction, and regression - explain the result?

## Primary hypothesis

Within each dataset, adaptive verify/escalate has higher paired exact
structured-decision accuracy than GPT-5.4-mini direct generation and is not
dominated on realized total-token use. We control the two dataset-level McNemar
tests with a Holm correction. We report support separately by dataset and call
the result replicated only if both datasets satisfy the accuracy, multiplicity,
and Pareto criteria. This hypothesis may be rejected. The paper remains
informative through preregistered mechanism and efficiency analyses.

## Secondary hypotheses

- A single-call checklist is a stronger and cheaper baseline than unstructured
  direct prompting.
- A one-call GPT-5.4 baseline separates gains from a stronger model from gains
  caused by orchestration around GPT-5.4-mini.
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
  as independent cases. A stratified 12-case pilot split is disjoint from the
  68-case confirmatory split.
- Stress test: a seeded, category-stratified sample of 218 MMLU-Pro test items,
  split into 18 disjoint pilot cases and 200 confirmatory cases.
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
3. one-call GPT-5.4 direct answer;
4. same-model draft, critique, and revision;
5. draft, GPT-5.4 external critique, and generator revision;
6. two candidates plus GPT-5.4 selection;
7. four candidates plus GPT-5.4 selection;
8. adaptive verify/escalate: direct draft, GPT-5.4 verification gate, and only
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
correctness and grounding on a frozen, balanced sample of at most 30 outputs per
dataset and system. This is an audit of rationale quality, not the primary
score. Judge coverage is reported as a numerator and denominator; missing
judgments never count as agreement.

## Pilot gates

The main sweep is blocked by the command-line runner unless, for at least one
dataset:

- structured-schema validity is at least 98%;
- direct accuracy is between 25% and 90%;
- at least 10% of cases show disagreement across systems.

Failed schema validity requires an operational correction and a new,
versioned experiment rather than selective deletion. A dataset that is too easy
or has insufficient disagreement may be reported as a boundary condition but
cannot support the primary effectiveness test, even if its main-split point
estimate is positive.

## Statistical analysis

- Paired case-level accuracy differences versus direct with 5,000 case bootstrap
  replicates.
- Exact two-sided McNemar tests from improved and regressed pairs, with a Holm
  correction across the two dataset-level tests for each system.
- Micro accuracy plus task-macro accuracy.
- Accuracy versus realized total-token Pareto frontiers.
- A descriptive budget policy that selects the highest observed accuracy among
  systems feasible at each observed mean-token operating point.
- Approximate 80%-power minimum detectable paired effects computed from observed
  discordance.
- Primary confirmatory comparison: adaptive versus direct.
- Smallest effect of practical interest: an accuracy gain of 5 percentage
  points. Confirmatory support requires the point estimate to meet this
  threshold, the paired interval to exclude zero, the Holm-adjusted McNemar
  value to be below 0.05, pilot eligibility, and Pareto efficiency.
- All other systems, task effects, mechanisms, and cross-dataset contrasts are
  secondary or exploratory.

No smooth crossover is estimated unless an architecture difference is monotonic
and a confidence interval supports a sign change. Discrete changes on the
empirical budget policy are called operating-point switches, not universal
thresholds. No result is rewritten to imply that a preregistered hypothesis was
supported when it was not.

## Runtime and stopping

- Pilot generation: maximum 1.25 hours.
- Main generation: maximum 4.5 hours.
- Secondary judging: maximum 1.5 hours. Balanced sampling limits this stage to
  at most 960 calls in the complete eight-system, two-dataset main run.
- Total live API work: maximum 7.25 hours, leaving 0.75 hours of contingency
  inside the eight-hour constraint.
- Every response is checkpointed. Deadline cancellation and failures are
  reported; missing jobs are never scored as incorrect.
- The two credential pairs have independent model allowlists and independent
  application-level CUBIC concurrency windows. The manual per-key ceiling is
  recorded in the local environment; the controller uses $C=0.4$ and
  $\beta=0.7$, and overload signals reduce the affected credential only.

## Reproducibility and malformed outputs

The first call freezes a run manifest containing the expanded configuration,
case-set hash, source-data hashes, source-tree hash, prompt version, model IDs,
seed, Git commit, and start time. A resume is rejected if any immutable field
changes. Secondary-judge run IDs are frozen in a separate manifest only after
the generation grid is complete.

A schema-invalid or token-cap-truncated response remains an observed system
output and is scored as incorrect when its structured decision is unavailable.
It is not selectively regenerated. The validator requires overall schema
validity of at least 98% and a truncation rate no greater than 2%; exceeding
either limit invalidates the operational setup and requires a new experiment
version.

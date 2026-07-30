# Preregistration: HMDA architecture--budget policy sandbox

Initially frozen: 2026-07-29, before any canonical study model call. The gateway was not
configured at freeze time. Repository tests and deterministic data preparation
are not model evaluations.

## Research question

When an entire mortgage-adjudication workflow is constrained by a case-level
total-token budget, which orchestration architecture produces the best joint
profile of policy accuracy, counterfactual compliance invariance, token use, and
latency?

The primary study compares five architectures at four nominal budgets while
holding the underlying model fixed. A separate routing ablation compares
always-primary, always-supervisor, and selective escalation. It does not assume that
more calls help, that a complex architecture is universally superior, or that a
larger completion ceiling is consumed compute.

## Dataset decision

The study selects HMDA (dataset option C), using the official CFPB/FFIEC source rather
than a Kaggle mirror.

Home Credit and Lending Club were rejected as the primary adjudication source
because their labels are post-credit performance among observed or funded loans.
Default and charge-off are useful risk outcomes, but neither is the correct label
for whether an application should have been approved at decision time. Home
Credit remains attractive for future relational-evidence work; Lending Club
remains attractive for future tool-selection work.

HMDA contains actual mortgage applications and action taken, plus loan, property,
income, and monitoring-demographic fields. Actual action is not normative truth:
it reflects institution policy, applicant behavior, data availability, and
selection. The experiment therefore retains action taken only for descriptive
concordance and never supplies it to a model.

## Source and frozen cohort

- Source endpoint:
  `https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DC,ND,VT,WY&years=2024&actions_taken=1,3`
- Dataset vintage: 2024 Data Browser export served on 2026-07-29. The endpoint
  may reflect official resubmissions; the recorded SHA-256 freezes the exact
  bytes used by this study.
- Frozen source SHA-256:
  `ed1f933f5b3487310c8364aebba8cb8b82d3f9870ff744648899e62baceaf4f5`.
- Geography: District of Columbia, North Dakota, Vermont, and Wyoming.
- Historical actions retained in the source cohort: originated (1) and denied
  (3).
- Product scope: conventional loan; home purchase; first lien; not reverse;
  closed end; not primarily business/commercial; site built; owner occupied; one
  to four units.

The resulting eligible cohort contains 15,181 source applications. The sample is
balanced first across the four research-policy decisions and then across states
as capacity allows.

The 24-application pilot produces 48 cases. The disjoint 192-application main
sample produces 384 cases. Each source application is the independent unit.

## Transparent research policy

The policy is an evaluation oracle, not a real mortgage policy:

1. Manual review if income, property value, LTV, or DTI is not reported.
2. Deny if DTI is at least 50%, LTV exceeds 100%, or term exceeds 360 months.
3. Conditional review if DTI is 43–49%, LTV exceeds 90% through 100%, or the
   application is nonconforming.
4. Approve otherwise.

The oracle emits the controlling decision and every applicable reason code at
that decision level. Rule precedence and parsing are deterministic and unit
tested. Race, ethnicity, sex, age band, tract minority share, and tract income
fields are monitoring-only and cannot affect the oracle.

HMDA does not expose the applicant's credit-score value in the public file.
The study does not impute one and makes no claim to reproduce a production
mortgage credit model.

## Counterfactual compliance probe

Every sampled application produces two case packets:

- observed monitoring attributes;
- a counterfactual variant changing exactly one of race, sex, or ethnicity.

The financial documents, quality notes, policy, and gold label are byte-identical
within a pair. The counterfactual test is metamorphic: a compliant system should
produce the same policy decision because the changed field is prohibited from the
research decision.

This probe tests individual workflow invariance under a controlled input change.
It is not an estimate of population disparity, disparate treatment in actual
lending, causal discrimination, or legal compliance.

## Evidence exposed to systems

Each case has five documents:

1. application intake;
2. collateral review;
3. credit and capacity;
4. segregated compliance monitoring;
5. data-quality notes.

Historical action, denial reasons, interest rate, rate spread, fees, purchaser
type, and other post-decision fields are withheld. The compliance-monitoring
document explicitly states that its fields cannot affect the research decision.

## Systems

Every primary architecture stage uses `gpt-5.4-mini`; architecture is therefore
the manipulated factor. The separate routing ablation uses
`claude-sonnet-4-6` only for the always-supervisor baseline and selective
escalation.

1. **Monolithic full-context**: one call receives policy and all five documents.
2. **Plan-and-retrieve**: a planning call selects at most three evidence tools;
   a second call adjudicates from only the returned documents.
3. **Specialist committee**: capacity, collateral, and compliance specialists run
   concurrently; a chair resolves their reports by policy precedence.
4. **Underwriter plus compliance guardrail**: a financial underwriter cannot see
   monitoring demographics; an independent compliance agent audits the draft;
   a finalizer applies supported corrections.
5. **Adaptive guarded routing**: a financial underwriter is accepted when it
   emits a valid approve decision with confidence at least 0.90 and no missing
   evidence marker. All other drafts route through compliance audit and
   finalization.

All primary-stage temperatures are zero. The main study repeats every cell twice
and clusters both repetitions with their source application.

## Token intervention

The independent resource variable is a case-level nominal **total-token budget**.
A high-budget calibration run proposes four pooled realized-cost quantiles,
constrained so the lowest primary budget is feasible for every fixed
architecture. The frozen candidates are 4,096, 6,144, 8,192, and 12,288 tokens,
covering input plus output across every internal call.

Before each call, the controller conservatively estimates prompt tokens as
`ceil(characters / 2.5) + 64`, then allocates no more than the remaining budget
to output. A call is not issued if fewer than 64 output tokens remain after the
estimate. Concurrent specialist calls are admitted only when the remaining
budget can fund all three estimated prompts and minimum outputs.

Gateway-reported prompt, completion, and total tokens are authoritative after a
call. A system never receives an extra retry because it exhausted its budget.
Budget exhaustion is an observed operational result. An actual overrun is
reported, retained, and disqualifies that operating point from the
budget-feasible frontier; it is never hidden by post hoc relabeling.

## Outcomes

Primary outcome:

- application-level paired success: both counterfactual variants have the correct
  policy decision.

Secondary outcomes:

- per-case policy-decision accuracy;
- exact decision-plus-reason-code accuracy;
- counterfactual decision flip rate and full-policy consistency;
- grid coverage, completed-decision accuracy, and intention-to-treat accuracy;
- schema validity and budget-exhaustion rate;
- prompt, completion, and total tokens;
- call count, wall time, and summed API latency;
- optional estimated dollar cost only when approved internal prices are supplied;
- accuracy by routine, threshold, and exception complexity;
- descriptive concordance with historical action, explicitly not correctness.

## Confirmatory hypothesis

Within each nominal budget, adaptive guarded routing will improve
application-level paired success over monolithic full-context by at least five
percentage points. Four exact paired McNemar tests are Holm-adjusted across the
budgets. Support at a budget requires:

- point difference at least +0.05;
- paired bootstrap 95% interval excluding zero;
- Holm-adjusted p-value below 0.05;
- budget-overrun rate at or below 1%;
- the adaptive operating point is not dominated on decision accuracy and realized
  total tokens within that nominal budget.

The hypothesis may fail at all budgets. A crossover is reported only if the sign
of the paired difference changes across adjacent tested budgets with uncertainty
consistent with a change. Otherwise, the paper reports discrete operating-point
comparisons and does not invent a continuous threshold.

All non-adaptive systems, reason-code accuracy, complexity interactions,
historical concordance, and architecture rankings are secondary or exploratory.

## Inference

- Source application, not counterfactual row, is the independent unit.
- Accuracy intervals use application-cluster bootstrap resampling.
- Paired system comparisons use the same application pairs.
- McNemar cells use binary success on both twins.
- The approximate 80%-power minimum detectable paired difference is reported
  from observed discordance; no underpowered null is described as equivalence.
- Pareto frontiers are descriptive and use realized mean total tokens.
- Missing or malformed decisions are incorrect. Execution failures are reported
  separately and must be resolved at the operational level, not by
  selectively deleting hard cases.

## Pilot gates

The main run is blocked unless:

- every source and counterfactual validation check passes;
- every model/eligible-credential completion preflight passes;
- the unique generation grid is complete with no unresolved infrastructure cell;
- every high-budget system has at least 95% schema validity;
- no operating point has more than 1% actual budget overrun;
- at least one budget exhibits accuracy disagreement across architectures.

A failed gate requires a documented, versioned operational change and a new
pilot. `--force` requires a written audit reason and does not retroactively make
the preregistered confirmatory claim valid.

## Reproducibility and stopping

The first model call freezes a manifest containing expanded configuration, case
hash, HMDA checksum, prompt revision, source hash, primary and supervisor model
IDs, resolved deployments, non-secret gateway protocol settings, seed, Git
commit, and dirty-worktree flag. Checkpointed JSONL makes runs
resumable; immutable-input changes block resume. Scored generations and
retryable execution errors are stored separately.

Pilot runtime is capped at 1.5 hours. Main runtime is capped at 6 hours, leaving
contingency inside an eight-hour execution window. Deadline cancellation and
missing cells are reported and are not scored as model errors.

## Interpretation boundary

This study evaluates whether LLM orchestration can execute a disclosed research
policy under resource and information-governance constraints. It does not
validate a production credit policy, determine creditworthiness, provide lending
advice, estimate market discrimination, or replace a fair-lending review.

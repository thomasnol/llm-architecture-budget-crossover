# Work remaining after the external gateway run

The experiment code, protocol figures, analysis package, and five-page LaTeX
protocol are finalized. Do not complete the paper from the offline scripted
smoke run.

## Required external execution

- Copy `.env.example` to `.env` and populate both OAuth client ID/secret pairs.
- Run `gateway-check`; both credential slots must report `status: ok`.
- Run the pilot, analyze it, and pass the pilot validation gate.
- Run all 3,456 main cells, then run `analyze-v3` and `validate-v3`.
- Preserve `generations.jsonl`, `errors.jsonl`, the run manifest, validation,
  analysis tables, and figures.

## Required empirical analysis

- Inspect missing cells, retry history, schema failures, budget exhaustion, and
  token overruns before reading aggregate accuracy.
- Review case-level errors by policy outcome, complexity, route, and changed
  monitoring attribute.
- Interpret adaptive versus monolith using the preregistered paired effect,
  clustered interval, McNemar test, Holm correction, five-point SESOI, and
  Pareto condition.
- Compare the strong monolith with guarded systems to separate model strength
  from orchestration.
- Report all operating points and null or negative results.

## Required paper completion

- Re-run `paper/build_paper.py`; verify the result gate switches only after a
  complete passing main validation.
- Replace the protocol abstract with purpose, method, measured results, and
  conclusion in past tense.
- Write the results and discussion from case-level and paired evidence, not only
  the summary table.
- Compare findings with the cited test-time compute, tool-use, behavioral
  testing, and self-correction literature.
- Recheck every factual claim and citation against a primary or official source.
- Confirm every table and figure is referenced, uses correct significant
  figures, and reports uncertainty where planned.
- State limitations plainly; do not call non-significance equivalence or the
  metamorphic probe a legal-compliance test.
- Compile, render all pages, inspect visual layout, proofread, and obtain one
  independent peer review before submission.

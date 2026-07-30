from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "experiments" / "runs" / "v3_main"
GENERATED = REPO / "paper" / "generated"


def _latex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _validated_results() -> tuple[bool, str]:
    validation_path = RUN_DIR / "validation.json"
    analysis_path = RUN_DIR / "analysis" / "analysis.json"
    if not validation_path.exists() or not analysis_path.exists():
        return False, "The external gateway run has not been completed and validated."
    validation = json.loads(validation_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    checks = [
        bool(validation.get("pass")),
        validation.get("requirements", {}).get("require_generations") is True,
        validation.get("requirements", {}).get("require_pilot_gate") is True,
        bool(analysis.get("results_are_empirical")),
        analysis.get("expected_generations") == analysis.get("observed_generations"),
        analysis.get("generation_completion_rate") == 1.0,
    ]
    if not all(checks):
        return False, "Run artifacts exist, but the completeness and validation gates do not pass."
    return True, "The external main run is complete and passed the frozen validation gates."


def _pending_section(reason: str) -> str:
    return rf"""\section{{Results Status}}

\begin{{statusbox}}
\textbf{{Protocol-stage artifact.}} {_latex(reason)} The repository tests and
offline scripted smoke run establish software behavior only. They are not model
evidence and do not answer the research question.
\end{{statusbox}}

The external run must execute all 3,456 cells in the case by system by budget
grid, regenerate the analysis bundle, and pass \texttt{{validate-v3}}. The final
paper will then report all operating points, including failures, budget
exhaustion, and null results.
Interpretation and the final abstract remain intentionally unwritten until those
artifacts exist.
"""


def _result_section() -> str:
    summary = pd.read_csv(RUN_DIR / "analysis" / "tables" / "system_summary.csv")
    rows = []
    for _, row in summary.sort_values(["token_budget", "system"]).iterrows():
        rows.append(
            "{} & {:,} & {:.3f} & {:.3f} & {:.0f} & {:.2f} \\\\".format(
                _latex(row["system_label"]),
                int(row["token_budget"]),
                float(row["both_twins_decision_correct"]),
                float(row["counterfactual_decision_consistency"]),
                float(row["mean_total_tokens"]),
                float(row["mean_call_count"]),
            )
        )
    table_rows = "\n".join(rows)
    return rf"""\section{{Results}}

\begin{{table}}[H]
\centering
\small
\caption{{Complete operating-point results. Paired accuracy requires both
counterfactual twins to receive the correct sandbox decision.}}
\label{{tab:results}}
\begin{{tabular}}{{lrrrrr}}
\toprule
System & Budget & Paired acc. & Invariance & Tokens & Calls \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{../experiments/runs/v3_main/analysis/figures/tokens_vs_accuracy.pdf}}
\caption{{Realized token use against policy-decision accuracy. Only validated
gateway outputs appear in this figure.}}
\label{{fig:results}}
\end{{figure}}

% TODO: Interpret the complete result set, compare it with the confirmatory
% criteria, and discuss limitations. Do not write this section from the table
% alone; inspect case-level errors and paired comparisons first.
"""


def main() -> None:
    valid, reason = _validated_results()
    GENERATED.mkdir(parents=True, exist_ok=True)
    section = _result_section() if valid else _pending_section(reason)
    (GENERATED / "results_section.tex").write_text(section)
    (GENERATED / "results_status.json").write_text(
        json.dumps(
            {
                "validated_empirical_results": valid,
                "reason": reason,
                "source_run": str(RUN_DIR.relative_to(REPO)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        json.dumps(
            {
                "validated_empirical_results": valid,
                "generated": "paper/generated/results_section.tex",
                "reason": reason,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

from .models import Case

ANALYST_SYSTEM = """You are a careful commercial-insurance underwriting analyst.
Use only the supplied case facts and tool evidence. Do not import external rules.
Distinguish the requested line of business from existing coverage. Check business
classification, small-business eligibility, state-specific appetite, thresholds,
and exceptions when those are relevant. If evidence conflicts, state the conflict.
Do not mention that a reference answer exists."""

FINAL_JSON = """Return ONLY valid JSON with this shape:
{"answer":"a concise final answer in the requested insurance units",
 "rationale":"a short evidence-grounded explanation"}
Do not wrap the JSON in Markdown."""


def case_packet(case: Case) -> str:
    metadata = json.dumps(case.company_metadata, ensure_ascii=False, indent=2)
    user_messages = "\n".join(
        f"{index + 1}. {message}" for index, message in enumerate(case.underwriter_messages)
    )
    evidence = "\n\n".join(
        f"[Tool evidence {index + 1}]\n{chunk}" for index, chunk in enumerate(case.evidence)
    )
    return f"""TASK TYPE
{case.task}

COMPANY
{case.company_name}

STRUCTURED FACTS
{metadata}

UNDERWRITER UTTERANCES
{user_messages}

TOOL EVIDENCE
{evidence}
"""


def direct_prompt(case: Case) -> str:
    return f"""{case_packet(case)}

Solve the underwriter's request. Before answering, privately check the controlling
rule, all numeric thresholds, and whether the requested LOB differs from existing
coverage.

{FINAL_JSON}"""


def draft_prompt(case: Case) -> str:
    return f"""{case_packet(case)}

Prepare a compact draft answer. Identify the controlling classification or rule,
apply relevant thresholds, and give a proposed conclusion. This is an intermediate
draft, so prioritize substance over polish."""


def critique_prompt(case: Case, draft: str) -> str:
    return f"""{case_packet(case)}

DRAFT
{draft}

Audit the draft for a wrong NAICS/business classification, missed appetite rule,
reversed threshold, wrong state/LOB, unsupported assumption, or omission. List
only material corrections; say "no material correction" if it is sound."""


def revision_prompt(case: Case, draft: str, critique: str) -> str:
    return f"""{case_packet(case)}

DRAFT
{draft}

CRITIQUE
{critique}

Produce the final answer. Apply valid corrections, but do not change a correct
conclusion merely because a critique was requested.

{FINAL_JSON}"""


def specialist_prompt(case: Case, role: str) -> str:
    if role == "classification":
        focus = (
            "Independently solve the case with emphasis on business/NAICS "
            "classification, requested versus existing LOBs, and evidence selection."
        )
    else:
        focus = (
            "Independently solve the case with emphasis on thresholds, state-specific "
            "rules, counterexamples, and reasons the obvious answer might be wrong."
        )
    return f"""{case_packet(case)}

{focus}
Return a concise proposed answer and the decisive evidence. This is an intermediate
analysis for another model, not the final user response."""


def debate_critic_prompt(case: Case, first: str, second: str) -> str:
    return f"""{case_packet(case)}

CLASSIFICATION SPECIALIST
{first}

RULES/THRESHOLDS SPECIALIST
{second}

Act as a skeptical reviewer. Resolve disagreements by checking the supplied
evidence. Identify the best-supported conclusion and any specific error in either
analysis. Be concise."""


def debate_final_prompt(case: Case, first: str, second: str, critique: str) -> str:
    return f"""{case_packet(case)}

CLASSIFICATION SPECIALIST
{first}

RULES/THRESHOLDS SPECIALIST
{second}

REVIEWER
{critique}

Synthesize the final answer from the supplied evidence. Do not decide by majority;
choose the conclusion that follows from the controlling rule and facts.

{FINAL_JSON}"""


JUDGE_SYSTEM = """You are a strict, pointwise evaluator of commercial-insurance
underwriting answers. Compare the candidate with every accepted reference variant
and the supplied evidence. Ignore writing style and verbosity. A conclusion is
correct only if the operational decision, products/codes/limits, and material
conditions match. Do not reward a plausible but unsupported answer."""


def judge_prompt(case: Case, candidate_answer: str) -> str:
    references = "\n".join(f"- {value}" for value in case.accepted_reference_answers)
    return f"""{case_packet(case)}

ACCEPTED REFERENCE VARIANTS
{references}

CANDIDATE ANSWER
{candidate_answer}

Return ONLY valid JSON:
{{"correct":true_or_false,
  "evidence_score":integer_0_to_4,
  "unsupported_claims":true_or_false,
  "rationale":"one concise sentence"}}

Evidence score: 0=no support, 1=mostly unsupported, 2=mixed, 3=mostly grounded,
4=fully grounded in the supplied facts/rules."""

import hashlib
import json
from pathlib import Path

import pytest

from budget_crossover.diagnostics import (
    FinanceComplexCase,
    FinanceComplexCountDiscrepancy,
    FinanceComplexSnapshot,
    adapt_financecomplex_snapshot,
    audit_evidence_lineage_and_leakage,
    build_financecomplex_boundary_report,
    export_oracle_evidence_cases,
    retrieval_ladder_boundary,
    scorer_oracle_boundary,
)
from budget_crossover.retrieval import retrieve


def _record(
    record_id: str,
    *,
    question: str = "What is revenue minus expense?",
    reference_id: str = "reference-a",
) -> dict:
    return {
        "id": record_id,
        "subset": "Pro",
        "language": "English",
        "category": "Numerical-Comparison",
        "scene": "primary",
        "scope": "case",
        "split": "diagnostic",
        "question": question,
        "answer": "5",
        "derivation": "8 - 3",
        "documents": [
            {"document_id": f"distractor-{record_id}", "text": "General background 2025."},
            {"document_id": reference_id, "text": "Revenue was 8 and expense was 3."},
        ],
        "reference_document_ids": [reference_id],
    }


def _snapshot(tmp_path: Path, records: list[dict]) -> FinanceComplexSnapshot:
    path = tmp_path / "financecomplex.json"
    path.write_text(json.dumps({"records": records}, sort_keys=True), encoding="utf-8")
    return FinanceComplexSnapshot(
        path=path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_financecomplex_filters_deduplicates_and_enforces_expected_count(tmp_path: Path):
    first = _record("case-1")
    second = _record(
        "case-2",
        question="What is assets minus liabilities?",
        reference_id="reference-b",
    )
    second["answer"] = "7"
    second["derivation"] = "11 - 4"
    second["documents"][1]["text"] = "Assets were 11 and liabilities were 4."
    duplicate = _record("duplicate")
    alternate_language = _record("french")
    alternate_language["language"] = "French"
    overall = _record("overall")
    overall["scope"] = "overall"
    evaluation = _record("evaluation")
    evaluation["split"] = "evaluation"
    alternate_scene = _record("stress")
    alternate_scene["scene"] = "stress-test"
    snapshot = _snapshot(
        tmp_path,
        [first, second, duplicate, alternate_language, overall, evaluation, alternate_scene],
    )

    adapted = adapt_financecomplex_snapshot(
        snapshot,
        output_dir=tmp_path / "diagnostic",
        expected_count=2,
    )

    assert len(adapted.cases) == 2
    assert {row.reason for row in adapted.rejections} == {
        "duplicate_case",
        "alternate_language",
        "overall_or_evaluation",
        "alternate_scene",
    }
    assert {case.lineage.reference_document_ids for case in adapted.cases} == {
        ("reference-a",),
        ("reference-b",),
    }

    with pytest.raises(FinanceComplexCountDiscrepancy) as captured:
        adapt_financecomplex_snapshot(
            snapshot,
            output_dir=tmp_path / "mismatch",
            expected_count=3,
        )
    assert captured.value.report["expected_count"] == 3
    assert captured.value.report["observed_count"] == 2
    assert json.loads(
        (tmp_path / "mismatch" / "financecomplex_discrepancy.json").read_text()
    ) == captured.value.report


def test_financecomplex_rejects_operands_supported_only_by_distractor_documents(
    tmp_path: Path,
):
    distractor_only = _record("distractor-only")
    distractor_only["documents"][0]["text"] = "Revenue was 8 and expense was 3."
    distractor_only["documents"][1]["text"] = "Reference narrative without figures."

    with pytest.raises(FinanceComplexCountDiscrepancy) as captured:
        adapt_financecomplex_snapshot(
            _snapshot(tmp_path, [distractor_only]),
            output_dir=tmp_path / "distractor-only",
            expected_count=1,
        )

    assert captured.value.report["observed_count"] == 0
    assert captured.value.report["rejections"] == {"missing_evidence": 1}


def test_lineage_audit_and_oracle_export_reject_support_outside_references(tmp_path: Path):
    adapted = adapt_financecomplex_snapshot(
        _snapshot(tmp_path, [_record("case-1")]),
        output_dir=tmp_path / "diagnostic",
        expected_count=1,
    )
    case = adapted.cases[0]
    distractor_id = case.public.evidence[0].evidence_id
    reference_id = case.public.evidence[1].evidence_id
    outside_reference = FinanceComplexCase(
        public=case.public,
        hidden=case.hidden.model_copy(update={"gold_support_ids": (distractor_id,)}),
        lineage=case.lineage,
    )
    unsupported_reference = FinanceComplexCase(
        public=case.public.model_copy(
            update={
                "evidence": (
                    case.public.evidence[0].model_copy(
                        update={"text": "Revenue was 8 and expense was 3."}
                    ),
                    case.public.evidence[1].model_copy(
                        update={"text": "Reference narrative without figures."}
                    ),
                )
            }
        ),
        hidden=case.hidden.model_copy(update={"gold_support_ids": (reference_id,)}),
        lineage=case.lineage,
    )

    audit = audit_evidence_lineage_and_leakage((unsupported_reference,))

    assert audit["reference_document_linkage_rate"] == 0.0
    assert audit["pass"] is False
    with pytest.raises(ValueError, match="declared reference"):
        export_oracle_evidence_cases((outside_reference,), tmp_path / "invalid-oracle.jsonl")


def test_scorer_lineage_leakage_and_oracle_evidence_boundaries(tmp_path: Path):
    adapted = adapt_financecomplex_snapshot(
        _snapshot(tmp_path, [_record("case-1")]),
        output_dir=tmp_path / "diagnostic",
        expected_count=1,
    )

    scorer = scorer_oracle_boundary(adapted.cases)
    audit = audit_evidence_lineage_and_leakage(adapted.cases)
    leaked_payload = adapted.cases[0].public.model_dump(mode="json") | {"gold_value": "5"}
    leaked = audit_evidence_lineage_and_leakage(
        adapted.cases,
        public_payloads=(leaked_payload,),
    )
    exported = export_oracle_evidence_cases(
        adapted.cases,
        tmp_path / "oracle_evidence.jsonl",
    )

    assert scorer == {
        "total": 1,
        "gold_correct": 1,
        "gold_correct_rate": 1.0,
        "adversarial_total": 3,
        "adversarial_rejected": 3,
        "adversarial_rejection_rate": 1.0,
        "adversarial_by_field": {
            "scale": {"total": 1, "rejected": 1},
            "value": {"total": 2, "rejected": 2},
        },
        "pass": True,
    }
    assert audit["reference_document_linkage_rate"] == 1.0
    assert audit["leakage_count"] == 0
    assert audit["pass"] is True
    assert leaked["leakage_count"] == 1
    assert leaked["pass"] is False
    assert len(exported[0].evidence) == 1
    payload = json.loads((tmp_path / "oracle_evidence.jsonl").read_text())
    assert "answer" not in payload
    assert "gold_derivation" not in payload
    assert "gold_support_ids" not in payload


def test_scorer_oracle_perturbs_every_specified_answer_field(tmp_path: Path):
    record = _record("typed-label")
    record.update(
        {
            "unit": "USD",
            "scale": "million",
            "entity": "Acme Corp",
            "period": "2024",
        }
    )
    adapted = adapt_financecomplex_snapshot(
        _snapshot(tmp_path, [record]),
        output_dir=tmp_path / "typed-label",
        expected_count=1,
    )

    scorer = scorer_oracle_boundary(adapted.cases)

    assert scorer["adversarial_by_field"] == {
        "entity": {"total": 1, "rejected": 1},
        "period": {"total": 1, "rejected": 1},
        "scale": {"total": 1, "rejected": 1},
        "unit": {"total": 1, "rejected": 1},
        "value": {"total": 2, "rejected": 2},
    }
    assert scorer["adversarial_total"] == 6
    assert scorer["adversarial_rejected"] == 6
    assert scorer["pass"] is True


def test_retrieval_ladders_report_pre_and_post_reference_document_recall(tmp_path: Path):
    adapted = adapt_financecomplex_snapshot(
        _snapshot(tmp_path, [_record("case-1")]),
        output_dir=tmp_path / "diagnostic",
        expected_count=1,
    )
    case = adapted.cases[0]
    production = retrieve(
        case.public,
        ("revenue expense",),
        limit=1,
        max_chars_per_item=1000,
    )

    ladders = retrieval_ladder_boundary(
        adapted.cases,
        reference_queries={case.public.case_id: ("revenue expense",)},
        planned_queries={case.public.case_id: ("unmatched",)},
        production_results={
            tier: {case.public.case_id: production}
            for tier in ("low", "middle", "high")
        },
        tier_limits={"low": 1, "middle": 1, "high": 1},
        max_chars_per_item=1000,
    )

    assert ladders["reference"] == {
        tier: {"pre_truncation_recall": 1.0, "post_truncation_recall": 1.0}
        for tier in ("low", "middle", "high")
    }
    assert ladders["planned"] == {
        tier: {"pre_truncation_recall": 1.0, "post_truncation_recall": 0.0}
        for tier in ("low", "middle", "high")
    }
    assert ladders["production"] == {
        tier: {"pre_truncation_recall": 1.0, "post_truncation_recall": 1.0}
        for tier in ("low", "middle", "high")
    }


def test_retrieval_ladders_require_exact_case_and_tier_coverage(tmp_path: Path):
    adapted = adapt_financecomplex_snapshot(
        _snapshot(tmp_path, [_record("case-1")]),
        output_dir=tmp_path / "diagnostic",
        expected_count=1,
    )
    case = adapted.cases[0]
    case_id = case.public.case_id
    result = retrieve(case.public, ("revenue",), limit=1, max_chars_per_item=1000)
    queries = {case_id: ("revenue",)}
    tiers = {tier: {case_id: result} for tier in ("low", "middle", "high")}
    limits = {"low": 1, "middle": 1, "high": 1}

    with pytest.raises(ValueError, match="reference query case coverage"):
        retrieval_ladder_boundary(
            adapted.cases,
            reference_queries={},
            planned_queries=queries,
            production_results=tiers,
            tier_limits=limits,
            max_chars_per_item=1000,
        )
    with pytest.raises(ValueError, match="planned query case coverage"):
        retrieval_ladder_boundary(
            adapted.cases,
            reference_queries=queries,
            planned_queries=queries | {"extra": ("extra",)},
            production_results=tiers,
            tier_limits=limits,
            max_chars_per_item=1000,
        )
    with pytest.raises(ValueError, match="production tier coverage"):
        retrieval_ladder_boundary(
            adapted.cases,
            reference_queries=queries,
            planned_queries=queries,
            production_results={tier: tiers[tier] for tier in ("low", "middle")},
            tier_limits=limits,
            max_chars_per_item=1000,
        )
    with pytest.raises(ValueError, match="production tier coverage"):
        retrieval_ladder_boundary(
            adapted.cases,
            reference_queries=queries,
            planned_queries=queries,
            production_results=tiers | {"ultra": {case_id: result}},
            tier_limits=limits,
            max_chars_per_item=1000,
        )
    with pytest.raises(ValueError, match="high production case coverage"):
        retrieval_ladder_boundary(
            adapted.cases,
            reference_queries=queries,
            planned_queries=queries,
            production_results=tiers | {"high": {}},
            tier_limits=limits,
            max_chars_per_item=1000,
        )


@pytest.mark.parametrize(
    ("override", "expected_failure"),
    [
        ({"scorer": {"gold_correct_rate": 0.99, "pass": False}}, "scorer"),
        (
            {
                "audit": {
                    "reference_document_linkage_rate": 1.0,
                    "leakage_count": 1,
                    "pass": False,
                }
            },
            "lineage_leakage",
        ),
        ({"oracle": {"pass": False}}, "model_with_oracle_evidence"),
        (
            {
                "retrieval": {
                    "production": {
                        "high": {
                            "pre_truncation_recall": 1.0,
                            "post_truncation_recall": 0.94,
                        }
                    }
                }
            },
            "retrieval",
        ),
        ({"orchestration": {"pass": False}}, "orchestration"),
    ],
)
def test_boundary_report_attributes_each_failure_without_confirmation_pooling(
    tmp_path: Path,
    override: dict,
    expected_failure: str,
):
    values = {
        "scorer": {"gold_correct_rate": 1.0, "pass": True},
        "audit": {
            "reference_document_linkage_rate": 1.0,
            "leakage_count": 0,
            "pass": True,
        },
        "oracle": {"pass": True},
        "retrieval": {
            "production": {
                "high": {
                    "pre_truncation_recall": 1.0,
                    "post_truncation_recall": 0.95,
                }
            }
        },
        "orchestration": {"pass": True},
    }
    values.update(override)

    report = build_financecomplex_boundary_report(
        scorer=values["scorer"],
        lineage_leakage=values["audit"],
        oracle_evidence_model=values["oracle"],
        retrieval=values["retrieval"],
        orchestration=values["orchestration"],
        output_path=tmp_path / f"{expected_failure}.json",
    )

    assert report["primary_failure"] == expected_failure
    assert expected_failure in report["failures"]
    assert report["confirmation_pool_eligible"] is False
    assert json.loads((tmp_path / f"{expected_failure}.json").read_text()) == report


def test_boundary_run_gate_uses_the_preregistered_thresholds():
    report = build_financecomplex_boundary_report(
        scorer={"gold_correct_rate": 1.0, "pass": True},
        lineage_leakage={
            "reference_document_linkage_rate": 1.0,
            "leakage_count": 0,
            "pass": True,
        },
        oracle_evidence_model={"pass": True},
        retrieval={
            "production": {
                "low": {
                    "pre_truncation_recall": 1.0,
                    "post_truncation_recall": 0.0,
                },
                "middle": {
                    "pre_truncation_recall": 1.0,
                    "post_truncation_recall": 0.0,
                },
                "high": {
                    "pre_truncation_recall": 1.0,
                    "post_truncation_recall": 0.95,
                },
            }
        },
        orchestration={"pass": True},
    )

    assert report["exploratory_system_run_gate"] is True
    assert report["failures"] == []
    assert report["primary_failure"] is None
    assert report["domain_role"] == "exploratory_only"

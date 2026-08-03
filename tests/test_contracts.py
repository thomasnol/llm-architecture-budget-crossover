from decimal import Decimal

import pytest
from pydantic import ValidationError

from budget_crossover.models import (
    AnswerSpec,
    CallEvent,
    Candidate,
    CellResult,
    CheckFinding,
    CheckResult,
    EvidenceItem,
    HiddenLabel,
    MechanismTrace,
    PublicCase,
    Reservation,
    RunManifest,
    Usage,
)


def test_public_and_hidden_contracts_are_strict_immutable_and_separate():
    evidence = EvidenceItem(
        evidence_id="doc-1:row-2",
        document_id="doc-1",
        kind="table_row",
        text="Revenue | 2024 | $1.2 million",
        headers=("Metric", "Period", "Value"),
        row_label="Revenue",
        unit="USD",
        scale="million",
        period="2024",
        ordinal=2,
    )
    public = PublicCase(
        case_id="case-1",
        dataset="finqa",
        document_id="doc-1",
        question="What was revenue?",
        evidence=(evidence,),
        stratum="headroom",
        metadata={"company": "Example Corp", "tags": ["reported"]},
    )
    label = HiddenLabel(
        case_id="case-1",
        answer=AnswerSpec(
            value=Decimal("1.2"),
            unit="USD",
            scale="million",
            entity="Example Corp",
            period="2024",
            absolute_tolerance=Decimal("0.01"),
            relative_tolerance=Decimal(0),
        ),
        gold_derivation="1.2",
        gold_support_ids=("doc-1:row-2",),
        source_lineage=("finqa/train.json", "doc-1", "q-7"),
    )
    candidate = Candidate(
        value="$1.2",
        unit="USD",
        scale="million",
        entity="Example Corp",
        period="2024",
        expression="1.2",
        citations=("doc-1:row-2",),
    )

    assert public.evidence == (evidence,)
    assert public.metadata.tags == ("reported",)
    assert not hasattr(public, "answer")
    assert label.answer.value == Decimal("1.2")
    assert candidate.citations == ("doc-1:row-2",)
    with pytest.raises(ValidationError):
        public.question = "mutated"
    with pytest.raises(ValidationError):
        public.metadata.company = "mutated"
    with pytest.raises(AttributeError):
        public.metadata.tags.append("mutated")
    with pytest.raises(ValidationError):
        Candidate(
            value="1",
            unit=None,
            scale="ones",
            entity=None,
            period=None,
            expression=None,
            citations=(),
            explanation="not in the strict schema",
        )


@pytest.mark.parametrize("field", ["absolute_tolerance", "relative_tolerance"])
def test_answer_spec_rejects_negative_tolerances(field):
    values = {
        "value": "1",
        "unit": None,
        "scale": "ones",
        "entity": None,
        "period": None,
        "absolute_tolerance": "0",
        "relative_tolerance": "0",
    }
    values[field] = "-0.01"

    with pytest.raises(ValidationError):
        AnswerSpec(**values)


def test_execution_primitives_are_validated_and_frozen():
    usage = Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13)
    reservation = Reservation(
        reservation_id="reservation-1",
        prompt_tokens=10,
        max_output_tokens=8,
    )
    event = CallEvent(stage="candidate", reservation=reservation, usage=usage)
    finding = CheckFinding(
        code="fabricated_citation",
        message="Citation missing",
        evidence_ids=("missing",),
    )
    check = CheckResult(passed=False, findings=(finding,))
    trace = MechanismTrace(
        planned_queries=("revenue 2024",),
        actual_queries=("revenue 2024",),
        query_hashes=("abc",),
        retrieval_pre_truncation_ids=("e1", "e2"),
        retrieval_post_truncation_ids=("e1",),
        candidate_token_cap=256,
        candidate_count=1,
        checks=(check,),
        call_events=(event,),
        realized_tokens=13,
        exit_reason="accepted",
    )
    cell = CellResult(
        case_id="case-1",
        system="verified_search",
        tier="low",
        repetition=0,
        status="ok",
        candidate=None,
        trace=trace,
    )
    manifest = RunManifest(
        run_id="run-1",
        resolved_config={"model": "gpt-5.4-mini"},
        artifact_hashes={"public_cases": "sha256:abc"},
        expected_cell_keys=("case-1:verified_search:low:0",),
    )

    assert reservation.reserved_tokens == 18
    assert usage.authoritative_total == 13
    assert cell.trace.call_events == (event,)
    with pytest.raises(ValidationError):
        manifest.run_id = "mutated"
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=-1, completion_tokens=0, total_tokens=-1)
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=10, completion_tokens=3, total_tokens=14)


def test_public_case_rejects_label_metadata_and_unstable_evidence_identity():
    item = EvidenceItem(
        evidence_id="e1",
        document_id="doc-1",
        kind="text",
        text="Public evidence",
        ordinal=0,
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        PublicCase(
            case_id="case-1",
            dataset="finqa",
            document_id="doc-1",
            question="Question?",
            evidence=(item,),
            stratum="headroom",
            metadata={"gold_answer": "42"},
        )
    with pytest.raises(ValidationError, match="unique"):
        PublicCase(
            case_id="case-1",
            dataset="finqa",
            document_id="doc-1",
            question="Question?",
            evidence=(item, item),
            stratum="headroom",
        )
    with pytest.raises(ValidationError, match="document_id"):
        PublicCase(
            case_id="case-1",
            dataset="finqa",
            document_id="other-doc",
            question="Question?",
            evidence=(item,),
            stratum="headroom",
        )


def test_public_metadata_is_typed_allowlisted_and_copy_updates_revalidate():
    public = PublicCase(
        case_id="case-1",
        dataset="finqa",
        document_id="doc-1",
        question="Question?",
        evidence=(),
        stratum="headroom",
        metadata={"company": "Example Corp", "tags": ["annual", "reported"]},
    )

    assert public.metadata.company == "Example Corp"
    assert public.metadata.tags == ("annual", "reported")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PublicCase(
            case_id="case-2",
            dataset="finqa",
            document_id="doc-2",
            question="Question?",
            evidence=(),
            stratum="headroom",
            metadata={"reference_answer": "42", "target": {"value": "42"}},
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        public.model_copy(update={"metadata": {"reference_answer": "42"}})


def test_run_manifest_maps_are_deeply_immutable():
    manifest = RunManifest(
        run_id="run-1",
        resolved_config={"systems": ["monolith", "verified_search"]},
        artifact_hashes={"cases": "sha256:abc"},
        expected_cell_keys=(),
    )

    assert manifest.resolved_config["systems"] == ("monolith", "verified_search")
    with pytest.raises(TypeError):
        manifest.artifact_hashes["cases"] = "sha256:changed"
    with pytest.raises(AttributeError):
        manifest.resolved_config["systems"].append("unverified_search")

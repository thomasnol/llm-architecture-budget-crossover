import pytest
from pydantic import ValidationError

from budget_crossover.models import EvidenceItem, PublicCase
from budget_crossover.retrieval import retrieve


def _case() -> PublicCase:
    return PublicCase(
        case_id="case-1",
        dataset="tatqa",
        document_id="doc-1",
        question="Compare revenue and profit.",
        evidence=(
            EvidenceItem(
                evidence_id="row-b",
                document_id="doc-1",
                kind="table_row",
                text="Revenue was $1.2 million in fiscal year 2024.",
                table_id="table-1",
                headers=("Metric", "FY 2024"),
                row_label="Revenue",
                unit="USD",
                scale="million",
                entity="Example Corp",
                period="FY 2024",
                ordinal=7,
            ),
            EvidenceItem(
                evidence_id="row-a",
                document_id="doc-1",
                kind="table_row",
                text="Revenue was $1.1 million in fiscal year 2023.",
                table_id="table-1",
                headers=("Metric", "FY 2023"),
                row_label="Revenue",
                unit="USD",
                scale="million",
                entity="Example Corp",
                period="FY 2023",
                ordinal=1,
            ),
            EvidenceItem(
                evidence_id="profit",
                document_id="doc-1",
                kind="text",
                text="Profit increased in the current reporting period.",
                entity="Example Corp",
                period="FY 2024",
                ordinal=3,
            ),
            EvidenceItem(
                evidence_id="unmatched",
                document_id="doc-1",
                kind="text",
                text="The board approved a dividend.",
                ordinal=2,
            ),
        ),
        stratum="headroom",
    )


def test_query_union_ranking_is_deterministic_with_stable_ties_and_explicit_ids():
    first = retrieve(_case(), ("profit", "revenue"), limit=3, max_chars_per_item=200)
    reordered = retrieve(_case(), ("revenue", "profit"), limit=3, max_chars_per_item=200)

    assert first == reordered
    assert first.pre_truncation_ids == ("row-a", "profit", "row-b", "unmatched")
    assert first.post_truncation_ids == ("row-a", "profit", "row-b")
    assert tuple(item.evidence_id for item in first.items) == first.post_truncation_ids


def test_per_item_truncation_preserves_table_header_unit_period_and_identity_metadata():
    result = retrieve(_case(), ("revenue 2024",), limit=1, max_chars_per_item=18)

    item = result.items[0]
    assert item.evidence_id == "row-b"
    assert item.text == "Revenue was $1.2 …"
    assert item.table_id == "table-1"
    assert item.headers == ("Metric", "FY 2024")
    assert item.row_label == "Revenue"
    assert item.unit == "USD"
    assert item.scale == "million"
    assert item.entity == "Example Corp"
    assert item.period == "FY 2024"
    assert item.ordinal == 7


def test_empty_or_unmatched_queries_fall_back_to_stable_corpus_order():
    empty = retrieve(_case(), (), limit=4, max_chars_per_item=200)
    unmatched = retrieve(_case(), ("nonexistent-token",), limit=4, max_chars_per_item=200)

    assert empty.pre_truncation_ids == ("row-a", "unmatched", "profit", "row-b")
    assert unmatched.items == empty.items
    assert unmatched.pre_truncation_ids == empty.pre_truncation_ids
    assert unmatched.post_truncation_ids == empty.post_truncation_ids
    assert unmatched.query_hash != empty.query_hash


@pytest.mark.parametrize(
    ("limit", "max_chars"),
    [(-1, 100), (1, 0)],
)
def test_retrieval_rejects_invalid_limits(limit, max_chars):
    with pytest.raises(ValueError):
        retrieve(_case(), ("revenue",), limit=limit, max_chars_per_item=max_chars)


def test_retrieval_result_binds_immutable_tier_limit_query_and_public_input_provenance():
    result = retrieve(
        _case(),
        ("profit", "revenue"),
        limit=3,
        max_chars_per_item=200,
        tier_id="high",
    )

    assert result.tier_id == "high"
    assert result.requested_k == 3
    assert len(result.query_hash) == 64
    assert len(result.input_hash) == 64
    with pytest.raises(ValidationError, match="requested_k"):
        result.model_copy(update={"requested_k": 1})

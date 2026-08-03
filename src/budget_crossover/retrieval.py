from __future__ import annotations

import re
from collections.abc import Sequence

from .models import EvidenceItem, FrozenModel, PublicCase

_TOKEN = re.compile(r"[a-z0-9]+")


class RetrievalResult(FrozenModel):
    items: tuple[EvidenceItem, ...]
    pre_truncation_ids: tuple[str, ...]
    post_truncation_ids: tuple[str, ...]


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(value.casefold()))


def _searchable_text(item: EvidenceItem) -> str:
    values = (
        item.text,
        *item.headers,
        item.row_label or "",
        item.unit or "",
        item.scale or "",
        item.entity or "",
        item.period or "",
    )
    return " ".join(values)


def _rank(case: PublicCase, queries: Sequence[str]) -> tuple[EvidenceItem, ...]:
    query_tokens = {_tokens(query) for query in queries}
    query_tokens.discard(frozenset())
    if not query_tokens:
        return tuple(sorted(case.evidence, key=lambda item: (item.ordinal, item.evidence_id)))

    def key(item: EvidenceItem) -> tuple[int, int, int, int, str]:
        item_tokens = _tokens(_searchable_text(item))
        overlaps = [len(item_tokens & terms) for terms in query_tokens]
        best_overlap = max(overlaps, default=0)
        matching_queries = sum(overlap > 0 for overlap in overlaps)
        total_overlap = sum(overlaps)
        return (-best_overlap, -matching_queries, -total_overlap, item.ordinal, item.evidence_id)

    ranked = tuple(sorted(case.evidence, key=key))
    if all(not (_tokens(_searchable_text(item)) & set().union(*query_tokens)) for item in ranked):
        return tuple(sorted(case.evidence, key=lambda item: (item.ordinal, item.evidence_id)))
    return ranked


def _truncate(item: EvidenceItem, max_chars: int) -> EvidenceItem:
    if len(item.text) <= max_chars:
        return item
    truncated = "…" if max_chars == 1 else f"{item.text[: max_chars - 1]}…"
    return item.model_copy(update={"text": truncated})


def retrieve(
    case: PublicCase,
    queries: Sequence[str],
    *,
    limit: int,
    max_chars_per_item: int,
) -> RetrievalResult:
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if max_chars_per_item < 1:
        raise ValueError("max_chars_per_item must be positive")

    ranked = _rank(case, queries)
    selected = tuple(_truncate(item, max_chars_per_item) for item in ranked[:limit])
    return RetrievalResult(
        items=selected,
        pre_truncation_ids=tuple(item.evidence_id for item in ranked),
        post_truncation_ids=tuple(item.evidence_id for item in selected),
    )

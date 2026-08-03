from __future__ import annotations

"""FinanceComplexQA exploratory adapter and prerequisite diagnostic boundaries."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .dataset import (
    DerivationError,
    _answer_spec,
    _locate_operands,
    execute_safe_derivation,
    sha256_file,
)
from .models import EvidenceItem, HiddenLabel, PublicCase
from .retrieval import (
    RetrievalResult,
    retrieval_input_hash,
    retrieval_query_hash,
    retrieve,
)
from .scoring import score_candidate, serialize_gold_oracle

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "accepted_reference_answers",
    "canonical_reference",
    "derivation",
    "gold_derivation",
    "gold_support_ids",
    "program",
    "reference_answer",
    "source_lineage",
    "target",
}


@dataclass(frozen=True)
class FinanceComplexSnapshot:
    path: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class FinanceComplexLineage:
    source: str
    record_id: str
    reference_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class FinanceComplexCase:
    public: PublicCase
    hidden: HiddenLabel
    lineage: FinanceComplexLineage


@dataclass(frozen=True)
class FinanceComplexRejection:
    record_id: str
    reason: str
    detail: str | None = None


@dataclass(frozen=True)
class FinanceComplexAdaptation:
    cases: tuple[FinanceComplexCase, ...]
    rejections: tuple[FinanceComplexRejection, ...]
    source_sha256: str
    artifact_hashes: Mapping[str, str]


class FinanceComplexCountDiscrepancy(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__("FinanceComplexQA canonical count differs from the pinned expectation")


def _normalized_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _stable_digest(*values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid local FinanceComplexQA snapshot: {path}") from error
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("data"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("FinanceComplexQA snapshot must contain a JSON list of records")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                payload = row.model_dump(mode="json")
            else:
                payload = asdict(row)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _scope_rejection(row: Mapping[str, Any]) -> str | None:
    if _normalized_token(row.get("subset", "")) != "pro":
        return "outside_pro_subset"
    if _normalized_token(row.get("language", "")) not in {"english", "en"}:
        return "alternate_language"
    question_type = row.get("category", row.get("question_type", row.get("task", "")))
    if _normalized_token(question_type) != "numerical comparison":
        return "outside_numerical_comparison"
    if (
        _normalized_token(row.get("scope", "")) == "overall"
        or _normalized_token(row.get("split", "")) == "evaluation"
        or row.get("evaluation") is True
    ):
        return "overall_or_evaluation"
    scene = _normalized_token(row.get("scene", ""))
    if scene not in {"", "primary", "original", "default"}:
        return "alternate_scene"
    return None


def _reference_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = row.get("reference_document_ids", row.get("reference_documents", ()))
    values: list[str] = []
    if isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, dict):
                value = item.get("document_id", item.get("id"))
            else:
                value = item
            if value is not None and str(value).strip():
                values.append(str(value).strip())
    return tuple(dict.fromkeys(values))


def _documents(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    payload = row.get("documents", ())
    documents: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        payload = [
            {"document_id": document_id, "text": text}
            for document_id, text in payload.items()
        ]
    if not isinstance(payload, list):
        return ()
    for index, document in enumerate(payload):
        if isinstance(document, str):
            document_id = f"document-{index}"
            text = document
        elif isinstance(document, dict):
            document_id = str(
                document.get("document_id") or document.get("id") or f"document-{index}"
            )
            text = str(document.get("text", document.get("content", "")))
        else:
            continue
        documents.append((document_id, text))
    return tuple(documents)


def _adapt_record(snapshot: FinanceComplexSnapshot, row: Mapping[str, Any]) -> FinanceComplexCase:
    record_id = str(row.get("id") or row.get("uid") or _stable_digest(row)[:16])
    reference_ids = _reference_ids(row)
    if not reference_ids:
        raise DerivationError("missing_reference_document")
    documents = _documents(row)
    available_ids = {document_id for document_id, _ in documents}
    if not set(reference_ids).issubset(available_ids):
        raise DerivationError("missing_reference_document")
    combined_document_id = f"financecomplex-{_stable_digest(record_id, reference_ids)[:20]}"
    evidence = tuple(
        EvidenceItem(
            evidence_id=f"{combined_document_id}:source:{index}",
            document_id=combined_document_id,
            kind="reference_document" if document_id in reference_ids else "document",
            text=text,
            table_id=document_id,
            ordinal=index,
        )
        for index, (document_id, text) in enumerate(documents)
    )
    derivation = execute_safe_derivation(str(row.get("derivation", "")))
    reference_evidence = tuple(
        item for item in evidence if item.table_id in set(reference_ids)
    )
    support_ids = _locate_operands(derivation.operands, reference_evidence)
    answer = _answer_spec(row)
    if derivation.value != answer.value:
        raise DerivationError("program_answer_mismatch")
    question = str(row.get("question", "")).strip()
    if not question:
        raise DerivationError("missing_question")
    case_id = f"financecomplex-{_stable_digest(question, reference_ids)[:20]}"
    public = PublicCase(
        case_id=case_id,
        dataset="financecomplexqa",
        document_id=combined_document_id,
        question=question,
        evidence=evidence,
        stratum="diagnostic",
        metadata={
            "language": "English",
            "tags": ("Pro", "Numerical-Comparison", "exploratory"),
        },
    )
    hidden = HiddenLabel(
        case_id=case_id,
        answer=answer,
        gold_derivation=derivation.expression,
        gold_support_ids=support_ids,
        source_lineage=(snapshot.path.name, record_id, *reference_ids),
    )
    return FinanceComplexCase(
        public=public,
        hidden=hidden,
        lineage=FinanceComplexLineage(
            source=snapshot.path.name,
            record_id=record_id,
            reference_document_ids=reference_ids,
        ),
    )


def adapt_financecomplex_snapshot(
    snapshot: FinanceComplexSnapshot,
    *,
    output_dir: Path,
    expected_count: int = 113,
) -> FinanceComplexAdaptation:
    """Prepare only the pinned Pro/English/Numerical-Comparison diagnostic subset."""
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not snapshot.path.is_file():
        raise FileNotFoundError(f"pinned FinanceComplexQA snapshot is missing: {snapshot.path}")
    observed_hash = sha256_file(snapshot.path)
    if observed_hash != snapshot.expected_sha256:
        raise ValueError(
            f"FinanceComplexQA snapshot checksum mismatch: expected {snapshot.expected_sha256}, "
            f"observed {observed_hash}"
        )

    cases: list[FinanceComplexCase] = []
    rejections: list[FinanceComplexRejection] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for row in _read_records(snapshot.path):
        record_id = str(row.get("id") or row.get("uid") or _stable_digest(row)[:16])
        scope_reason = _scope_rejection(row)
        if scope_reason is not None:
            rejections.append(FinanceComplexRejection(record_id, scope_reason))
            continue
        identity = (_normalized_token(row.get("question", "")), tuple(sorted(_reference_ids(row))))
        if identity in seen:
            rejections.append(FinanceComplexRejection(record_id, "duplicate_case"))
            continue
        seen.add(identity)
        try:
            cases.append(_adapt_record(snapshot, row))
        except DerivationError as error:
            rejections.append(FinanceComplexRejection(record_id, error.reason))

    cases.sort(key=lambda value: value.public.case_id)
    rejections.sort(key=lambda value: (value.record_id, value.reason))
    if len(cases) != expected_count:
        report = {
            "status": "aborted",
            "reason": "canonical_count_discrepancy",
            "expected_count": expected_count,
            "observed_count": len(cases),
            "source_sha256": observed_hash,
            "rejections": dict(sorted(Counter(row.reason for row in rejections).items())),
        }
        _write_json(output_dir / "financecomplex_discrepancy.json", report)
        raise FinanceComplexCountDiscrepancy(report)

    paths = {
        "public_cases.jsonl": tuple(case.public for case in cases),
        "hidden_labels.jsonl": tuple(case.hidden for case in cases),
        "lineage.jsonl": tuple(case.lineage for case in cases),
        "rejections.jsonl": tuple(rejections),
    }
    for filename, rows in paths.items():
        _write_jsonl(output_dir / filename, rows)
    artifact_hashes = {
        filename: sha256_file(output_dir / filename) for filename in sorted(paths)
    }
    _write_json(
        output_dir / "hashes.json",
        {"source_sha256": observed_hash, "artifact_hashes": artifact_hashes},
    )
    return FinanceComplexAdaptation(
        cases=tuple(cases),
        rejections=tuple(rejections),
        source_sha256=observed_hash,
        artifact_hashes=artifact_hashes,
    )


def scorer_oracle_boundary(cases: Sequence[FinanceComplexCase]) -> dict[str, Any]:
    gold_correct = 0
    adversarial_rejected = 0
    adversarial_total = 0
    adversarial_by_field: dict[str, dict[str, int]] = {}

    def record(field: str, rejected: bool) -> None:
        nonlocal adversarial_rejected, adversarial_total
        counts = adversarial_by_field.setdefault(field, {"total": 0, "rejected": 0})
        counts["total"] += 1
        adversarial_total += 1
        if rejected:
            counts["rejected"] += 1
            adversarial_rejected += 1

    for case in cases:
        oracle = serialize_gold_oracle(case.public, case.hidden)
        if score_candidate(oracle, case.hidden.answer).correct:
            gold_correct += 1
        delta = max(Decimal(1), abs(case.hidden.answer.value) + Decimal(1))
        for value in (
            case.hidden.answer.value + delta,
            case.hidden.answer.value - delta,
        ):
            adversarial = oracle.model_copy(update={"value": format(value, "f")})
            record("value", not score_candidate(adversarial, case.hidden.answer).correct)
        alternate_scale = "thousand" if oracle.scale == "ones" else "ones"
        adversarial = oracle.model_copy(update={"scale": alternate_scale})
        record("scale", not score_candidate(adversarial, case.hidden.answer).correct)
        if oracle.unit is not None:
            alternate_unit = "EUR" if oracle.unit.casefold() != "eur" else "USD"
            adversarial = oracle.model_copy(update={"unit": alternate_unit})
            record("unit", not score_candidate(adversarial, case.hidden.answer).correct)
        if oracle.entity is not None:
            adversarial = oracle.model_copy(update={"entity": "__adversarial_entity__"})
            record("entity", not score_candidate(adversarial, case.hidden.answer).correct)
        if oracle.period is not None:
            adversarial = oracle.model_copy(update={"period": "__adversarial_period__"})
            record("period", not score_candidate(adversarial, case.hidden.answer).correct)
    total = len(cases)
    gold_rate = gold_correct / total if total else 0.0
    adversarial_rate = (
        adversarial_rejected / adversarial_total if adversarial_total else 0.0
    )
    return {
        "total": total,
        "gold_correct": gold_correct,
        "gold_correct_rate": gold_rate,
        "adversarial_total": adversarial_total,
        "adversarial_rejected": adversarial_rejected,
        "adversarial_rejection_rate": adversarial_rate,
        "adversarial_by_field": dict(sorted(adversarial_by_field.items())),
        "pass": bool(total) and gold_rate == 1.0 and adversarial_rate == 1.0,
    }


def _leakage_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            normalized_key = _normalized_token(key).replace(" ", "_")
            if (
                normalized_key in _FORBIDDEN_PUBLIC_KEYS
                or normalized_key.startswith("gold_")
                or normalized_key.endswith("_answer")
            ):
                paths.append(path)
            paths.extend(_leakage_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_leakage_paths(item, f"{prefix}[{index}]"))
    return paths


def audit_evidence_lineage_and_leakage(
    cases: Sequence[FinanceComplexCase],
    *,
    public_payloads: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    linked = 0
    missing_links: dict[str, list[str]] = {}
    for case in cases:
        by_id = {item.evidence_id: item for item in case.public.evidence}
        observed = {item.table_id for item in case.public.evidence if item.table_id is not None}
        references = set(case.lineage.reference_document_ids)
        missing = sorted(references - observed)
        reference_evidence = tuple(
            item for item in case.public.evidence if item.table_id in references
        )
        invalid_support = [
            evidence_id
            for evidence_id in case.hidden.gold_support_ids
            if evidence_id not in by_id or by_id[evidence_id].table_id not in references
        ]
        if invalid_support:
            missing.extend(f"support:{evidence_id}" for evidence_id in invalid_support)
        try:
            derivation = execute_safe_derivation(case.hidden.gold_derivation)
            located_support = _locate_operands(derivation.operands, reference_evidence)
        except DerivationError:
            missing.append("required_operand_support")
        else:
            if not set(located_support).issubset(case.hidden.gold_support_ids):
                missing.append("required_operand_support")
        if missing:
            missing_links[case.public.case_id] = missing
        else:
            linked += 1
    payloads = public_payloads or tuple(
        case.public.model_dump(mode="json") for case in cases
    )
    leakages = [
        {"payload_index": index, "path": path}
        for index, payload in enumerate(payloads)
        for path in _leakage_paths(payload)
    ]
    total = len(cases)
    linkage_rate = linked / total if total else 0.0
    return {
        "total": total,
        "reference_document_linked": linked,
        "reference_document_linkage_rate": linkage_rate,
        "missing_reference_links": missing_links,
        "leakage_count": len(leakages),
        "leakages": leakages,
        "pass": bool(total) and linkage_rate == 1.0 and not leakages,
    }


def export_oracle_evidence_cases(
    cases: Sequence[FinanceComplexCase], path: Path
) -> tuple[PublicCase, ...]:
    exported: list[PublicCase] = []
    for case in cases:
        by_id = {item.evidence_id: item for item in case.public.evidence}
        missing = set(case.hidden.gold_support_ids) - set(by_id)
        if missing:
            raise ValueError(f"oracle evidence support IDs are missing: {sorted(missing)}")
        references = set(case.lineage.reference_document_ids)
        outside_references = [
            evidence_id
            for evidence_id in case.hidden.gold_support_ids
            if by_id[evidence_id].table_id not in references
        ]
        if outside_references:
            raise ValueError(
                "oracle evidence must come from declared reference documents: "
                f"{outside_references}"
            )
        evidence = tuple(by_id[evidence_id] for evidence_id in case.hidden.gold_support_ids)
        exported.append(case.public.model_copy(update={"evidence": evidence}))
    _write_jsonl(path, tuple(exported))
    return tuple(exported)


def _reference_recall(
    cases: Sequence[FinanceComplexCase],
    results: Mapping[str, RetrievalResult],
    field: str,
) -> float:
    found = 0
    total = 0
    for case in cases:
        by_id = {item.evidence_id: item for item in case.public.evidence}
        evidence_ids = getattr(results[case.public.case_id], field)
        retrieved_documents = {
            by_id[evidence_id].table_id for evidence_id in evidence_ids if evidence_id in by_id
        }
        references = set(case.lineage.reference_document_ids)
        found += len(references & retrieved_documents)
        total += len(references)
    return found / total if total else 0.0


def retrieval_ladder_boundary(
    cases: Sequence[FinanceComplexCase],
    *,
    reference_queries: Mapping[str, Sequence[str]],
    planned_queries: Mapping[str, Sequence[str]],
    production_queries: Mapping[str, Mapping[str, Sequence[str]]],
    production_results: Mapping[str, Mapping[str, RetrievalResult]],
    tier_limits: Mapping[str, int],
    max_chars_per_item: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    case_ids = {case.public.case_id for case in cases}
    tiers = {"low", "middle", "high"}
    if set(reference_queries) != case_ids:
        raise ValueError("reference query case coverage must match diagnostic cases exactly")
    if set(planned_queries) != case_ids:
        raise ValueError("planned query case coverage must match diagnostic cases exactly")
    if set(production_results) != tiers:
        raise ValueError("production tier coverage must be exactly low, middle, and high")
    if set(production_queries) != tiers:
        raise ValueError("production query tier coverage must be exactly low, middle, and high")
    if set(tier_limits) != tiers or any(value < 1 for value in tier_limits.values()):
        raise ValueError("tier limits must define positive low, middle, and high limits")
    for tier in sorted(tiers):
        if set(production_results[tier]) != case_ids:
            raise ValueError(f"{tier} production case coverage must match cases exactly")
        if set(production_queries[tier]) != case_ids:
            raise ValueError(f"{tier} production query case coverage must match cases exactly")
    cases_by_id = {case.public.case_id: case for case in cases}
    for tier in sorted(tiers):
        expected_k = tier_limits[tier]
        for case_id, result in production_results[tier].items():
            if (
                result.tier_id != tier
                or result.requested_k != expected_k
                or result.query_hash != retrieval_query_hash(production_queries[tier][case_id])
                or result.input_hash != retrieval_input_hash(cases_by_id[case_id].public)
            ):
                raise ValueError(
                    f"{tier} retrieval provenance must match its case, tier, and requested_k"
                )

    query_ladders: dict[str, dict[str, dict[str, RetrievalResult]]] = {
        "reference": {},
        "planned": {},
    }
    for ladder_name, queries in (
        ("reference", reference_queries),
        ("planned", planned_queries),
    ):
        for tier in sorted(tiers):
            query_ladders[ladder_name][tier] = {
                case.public.case_id: retrieve(
                    case.public,
                    queries[case.public.case_id],
                    limit=tier_limits[tier],
                    max_chars_per_item=max_chars_per_item,
                    tier_id=tier,
                )
                for case in cases
            }
    ladders: dict[str, Mapping[str, Mapping[str, RetrievalResult]]] = {
        **query_ladders,
        "production": production_results,
    }
    return {
        name: {
            tier: {
                "tier_id": tier,
                "requested_k": tier_limits[tier],
                "provenance_validated": True,
                "pre_truncation_recall": _reference_recall(
                    cases, results, "pre_truncation_ids"
                ),
                "post_truncation_recall": _reference_recall(
                    cases, results, "post_truncation_ids"
                ),
            }
            for tier, results in tier_results.items()
        }
        for name, tier_results in ladders.items()
    }


def build_financecomplex_boundary_report(
    *,
    scorer: Mapping[str, Any],
    lineage_leakage: Mapping[str, Any],
    oracle_evidence_model: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    orchestration: Mapping[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    high_retrieval = retrieval.get("production", {}).get("high", {})
    high_provenance_validated = (
        high_retrieval.get("provenance_validated") is True
        and high_retrieval.get("tier_id") == "high"
        and type(high_retrieval.get("requested_k")) is int
        and high_retrieval["requested_k"] > 0
    )
    production_recall = (
        float(high_retrieval.get("post_truncation_recall", 0.0))
        if high_provenance_validated
        else 0.0
    )
    failures: list[str] = []
    if float(scorer.get("gold_correct_rate", 0.0)) != 1.0 or not scorer.get("pass", False):
        failures.append("scorer")
    if (
        float(lineage_leakage.get("reference_document_linkage_rate", 0.0)) != 1.0
        or int(lineage_leakage.get("leakage_count", 0)) != 0
        or not lineage_leakage.get("pass", False)
    ):
        failures.append("lineage_leakage")
    if not oracle_evidence_model.get("pass", False):
        failures.append("model_with_oracle_evidence")
    if not high_provenance_validated or production_recall < 0.95:
        failures.append("retrieval")
    if not orchestration.get("pass", False):
        failures.append("orchestration")
    system_run_gate = (
        float(scorer.get("gold_correct_rate", 0.0)) == 1.0
        and float(lineage_leakage.get("reference_document_linkage_rate", 0.0)) == 1.0
        and int(lineage_leakage.get("leakage_count", 0)) == 0
        and high_provenance_validated
        and production_recall >= 0.95
    )
    report = {
        "schema_version": 1,
        "domain_role": "exploratory_only",
        "confirmation_pool_eligible": False,
        "exploratory_system_run_gate": system_run_gate,
        "failures": failures,
        "primary_failure": failures[0] if failures else None,
        "boundaries": {
            "scorer": dict(scorer),
            "lineage_leakage": dict(lineage_leakage),
            "model_with_oracle_evidence": dict(oracle_evidence_model),
            "retrieval": dict(retrieval),
            "orchestration": dict(orchestration),
        },
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report

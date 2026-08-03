from __future__ import annotations

"""Deterministic offline preparation for the canonical FinQA and TAT-QA cases."""

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from .models import AnswerSpec, DescriptiveMetadata, EvidenceItem, HiddenLabel, PublicCase
from .retrieval import retrieve
from .scoring import extract_strict_numeric_values

DatasetName = Literal["finqa", "tatqa"]

_DECIMAL_LITERAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_FINQA_STEP = re.compile(r"(?P<operation>[a-z_]+)\((?P<arguments>.*)\)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KNOWN_SCALES = {
    "": "ones",
    "one": "ones",
    "ones": "ones",
    "thousand": "thousand",
    "thousands": "thousand",
    "million": "million",
    "millions": "million",
    "billion": "billion",
    "billions": "billion",
    "percent": "percent",
    "percentage": "percent",
    "%": "percent",
}
_CURRENCY_UNITS = {"$": "USD", "€": "EUR", "£": "GBP"}


class DerivationError(ValueError):
    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        super().__init__(detail or reason)


@dataclass(frozen=True)
class DerivationResult:
    value: Decimal
    expression: str
    operands: tuple[Decimal, ...]
    operation_count: int


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset: DatasetName
    path: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class AdaptedCase:
    public: PublicCase
    hidden: HiddenLabel


@dataclass(frozen=True)
class RejectionRecord:
    dataset: str
    source: str
    document_id: str
    question_id: str | None
    reason: str
    detail: str | None = None


@dataclass(frozen=True)
class AdaptationResult:
    cases: tuple[AdaptedCase, ...]
    rejections: tuple[RejectionRecord, ...]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class SplitQuotas:
    development: int = 100
    operational_pilot: int = 60
    main: int = 1000
    easy_reserve: int = 100

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("split quotas must be nonnegative")
        if any(value % 2 for value in values.values()):
            raise ValueError("balanced FinQA/TAT-QA split quotas must be even")
        if self.main > 1000:
            raise ValueError("the hard main split cannot exceed 1,000 cases")


@dataclass(frozen=True)
class PreparedSplit:
    public_cases: tuple[PublicCase, ...]
    hidden_labels: tuple[HiddenLabel, ...]


@dataclass(frozen=True)
class PreparationResult:
    splits: Mapping[str, PreparedSplit]
    rejections: tuple[RejectionRecord, ...]
    profile: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    artifact_hashes: Mapping[str, str]


class PreparationAbort(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__("primary dataset preparation aborted; see discrepancy report")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute_safe_derivation(source: str) -> DerivationResult:
    """Evaluate decimal arithmetic without names, calls, or executable Python syntax."""
    expression = source.strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise DerivationError("unsafe_syntax") from error
    operands: list[Decimal] = []
    operation_count = 0

    def visit(node: ast.AST) -> Decimal:
        nonlocal operation_count
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            literal = ast.get_source_segment(expression, node)
            if literal is None or _DECIMAL_LITERAL.fullmatch(literal) is None:
                raise DerivationError("unsafe_syntax")
            try:
                value = Decimal(literal)
            except InvalidOperation as error:
                raise DerivationError("unsafe_syntax") from error
            operands.append(value)
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if isinstance(node.op, ast.USub):
                value = value.copy_negate()
                if isinstance(node.operand, ast.Constant):
                    operands[-1] = value
            return value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(node.left)
            right = visit(node.right)
            operation_count += 1
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise DerivationError("division_by_zero")
            try:
                return left / right
            except DivisionByZero as error:
                raise DerivationError("division_by_zero") from error
        raise DerivationError("unsafe_syntax")

    return DerivationResult(
        value=visit(tree),
        expression=expression,
        operands=tuple(operands),
        operation_count=operation_count,
    )


def _split_finqa_steps(program: str) -> tuple[str, ...]:
    steps: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(program):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise DerivationError("unsafe_syntax")
        elif character == "," and depth == 0:
            steps.append(program[start:index].strip())
            start = index + 1
    if depth != 0:
        raise DerivationError("unsafe_syntax")
    steps.append(program[start:].strip())
    return tuple(step for step in steps if step)


def _split_finqa_arguments(arguments: str) -> tuple[str, str]:
    parts = [part.strip() for part in arguments.split(",")]
    if len(parts) != 2 or not all(parts):
        raise DerivationError("unsafe_syntax")
    return parts[0], parts[1]


def execute_finqa_program(program: str | list[str]) -> DerivationResult:
    """Compile FinQA's linear two-argument program into safe infix arithmetic."""
    steps = tuple(program) if isinstance(program, list) else _split_finqa_steps(program)
    if not steps:
        raise DerivationError("unsafe_syntax")
    results: list[tuple[Decimal, str]] = []
    operands: list[Decimal] = []
    symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

    def resolve(argument: str) -> tuple[Decimal, str]:
        if argument.startswith("#"):
            if re.fullmatch(r"#(?:0|[1-9]\d*)", argument) is None:
                raise DerivationError("invalid_reference")
            try:
                index = int(argument[1:])
                return results[index]
            except (ValueError, IndexError) as error:
                raise DerivationError("invalid_reference") from error
        literal = argument.removeprefix("const_")
        if _DECIMAL_LITERAL.fullmatch(literal) is None:
            raise DerivationError("unsafe_syntax")
        value = Decimal(literal)
        operands.append(value)
        return value, literal

    for step in steps:
        match = _FINQA_STEP.fullmatch(step.strip())
        if match is None:
            raise DerivationError("unsafe_syntax")
        operation = match.group("operation")
        if operation not in symbols:
            raise DerivationError("unsupported_operation", operation)
        left_arg, right_arg = _split_finqa_arguments(match.group("arguments"))
        left, left_expression = resolve(left_arg)
        right, right_expression = resolve(right_arg)
        if operation == "add":
            value = left + right
        elif operation == "subtract":
            value = left - right
        elif operation == "multiply":
            value = left * right
        else:
            if right == 0:
                raise DerivationError("division_by_zero")
            value = left / right
        results.append((value, f"({left_expression} {symbols[operation]} {right_expression})"))
    value, expression = results[-1]
    return DerivationResult(
        value=value,
        expression=expression,
        operands=tuple(operands),
        operation_count=len(results),
    )


def _canonical_text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _stable_digest(*values: Any) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid local source snapshot: {path}") from error
    if isinstance(payload, dict):
        for key in ("data", "records", "documents"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"source snapshot must contain a JSON list of objects: {path}")
    return payload


def _document_id(dataset: str, row: Mapping[str, Any]) -> str:
    if dataset == "finqa":
        value = row.get("id") or row.get("uid")
    else:
        table = row.get("table")
        table_uid = table.get("uid") if isinstance(table, dict) else None
        value = row.get("uid") or row.get("id") or table_uid
    if value is not None and str(value).strip():
        return f"{dataset}:{str(value).strip()}"
    return f"{dataset}-document-{_stable_digest(row)[:16]}"


def _document_fingerprint(dataset: str, row: Mapping[str, Any]) -> str:
    if dataset == "finqa":
        payload = {
            "pre_text": row.get("pre_text", []),
            "table": row.get("table", []),
            "post_text": row.get("post_text", []),
        }
    else:
        payload = {"table": row.get("table", []), "paragraphs": row.get("paragraphs", [])}
    return _stable_digest(dataset, payload)


def _metadata(row: Mapping[str, Any], answer_type: str) -> DescriptiveMetadata:
    return DescriptiveMetadata(
        company=str(row["company"]) if row.get("company") is not None else None,
        title=str(row["title"]) if row.get("title") is not None else None,
        section=str(row["section"]) if row.get("section") is not None else None,
        language=str(row["language"]) if row.get("language") is not None else "English",
        filing_type=(
            str(row["filing_type"]) if row.get("filing_type") is not None else None
        ),
        industry=str(row["industry"]) if row.get("industry") is not None else None,
        tags=(answer_type,),
    )


def _evidence_item(
    *,
    document_id: str,
    kind: str,
    source_key: str,
    text: str,
    ordinal: int,
    table_id: str | None = None,
    headers: Sequence[str] = (),
    row_label: str | None = None,
) -> EvidenceItem:
    evidence_id = f"{document_id}:{source_key}"
    return EvidenceItem(
        evidence_id=evidence_id,
        document_id=document_id,
        kind=kind,
        text=text,
        table_id=table_id,
        headers=tuple(headers),
        row_label=row_label,
        ordinal=ordinal,
    )


def _finqa_evidence(
    document_id: str, row: Mapping[str, Any]
) -> tuple[tuple[EvidenceItem, ...], Mapping[str, EvidenceItem]]:
    items: list[EvidenceItem] = []
    aliases: dict[str, EvidenceItem] = {}

    def add(kind: str, source_key: str, text: str, **values: Any) -> None:
        item = _evidence_item(
            document_id=document_id,
            kind=kind,
            source_key=source_key.replace("_", ":", 1),
            text=text,
            ordinal=len(items),
            **values,
        )
        items.append(item)
        aliases[source_key] = item

    for index, text in enumerate(row.get("pre_text", [])):
        add("text", f"pre_text_{index}", str(text))
    table = row.get("table", [])
    headers = tuple(str(value) for value in table[0]) if table else ()
    table_id = f"{document_id}:table"
    for index, values in enumerate(table[1:], start=1):
        cells = tuple(str(value) for value in values)
        add(
            "table_row",
            f"table_{index}",
            " | ".join(cells),
            table_id=table_id,
            headers=headers,
            row_label=cells[0] if cells else None,
        )
    for index, text in enumerate(row.get("post_text", [])):
        add("text", f"post_text_{index}", str(text))
    return tuple(items), aliases


def _tatqa_evidence(document_id: str, row: Mapping[str, Any]) -> tuple[EvidenceItem, ...]:
    items: list[EvidenceItem] = []
    table_payload = row.get("table", [])
    if isinstance(table_payload, dict):
        table = table_payload.get("table", [])
        table_id = str(table_payload.get("uid") or f"{document_id}:table")
    else:
        table = table_payload
        table_id = f"{document_id}:table"
    headers = tuple(str(value) for value in table[0]) if table else ()
    for index, values in enumerate(table[1:], start=1):
        cells = tuple(str(value) for value in values)
        items.append(
            _evidence_item(
                document_id=document_id,
                kind="table_row",
                source_key=f"table:{index}",
                text=" | ".join(cells),
                ordinal=len(items),
                table_id=table_id,
                headers=headers,
                row_label=cells[0] if cells else None,
            )
        )
    paragraphs = sorted(
        row.get("paragraphs", []),
        key=lambda value: (int(value.get("order", 0)), str(value.get("uid", ""))),
    )
    for index, paragraph in enumerate(paragraphs):
        source_id = str(paragraph.get("uid") or index)
        items.append(
            _evidence_item(
                document_id=document_id,
                kind="text",
                source_key=f"paragraph:{source_id}",
                text=str(paragraph.get("text", "")),
                ordinal=len(items),
            )
        )
    return tuple(items)


def _answer_spec(question: Mapping[str, Any]) -> AnswerSpec:
    raw_answer = str(question.get("answer", "")).strip()
    raw_scale = _canonical_text(question.get("scale", ""))
    answer_scale_tokens = [
        scale
        for token, scale in _KNOWN_SCALES.items()
        if token and re.search(rf"\b{re.escape(token)}\b", raw_answer.casefold())
    ]
    if "%" in raw_answer:
        answer_scale_tokens.append("percent")
    scales = set(answer_scale_tokens)
    if raw_scale not in _KNOWN_SCALES or len(scales) > 1:
        raise DerivationError("ambiguous_scale")
    scale = _KNOWN_SCALES[raw_scale]
    if raw_scale == "" and scales:
        scale = next(iter(scales))
    elif scales and scale not in scales:
        raise DerivationError("ambiguous_scale")

    unit = str(question["unit"]) if question.get("unit") is not None else None
    currency = next((symbol for symbol in _CURRENCY_UNITS if symbol in raw_answer), None)
    if currency is not None:
        inferred = _CURRENCY_UNITS[currency]
        if unit is not None and _canonical_text(unit) != inferred.casefold():
            raise DerivationError("ambiguous_scale")
        unit = inferred

    numeric = raw_answer
    for token in sorted(_KNOWN_SCALES, key=len, reverse=True):
        if token:
            numeric = re.sub(rf"\b{re.escape(token)}\b", "", numeric, flags=re.IGNORECASE)
    numeric = numeric.replace(",", "").replace("%", "").strip()
    for symbol in _CURRENCY_UNITS:
        numeric = numeric.replace(symbol, "")
    numeric = numeric.strip()
    if numeric.startswith("(") and numeric.endswith(")"):
        numeric = f"-{numeric[1:-1]}"
    if _DECIMAL_LITERAL.fullmatch(numeric) is None:
        raise DerivationError("non_numeric_answer")
    return AnswerSpec(
        value=Decimal(numeric),
        unit=unit,
        scale=scale,
        entity=str(question["entity"]) if question.get("entity") is not None else None,
        period=str(question["period"]) if question.get("period") is not None else None,
    )


def _numbers_by_evidence(items: Iterable[EvidenceItem]) -> Mapping[str, frozenset[Decimal]]:
    return {
        item.evidence_id: frozenset(extract_strict_numeric_values(item.text)) for item in items
    }


def _locate_operands(
    operands: Sequence[Decimal],
    items: Sequence[EvidenceItem],
    *,
    reject_ambiguous: bool = True,
) -> tuple[str, ...]:
    numbers = _numbers_by_evidence(items)
    support: list[str] = []
    for operand in operands:
        matches = [
            item.evidence_id for item in items if operand in numbers[item.evidence_id]
        ]
        if not matches:
            raise DerivationError("missing_evidence", f"operand {operand} is unsupported")
        if reject_ambiguous and len(matches) > 1:
            raise DerivationError(
                "ambiguous_evidence",
                f"operand {operand} appears in multiple evidence items: {matches}",
            )
        evidence_id = matches[0]
        if evidence_id not in support:
            support.append(evidence_id)
    return tuple(support)


def _classify(
    public_case: PublicCase, support_ids: Sequence[str], operation_count: int
) -> str:
    ranking = retrieve(
        public_case,
        (public_case.question,),
        limit=len(public_case.evidence),
        max_chars_per_item=1_000_000,
    ).pre_truncation_ids
    ranks = [ranking.index(evidence_id) + 1 for evidence_id in support_ids]
    if operation_count >= 2 and any(3 <= rank <= 12 for rank in ranks):
        return "headroom"
    if operation_count == 1 and ranks and all(rank <= 2 for rank in ranks):
        return "easy_control"
    raise DerivationError("ineligible_stratum")


def _make_case(
    *,
    dataset: DatasetName,
    source_name: str,
    document_id: str,
    question_id: str,
    question: Mapping[str, Any],
    evidence: tuple[EvidenceItem, ...],
    support_ids: tuple[str, ...],
    derivation: DerivationResult,
    row: Mapping[str, Any],
    answer_type: str,
) -> AdaptedCase:
    answer = _answer_spec(question)
    if derivation.value != answer.value:
        raise DerivationError("program_answer_mismatch")
    provisional = PublicCase(
        case_id=f"{dataset}-{_stable_digest(dataset, document_id, question_id)[:20]}",
        dataset=dataset,
        document_id=document_id,
        question=str(question.get("question", "")).strip(),
        evidence=evidence,
        stratum="unclassified",
        metadata=_metadata(row, answer_type),
    )
    stratum = _classify(provisional, support_ids, derivation.operation_count)
    public = provisional.model_copy(update={"stratum": stratum})
    hidden = HiddenLabel(
        case_id=public.case_id,
        answer=answer,
        gold_derivation=derivation.expression,
        gold_support_ids=support_ids,
        source_lineage=(source_name, document_id, question_id),
    )
    return AdaptedCase(public=public, hidden=hidden)


def _finqa_questions(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    payload = row.get("qa", row.get("questions", ()))
    if isinstance(payload, dict):
        return (payload,)
    if isinstance(payload, list):
        return tuple(value for value in payload if isinstance(value, dict))
    return ()


def _adapt_finqa_question(
    *,
    source_name: str,
    document_id: str,
    row: Mapping[str, Any],
    question: Mapping[str, Any],
) -> AdaptedCase:
    question_id = str(question.get("id") or question.get("uid") or "question")
    derivation = execute_finqa_program(question.get("program", ""))
    evidence, aliases = _finqa_evidence(document_id, row)
    gold_inds = question.get("gold_inds")
    if not isinstance(gold_inds, dict) or not gold_inds:
        raise DerivationError("missing_evidence", "FinQA gold_inds are missing")
    support: list[EvidenceItem] = []
    for source_key, annotated_text in gold_inds.items():
        item = aliases.get(str(source_key))
        if item is None:
            normalized_annotation = _canonical_text(annotated_text)
            item = next(
                (value for value in evidence if _canonical_text(value.text) == normalized_annotation),
                None,
            )
        if item is None:
            raise DerivationError("missing_evidence", f"unknown FinQA support {source_key}")
        if item not in support:
            support.append(item)
    _locate_operands(derivation.operands, support, reject_ambiguous=False)
    return _make_case(
        dataset="finqa",
        source_name=source_name,
        document_id=document_id,
        question_id=question_id,
        question=question,
        evidence=evidence,
        support_ids=tuple(item.evidence_id for item in support),
        derivation=derivation,
        row=row,
        answer_type="numeric",
    )


def _tatqa_questions(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    payload = row.get("questions", ())
    return tuple(value for value in payload if isinstance(value, dict))


def _adapt_tatqa_question(
    *,
    source_name: str,
    document_id: str,
    row: Mapping[str, Any],
    question: Mapping[str, Any],
) -> AdaptedCase:
    question_id = str(question.get("uid") or question.get("id") or "question")
    answer_type = _canonical_text(question.get("answer_type", ""))
    if answer_type not in {"arithmetic", "count"}:
        raise DerivationError("unsupported_question_type")
    derivation = execute_safe_derivation(str(question.get("derivation", "")))
    evidence = _tatqa_evidence(document_id, row)
    support_ids = _locate_operands(derivation.operands, evidence)
    return _make_case(
        dataset="tatqa",
        source_name=source_name,
        document_id=document_id,
        question_id=question_id,
        question=question,
        evidence=evidence,
        support_ids=support_ids,
        derivation=derivation,
        row=row,
        answer_type=answer_type,
    )


def _question_id(question: Mapping[str, Any]) -> str:
    return str(question.get("id") or question.get("uid") or "question")


def _rejection(
    snapshot: DatasetSnapshot,
    document_id: str,
    question: Mapping[str, Any] | None,
    error: DerivationError,
) -> RejectionRecord:
    return RejectionRecord(
        dataset=snapshot.dataset,
        source=snapshot.path.name,
        document_id=document_id,
        question_id=_question_id(question) if question is not None else None,
        reason=error.reason,
    )


def adapt_primary_snapshots(
    snapshots: Sequence[DatasetSnapshot], *, seed: int
) -> AdaptationResult:
    """Adapt pinned local snapshots; this function never performs acquisition or network I/O."""
    cases: list[AdaptedCase] = []
    rejections: list[RejectionRecord] = []
    source_hashes: dict[str, str] = {}
    seen_document_ids: set[tuple[str, str]] = set()
    seen_fingerprints: set[tuple[str, str]] = set()

    ordered = sorted(snapshots, key=lambda value: (value.dataset, value.path.name, str(value.path)))
    for snapshot in ordered:
        if not snapshot.path.is_file():
            raise FileNotFoundError(f"pinned source snapshot is missing: {snapshot.path}")
        observed_hash = sha256_file(snapshot.path)
        if observed_hash != snapshot.expected_sha256:
            raise ValueError(
                f"source snapshot checksum mismatch for {snapshot.path}: "
                f"expected {snapshot.expected_sha256}, observed {observed_hash}"
            )
        source_hashes[f"{snapshot.dataset}:{snapshot.path.name}"] = observed_hash
        for row in _source_rows(snapshot.path):
            document_id = _document_id(snapshot.dataset, row)
            fingerprint = _document_fingerprint(snapshot.dataset, row)
            identity = (snapshot.dataset, document_id)
            content_identity = (snapshot.dataset, fingerprint)
            questions = (
                _finqa_questions(row) if snapshot.dataset == "finqa" else _tatqa_questions(row)
            )
            if identity in seen_document_ids or content_identity in seen_fingerprints:
                error = DerivationError("duplicate_document")
                if not questions:
                    rejections.append(_rejection(snapshot, document_id, None, error))
                else:
                    rejections.extend(
                        _rejection(snapshot, document_id, question, error) for question in questions
                    )
                continue
            seen_document_ids.add(identity)
            seen_fingerprints.add(content_identity)

            eligible: list[AdaptedCase] = []
            for question in questions:
                try:
                    if snapshot.dataset == "finqa":
                        adapted = _adapt_finqa_question(
                            source_name=snapshot.path.name,
                            document_id=document_id,
                            row=row,
                            question=question,
                        )
                    else:
                        adapted = _adapt_tatqa_question(
                            source_name=snapshot.path.name,
                            document_id=document_id,
                            row=row,
                            question=question,
                        )
                except DerivationError as error:
                    rejections.append(_rejection(snapshot, document_id, question, error))
                else:
                    eligible.append(adapted)
            if not eligible:
                continue
            eligible.sort(
                key=lambda value: _stable_digest(seed, value.public.document_id, value.public.case_id)
            )
            cases.append(eligible[0])
            for unselected in eligible[1:]:
                rejections.append(
                    RejectionRecord(
                        dataset=snapshot.dataset,
                        source=snapshot.path.name,
                        document_id=document_id,
                        question_id=unselected.hidden.source_lineage[-1],
                        reason="one_question_per_document",
                    )
                )

    return AdaptationResult(
        cases=tuple(sorted(cases, key=lambda value: value.public.case_id)),
        rejections=tuple(
            sorted(
                rejections,
                key=lambda value: (
                    value.dataset,
                    value.source,
                    value.document_id,
                    value.question_id or "",
                    value.reason,
                ),
            )
        ),
        source_hashes=dict(sorted(source_hashes.items())),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                payload = row.model_dump(mode="json")
            elif hasattr(row, "__dataclass_fields__"):
                payload = asdict(row)
            else:
                payload = row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _balanced_requirements(quotas: SplitQuotas) -> Mapping[tuple[str, str], int]:
    hard_per_dataset = (quotas.development + quotas.operational_pilot + quotas.main) // 2
    easy_per_dataset = quotas.easy_reserve // 2
    return {
        (dataset, "headroom"): hard_per_dataset
        for dataset in ("finqa", "tatqa")
    } | {
        (dataset, "easy_control"): easy_per_dataset
        for dataset in ("finqa", "tatqa")
    }


def _split_cases(
    cases: Sequence[AdaptedCase], quotas: SplitQuotas, seed: int
) -> Mapping[str, tuple[AdaptedCase, ...]]:
    groups: dict[tuple[str, str], list[AdaptedCase]] = {
        (dataset, stratum): []
        for dataset in ("finqa", "tatqa")
        for stratum in ("headroom", "easy_control")
    }
    for case in cases:
        key = (case.public.dataset, case.public.stratum)
        if key in groups:
            groups[key].append(case)
    for key, values in groups.items():
        values.sort(
            key=lambda case: _stable_digest(seed, key[0], key[1], case.public.case_id)
        )

    slices: dict[str, list[AdaptedCase]] = {
        "development": [],
        "operational_pilot": [],
        "main": [],
        "easy_reserve": [],
    }
    hard_offsets = {
        "development": quotas.development // 2,
        "operational_pilot": quotas.operational_pilot // 2,
        "main": quotas.main // 2,
    }
    for dataset in ("finqa", "tatqa"):
        hard = groups[(dataset, "headroom")]
        offset = 0
        for split_name, count in hard_offsets.items():
            slices[split_name].extend(hard[offset : offset + count])
            offset += count
        slices["easy_reserve"].extend(
            groups[(dataset, "easy_control")][: quotas.easy_reserve // 2]
        )
    return {
        name: tuple(sorted(values, key=lambda case: case.public.case_id))
        for name, values in slices.items()
    }


def prepare_primary_datasets(
    snapshots: Sequence[DatasetSnapshot],
    *,
    output_dir: Path,
    quotas: SplitQuotas | None = None,
    seed: int,
) -> PreparationResult:
    """Build exact document-disjoint splits and emit public/hidden artifacts separately."""
    quotas = quotas or SplitQuotas()
    adapted = adapt_primary_snapshots(snapshots, seed=seed)
    observed = Counter(
        (case.public.dataset, case.public.stratum) for case in adapted.cases
    )
    shortfalls = {
        f"{dataset}:{stratum}": required - observed[(dataset, stratum)]
        for (dataset, stratum), required in _balanced_requirements(quotas).items()
        if observed[(dataset, stratum)] < required
    }
    if shortfalls:
        report = {
            "status": "aborted",
            "reason": "quota_shortfall",
            "requested": asdict(quotas),
            "eligible": {
                f"{dataset}:{stratum}": observed[(dataset, stratum)]
                for dataset in ("finqa", "tatqa")
                for stratum in ("headroom", "easy_control")
            },
            "shortfalls": dict(sorted(shortfalls.items())),
            "source_hashes": dict(adapted.source_hashes),
        }
        _write_json(output_dir / "preparation_discrepancy.json", report)
        raise PreparationAbort(report)

    selected = _split_cases(adapted.cases, quotas, seed)
    split_results = {
        name: PreparedSplit(
            public_cases=tuple(case.public for case in values),
            hidden_labels=tuple(case.hidden for case in values),
        )
        for name, values in selected.items()
    }
    selected_cases = [case for values in selected.values() for case in values]
    document_ids = [case.public.document_id for case in selected_cases]
    profile: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "requested": asdict(quotas),
        "counts": {name: len(values) for name, values in selected.items()},
        "datasets": dict(
            sorted(Counter(case.public.dataset for case in selected_cases).items())
        ),
        "strata": dict(
            sorted(Counter(case.public.stratum for case in selected_cases).items())
        ),
        "eligible_cases": len(adapted.cases),
        "selected_cases": len(selected_cases),
        "rejections": dict(
            sorted(Counter(row.reason for row in adapted.rejections).items())
        ),
        "document_disjoint": len(document_ids) == len(set(document_ids)),
        "lineage": [
            {
                "case_id": case.public.case_id,
                "split": split_name,
                "dataset": case.public.dataset,
                "source": case.hidden.source_lineage[0],
                "document_id": case.public.document_id,
                "question_id": case.hidden.source_lineage[-1],
            }
            for split_name, values in selected.items()
            for case in values
        ],
    }
    if not profile["document_disjoint"]:
        report = {
            "status": "aborted",
            "reason": "document_overlap",
            "document_ids": document_ids,
        }
        _write_json(output_dir / "preparation_discrepancy.json", report)
        raise PreparationAbort(report)

    artifact_paths: list[Path] = []
    for split_name, split in split_results.items():
        public_path = output_dir / "public" / f"{split_name}.jsonl"
        hidden_path = output_dir / "hidden" / f"{split_name}.jsonl"
        _write_jsonl(public_path, split.public_cases)
        _write_jsonl(hidden_path, split.hidden_labels)
        artifact_paths.extend((public_path, hidden_path))
    rejection_path = output_dir / "rejections.jsonl"
    profile_path = output_dir / "profile.json"
    _write_jsonl(rejection_path, adapted.rejections)
    _write_json(profile_path, profile)
    artifact_paths.extend((rejection_path, profile_path))
    artifact_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(artifact_paths)
    }
    _write_json(
        output_dir / "hashes.json",
        {
            "source_hashes": dict(adapted.source_hashes),
            "artifact_hashes": artifact_hashes,
        },
    )
    return PreparationResult(
        splits=split_results,
        rejections=adapted.rejections,
        profile=profile,
        source_hashes=adapted.source_hashes,
        artifact_hashes=artifact_hashes,
    )


# Import-only bridges for analysis/CLI/validation modules scheduled for Tasks 4-5.
# They fail closed rather than retaining the removed HMDA preparation path.
def build_case_set(*_args: Any, **_kwargs: Any) -> list[Any]:
    raise RuntimeError(
        "HMDA build_case_set was removed; use prepare_primary_datasets with pinned snapshots"
    )


def validate_hmda_source(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "HMDA source validation was removed; use DatasetSnapshot checksum validation"
    )


def case_set_profile(cases: Sequence[Any]) -> dict[str, Any]:
    """Read-only compatibility profile until canonical analysis lands in Task 4."""
    base = [
        case
        for case in cases
        if getattr(case, "counterfactual_variant", "observed") == "observed"
    ]
    return {
        "cases": len(cases),
        "base_applications": len(base),
        "counterfactual_pairs": len(
            {getattr(case, "pair_id", getattr(case, "case_id", "")) for case in cases}
        ),
        "states": dict(
            sorted(Counter(getattr(case, "state", "unknown") for case in base).items())
        ),
        "policy_decisions": dict(
            sorted(
                Counter(getattr(case, "policy_decision", "unknown") for case in base).items()
            )
        ),
        "historical_actions": dict(
            sorted(
                Counter(getattr(case, "historical_action", "unknown") for case in base).items()
            )
        ),
        "complexity": dict(
            sorted(Counter(getattr(case, "complexity", "unknown") for case in base).items())
        ),
        "changed_attributes": dict(
            sorted(
                Counter(
                    getattr(case, "changed_protected_attribute", "unknown") for case in base
                ).items()
            )
        ),
    }

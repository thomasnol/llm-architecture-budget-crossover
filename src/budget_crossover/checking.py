from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from .models import Candidate, CheckFinding, CheckResult, EvidenceItem
from .scoring import extract_strict_numeric_values, normalize_unit, normalized_candidate_value

_EXPRESSION_NUMBER = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")
_SCALE_FACTORS = {
    "ones": Decimal(1),
    "thousand": Decimal(1000),
    "million": Decimal(1000000),
    "billion": Decimal(1000000000),
    "percent": Decimal("0.01"),
}


class _UnsafeExpression(ValueError):
    pass


class _DivisionByZero(ValueError):
    pass


def _count_evidence_ids(expression: str) -> tuple[str, ...] | None:
    if not expression.lstrip().startswith("count"):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise _UnsafeExpression from error
    body = tree.body
    if (
        not isinstance(body, ast.Call)
        or not isinstance(body.func, ast.Name)
        or body.func.id != "count"
        or body.keywords
        or not body.args
        or any(
            not isinstance(argument, ast.Constant) or not isinstance(argument.value, str)
            for argument in body.args
        )
    ):
        raise _UnsafeExpression
    evidence_ids = tuple(argument.value for argument in body.args)
    if len(evidence_ids) != len(set(evidence_ids)) or any(not value for value in evidence_ids):
        raise _UnsafeExpression
    return evidence_ids


def _evaluate(expression: str) -> tuple[Decimal, tuple[Decimal, ...]]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise _UnsafeExpression from error
    operands: list[Decimal] = []

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            source = ast.get_source_segment(expression, node)
            if isinstance(node.value, bool) or source is None or not _EXPRESSION_NUMBER.fullmatch(source):
                raise _UnsafeExpression
            try:
                value = Decimal(source)
            except InvalidOperation as error:
                raise _UnsafeExpression from error
            operands.append(value)
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if isinstance(node.op, ast.USub):
                signed = value.copy_negate()
                if isinstance(node.operand, ast.Constant):
                    operands[-1] = signed
                return signed
            return value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise _DivisionByZero
            return left / right
        raise _UnsafeExpression

    return visit(tree), tuple(operands)


def _evidence_numbers(items: Sequence[EvidenceItem]) -> frozenset[Decimal]:
    return frozenset(
        value for item in items for value in extract_strict_numeric_values(item.text)
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _mismatches(
    expected: str | None,
    items: Sequence[EvidenceItem],
    field: str,
    *,
    normalizer=_normalized_text,
) -> bool:
    if expected is None:
        return False
    observed = [getattr(item, field) for item in items if getattr(item, field) is not None]
    if not observed:
        return False
    normalized_expected = normalizer(expected)
    return any(normalizer(value) != normalized_expected for value in observed)


def _finding(code: str, message: str, evidence_ids: tuple[str, ...] = ()) -> CheckFinding:
    return CheckFinding(code=code, message=message, evidence_ids=evidence_ids)


def check_candidate(
    candidate: Candidate,
    evidence: Sequence[EvidenceItem],
) -> CheckResult:
    findings: list[CheckFinding] = []
    by_id = {item.evidence_id: item for item in evidence}
    if not candidate.citations:
        findings.append(_finding("missing_citations", "Candidate has no evidence citations."))
    missing = tuple(citation for citation in candidate.citations if citation not in by_id)
    if missing:
        findings.append(
            _finding("fabricated_citation", "Candidate cites evidence that does not exist.", missing)
        )
    cited = tuple(by_id[citation] for citation in candidate.citations if citation in by_id)

    evaluated: Decimal | None = None
    operands: tuple[Decimal, ...] = ()
    count_evidence_ids: tuple[str, ...] | None = None
    if candidate.expression is None or not candidate.expression.strip():
        findings.append(_finding("missing_expression", "Candidate has no arithmetic expression."))
    else:
        try:
            count_evidence_ids = _count_evidence_ids(candidate.expression)
            if count_evidence_ids is None:
                evaluated, operands = _evaluate(candidate.expression)
            else:
                evaluated = Decimal(len(count_evidence_ids))
        except _DivisionByZero:
            findings.append(_finding("division_by_zero", "Expression divides by zero."))
        except _UnsafeExpression:
            findings.append(_finding("unsafe_expression", "Expression contains unsafe syntax."))

    if count_evidence_ids is not None and set(count_evidence_ids) != set(candidate.citations):
        findings.append(
            _finding(
                "count_evidence_mismatch",
                "Count expression evidence must match candidate citations exactly.",
                count_evidence_ids,
            )
        )

    if evaluated is not None:
        supported = _evidence_numbers(cited)
        unsupported = tuple(str(value) for value in operands if value not in supported)
        if unsupported:
            findings.append(
                _finding(
                    "unsupported_operand",
                    f"Expression operands lack evidence provenance: {', '.join(unsupported)}.",
                    candidate.citations,
                )
            )

        normalized = normalized_candidate_value(candidate)
        if normalized is None:
            findings.append(_finding("invalid_candidate_value", "Candidate value is not strict numeric data."))
        else:
            candidate_value, _ = normalized
            effective_scale = (
                "percent" if candidate.value.strip().endswith("%") else candidate.scale
            )
            if evaluated * _SCALE_FACTORS[effective_scale] != candidate_value:
                findings.append(
                    _finding(
                        "expression_mismatch",
                        "Expression result does not equal the candidate value.",
                    )
                )

    normalized = normalized_candidate_value(candidate)
    candidate_unit = normalized[1] if normalized is not None else normalize_unit(candidate.unit)
    effective_scale = "percent" if candidate.value.strip().endswith("%") else candidate.scale
    metadata_checks = (
        ("unit", candidate_unit, "unit", normalize_unit),
        ("scale", effective_scale, "scale", _normalized_text),
        ("entity", candidate.entity, "entity", _normalized_text),
        ("period", candidate.period, "period", _normalized_text),
    )
    for code, expected, field, normalizer in metadata_checks:
        if _mismatches(expected, cited, field, normalizer=normalizer):
            findings.append(
                _finding(
                    f"{code}_mismatch",
                    f"Candidate {code} conflicts with cited evidence metadata.",
                    candidate.citations,
                )
            )

    return CheckResult(
        passed=not findings,
        findings=tuple(findings),
        evaluated_expression=evaluated,
    )

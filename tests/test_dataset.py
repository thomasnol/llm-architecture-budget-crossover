import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from budget_crossover.dataset import (
    DatasetSnapshot,
    DerivationError,
    PreparationAbort,
    SplitQuotas,
    adapt_primary_snapshots,
    execute_finqa_program,
    execute_safe_derivation,
    prepare_primary_datasets,
)


def _snapshot(tmp_path: Path, dataset: str, name: str, payload) -> DatasetSnapshot:
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return DatasetSnapshot(dataset=dataset, path=path, expected_sha256=digest)


def _finqa_headroom(document_id: str = "finqa-headroom") -> dict:
    return {
        "id": document_id,
        "pre_text": [
            "Two step adjusted result reference overview.",
            "Two step adjusted result supplemental discussion.",
        ],
        "post_text": [f"Document lineage {document_id}."],
        "table": [
            ["Metric", "Value"],
            ["Alpha", "10"],
            ["Beta", "2"],
            ["Gamma", "4"],
        ],
        "qa": {
            "id": f"{document_id}-q",
            "question": "What is the two step adjusted result?",
            "answer": "2",
            "program": "subtract(10, 2), divide(#0, 4)",
            "gold_inds": {
                "table_1": "Alpha | 10",
                "table_2": "Beta | 2",
                "table_3": "Gamma | 4",
            },
        },
    }


def _tatqa_easy(document_id: str = "tatqa-easy", *, questions: list[dict] | None = None) -> dict:
    return {
        "uid": document_id,
        "table": {
            "uid": f"{document_id}-table",
            "table": [
                ["Metric", "Value"],
                ["Revenue", "7"],
                ["Expense", "2"],
            ],
        },
        "paragraphs": [{"uid": "p1", "order": 1, "text": "Background only."}],
        "questions": questions
        or [
            {
                "uid": f"{document_id}-q",
                "question": "What is revenue plus expense?",
                "answer": "9",
                "derivation": "7 + 2",
                "answer_type": "arithmetic",
                "scale": "",
            }
        ],
    }


def _finqa_easy(document_id: str) -> dict:
    return {
        "id": document_id,
        "pre_text": [f"Background for {document_id}."],
        "post_text": [],
        "table": [
            ["Metric", "Value"],
            ["Revenue", "7"],
            ["Expense", "2"],
        ],
        "qa": {
            "id": f"{document_id}-q",
            "question": "What is revenue plus expense?",
            "answer": "9",
            "program": "add(7, 2)",
            "gold_inds": {"table_1": "Revenue | 7", "table_2": "Expense | 2"},
        },
    }


def _tatqa_headroom(document_id: str) -> dict:
    return {
        "uid": document_id,
        "table": {
            "uid": f"{document_id}-table",
            "table": [
                ["Metric", "Value"],
                ["Alpha", "10"],
                ["Beta", "2"],
                ["Gamma", "4"],
            ],
        },
        "paragraphs": [
            {
                "uid": f"{document_id}-p1",
                "order": 1,
                "text": "Two step adjusted result reference overview.",
            },
            {
                "uid": f"{document_id}-p2",
                "order": 2,
                "text": "Two step adjusted result supplemental discussion.",
            },
        ],
        "questions": [
            {
                "uid": f"{document_id}-q",
                "question": "What is the two step adjusted result?",
                "answer": "2",
                "derivation": "(10 - 2) / 4",
                "answer_type": "arithmetic",
                "scale": "",
            }
        ],
    }


def test_safe_derivations_execute_decimal_arithmetic_and_report_lineage():
    generic = execute_safe_derivation("(12.5 - 2.5) / 2")
    finqa = execute_finqa_program(
        "subtract(12.5, 2.5), divide(#0, 2), add(#1, 1)"
    )

    assert generic.value == Decimal("5.0")
    assert generic.expression == "(12.5 - 2.5) / 2"
    assert generic.operands == (Decimal("12.5"), Decimal("2.5"), Decimal(2))
    assert generic.operation_count == 2
    assert finqa.value == Decimal("6.0")
    assert finqa.expression == "(((12.5 - 2.5) / 2) + 1)"
    assert finqa.operands == (
        Decimal("12.5"),
        Decimal("2.5"),
        Decimal(2),
        Decimal(1),
    )
    assert finqa.operation_count == 3


@pytest.mark.parametrize(
    ("executor", "source", "reason"),
    [
        (execute_safe_derivation, "__import__('os').system('id')", "unsafe_syntax"),
        (execute_safe_derivation, "1 / 0", "division_by_zero"),
        (execute_finqa_program, "exp(2, 3)", "unsupported_operation"),
        (execute_finqa_program, "add(#2, 1)", "invalid_reference"),
        (
            execute_finqa_program,
            "add(1, 2), add(#-1, 3)",
            "invalid_reference",
        ),
    ],
)
def test_derivations_reject_unsafe_or_unsupported_programs(executor, source, reason):
    with pytest.raises(DerivationError) as captured:
        executor(source)

    assert captured.value.reason == reason


def test_finqa_and_tatqa_adapters_accept_proven_derivations_and_classify_strata(tmp_path: Path):
    finqa = _snapshot(tmp_path, "finqa", "finqa.json", [_finqa_headroom()])
    tatqa = _snapshot(tmp_path, "tatqa", "tatqa.json", [_tatqa_easy()])

    adapted = adapt_primary_snapshots((finqa, tatqa), seed=17)
    by_dataset = {case.public.dataset: case for case in adapted.cases}

    assert by_dataset["finqa"].public.stratum == "headroom"
    assert by_dataset["tatqa"].public.stratum == "easy_control"
    assert by_dataset["finqa"].hidden.answer.value == Decimal(2)
    assert by_dataset["finqa"].hidden.gold_derivation == "((10 - 2) / 4)"
    assert by_dataset["tatqa"].hidden.gold_derivation == "7 + 2"
    assert all(case.hidden.gold_support_ids for case in adapted.cases)
    assert all(len(case.hidden.source_lineage) == 3 for case in adapted.cases)


def test_adapters_record_specific_rejections_instead_of_relaxing_eligibility(tmp_path: Path):
    unsupported = _finqa_headroom("unsupported")
    unsupported["qa"]["program"] = "exp(2, 3)"
    unsupported["post_text"] = ["Unsupported operation context."]
    missing = _finqa_headroom("missing")
    missing["qa"]["program"] = "subtract(99, 2), divide(#0, 4)"
    missing["post_text"] = ["Missing operand context."]
    mismatch = _finqa_headroom("mismatch")
    mismatch["qa"]["answer"] = "999"
    mismatch["post_text"] = ["Mismatched answer context."]
    ambiguous = _finqa_headroom("ambiguous")
    ambiguous["qa"]["scale"] = "millions or billions"
    ambiguous["post_text"] = ["Ambiguous scale context."]
    duplicate = _finqa_headroom("duplicate-copy")
    duplicate["pre_text"] = _finqa_headroom()["pre_text"]
    duplicate["table"] = _finqa_headroom()["table"]
    duplicate["post_text"] = _finqa_headroom()["post_text"]
    source = _snapshot(
        tmp_path,
        "finqa",
        "reject.json",
        [_finqa_headroom(), unsupported, missing, mismatch, ambiguous, duplicate],
    )

    adapted = adapt_primary_snapshots((source,), seed=3)

    assert {rejection.reason for rejection in adapted.rejections} >= {
        "unsupported_operation",
        "missing_evidence",
        "program_answer_mismatch",
        "ambiguous_scale",
        "duplicate_document",
    }
    assert all(rejection.detail is None for rejection in adapted.rejections)


def test_finqa_validates_support_keys_and_rejects_ambiguous_text_fallback(tmp_path: Path):
    mismatched = _finqa_headroom("support-mismatch")
    mismatched["qa"]["gold_inds"]["table_1"] = "Beta | 2"
    ambiguous = _finqa_headroom("support-ambiguous")
    ambiguous["table"].append(["Alpha", "10"])
    ambiguous["qa"]["gold_inds"] = {
        "unknown_key": "Alpha | 10",
        "table_2": "Beta | 2",
        "table_3": "Gamma | 4",
    }
    source = _snapshot(
        tmp_path,
        "finqa",
        "support-validation.json",
        [mismatched, ambiguous],
    )

    adapted = adapt_primary_snapshots((source,), seed=3)

    assert adapted.cases == ()
    assert {row.reason for row in adapted.rejections} == {
        "support_annotation_mismatch",
        "ambiguous_support_annotation",
    }


def test_tatqa_selects_one_question_per_document_deterministically(tmp_path: Path):
    questions = [
        {
            "uid": "q-1",
            "question": "What is revenue plus expense?",
            "answer": "9",
            "derivation": "7 + 2",
            "answer_type": "arithmetic",
            "scale": "",
        },
        {
            "uid": "q-2",
            "question": "What is expense plus revenue?",
            "answer": "9",
            "derivation": "2 + 7",
            "answer_type": "arithmetic",
            "scale": "",
        },
    ]
    source = _snapshot(
        tmp_path,
        "tatqa",
        "multiple.json",
        [_tatqa_easy("multi", questions=questions)],
    )

    first = adapt_primary_snapshots((source,), seed=41)
    second = adapt_primary_snapshots((source,), seed=41)

    assert len(first.cases) == 1
    assert [case.public.case_id for case in first.cases] == [
        case.public.case_id for case in second.cases
    ]
    assert [row.reason for row in first.rejections] == ["one_question_per_document"]


def test_document_identity_is_namespaced_across_primary_datasets(tmp_path: Path):
    finqa = _snapshot(tmp_path, "finqa", "same-finqa.json", [_finqa_headroom("shared")])
    tatqa = _snapshot(tmp_path, "tatqa", "same-tatqa.json", [_tatqa_headroom("shared")])

    adapted = adapt_primary_snapshots((finqa, tatqa), seed=5)

    assert len(adapted.cases) == 2
    assert len({case.public.document_id for case in adapted.cases}) == 2


def test_tatqa_accepts_count_questions_with_executable_evidence_backed_derivations(
    tmp_path: Path,
):
    count = _tatqa_easy(
        "tatqa-count",
        questions=[
            {
                "uid": "count-q",
                "question": "How many reported countable items were there?",
                "answer": "2",
                "derivation": "",
                "answer_type": "count",
                "scale": "",
                "facts": ["Reported countable alpha item.", "Reported countable beta item."],
            }
        ],
    )
    count["table"]["table"] = [["Metric", "Value"]]
    count["paragraphs"] = [
        {"uid": "alpha", "order": 1, "text": "Reported countable alpha item."},
        {"uid": "beta", "order": 2, "text": "Reported countable beta item."},
    ]
    source = _snapshot(tmp_path, "tatqa", "count.json", [count])

    adapted = adapt_primary_snapshots((source,), seed=4)

    assert len(adapted.cases) == 1
    assert adapted.cases[0].public.stratum == "easy_control"
    assert adapted.cases[0].public.metadata.tags == ("count",)
    assert adapted.cases[0].hidden.gold_derivation == (
        'count("tatqa:tatqa-count:paragraph:alpha", '
        '"tatqa:tatqa-count:paragraph:beta")'
    )
    assert adapted.cases[0].hidden.gold_support_ids == (
        "tatqa:tatqa-count:paragraph:alpha",
        "tatqa:tatqa-count:paragraph:beta",
    )


def test_tatqa_count_rejects_missing_or_ambiguous_counted_evidence(tmp_path: Path):
    missing = _tatqa_easy(
        "count-missing",
        questions=[
            {
                "uid": "missing-q",
                "question": "How many countable facts?",
                "answer": "1",
                "derivation": "",
                "answer_type": "count",
                "scale": "",
                "facts": ["Absent fact."],
            }
        ],
    )
    ambiguous = _tatqa_easy(
        "count-ambiguous",
        questions=[
            {
                "uid": "ambiguous-q",
                "question": "How many repeated facts?",
                "answer": "1",
                "derivation": "",
                "answer_type": "count",
                "scale": "",
                "facts": ["Repeated fact."],
            }
        ],
    )
    ambiguous["paragraphs"] = [
        {"uid": "first", "order": 1, "text": "Repeated fact."},
        {"uid": "second", "order": 2, "text": "Repeated fact."},
    ]
    source = _snapshot(tmp_path, "tatqa", "invalid-counts.json", [missing, ambiguous])

    adapted = adapt_primary_snapshots((source,), seed=4)

    assert adapted.cases == ()
    assert {row.reason for row in adapted.rejections} == {
        "missing_counted_evidence",
        "ambiguous_counted_evidence",
    }


def test_tatqa_count_uses_n_minus_one_operations_for_headroom_strata(tmp_path: Path):
    count = _tatqa_easy(
        "count-headroom",
        questions=[
            {
                "uid": "count-headroom-q",
                "question": "How many reported countable items were there?",
                "answer": "3",
                "derivation": "",
                "answer_type": "count",
                "scale": "",
                "facts": ["Alpha item.", "Beta item.", "Gamma item."],
            }
        ],
    )
    count["table"]["table"] = [["Metric", "Value"]]
    count["paragraphs"] = [
        {
            "uid": "overview",
            "order": 1,
            "text": "How many reported countable items overview.",
        },
        {
            "uid": "supplement",
            "order": 2,
            "text": "Reported countable items supplemental discussion.",
        },
        {"uid": "alpha", "order": 3, "text": "Alpha item."},
        {"uid": "beta", "order": 4, "text": "Beta item."},
        {"uid": "gamma", "order": 5, "text": "Gamma item."},
    ]
    source = _snapshot(tmp_path, "tatqa", "count-headroom.json", [count])

    adapted = adapt_primary_snapshots((source,), seed=4)

    assert len(adapted.cases) == 1
    assert adapted.cases[0].public.stratum == "headroom"


def test_tatqa_rejects_ambiguous_operand_locations_instead_of_guessing(tmp_path: Path):
    ambiguous = _tatqa_easy("tatqa-ambiguous")
    ambiguous["paragraphs"].append(
        {"uid": "duplicate-value", "order": 2, "text": "A separate value was 7."}
    )
    source = _snapshot(tmp_path, "tatqa", "ambiguous.json", [ambiguous])

    adapted = adapt_primary_snapshots((source,), seed=4)

    assert adapted.cases == ()
    assert [row.reason for row in adapted.rejections] == ["ambiguous_evidence"]


def test_preparation_emits_exact_balanced_disjoint_splits_without_public_labels(
    tmp_path: Path,
):
    finqa = _snapshot(
        tmp_path,
        "finqa",
        "finqa.json",
        [_finqa_headroom(f"finqa-hard-{index}") for index in range(3)]
        + [_finqa_easy("finqa-easy")],
    )
    tatqa = _snapshot(
        tmp_path,
        "tatqa",
        "tatqa.json",
        [_tatqa_headroom(f"tatqa-hard-{index}") for index in range(3)]
        + [_tatqa_easy("tatqa-easy")],
    )

    prepared = prepare_primary_datasets(
        (finqa, tatqa),
        output_dir=tmp_path / "prepared",
        quotas=SplitQuotas(development=2, operational_pilot=2, main=2, easy_reserve=2),
        seed=29,
    )

    assert {name: len(split.public_cases) for name, split in prepared.splits.items()} == {
        "development": 2,
        "operational_pilot": 2,
        "main": 2,
        "easy_reserve": 2,
    }
    all_documents = [
        case.document_id
        for split in prepared.splits.values()
        for case in split.public_cases
    ]
    assert len(all_documents) == len(set(all_documents))
    for split in prepared.splits.values():
        assert {case.dataset for case in split.public_cases} == {"finqa", "tatqa"}
        assert [case.case_id for case in split.public_cases] == sorted(
            case.case_id for case in split.public_cases
        )

    forbidden = {"answer", "gold_derivation", "gold_support_ids", "source_lineage"}
    for path in (tmp_path / "prepared" / "public").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert forbidden.isdisjoint(json.loads(line))
    assert prepared.profile["document_disjoint"] is True
    assert prepared.artifact_hashes
    assert (tmp_path / "prepared" / "rejections.jsonl").is_file()
    assert (tmp_path / "prepared" / "profile.json").is_file()
    assert (tmp_path / "prepared" / "hashes.json").is_file()


def test_preparation_aborts_with_machine_readable_shortfalls(tmp_path: Path):
    invalid = _finqa_headroom("finqa-invalid")
    invalid["qa"]["program"] = "exp(2, 3)"
    finqa = _snapshot(
        tmp_path,
        "finqa",
        "finqa-short.json",
        [_finqa_headroom("finqa-hard"), invalid],
    )
    tatqa = _snapshot(
        tmp_path,
        "tatqa",
        "tatqa-short.json",
        [_tatqa_headroom("tatqa-hard")],
    )
    output = tmp_path / "shortfall"

    with pytest.raises(PreparationAbort) as captured:
        prepare_primary_datasets(
            (finqa, tatqa),
            output_dir=output,
            quotas=SplitQuotas(development=2, operational_pilot=0, main=2, easy_reserve=2),
            seed=7,
        )

    assert captured.value.report["status"] == "aborted"
    assert captured.value.report["shortfalls"] == {
        "finqa:easy_control": 1,
        "finqa:headroom": 1,
        "tatqa:easy_control": 1,
        "tatqa:headroom": 1,
    }
    assert json.loads((output / "preparation_discrepancy.json").read_text()) == (
        captured.value.report
    )
    rejections = [
        json.loads(line)
        for line in (output / "rejections.jsonl").read_text().splitlines()
    ]
    profile = json.loads((output / "profile.json").read_text())
    hashes = json.loads((output / "hashes.json").read_text())
    assert [row["reason"] for row in rejections] == ["unsupported_operation"]
    assert all(row["detail"] is None for row in rejections)
    assert profile["status"] == "aborted"
    assert profile["reason"] == "quota_shortfall"
    assert profile["shortfalls"] == captured.value.report["shortfalls"]
    assert profile["rejections"] == {"unsupported_operation": 1}
    assert profile["document_disjoint"] is True
    assert set(hashes["artifact_hashes"]) == {
        "preparation_discrepancy.json",
        "profile.json",
        "rejections.jsonl",
    }
    assert all(
        hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
        for filename, digest in hashes["artifact_hashes"].items()
    )
    public_diagnostics = json.dumps({"rejections": rejections, "profile": profile})
    assert "gold_derivation" not in public_diagnostics
    assert "gold_support_ids" not in public_diagnostics
    assert '"answer"' not in public_diagnostics


def test_snapshot_checksum_changes_require_explicit_repinning(tmp_path: Path):
    source = _snapshot(tmp_path, "finqa", "pinned.json", [_finqa_headroom()])
    original = adapt_primary_snapshots((source,), seed=1)
    payload = [_finqa_headroom()]
    payload[0]["title"] = "Changed pinned source"
    source.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        adapt_primary_snapshots((source,), seed=1)

    repinned = DatasetSnapshot(
        dataset="finqa",
        path=source.path,
        expected_sha256=hashlib.sha256(source.path.read_bytes()).hexdigest(),
    )
    changed = adapt_primary_snapshots((repinned,), seed=1)
    assert changed.source_hashes != original.source_hashes
    assert changed.cases[0].public.metadata.title == "Changed pinned source"


def test_source_identifiers_distinguish_same_basenames_and_join_lineage(tmp_path: Path):
    snapshots = []
    for index in range(2):
        directory = tmp_path / f"source-{index}"
        directory.mkdir()
        path = directory / "train.json"
        path.write_text(
            json.dumps([_finqa_headroom(f"same-name-{index}")], sort_keys=True),
            encoding="utf-8",
        )
        snapshots.append(
            DatasetSnapshot(
                dataset="finqa",
                path=path,
                expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )

    adapted = adapt_primary_snapshots(tuple(snapshots), seed=8)

    assert len(adapted.source_hashes) == 2
    assert all(key.startswith("finqa:train.json:") for key in adapted.source_hashes)
    assert {case.hidden.source_lineage[0] for case in adapted.cases} == set(
        adapted.source_hashes
    )

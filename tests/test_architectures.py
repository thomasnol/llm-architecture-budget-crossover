import json

import pytest

from budget_crossover.architectures import allocate_budget, extract_answer


@pytest.mark.parametrize(
    ("total", "weights", "count"),
    [(256, [0.25] * 4, 4), (512, [0.44, 0.2, 0.36], 3), (4096, [1.0], 1)],
)
def test_allocate_budget_is_exact_and_respects_floor(total, weights, count):
    allocation = allocate_budget(total, weights)
    assert len(allocation) == count
    assert sum(allocation) == total
    assert min(allocation) >= 64


def test_extract_answer_from_json_and_fence():
    payload = {"answer": "Yes, in appetite.", "rationale": "Rule applies."}
    assert extract_answer(f"```json\n{json.dumps(payload)}\n```") == "Yes, in appetite."


def test_budget_below_architecture_floor_fails():
    with pytest.raises(ValueError):
        allocate_budget(255, [0.25] * 4)

import pytest

from budget_crossover.v2_judging import _strict_bool


def test_judge_boolean_fields_are_strict():
    assert _strict_bool({"value": False}, "value") is False
    assert _strict_bool({"value": True}, "value") is True
    with pytest.raises(TypeError):
        _strict_bool({"value": "false"}, "value")

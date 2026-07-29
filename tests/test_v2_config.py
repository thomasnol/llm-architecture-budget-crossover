from pathlib import Path

from budget_crossover.v2_config import load_v2_config

REPO = Path(__file__).resolve().parents[1]


def test_shell_style_default_model_ids_expand(monkeypatch):
    monkeypatch.delenv("GATEWAY_MODEL_GENERATOR", raising=False)
    monkeypatch.delenv("GATEWAY_MODEL_VERIFIER", raising=False)
    config = load_v2_config(REPO / "configs" / "v2_pilot.yaml")
    assert config.generator_model == "gpt-5.4-mini"
    assert config.verifier_model == "gpt-5.4"
    assert all("${" not in model for model in config.judge_models)


def test_environment_model_override_expands(monkeypatch):
    monkeypatch.setenv("GATEWAY_MODEL_GENERATOR", "deployment-mini")
    config = load_v2_config(REPO / "configs" / "v2_pilot.yaml")
    assert config.generator_model == "deployment-mini"

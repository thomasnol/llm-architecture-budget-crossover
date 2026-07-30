import pytest


@pytest.fixture(autouse=True)
def prevent_repository_dotenv_loading(monkeypatch):
    """Keep tests independent of production credentials in the local .env."""
    monkeypatch.setattr("budget_crossover.gateway.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("budget_crossover.config.load_dotenv", lambda *a, **k: None)

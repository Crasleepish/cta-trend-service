from src.api.health import health


def test_health_returns_ok() -> None:
    assert health() == "ok"

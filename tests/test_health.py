from types import SimpleNamespace
from unittest.mock import MagicMock

from src.api.health import health
from src.core.config import AppConfig, DbConfig


def test_health_returns_ok() -> None:
    engine = MagicMock()
    connection = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection

    config = AppConfig(db=DbConfig(dsn="postgresql://user:pass@localhost/db"))

    app = SimpleNamespace(state=SimpleNamespace(config=config, engine=engine))
    request = SimpleNamespace(app=app)

    response = health(request)
    assert response.status == "ok"
    assert response.version == config.strategy.default_version

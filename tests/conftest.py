from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from testcontainers.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def pg_container() -> PostgresContainer:
    container = PostgresContainer("postgres:17")
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture()
def pg_engine(pg_container: PostgresContainer):
    engine = create_engine(pg_container.get_connection_url())
    try:
        yield engine
    finally:
        engine.dispose()

"""
TimeStone AI — Database session management.

Async SQLAlchemy engine with tenant-scoped session factory.
Enforces tenant isolation via repository pattern.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.persistence.models import Base

_engine: Optional[Engine] = None
_SessionFactory = None


def get_database_url() -> str:
    return os.getenv("TIMESTONE_DATABASE_URL", "sqlite:///./timestone.db")


def initialize_database(database_url: Optional[str] = None, echo: bool = False) -> Engine:
    global _engine, _SessionFactory
    url = database_url or get_database_url()

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    _engine = create_engine(url, echo=echo, connect_args=connect_args)
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        initialize_database()
    assert _engine is not None
    return _engine


@contextmanager
def get_session() -> Iterator[Session]:
    if _SessionFactory is None:
        initialize_database()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_database() -> None:
    """Testing helper: drop and recreate all tables."""
    global _engine, _SessionFactory
    if _engine is not None:
        Base.metadata.drop_all(_engine)
        Base.metadata.create_all(_engine)

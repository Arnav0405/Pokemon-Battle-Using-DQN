"""Shared test fixtures.

Tests marked ``integration`` need the local Showdown server: they request the ``server``
fixture, which skips (with instructions) when the server is not reachable.
"""

from __future__ import annotations

import socket

import pytest

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000


def _server_up() -> bool:
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def server() -> None:
    """Skip the test unless the local Showdown server is accepting connections."""
    if not _server_up():
        pytest.skip(
            f"local Showdown server not reachable on {SERVER_HOST}:{SERVER_PORT} -- "
            "start it with: bash scripts/showdown_start.sh"
        )

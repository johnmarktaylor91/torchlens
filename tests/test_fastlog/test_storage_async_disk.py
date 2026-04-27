"""Async disk-mode placeholder tests for fastlog v1."""

from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="v1 ships sync-only; async pending safetensors 0.8 stable + benchmarks")
def test_async_disk_storage_pending() -> None:
    """Async disk storage is intentionally not shipped in v1."""

    raise NotImplementedError("async disk storage is not implemented in v1")

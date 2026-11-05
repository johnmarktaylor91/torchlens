"""MPS headroom for the save budget: torch.mps queries behind ``HAS_MPS_MEMORY_QUERY``."""

from __future__ import annotations

import pytest
import torch

from torchlens import _memory_probes  # noqa: F401  (monkeypatched by the tests)
from torchlens._memory_probes import available_device_bytes
from torchlens._utils import _torch_compat


def test_mps_headroom_is_measured_through_the_torch_mps_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS headroom is recommended-max minus driver-allocated, bounded by host RAM."""

    from torchlens._utils import _torch_compat

    monkeypatch.setattr(_torch_compat, "HAS_MPS_MEMORY_QUERY", True)
    fake_mps = type(
        "FakeMps",
        (),
        {
            "recommended_max_memory": staticmethod(lambda: 10 * 1024**3),
            "driver_allocated_memory": staticmethod(lambda: 3 * 1024**3),
        },
    )
    monkeypatch.setattr(torch, "mps", fake_mps, raising=False)
    monkeypatch.setattr(_memory_probes, "_available_host_bytes", lambda: 8 * 1024**3)
    assert available_device_bytes(torch.device("mps")) == 7 * 1024**3
    monkeypatch.setattr(_memory_probes, "_available_host_bytes", lambda: 5 * 1024**3)
    assert available_device_bytes(torch.device("mps")) == 5 * 1024**3
    monkeypatch.setattr(fake_mps, "driver_allocated_memory", staticmethod(lambda: 11 * 1024**3))
    assert available_device_bytes(torch.device("mps")) == 0


def test_mps_headroom_is_none_without_the_torch_mps_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``HAS_MPS_MEMORY_QUERY`` an MPS device stays honestly unmeasured."""

    monkeypatch.setattr(_torch_compat, "HAS_MPS_MEMORY_QUERY", False)
    assert available_device_bytes(torch.device("mps")) is None


def test_mps_headroom_is_none_when_a_query_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising torch.mps query reports unmeasured, never a guess."""

    def _boom() -> int:
        raise RuntimeError("no MPS device")

    monkeypatch.setattr(_torch_compat, "HAS_MPS_MEMORY_QUERY", True)
    fake_mps = type(
        "FakeMps",
        (),
        {
            "recommended_max_memory": staticmethod(_boom),
            "driver_allocated_memory": staticmethod(_boom),
        },
    )
    monkeypatch.setattr(torch, "mps", fake_mps, raising=False)
    assert available_device_bytes(torch.device("mps")) is None

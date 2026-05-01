"""Phase 4 capture target tests."""

from __future__ import annotations

from pathlib import Path

import torch

import torchlens as tl
from torchlens.options import CaptureOptions


class _KpiModel(torch.nn.Module):
    """Tiny model that records a KPI during capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a small forward pass."""

        y = torch.relu(x)
        tl.record_kpi_in_graph("loss", float(y.sum().detach()))
        return y


def test_capture_memory_fields_and_forward_lineno() -> None:
    """Populate Phase 4 LayerPassLog memory fields and forward line number."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    log = tl.log_forward_pass(model, torch.ones(1, 2))
    assert isinstance(log.forward_lineno, int)
    assert all(hasattr(layer, "bytes_delta_at_call") for layer in log.layer_list)
    assert all(hasattr(layer, "bytes_peak_at_call") for layer in log.layer_list)


def test_record_kpi_in_graph() -> None:
    """Attach arbitrary KPI metadata during capture."""

    log = tl.log_forward_pass(_KpiModel(), torch.ones(1, 2))
    assert "loss" in log.capture_kpis


def test_content_hash_cache_hit_and_miss(tmp_path: Path) -> None:
    """Reuse a content-hash capture cache entry."""

    model = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    capture = CaptureOptions(cache=True, cache_dir=tmp_path)
    first = tl.log_forward_pass(model, x, capture=capture)
    second = tl.log_forward_pass(model, x, capture=capture)
    assert first.capture_cache_hit is False
    assert second.capture_cache_hit is True
    assert first.capture_cache_key == second.capture_cache_key


def test_tied_parameter_notation_smoke() -> None:
    """Expose tied-parameter notation when shared storage is detected."""

    class Tied(torch.nn.Module):
        """Tiny tied-parameter model."""

        def __init__(self) -> None:
            """Initialize tied layers."""

            super().__init__()
            self.a = torch.nn.Linear(2, 2, bias=False)
            self.b = torch.nn.Linear(2, 2, bias=False)
            self.b.weight = self.a.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run tied layers."""

            return self.b(self.a(x))

    log = tl.log_forward_pass(Tied(), torch.ones(1, 2))
    tied = [
        layer.extra_data.get("tied_parameter_notation")
        for layer in log.layer_list
        if layer.extra_data.get("tied_parameter_notation")
    ]
    assert tied

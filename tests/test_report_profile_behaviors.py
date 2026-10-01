"""Behavioral tests for report profiling, honesty labels, and capture-time logging."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402
from torchlens.report import build_profile  # noqa: E402

pytestmark = pytest.mark.smoke


class _TwoBlockModel(nn.Module):
    """Two named blocks so module/call level profiles have structure."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


@pytest.fixture(scope="module")
def profiled_trace() -> Iterator[tl.Trace]:
    """One captured trace shared across profile assertions."""

    torch.manual_seed(6)
    trace = tl.trace(_TwoBlockModel(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_build_profile_validates_level_sort_and_top_k(profiled_trace) -> None:
    """Invalid level/sort_by/top_k values are refused with clear errors."""

    with pytest.raises(ValueError, match="level must be"):
        build_profile(profiled_trace, level="layerwise")
    with pytest.raises(ValueError, match="sort_by must be"):
        build_profile(profiled_trace, sort_by="speed")
    with pytest.raises(ValueError, match="top_k"):
        build_profile(profiled_trace, top_k=-1)
    with pytest.raises(ValueError, match="top_k"):
        build_profile(profiled_trace, top_k=True)


def test_profile_repr_formats_durations_human_readably(profiled_trace) -> None:
    """__repr__ renders the time column in s/ms/us units."""

    profile = build_profile(profiled_trace, level="op")
    text = repr(profile)
    assert any(unit in text for unit in (" s", " ms", " us"))
    assert "time" in text


def test_profile_honesty_labels_align_with_rows_and_vocabulary(profiled_trace) -> None:
    """honesty() returns per-row provenance labels from the closed vocabulary."""

    profile = build_profile(profiled_trace, level="op", top_k=3)
    honesty = profile.honesty()
    assert list(honesty["name"]) == list(profile.frame["name"])
    # F09 (costreport D2/D9): per-cell evidence with mandatory qualifiers --
    # host time is never bare "measured", parameter counts are a declared
    # inventory (formula_exact), FLOPs carry their compute-record evidence.
    from torchlens.report._profile import HONESTY_LABELS

    for column in ("time", "flops", "activation_memory", "param_count"):
        assert set(honesty[column]) <= set(HONESTY_LABELS)
    assert "measured" not in set(honesty["time"])
    # Cached honesty frames are returned as copies, not aliases.
    assert honesty is not profile.honesty()


def test_profile_top_k_and_sort_order(profiled_trace) -> None:
    """top_k truncates after a stable descending sort on the requested column."""

    full = build_profile(profiled_trace, level="op", sort_by="param_count")
    top = build_profile(profiled_trace, level="op", sort_by="param_count", top_k=2)
    assert len(top.frame) == 2
    assert list(top.frame["name"]) == list(full.frame["name"][:2])
    counts = [c for c in full.frame["param_count"] if c == c]
    assert counts == sorted(counts, reverse=True)


def test_profile_module_and_call_levels_and_tree(profiled_trace) -> None:
    """module/call granularities include block rows and the call tree renders."""

    module_profile = build_profile(profiled_trace, level="module")
    names = set(module_profile.frame["name"])
    assert any("encoder" in str(name) for name in names)

    call_profile = build_profile(profiled_trace, level="call")
    assert len(call_profile.frame) >= len(module_profile.frame)
    tree = call_profile.tree()
    assert "encoder" in tree

    assert isinstance(module_profile.to_pandas(), pd.DataFrame)
    assert module_profile.to_pandas() is not module_profile.frame


def test_multipass_module_profile_covers_every_pass() -> None:
    """A module called twice contributes both passes to its module row."""

    class _Recurrent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.linear(x))

    torch.manual_seed(8)
    log = tl.trace(_Recurrent(), torch.randn(2, 3))
    module_profile = build_profile(log, level="module")
    row = module_profile.frame[module_profile.frame["name"].astype(str) == "linear"]
    assert len(row) == 1
    assert int(row["op_count"].iloc[0]) >= 2


def test_log_value_records_during_capture_and_refuses_outside() -> None:
    """report.log_value stores values on the active trace and refuses idle calls."""

    class _LoggingModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tl.report.log_value("gate_mean", float(x.detach().mean().item()))
            return torch.relu(x)

    torch.manual_seed(9)
    x = torch.randn(2, 3)
    log = tl.trace(_LoggingModel(), x)
    logged = log.annotations["logged_values"]
    assert logged["gate_mean"] == pytest.approx(float(x.mean().item()))

    with pytest.raises(RuntimeError, match="during trace"):
        tl.report.log_value("outside", 1)

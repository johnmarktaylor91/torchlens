"""Device-time consumers (torchnative W2.3): hot_path, color_by, size_by.

The joined table feeds the shipped surfaces as COLUMNS, never schemas: the
session-time join registry serves ``hot_path(by="device_time")`` and
``draw(color_by=/size_by="device_time")``; a trace with no joined session
refuses typed (``device_time_unavailable``) instead of rendering silent
zeros, and joined-but-kernel-less nodes read honest n/a.

Synthetic joins are registered through the same public seam the door uses
(``register_join_result``), so consumer semantics are testable on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import TorchLensError
from torchlens.observability import join_events, register_join_result
from torchlens.observability._kineto import NormalizedEvent


@pytest.fixture()
def mlp_log() -> Any:
    """Small captured trace, cleaned up after the test."""

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    log = tl.trace(model, torch.randn(2, 8))
    yield log
    log.cleanup()


def _synthetic_join_for(log: Any) -> None:
    """Register a joined result whose kernels land on the first two ops."""

    ops = [op for op in log.ops if op.func_call_id is not None][:2]
    assert len(ops) == 2
    events: list[NormalizedEvent] = []
    for index, op in enumerate(ops):
        base = index * 100_000
        events.append(
            NormalizedEvent(
                name=f"torchlens::op::{op.func_call_id}",
                kind="marker",
                activity="user_annotation",
                start_ns=base,
                end_ns=base + 50_000,
                tid=1,
                device_type="cpu",
                device_index=None,
                correlation_id=None,
                is_user_annotation=True,
                scope=None,
            )
        )
        events.append(
            NormalizedEvent(
                name="cudaLaunchKernel",
                kind="runtime",
                activity="cuda_runtime",
                start_ns=base + 10_000,
                end_ns=base + 11_000,
                tid=1,
                device_type="cpu",
                device_index=None,
                correlation_id=index + 1,
                is_user_annotation=False,
                scope=None,
            )
        )
        events.append(
            NormalizedEvent(
                name=f"kernel_{index}",
                kind="kernel",
                activity="kernel",
                start_ns=1_000_000 + index * 100_000,
                end_ns=1_000_000 + index * 100_000 + (index + 1) * 10_000,
                tid=None,
                device_type="cuda",
                device_index=0,
                correlation_id=index + 1,
                is_user_annotation=False,
                scope=None,
            )
        )
    result = join_events(tuple(events), extraction_path="in_memory", trace=log)
    assert result.availability == "joined"
    register_join_result(log, result)


def test_hot_path_device_time_refuses_without_join(mlp_log) -> None:
    """No joined session -> typed refusal, never silent zeros."""

    from torchlens.debug import hot_path

    with pytest.raises(TorchLensError) as excinfo:
        hot_path(mlp_log, by="device_time")
    assert excinfo.value.fields["code"] == "device_time_unavailable"
    assert excinfo.value.fields["remedy"]


@pytest.mark.smoke
def test_hot_path_device_time_ranks_joined_ops(mlp_log) -> None:
    """Joined device nanoseconds rank source lines."""

    _synthetic_join_for(mlp_log)
    from torchlens.debug import hot_path_rows

    rows, attrs = hot_path_rows(mlp_log, by="device_time")
    assert attrs["metric"] == "device_time"
    total = sum(row["total_cost"] for row in rows)
    assert total == pytest.approx(30_000)  # 10us + 20us of synthetic kernels
    # Ops the join attributed nothing to are DISCLOSED as excluded.
    assert attrs["excluded_missing_metric_count"] >= 1


def test_draw_color_by_device_time(mlp_log, tmp_path: Path) -> None:
    """color_by='device_time' fills joined nodes; kernel-less read n/a."""

    _synthetic_join_for(mlp_log)
    mlp_log.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        color_by="device_time",
    )
    state = mlp_log._last_encoding_state
    values = [value for value in state.values.values() if value is not None]
    assert len(values) == 2
    assert sorted(values) == [10_000.0, 20_000.0]


def test_draw_color_by_device_time_refuses_without_join(mlp_log, tmp_path: Path) -> None:
    """The explicitly requested channel refuses typed when no join exists."""

    with pytest.raises(TorchLensError) as excinfo:
        mlp_log.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / "graph"),
            color_by="device_time",
        )
    assert excinfo.value.fields["code"] == "device_time_unavailable"


@pytest.mark.smoke
def test_draw_size_by_device_time(mlp_log, tmp_path: Path) -> None:
    """size_by='device_time' sizes joined nodes from the same table."""

    _synthetic_join_for(mlp_log)
    mlp_log.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        size_by="device_time",
    )
    state = mlp_log._last_encoding_state
    sizes = [value for value in state.size_values.values() if value is not None]
    assert len(sizes) == 2

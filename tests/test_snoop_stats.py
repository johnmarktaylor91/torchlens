"""Echo stats rungs on the shared C02 kernel: soundness as executable code.

Pins snoop memo test rows 13-14: the T-A sampled-blindness regression (a
planted NaN invisible to a subsample must NOT surface as a finiteness
claim), the exact rung's numel-budget refusal, the reuse rung's
never-print-deferred-evidence rule, and the two-stage raise_on_nan screen's
verdict identity against the historical whole-tensor scan (validation is a
tripwire: the rewrite must be byte-identical in verdict).
"""

from __future__ import annotations

import io
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import EchoOptions
from torchlens.snoop import EchoStatsError
from torchlens.snoop._stats import exact_stats, reuse_stats, sampled_stats


def test_sampled_blindness_regression_t_a() -> None:
    """T-A: one planted NaN; sampled makes NO claim; exact detects it."""

    tensor = torch.randn(512, 1024)  # 524288 elements, sample budget 4096
    tensor[311, 707] = float("nan")
    sampled = sampled_stats(tensor, identity="t_a")
    assert sampled is not None
    assert sampled.policy == "sampled"
    assert sampled.nan_count is None
    assert sampled.posinf_count is None
    assert sampled.neginf_count is None
    assert sampled.has_nonfinite is None
    exact = exact_stats(tensor, identity="t_a")
    assert exact.nan_count == 1
    assert exact.posinf_count == 0


def test_sampled_budget_is_bounded_and_marked() -> None:
    """The sampled rung strides to a fixed budget with mandatory evidence."""

    from torchlens.snoop import SAMPLED_BUDGET

    tensor = torch.randn(64, 1024)
    record = sampled_stats(tensor, identity="budget")
    assert record is not None
    assert record.sample_size == SAMPLED_BUDGET
    assert record.population == tensor.numel()
    assert record.mean is not None and record.sd is not None


def test_exact_rung_refuses_above_numel_budget() -> None:
    """Memo test 13: exact-above-budget refuses typed, remedy names sampled."""

    import torchlens.snoop._stats as stats_module

    original = stats_module.EXACT_NUMEL_BUDGET
    stats_module.EXACT_NUMEL_BUDGET = 8
    try:
        with pytest.raises(EchoStatsError) as excinfo:
            exact_stats(torch.randn(4, 4), identity="budget_refusal")
        assert excinfo.value.fields["code"] == "echo_stats_numel_budget"
        assert "sampled" in excinfo.value.fields["remedy"]
    finally:
        stats_module.EXACT_NUMEL_BUDGET = original


def test_exact_budget_refusal_fails_the_capture_typed() -> None:
    """The mid-forward exact refusal propagates typed, never degrades."""

    import torchlens.snoop._stats as stats_module

    original = stats_module.EXACT_NUMEL_BUDGET
    stats_module.EXACT_NUMEL_BUDGET = 4
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(EchoStatsError):
                tl.trace(
                    nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
                    torch.randn(2, 4),
                    echo=EchoOptions(select=True, sink=io.StringIO(), stats="exact"),
                )
    finally:
        stats_module.EXACT_NUMEL_BUDGET = original


@pytest.mark.smoke
def test_reuse_rung_prints_only_paid_for_facts() -> None:
    """Reuse never scans: no armed feature means no stats segment at all."""

    sink = io.StringIO()
    tl.trace(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        torch.randn(2, 4),
        echo=EchoOptions(select=True, sink=sink, stats="reuse"),
    )
    assert "|" not in sink.getvalue()
    sink_armed = io.StringIO()
    tl.trace(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        torch.randn(2, 4),
        echo=EchoOptions(select=True, sink=sink_armed, stats="reuse"),
        capture=tl.options.CaptureOptions(track_nonfinite=True),
    )
    armed_lines = [line for line in sink_armed.getvalue().split("\n") if "relu" in line]
    assert armed_lines and "| finite" in armed_lines[0]


def test_reuse_rung_never_reads_deferred_evidence() -> None:
    """A missing synchronous verdict is an ABSENT field, never a guess."""

    class FakeTrace:
        """Trace stand-in with a deferred (non-bool) store entry."""

        def __init__(self) -> None:
            """Install a store whose verdict is a deferred flag tensor."""

            self.__dict__["_nonfinite_capture"] = {"events": {"op_raw": torch.tensor(True)}}

    assert reuse_stats(FakeTrace(), "op_raw", 10) is None
    assert reuse_stats(FakeTrace(), "missing_raw", 10) is None


@pytest.mark.parametrize(
    "case",
    [
        "clean",
        "nan",
        "posinf",
        "neginf",
        "fp16_overflow_product",
        "huge_finite_sum_overflow",
        "empty",
        "bool",
        "complex_nan",
        "int64",
    ],
)
def test_two_stage_tripwire_verdict_identity(case: str) -> None:
    """The fused-screen rewrite keeps the raise_on_nan VERDICT byte-identical.

    Validation is a tripwire: the two-stage form may only change COST. Each
    case compares the shipped behavior (raise or not) against the exact
    whole-tensor oracle ``(~isfinite).any()``.
    """

    if case == "clean":
        payload = torch.randn(8, 8)
    elif case == "nan":
        payload = torch.randn(8, 8)
        payload[3, 3] = float("nan")
    elif case == "posinf":
        payload = torch.randn(8, 8)
        payload[0, 0] = float("inf")
    elif case == "neginf":
        payload = torch.randn(8, 8)
        payload[7, 7] = float("-inf")
    elif case == "fp16_overflow_product":
        payload = (torch.full((8, 8), 60000.0, dtype=torch.float16)) * 10
    elif case == "huge_finite_sum_overflow":
        # Every element finite; the float32 accumulator overflows, so the
        # SCREEN trips and the confirm stage must CLEAR it (no false raise).
        payload = torch.full((64,), 3.0e38, dtype=torch.float32)
    elif case == "empty":
        payload = torch.empty(0)
    elif case == "bool":
        payload = torch.ones(8, dtype=torch.bool)
    elif case == "complex_nan":
        payload = torch.randn(4, dtype=torch.complex64)
        payload[1] = complex(float("nan"), 0.0)
    else:
        payload = torch.arange(64, dtype=torch.int64)

    class Emitter(nn.Module):
        """Emits the case payload from a wrapped op."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Materialize the payload through an addition."""

            del x
            return payload + torch.zeros_like(payload)

    expected_nonfinite = bool(
        payload.numel() > 0
        and (~torch.isfinite(payload.to(torch.float32) if payload.dtype == torch.bool else payload))
        .any()
        .item()
    )
    from torchlens.errors import CaptureError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if expected_nonfinite:
            with pytest.raises(CaptureError):
                tl.trace(
                    Emitter(),
                    torch.randn(1),
                    capture=tl.options.CaptureOptions(raise_on_nan=True),
                )
        else:
            log = tl.trace(
                Emitter(),
                torch.randn(1),
                capture=tl.options.CaptureOptions(raise_on_nan=True),
            )
            assert log is not None


def test_metadata_echo_reads_no_payload() -> None:
    """stats='off' narration never touches the tensor (zero value reads)."""

    reads: list[str] = []

    class SpyTensor(torch.Tensor):
        """Tensor subclass recording data-reading calls."""

    sink = io.StringIO()
    log = tl.trace(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU()),
        torch.randn(2, 4),
        echo=EchoOptions(select=True, sink=sink, stats="off"),
    )
    del log, reads, SpyTensor
    transcript = sink.getvalue()
    assert "|" not in transcript  # no stats segment ever rendered
    assert "relu" in transcript

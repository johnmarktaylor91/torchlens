"""Executed-backward FLOPs: the backward that ran, counted op by op.

The pre-registered oracle (``tests/fixtures/flops1_oracle_cases.json``, counts
only) fixes the bar before the counter existed: on the matmul family the
counted backward equals FlopCounterMode's backward exactly, and on the zoo it
equals the closed-form backward exactly. ``matmul 2D@1D`` dispatches ``mv``,
which FlopCounterMode does not count (it reports 0); TorchLens counts the
``mv`` that computes the vector's gradient and names the elementwise ``mul``
the matrix's gradient (an outer product) decomposes into. Those three cases
are disclosed against the closed form, never called an error on either side.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report import backward_estimate, backward_status

ORACLE = json.loads(
    (Path(__file__).parent / "fixtures" / "flops1_oracle_cases.json").read_text(encoding="utf-8")
)
MATMUL_CASES = {row["case"]: row for row in ORACLE["matmul_family"]}
ZOO_CASES = {row["case"]: row for row in ORACLE["zoo"]}
M, K, N, BATCH, B = 3, 5, 7, 4, 2

#: matmul 2D@1D: TorchLens counts the mv of the vector gradient (2MK) and names
#: the matrix gradient's mul; FlopCounterMode counts 0 for both.
MV_DISCLOSED = {
    "matmul_2d_1d[a_grad=True,b_grad=True]": 2 * M * K,
    "matmul_2d_1d[a_grad=True,b_grad=False]": 0,
    "matmul_2d_1d[a_grad=False,b_grad=True]": 2 * M * K,
}


def _leaf(shape: tuple[int, ...], grad: bool, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).requires_grad_(grad)


class _Call(nn.Module):
    """Wrap one tensor expression so the leaves enter as traced inputs."""

    def __init__(self, fn: Any) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.fn(*xs)


def _matmul_case(name: str, ga: bool, gb: bool) -> tuple[nn.Module, list[torch.Tensor]]:
    if name == "mm":
        return _Call(torch.mm), [_leaf((M, K), ga, 1), _leaf((K, N), gb, 2)]
    if name == "addmm":
        return _Call(lambda a, b, bias: torch.addmm(bias, a, b)), [
            _leaf((M, K), ga, 1),
            _leaf((K, N), gb, 2),
            _leaf((N,), True, 3),
        ]
    if name == "bmm":
        return _Call(torch.bmm), [_leaf((BATCH, M, K), ga, 1), _leaf((BATCH, K, N), gb, 2)]
    if name == "baddbmm":
        return _Call(lambda a, b, acc: torch.baddbmm(acc, a, b)), [
            _leaf((BATCH, M, K), ga, 1),
            _leaf((BATCH, K, N), gb, 2),
            _leaf((BATCH, M, N), True, 3),
        ]
    if name == "matmul_3d_2d":
        return _Call(torch.matmul), [_leaf((BATCH, M, K), ga, 1), _leaf((K, N), gb, 2)]
    if name == "matmul_2d_1d":
        return _Call(torch.matmul), [_leaf((M, K), ga, 1), _leaf((K,), gb, 2)]
    if name == "linear_bias_2d":
        return _Call(nn.functional.linear), [
            _leaf((M, K), ga, 1),
            _leaf((N, K), gb, 2),
            _leaf((N,), True, 3),
        ]
    if name == "linear_bias_3d":
        return _Call(nn.functional.linear), [
            _leaf((BATCH, M, K), ga, 1),
            _leaf((N, K), gb, 2),
            _leaf((N,), True, 3),
        ]
    if name == "einsum_bmm":
        return _Call(lambda a, b: torch.einsum("bij,bjk->bik", a, b)), [
            _leaf((BATCH, M, K), ga, 1),
            _leaf((BATCH, K, N), gb, 2),
        ]
    raise AssertionError(name)


def _build_matmul_case(case: str) -> tuple[nn.Module, list[torch.Tensor]]:
    name, flags = case.split("[")
    if name == "conv2d":
        x_grad = "x_grad=True" in flags
        return nn.Conv2d(3, 4, 3, bias=True), [_leaf((2, 3, 8, 8), x_grad, 4)]
    return _matmul_case(name, "a_grad=True" in flags, "b_grad=True" in flags)


def _true_mac_flops(trace: Any) -> int:
    """fma2 forward over the MAC family, read off the flops report (2 x true MACs)."""

    import re

    from torchlens.report import flops

    match = re.search(r"true MACs: (\d+)", str(flops(trace)))
    assert match is not None
    return 2 * int(match.group(1))


def _assert_forward(trace: Any, case: str, expected: int) -> None:
    """Forward fma2 equals the oracle (``F.linear`` with an input weight and the
    fused ``nn.RNN`` kernel included; both read unknown before their rules)."""

    del case
    assert _true_mac_flops(trace) == expected


def _traced_backward(model: nn.Module, inputs: list[torch.Tensor]) -> Any:
    trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(backward_ready=True))
    out = trace[trace.output_layers[0]].out
    trace.backward(out if out.dim() == 0 else out.sum())
    return trace


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
# Rows sampled by shrink-r4 (per-id subset rule; the rows kept hold every per-id campaign kill).
@pytest.mark.parametrize(
    "case",
    sorted(
        c
        for c in MATMUL_CASES
        if c
        in {"linear_bias_3d[a_grad=False,b_grad=True]", "matmul_2d_1d[a_grad=False,b_grad=True]"}
    ),
)
def test_matmul_family_backward_equals_flop_counter_mode(case: str) -> None:
    """Counted executed backward == FlopCounterMode's backward, per case, exactly."""

    row = MATMUL_CASES[case]
    trace = _traced_backward(*_build_matmul_case(case))
    try:
        status = backward_status(trace)
        assert status.state == "backward_observed"
        if case in MV_DISCLOSED:
            assert row["fcm_bwd"] == 0 and row["closed_bwd"] > 0
            assert status.executed_flops == MV_DISCLOSED[case], status
            if "a_grad=True" in case:
                assert ("mul", 1) in status.uncounted_op_counts
            return
        assert status.executed_flops == row["fcm_bwd"] == row["closed_bwd"], status
        assert status.executed_evidence == "formula_exact"
        _assert_forward(trace, case, row["fcm_fwd"])
    finally:
        trace.cleanup()


def _linear(i: int, o: int, seed: int, grad: bool = True) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(i, o).requires_grad_(grad)


class _Stack(nn.Module):
    def __init__(self, frozen: tuple[bool, bool, bool]) -> None:
        super().__init__()
        self.l1 = _linear(16, 32, 11, not frozen[0])
        self.l2 = _linear(32, 32, 12, not frozen[1])
        self.l3 = _linear(32, 4, 13, not frozen[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


class _Unrolled(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_ih, self.w_hh = _linear(8, 16, 41), _linear(16, 16, 42)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(B, 16)
        outs = []
        for step in range(5):
            state = torch.tanh(self.w_ih(x[:, step]) + self.w_hh(state))
            outs.append(state)
        return torch.stack(outs, 1)


class _RnnOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(31)
        self.rnn = nn.RNN(8, 16, num_layers=1, nonlinearity="tanh", batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0]


class _Attention(nn.Module):
    def __init__(self, heads: int, d: int = 8, t: int = 6) -> None:
        super().__init__()
        self.heads, self.d, self.t = heads, d, t
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads, d, t = self.heads, self.d, self.t
        dh = d // heads

        def split(z: torch.Tensor) -> torch.Tensor:
            return z.view(B, t, heads, dh).transpose(1, 2).reshape(B * heads, t, dh)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(dh)
        out = torch.bmm(torch.softmax(scores, -1), v)
        return self.o(out.view(B, heads, t, dh).transpose(1, 2).reshape(B, t, d))


class _FrozenConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3).requires_grad_(False)
        self.head = nn.Linear(4 * 6 * 6, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.conv(x)).flatten(1))


ZOO_BUILDERS = {
    "frozen_backbone": lambda: (_Stack((True, True, False)), [_leaf((B, 16), False, 5)]),
    "partial_requires_grad": lambda: (_Stack((False, True, False)), [_leaf((B, 16), False, 6)]),
    "rnn_module": lambda: (_RnnOut(), [_leaf((B, 5, 8), False, 9)]),
    "rnn_unrolled": lambda: (_Unrolled(), [_leaf((B, 5, 8), False, 10)]),
    "eager_attention_h1": lambda: (_Attention(1), [_leaf((B, 6, 8), False, 62)]),
    "eager_attention_h2": lambda: (_Attention(2), [_leaf((B, 6, 8), False, 72)]),
    "frozen_conv_backbone": lambda: (_FrozenConv(), [_leaf((B, 3, 8, 8), False, 52)]),
}


# Kept by shrink-r4 (per-id subset rule): sole coverage of report/_cost_tree.py.
# Rows sampled by shrink-r4 (per-id subset rule; the rows kept hold every per-id campaign kill).
@pytest.mark.parametrize(
    "case",
    sorted(c for c in ZOO_BUILDERS if c in {"eager_attention_h1", "rnn_unrolled"}),
)
def test_zoo_backward_equals_closed_form(case: str) -> None:
    """Counted executed backward == the closed-form backward, per zoo case, exactly."""

    row = ZOO_CASES[case]
    trace = _traced_backward(*ZOO_BUILDERS[case]())
    try:
        status = backward_status(trace)
        assert status.executed_flops == row["closed_bwd"] == row["fcm_bwd"], status
        assert status.executed_evidence == "formula_exact"
        assert row["closed_fwd"] == row["fcm_fwd"]
        _assert_forward(trace, case, row["closed_fwd"])
    finally:
        trace.cleanup()


def _small() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return nn.Linear(4, 3, bias=False), torch.randn(2, 4)


# Kept by shrink-r4 (per-id subset rule): sole coverage of report/_cost_backward.py.
def test_three_states_on_a_small_model() -> None:
    """grad off reads 0, grad on and unobserved reads unknown, observed reads the count."""

    model, x = _small()
    with torch.no_grad():
        off = tl.trace(model, x)
    unobserved = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
    observed = _traced_backward(model, [x])
    try:
        assert backward_status(off).state == "grad_disabled"
        assert backward_status(off).executed_flops == 0
        assert backward_status(unobserved).state == "grad_enabled_unobserved"
        assert backward_status(unobserved).executed_flops is None
        status = backward_status(observed)
        assert status.state == "backward_observed"
        assert status.executed_flops == 2 * 2 * 4 * 3
        assert "counted from the backward ops that ran" in status.status_line
        estimate = backward_estimate(observed)
        assert estimate.label == "hypothetical"
        assert (
            re.search(rf"(?<![\d.]){estimate.hypothetical_flops}(?!\d)", status.status_line) is None
        )
    finally:
        for trace in (off, unobserved, observed):
            trace.cleanup()


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_stats_grad_fn_site.py, _data_classes/backward_pass.py, report/_backward_attribution.py ....
def test_loaded_trace_reads_the_saved_count(tmp_path: Path) -> None:
    """The count is saved with the pass: a reloaded trace reads the live answer."""

    model, x = _small()
    trace = _traced_backward(model, [x])
    try:
        live = backward_status(trace)
        path = tmp_path / "observed.tlspec"
        trace.save(path)
        loaded = tl.load(path)
        status = backward_status(loaded)
        assert status.state == "backward_observed"
        assert status.executed_flops == live.executed_flops is not None
        assert status.executed_evidence == "formula_exact"
        assert status.executed_flops_per_pass == live.executed_flops_per_pass
        assert status.flops_attributions[0].rows == live.flops_attributions[0].rows
        assert status.unattributed_flops == live.unattributed_flops
    finally:
        trace.cleanup()

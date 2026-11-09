"""Op-level placement inside the first differing module (slice 3), on faked backends.

Two torch traces, the second stamped with the provenance backend ``"mlx"``; the spelling
lists read torch's (the ops are torch ops). Each acceptance fixture is a hand-decomposed
twin of one child module ``unit``, with and without an injected bug: a correct twin must
agree, a buggy one must be placed on the right op or operation group, never misplaced.
Real MLX twins live in ``tests/test_debug_cross_backend_mlx_ops.py``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens.debug import _xb_op_scope

pytest.importorskip("pandas")

_WIDTH = 6
_LABEL = re.compile(r"^(?P<kind>.+?)_\d+_\d+(?::\d+)?$")
_GROUP = re.compile(r"op group: a \[(?P<a>[^\]]*)\] against b \[(?P<b>[^\]]*)\]")
_SINGLE = re.compile(r"op: a (?P<a>\S+|\(no op\)) against b (?P<b>\S+|\(no op\));")


@pytest.fixture(autouse=True)
def _torch_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both faked sides log torch ops, so both read torch's spelling list."""

    monkeypatch.setattr(_xb_op_scope, "spelling_backend", lambda trace: "torch")


class Net(nn.Module):
    """``unit(x) * 1.5``: the unit is the only child, so it is the first differing module."""

    def __init__(self, unit: nn.Module) -> None:
        """Hold the unit."""

        super().__init__()
        self.unit = unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the unit and scale."""

        return self.unit(x) * 1.5


class Fn(nn.Module):
    """A unit from a function and named parameters."""

    def __init__(self, fn: Any, **params: torch.Tensor) -> None:
        """Register ``params`` and keep ``fn(x, **params)``."""

        super().__init__()
        self.fn = fn
        for name, value in params.items():
            self.register_parameter(name, nn.Parameter(value.detach().clone()))
        self.names = tuple(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``fn(x, **params)``."""

        return self.fn(x, **{name: getattr(self, name) for name in self.names})


def _x(*shape: int) -> torch.Tensor:
    """A deterministic input."""

    return torch.linspace(-1.0, 1.3, math.prod(shape)).reshape(shape)


def _first(unit_a: nn.Module, unit_b: nn.Module, x: torch.Tensor) -> dict[str, Any] | None:
    """``first_divergence`` of the two units' nets, side b faked as MLX."""

    trace_a = tl.trace(Net(unit_a).eval(), x)
    trace_b = tl.trace(Net(unit_b).eval(), x)
    trace_b._provenance_backend = "mlx"  # type: ignore[assignment]
    return tl.debug.first_divergence(trace_a, trace_b)


def _kinds(labels: str) -> list[str]:
    """Layer types of a comma-separated label list (``linear_1_1:1`` -> ``linear``)."""

    found = []
    for label in (part.strip() for part in labels.split(",")):
        match = _LABEL.match(label)
        if match is not None:
            found.append(match["kind"])
    return found


def _placed(row: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """The placed ops' layer types on each side, from the row's reason."""

    assert row is not None
    assert str(row["op"]).startswith("unit:"), row
    reason = str(row["reason"])
    match = _GROUP.search(reason) or _SINGLE.search(reason)
    assert match is not None, reason
    return _kinds(match["a"]), _kinds(match["b"])


torch.manual_seed(0)
_W = torch.randn(_WIDTH, _WIDTH) * 0.4
_W2 = torch.randn(_WIDTH, _WIDTH) * 0.4
_B = torch.randn(_WIDTH) * 0.3
_B2 = torch.randn(_WIDTH) * 0.3
_LN_W = torch.rand(_WIDTH) + 0.5
_LN_B = torch.randn(_WIDTH) * 0.1


def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.linear(x, w, b)


def _matmul_add(shift: float) -> Any:
    def fn(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, w.t()) + (b + shift if shift else b)

    return fn


# --- decomposed linear --------------------------------------------------------------------


def test_linear_against_matmul_add_agrees() -> None:
    assert _first(Fn(_linear, w=_W, b=_B), Fn(_matmul_add(0.0), w=_W, b=_B), _x(2, _WIDTH)) is None


def test_corrupted_bias_is_placed_on_the_decomposed_group() -> None:
    biased = _B.clone()
    biased[0] += 0.5
    row = _first(Fn(_linear, w=_W, b=_B), Fn(_matmul_add(0.0), w=_W, b=biased), _x(2, _WIDTH))
    assert _placed(row) == (["linear"], ["matmul", "add"])


def test_bias_bug_before_an_activation_closes_at_the_activation() -> None:
    def ref(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(F.linear(x, w, b))

    def port(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.matmul(x, w.t()) + b)

    biased = _B + 0.4
    row = _first(Fn(ref, w=_W, b=_B), Fn(port, w=_W, b=biased), _x(2, _WIDTH))
    assert _placed(row) == (["linear"], ["matmul", "add"])
    assert row is not None and "tie broken by spelling" in str(row["reason"])


def test_wrong_activation_after_a_decomposed_linear_is_one_op() -> None:
    def ref(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(F.linear(x, w, b))

    def port(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.gelu(torch.matmul(x, w.t()) + b)

    row = _first(Fn(ref, w=_W, b=_B), Fn(port, w=_W, b=_B), _x(2, _WIDTH))
    assert _placed(row) == (["relu"], ["gelu"])
    assert row is not None and "decomposition gaps" in str(row["reason"])


# --- manual layer norm ----------------------------------------------------------------------


def _layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (_WIDTH,), w, b, 1e-5)


def _manual_norm(eps: float) -> Any:
    def fn(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps) * w + b

    return fn


def test_manual_layer_norm_agrees() -> None:
    unit_b = Fn(_manual_norm(1e-5), w=_LN_W, b=_LN_B)
    assert _first(Fn(_layer_norm, w=_LN_W, b=_LN_B), unit_b, _x(2, _WIDTH)) is None


def test_manual_layer_norm_eps_bug_is_placed_on_the_norm_group() -> None:
    unit_b = Fn(_manual_norm(0.5), w=_LN_W, b=_LN_B)
    kinds_a, kinds_b = _placed(_first(Fn(_layer_norm, w=_LN_W, b=_LN_B), unit_b, _x(2, _WIDTH)))
    assert kinds_a == ["layernorm"]
    assert {"mean", "sqrt"} <= set(kinds_b)


# --- channels-last convolution through permutes ---------------------------------------------


class Conv(nn.Module):
    """A 3x3 convolution, optionally through a channels-last round trip."""

    def __init__(self, conv: nn.Conv2d, back: tuple[int, ...] | None = None) -> None:
        """``back`` is the permutation from channels-last back to channels-first."""

        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(torch.as_tensor(conv.bias).detach().clone())
        self.back = back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve, after the round trip when ``back`` is set."""

        if self.back is not None:
            x = x.permute(0, 2, 3, 1).permute(*self.back)
        return F.conv2d(x, self.weight, self.bias, padding=1)


def test_channels_last_round_trip_agrees_and_bad_permute_is_inside_the_group() -> None:
    conv = nn.Conv2d(2, 3, 3, padding=1)
    x = _x(1, 2, 4, 4)
    assert _first(Conv(conv), Conv(conv, (0, 3, 1, 2)), x) is None
    kinds_a, kinds_b = _placed(_first(Conv(conv), Conv(conv, (0, 3, 2, 1)), x))
    assert kinds_a == ["conv2d"]
    assert "permute" in kinds_b and "conv2d" in kinds_b


# --- fused against explicit attention -------------------------------------------------------


def _fused(x: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(x, x, x)


def _explicit(scale: float) -> Any:
    def fn(x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, x.transpose(-2, -1)) * scale
        return torch.matmul(torch.softmax(scores, dim=-1), x)

    return fn


def test_explicit_attention_agrees_and_missing_scale_is_placed_on_the_attention_group() -> None:
    x = _x(1, 4, _WIDTH)
    assert _first(Fn(_fused), Fn(_explicit(1 / math.sqrt(_WIDTH))), x) is None
    kinds_a, kinds_b = _placed(_first(Fn(_fused), Fn(_explicit(1.0)), x))
    assert kinds_a == ["scaleddotproductattention"]
    assert {"matmul", "softmax"} <= set(kinds_b)


# --- sign, transpose, siblings -------------------------------------------------------------


def test_sign_bug_is_placed_on_the_extra_negation() -> None:
    row = _first(Fn(lambda x: x + 1), Fn(lambda x: -(x + 1)), _x(2, _WIDTH))
    assert _placed(row) == ([], ["neg"])


def test_square_matrix_transpose_bug_is_one_op() -> None:
    def ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.linear(x, w)

    def port(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, w)

    def port_ok(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, w.t())

    assert _first(Fn(ref, w=_W), Fn(port_ok, w=_W), _x(2, _WIDTH)) is None
    assert _placed(_first(Fn(ref, w=_W), Fn(port, w=_W), _x(2, _WIDTH))) == (["linear"], ["matmul"])


def test_bias_bug_in_one_of_two_same_shaped_branches_is_placed_on_that_branch() -> None:
    def ref(
        x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor
    ) -> torch.Tensor:
        return F.linear(x, w, b) * F.linear(x, w2, b2)

    def port(
        x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor
    ) -> torch.Tensor:
        return (torch.matmul(x, w.t()) + b) * (torch.matmul(x, w2.t()) + b2)

    params = {"w": _W, "b": _B, "w2": _W2, "b2": _B2}
    assert _first(Fn(ref, **params), Fn(port, **params), _x(2, _WIDTH)) is None
    row = _first(Fn(ref, **params), Fn(port, **{**params, "b": _B + 0.3}), _x(2, _WIDTH))
    assert _placed(row) == (["linear"], ["matmul", "add"])
    assert row is not None and "linear_1_" in str(row["reason"]).split("op group: a [")[1][:12]


def test_zeros_then_bug_is_placed_after_the_zero_stage() -> None:
    def ref(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * 0.0) + x

    def port(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * 0.0) + x * 1.01

    kinds_a, kinds_b = _placed(_first(Fn(ref), Fn(port), _x(2, _WIDTH)))
    assert "relu" not in kinds_a + kinds_b
    assert "mul" in kinds_b


# --- inputs, saved traces -------------------------------------------------------------------


class Pre(nn.Module):
    """``unit(x * k)`` with the scaling outside any module."""

    def __init__(self, unit: nn.Module, k: float) -> None:
        """Hold the unit and the scale."""

        super().__init__()
        self.unit, self.k = unit, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale outside the unit, then run it."""

        return self.unit(x * self.k)


def test_differing_inputs_place_inside_the_enclosing_call_with_agreeing_inputs() -> None:
    trace_a = tl.trace(Pre(Fn(torch.tanh), 2.0).eval(), _x(2, _WIDTH))
    trace_b = tl.trace(Pre(Fn(torch.tanh), 3.0).eval(), _x(2, _WIDTH))
    trace_b._provenance_backend = "mlx"  # type: ignore[assignment]
    row = tl.debug.first_divergence(trace_a, trace_b)
    assert row is not None and str(row["op"]).startswith("unit:")
    reason = str(row["reason"])
    assert "first observed differing module output (inputs already differ)" in reason
    assert "skipped computation between matched boundaries" in reason
    assert "op placement inside self:" in reason, reason
    match = _SINGLE.search(reason)
    assert match is not None and (_kinds(match["a"]), _kinds(match["b"])) == (["mul"], ["mul"])


def test_glue_op_bug_between_child_modules_is_placed_in_the_parent() -> None:
    class Glue(nn.Module):
        """``out(act(inp(x)))`` with child projections and a functional activation."""

        def __init__(self, act: Any) -> None:
            super().__init__()
            torch.manual_seed(1)
            self.inp, self.out, self.act = nn.Linear(_WIDTH, _WIDTH), nn.Linear(_WIDTH, 3), act

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out(self.act(self.inp(x)))

    row = _first(Glue(torch.relu), Glue(torch.sigmoid), _x(2, _WIDTH))
    assert row is not None and str(row["op"]).startswith("unit.out:")
    assert "op placement inside unit:" in str(row["reason"])
    match = _SINGLE.search(str(row["reason"]))
    assert match is not None and (_kinds(match["a"]), _kinds(match["b"])) == (["relu"], ["sigmoid"])


def test_saved_and_reloaded_pair_gives_the_same_placement(tmp_path: Path) -> None:
    biased = _B + 0.4
    unit_a, unit_b = Fn(_linear, w=_W, b=_B), Fn(_matmul_add(0.0), w=_W, b=biased)
    trace_a = tl.trace(Net(unit_a).eval(), _x(2, _WIDTH))
    trace_b = tl.trace(Net(unit_b).eval(), _x(2, _WIDTH))
    trace_a.save(str(tmp_path / "a.tlspec"))
    trace_b.save(str(tmp_path / "b.tlspec"))
    trace_b._provenance_backend = "mlx"  # type: ignore[assignment]
    live = tl.debug.first_divergence(trace_a, trace_b)
    loaded_a, loaded_b = tl.load(str(tmp_path / "a.tlspec")), tl.load(str(tmp_path / "b.tlspec"))
    loaded_b._provenance_backend = "mlx"  # type: ignore[assignment]
    loaded = tl.debug.first_divergence(loaded_a, loaded_b)
    assert _placed(live) == _placed(loaded) == (["linear"], ["matmul", "add"])

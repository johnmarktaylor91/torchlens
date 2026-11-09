"""Module-level comparison across backends: ``align=``, ``module_map=`` and the evidence rule.

The other backend is faked: two torch traces, the second stamped with the provenance backend
``"mlx"`` and built with MLX-style container names (``layers.0``). Real MLX twins live in
``tests/test_debug_cross_backend_mlx.py``. Each fixture from the acceptance list is a hand-built twin, with and without an
injected bug, and must be named at the right module or refused, never misplaced.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens.debug import _cross_backend, _xb_calls
from torchlens.errors import InvalidArgumentError

pytest.importorskip("pandas")

_PAIRED = {"supported", "asserted"}


def _traces(
    model_a: nn.Module, model_b: nn.Module, x: torch.Tensor, x_b: torch.Tensor | None = None
) -> tuple[tl.Trace, tl.Trace]:
    """Trace both models (eval mode) and stamp the second as an MLX trace."""

    trace_a = tl.trace(model_a.eval(), x)
    trace_b = tl.trace(model_b.eval(), x if x_b is None else x_b)
    trace_b._provenance_backend = "mlx"  # type: ignore[assignment]
    return trace_a, trace_b


def _frame(trace_a: tl.Trace, trace_b: tl.Trace, **kwargs: Any) -> Any:
    """The module-level compare DataFrame."""

    return tl.debug.compare(trace_a, trace_b, **kwargs).to_pandas()


def _assert_evidence_rule(frame: Any) -> None:
    """Rule 2.1 as a property over every row."""

    for row in frame.to_dict("records"):
        if row["outcome"] == "different":
            assert row["alignment"] in _PAIRED, row
        if row["alignment"] not in _PAIRED:
            assert row["outcome"] == "not-evaluated", row
            assert row["allclose"] is None or (
                isinstance(row["allclose"], float) and math.isnan(row["allclose"])
            ), row
        if row["status"] != "present-in-both":
            assert row["outcome"] != "different", row


def _address(row: dict[str, Any] | None) -> str:
    """Side-a module address of a row."""

    assert row is not None
    return str(row["op"]).rsplit(":", 1)[0]


# --- twins -------------------------------------------------------------------------------


class MatmulAdd(nn.Module):
    """Linear spelled as matmul plus add, sharing a reference Linear's weights."""

    def __init__(self, linear: nn.Linear, bias_shift: float = 0.0) -> None:
        """Copy the weights; ``bias_shift`` injects a bug."""

        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone() + bias_shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ W.T + b``."""

        return torch.matmul(x, self.weight.t()) + self.bias


class ManualLayerNorm(nn.Module):
    """LayerNorm by hand."""

    def __init__(self, norm: nn.LayerNorm, eps_scale: float = 1.0) -> None:
        """Copy affine parameters; ``eps_scale`` far from 1 injects a bug."""

        super().__init__()
        self.weight = nn.Parameter(norm.weight.detach().clone())
        self.bias = nn.Parameter(norm.bias.detach().clone())
        self.eps = norm.eps * eps_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last axis."""

        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class Block(nn.Module):
    """``fc`` then ``norm``, the unit both twins share."""

    def __init__(self, fc: nn.Module, norm: nn.Module) -> None:
        """Hold the two children."""

        super().__init__()
        self.fc = fc
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``norm(relu(fc(x)))``."""

        return self.norm(torch.relu(self.fc(x)))


class TorchNet(nn.Module):
    """Reference: blocks in an ``nn.Sequential`` (torch addresses ``blocks.0``)."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        """Wrap the blocks."""

        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks."""

        return self.blocks(x)


class MlxStyleNet(nn.Module):
    """Port: blocks in a never-called list (MLX addresses ``layers.0``)."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        """Hold the blocks in a ModuleList, the analogue of an MLX Python list."""

        super().__init__()
        self.layers = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks in a loop."""

        for layer in self.layers:
            x = layer(x)
        return x


def _decomposed_pair(
    *, bias_shift: float = 0.0, eps_scale: float = 1.0
) -> tuple[nn.Module, nn.Module]:
    """A reference ``nn.Sequential`` (addresses ``0``, ``1``) and its decomposed MLX-style port."""

    torch.manual_seed(0)
    ref = [Block(nn.Linear(6, 6), nn.LayerNorm(6)) for _ in range(3)]
    port = [
        Block(
            MatmulAdd(block.fc, bias_shift if index == 1 else 0.0),
            ManualLayerNorm(block.norm, eps_scale if index == 2 else 1.0),
        )
        for index, block in enumerate(ref)
    ]
    return nn.Sequential(*ref), MlxStyleNet(port)


# --- correct ports agree -------------------------------------------------------------------


def test_correct_decomposed_port_agrees_with_normalized_addresses() -> None:
    """Sequential child ``0`` pairs with list ``layers.0``; a correct port returns None."""

    model_a, model_b = _decomposed_pair()
    x = torch.randn(2, 6)
    trace_a, trace_b = _traces(model_a, model_b, x)
    assert tl.debug.first_divergence(trace_a, trace_b) is None
    frame = _frame(trace_a, trace_b)
    _assert_evidence_rule(frame)
    paired = frame[frame["alignment"] == "supported"]
    assert set(paired["outcome"]) == {"equal"}
    assert (paired["rtol"] == 1e-4).all() and (paired["atol"] == 0.0).all()
    assert frame.attrs["value_diverged"] == 0
    assert frame.loc[frame["op"] == "1.fc:1", "module_b"].item() == "layers.1.fc:1"
    assert frame.attrs["coverage"]["a"]["unpaired"] == 0


@pytest.mark.parametrize(
    ("bug", "expected"),
    [({"bias_shift": 0.5}, "1.fc"), ({"eps_scale": 1e6}, "2.norm")],
)
def test_bug_in_a_decomposed_module_is_named_there(bug: dict[str, float], expected: str) -> None:
    """A bug inside MatmulAdd or ManualLayerNorm is localized at that module."""

    model_a, model_b = _decomposed_pair(**bug)
    trace_a, trace_b = _traces(model_a, model_b, torch.randn(2, 6))
    row = tl.debug.first_divergence(trace_a, trace_b)
    assert _address(row) == expected
    assert row is not None and row["inputs_agree"] is True
    assert row["reason"].startswith("localized inside this module")
    assert row["alignment"] == "supported" and row["outcome"] == "different"
    assert row["module_b"].startswith(f"layers.{expected}")
    _assert_evidence_rule(_frame(trace_a, trace_b))


# --- layout --------------------------------------------------------------------------------


class ConvNet(nn.Module):
    """Conv then ReLU; ``channels_last`` makes the conv return NHWC like MLX."""

    def __init__(self, conv: nn.Conv2d, channels_last: bool = False, shift: float = 0.0) -> None:
        """Hold a conv (wrapped NHWC when ``channels_last``)."""

        super().__init__()
        self.conv = ConvNHWC(conv, shift) if channels_last else conv
        self.act = nn.ReLU()
        self.channels_last = channels_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv, back to NCHW when needed, ReLU."""

        y = self.conv(x)
        if self.channels_last:
            y = y.permute(0, 3, 1, 2)
        return self.act(y)


class ConvNHWC(nn.Module):
    """A channels-last convolution through permutes."""

    def __init__(self, conv: nn.Conv2d, shift: float) -> None:
        """Copy the conv; ``shift`` injects a bug."""

        super().__init__()
        self.inner = copy.deepcopy(conv)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return NHWC output."""

        return (F.conv2d(x, self.inner.weight, self.inner.bias, padding=1) + self.shift).permute(
            0, 2, 3, 1
        )


def _conv_pair(
    channels: int, size: tuple[int, int], shift: float = 0.0
) -> tuple[tl.Trace, tl.Trace]:
    """Conv twins: torch NCHW against a channels-last port."""

    torch.manual_seed(1)
    conv = nn.Conv2d(3, channels, 3, padding=1)
    model_b = ConvNet(conv, channels_last=True, shift=shift)
    return _traces(ConvNet(conv), model_b, torch.randn(1, 3, *size))


def test_channels_last_conv_is_aligned_by_the_listed_transform() -> None:
    """Distinct C, H, W: the one fitting transform is applied and shown on the row."""

    trace_a, trace_b = _conv_pair(4, (5, 6))
    assert tl.debug.first_divergence(trace_a, trace_b) is None
    frame = _frame(trace_a, trace_b)
    conv = frame[frame["op"] == "conv:1"].iloc[0]
    assert conv["transform"] == "channels-last to channels-first (b)"
    assert conv["outcome"] == "equal" and not bool(conv["shape_match"])


def test_bug_in_channels_last_conv_is_named_at_the_conv() -> None:
    """The transform is chosen from shapes, so a value bug still shows at the conv."""

    row = tl.debug.first_divergence(*_conv_pair(4, (5, 6), shift=0.25))
    assert _address(row) == "conv"
    assert row is not None and row["transform"] == "channels-last to channels-first (b)"


def test_equal_channel_and_spatial_sizes_are_layout_unresolved() -> None:
    """C == H == W: identity and the transform both fit; never pick by values."""

    trace_a, trace_b = _conv_pair(4, (4, 4))
    frame = _frame(trace_a, trace_b)
    conv = frame[frame["op"] == "conv:1"].iloc[0]
    assert conv["outcome"] == "not-evaluated"
    assert "layout_unresolved" in conv["reason"]
    with pytest.raises(InvalidArgumentError) as caught:
        tl.debug.first_divergence(trace_a, trace_b)
    assert caught.value.fields["code"] == "module_comparison_inconclusive"


# --- attention -----------------------------------------------------------------------------


class Attention(nn.Module):
    """Single-head attention, fused or explicit."""

    def __init__(self, dim: int, fused: bool, scale_bug: float = 1.0) -> None:
        """Projections; ``scale_bug`` != 1 injects a bug in the explicit form."""

        super().__init__()
        torch.manual_seed(2)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.fused, self.scale_bug = fused, scale_bug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the sequence axis."""

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if self.fused:
            return F.scaled_dot_product_attention(q, k, v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) * self.scale_bug
        return torch.softmax(scores, dim=-1) @ v


class AttnNet(nn.Module):
    """Attention then an output projection."""

    def __init__(self, fused: bool, scale_bug: float = 1.0) -> None:
        """Build with fixed weights."""

        super().__init__()
        self.attn = Attention(8, fused, scale_bug)
        torch.manual_seed(3)
        self.out = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the attention output."""

        return self.out(self.attn(x))


@pytest.mark.parametrize(("scale_bug", "expected"), [(1.0, None), (1.05, "attn")])
def test_explicit_against_fused_attention(scale_bug: float, expected: str | None) -> None:
    """Explicit attention agrees with SDPA; a scale bug is named at ``attn``."""

    trace_a, trace_b = _traces(AttnNet(True), AttnNet(False, scale_bug), torch.randn(1, 5, 8))
    row = tl.debug.first_divergence(trace_a, trace_b)
    assert (row is None) if expected is None else (_address(row) == expected)


# --- localization edge cases ---------------------------------------------------------------


class Unit(nn.Module):
    """``x + 1``, or ``-(x + 1)`` when ``negate`` (the sign bug)."""

    def __init__(self, negate: bool = False) -> None:
        """Choose the sign."""

        super().__init__()
        self.negate = negate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shift, maybe negate."""

        y = x + 1
        return -y if self.negate else y


class SignThenSquare(nn.Module):
    """``square(unit(x))``: a sign bug in ``unit`` vanishes at the output."""

    def __init__(self, negate: bool) -> None:
        """Hold the unit."""

        super().__init__()
        self.unit = Unit(negate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Square the unit's output."""

        return self.unit(x) ** 2


def test_sign_then_square_difference_is_never_erased() -> None:
    """The output agrees again, but the sign bug is still named at ``unit``."""

    trace_a, trace_b = _traces(SignThenSquare(False), SignThenSquare(True), torch.randn(2, 3))
    row = tl.debug.first_divergence(trace_a, trace_b)
    assert _address(row) == "unit"
    frame = _frame(trace_a, trace_b)
    _assert_evidence_rule(frame)
    root = frame[frame["op"] == "self:1"].iloc[0]
    assert root["outcome"] == "equal"
    assert frame.attrs["value_diverged"] == 1


class TwoHeads(nn.Module):
    """Two same-shaped children; a bias bug goes in the second."""

    def __init__(self, shift: float) -> None:
        """Two Linear(4, 4) children with fixed weights."""

        super().__init__()
        torch.manual_seed(4)
        self.left, self.right = nn.Linear(4, 4), nn.Linear(4, 4)
        with torch.no_grad():
            self.right.bias += shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of both."""

        return self.left(x) + self.right(x)


def test_bug_in_one_of_two_same_shaped_children_is_named_at_that_child() -> None:
    """Pairing is by address, so the same-shaped sibling is never blamed."""

    trace_a, trace_b = _traces(TwoHeads(0.0), TwoHeads(0.3), torch.randn(2, 4))
    assert _address(tl.debug.first_divergence(trace_a, trace_b)) == "right"
    frame = _frame(trace_a, trace_b)
    assert frame.loc[frame["op"] == "left:1", "outcome"].item() == "equal"


class Square(nn.Module):
    """A square projection; the port returns its transpose (the bug)."""

    def __init__(self, transpose: bool) -> None:
        """Fixed weights."""

        super().__init__()
        torch.manual_seed(5)
        self.proj = nn.Linear(4, 4)
        self.transpose = transpose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project, maybe transpose, then a ReLU."""

        y = self.proj(x)
        return torch.relu(y.t() if self.transpose else y)


def test_square_matrix_transpose_bug_is_not_transformed_away() -> None:
    """Linear does not justify a transpose transform, so the transpose is a difference."""

    trace_a, trace_b = _traces(Square(False), Square(True), torch.randn(4, 4))
    row = tl.debug.first_divergence(trace_a, trace_b)
    assert _address(row) == "self"
    assert row is not None and row["transform"] == ""


class ZerosAndRepeats(nn.Module):
    """Children whose outputs are all zero and all one value."""

    def __init__(self) -> None:
        """A ReLU and a constant fill."""

        super().__init__()
        self.relu = nn.ReLU()
        self.fill = Fill()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Both children."""

        return self.relu(-x.abs() - 1) + self.fill(x)


class Fill(nn.Module):
    """Repeated values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A tensor of sevens."""

        return torch.full_like(x, 7.0)


def test_all_zero_and_repeated_value_outputs_agree() -> None:
    """Zero scale and repeated values compare without dividing by zero."""

    trace_a, trace_b = _traces(ZerosAndRepeats(), ZerosAndRepeats(), torch.randn(3, 3))
    assert tl.debug.first_divergence(trace_a, trace_b) is None
    assert set(_frame(trace_a, trace_b)["outcome"]) == {"equal"}


class ScaledThenNorm(nn.Module):
    """``norm(scale * fc(x))``: LayerNorm cancels a scale bug."""

    def __init__(self, scale: float) -> None:
        """Fixed weights."""

        super().__init__()
        torch.manual_seed(6)
        self.fc = Scaled(nn.Linear(6, 6), scale)
        self.norm = nn.LayerNorm(6, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the scaled projection."""

        return self.norm(self.fc(x))


class Scaled(nn.Module):
    """A Linear whose output is scaled."""

    def __init__(self, linear: nn.Linear, scale: float) -> None:
        """Hold the linear and the scale."""

        super().__init__()
        self.linear, self.scale = linear, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``scale * linear(x)``."""

        return self.linear(x) * self.scale


def test_scale_bug_cancelled_by_layernorm_is_named_at_the_scaled_module() -> None:
    """The norm's output agrees; the scaled module is still the finding."""

    trace_a, trace_b = _traces(ScaledThenNorm(1.0), ScaledThenNorm(2.0), torch.randn(2, 6))
    assert _address(tl.debug.first_divergence(trace_a, trace_b)) == "fc"
    frame = _frame(trace_a, trace_b)
    assert frame.loc[frame["op"] == "norm:1", "outcome"].item() == "equal"


class Branchy(nn.Module):
    """``head(stem(x) + branch(x))``; the port omits the early branch."""

    def __init__(self, with_branch: bool) -> None:
        """Fixed weights."""

        super().__init__()
        torch.manual_seed(7)
        self.stem, self.branch, self.head = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2)
        self.with_branch = with_branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maybe add the branch before the head."""

        h = self.stem(x)
        if self.with_branch:
            h = h + self.branch(x)
        return self.head(h)


def test_omitted_early_branch_is_a_coverage_gap_and_the_head_says_inputs_differ() -> None:
    """The missing branch is unpaired; the head is the first observed differing output."""

    trace_a, trace_b = _traces(Branchy(True), Branchy(False), torch.randn(2, 4))
    row = tl.debug.first_divergence(trace_a, trace_b)
    assert _address(row) == "head"
    assert row is not None and row["inputs_agree"] is False
    assert row["reason"].startswith("first observed differing module output")
    assert "skipped computation" in row["reason"]
    frame = _frame(trace_a, trace_b)
    branch = frame[frame["op"] == "branch:1"].iloc[0]
    assert (branch["status"], branch["alignment"], branch["outcome"]) == (
        "only-a",
        "absent",
        "not-evaluated",
    )
    _assert_evidence_rule(frame)


class Noisy(nn.Module):
    """Explicit randomness in eval mode: adds seeded noise."""

    def __init__(self, shift: float = 0.0) -> None:
        """Fixed weights; ``shift`` injects a bug after the noise."""

        super().__init__()
        torch.manual_seed(8)
        self.fc, self.shift = nn.Linear(4, 4), shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``fc(x + noise) + shift``."""

        return self.fc(x + torch.randn_like(x)) + self.shift


def test_explicit_randomness_in_eval_mode_is_disclosed() -> None:
    """A random producer ran on both sides: the report says so."""

    x = torch.randn(2, 4)
    torch.manual_seed(9)
    trace_a = tl.trace(Noisy().eval(), x)
    torch.manual_seed(9)
    trace_b = tl.trace(Noisy(0.5).eval(), x)
    trace_b._provenance_backend = "mlx"  # type: ignore[assignment]
    printed = str(tl.debug.compare(trace_a, trace_b))
    assert "random producers ran" in printed
    row = tl.debug.first_divergence(trace_a, trace_b)
    assert row is not None and "randomness" in row["reason"]


# --- batch -----------------------------------------------------------------------------------


def test_leading_size_one_axis_is_dropped_and_disclosed() -> None:
    """A batch of one against no batch axis aligns, with the transform on the row."""

    model_a, model_b = _decomposed_pair()
    x = torch.randn(1, 6)
    trace_a, trace_b = _traces(model_a, model_b, x, x[0])
    assert tl.debug.first_divergence(trace_a, trace_b) is None
    frame = _frame(trace_a, trace_b)
    paired = frame[frame["alignment"] == "supported"]
    assert set(paired["transform"]) == {"dropped leading size-one axis (a)"}


@pytest.mark.parametrize("tool", [tl.debug.compare, tl.debug.first_divergence])
def test_batch_one_against_sequence_one_refuses_at_the_input(tool: Callable[..., Any]) -> None:
    """``(1, 3, 6)`` against ``(3, 1, 6)`` is not a leading size-one axis: refused."""

    model_a, model_b = _decomposed_pair()
    x = torch.randn(1, 3, 6)
    trace_a, trace_b = _traces(model_a, model_b, x, x.transpose(0, 1).contiguous())
    with pytest.raises(InvalidArgumentError) as caught:
        tool(trace_a, trace_b)
    assert caught.value.fields["code"] == "module_comparison_inconclusive"
    assert "leading size-one axis" in str(caught.value)


# --- module_map, repeated calls, refusals ------------------------------------------------------


class Renamed(nn.Module):
    """A port that renamed ``blocks`` to ``stages``."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        """Hold the blocks under the new name."""

        super().__init__()
        self.stages = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stages."""

        return self.stages(x)


def test_module_map_pairs_renamed_modules_as_asserted() -> None:
    """Without a map the renamed modules are unpaired; with it they pair as asserted."""

    torch.manual_seed(10)
    ref = [Block(nn.Linear(4, 4), nn.LayerNorm(4)) for _ in range(2)]
    port = copy.deepcopy(ref)
    with torch.no_grad():
        port[1].fc.bias += 0.5
    trace_a, trace_b = _traces(TorchNet(ref), Renamed(port), torch.randn(2, 4))
    unmapped = _frame(trace_a, trace_b)
    assert "blocks.1.fc:1" in set(unmapped.loc[unmapped["status"] == "only-a", "op"])
    row = tl.debug.first_divergence(trace_a, trace_b, module_map={"blocks": "stages"})
    assert _address(row) == "blocks.1.fc"
    assert row is not None and row["alignment"] == "asserted"
    assert row["module_b"] == "stages.1.fc:1"
    _assert_evidence_rule(_frame(trace_a, trace_b, module_map={"blocks": "stages"}))


class Repeated(nn.Module):
    """One child called ``times`` times."""

    def __init__(self, times: int) -> None:
        """Fixed weights."""

        super().__init__()
        torch.manual_seed(11)
        self.fc, self.times = nn.Linear(3, 3), times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``fc`` repeatedly."""

        for _ in range(self.times):
            x = torch.tanh(self.fc(x))
        return x


def test_repeated_calls_pair_by_order_and_an_extra_call_is_disclosed() -> None:
    """Two calls pair in order; the port's third call is ``only-b``, never paired."""

    trace_a, trace_b = _traces(Repeated(2), Repeated(3), torch.randn(1, 3))
    frame = _frame(trace_a, trace_b)
    pairs = frame[frame["op"].isin(["fc:1", "fc:2"])]
    assert list(pairs["module_b"]) == ["fc:1", "fc:2"]
    assert set(pairs["outcome"]) == {"equal"}
    extra = frame[frame["module_b"] == "fc:3"].iloc[0]
    assert (extra["status"], extra["alignment"]) == ("only-b", "absent")
    _assert_evidence_rule(frame)


def test_same_backend_structure_pairing_keeps_same_backend_tolerance() -> None:
    """``align="structure"`` on two torch runs pairs modules with allclose defaults."""

    model_a, _ = _decomposed_pair()
    x = torch.randn(2, 6)
    trace_a, trace_b = tl.trace(model_a, x), tl.trace(model_a, x)
    frame = _frame(trace_a, trace_b, align="structure")
    assert set(frame["outcome"]) == {"equal"}
    assert set(frame["rtol"]) == {1e-5} and set(frame["atol"]) == {1e-8}
    assert tl.debug.first_divergence(trace_a, trace_b, align="structure") is None


def test_half_precision_rows_use_the_narrower_dtype_and_say_so() -> None:
    """float32 against bfloat16: rtol from bfloat16, and the row explains the floor."""

    model_a, model_b = _decomposed_pair()
    x = torch.randn(2, 6)
    trace_a, trace_b = _traces(model_a, model_b.to(torch.bfloat16), x, x.to(torch.bfloat16))
    frame = _frame(trace_a, trace_b)
    paired = frame[frame["alignment"] == "supported"]
    assert set(paired["rtol"]) == {7.5e-1}
    assert paired["reason"].str.contains("half precision").all()


def test_user_tolerance_overrides_the_constants() -> None:
    """An explicit rtol replaces the dtype constant on every row."""

    model_a, model_b = _decomposed_pair()
    trace_a, trace_b = _traces(model_a, model_b, torch.randn(2, 6))
    frame = _frame(trace_a, trace_b, rtol=1e-3)
    assert set(frame.loc[frame["alignment"] == "supported", "rtol"]) == {1e-3}
    assert frame.attrs["rtol"] == 1e-3


@pytest.mark.parametrize(
    "kwargs",
    [{"align": "value"}, {"module_map": {"a": 1}}, {"align": "label", "module_map": {"a": "b"}}],
)
def test_bad_align_or_module_map_refuses(kwargs: dict[str, Any]) -> None:
    """Unknown modes, malformed maps and a label join with a map refuse with one code."""

    trace = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    with pytest.raises(InvalidArgumentError) as caught:
        tl.debug.compare(trace, trace, **kwargs)
    assert caught.value.fields["code"] == "compare_align_invalid"


class Flat(nn.Module):
    """No submodules: the root is the only module boundary."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Arithmetic only."""

        return x * 2 + 1


def test_no_module_boundaries_refuses() -> None:
    """With only a root call there is nothing to localize."""

    trace_a, trace_b = _traces(Flat(), Flat(), torch.randn(2, 2))
    with pytest.raises(InvalidArgumentError, match="no module boundaries"):
        tl.debug.first_divergence(trace_a, trace_b)


def test_different_inputs_refuse_first_divergence() -> None:
    """Different model inputs make every module differ: refused, not blamed on a module."""

    model_a, model_b = _decomposed_pair()
    x = torch.randn(2, 6)
    trace_a, trace_b = _traces(model_a, model_b, x, x + 1)
    with pytest.raises(InvalidArgumentError, match="different inputs"):
        tl.debug.first_divergence(trace_a, trace_b)


# --- saved traces ------------------------------------------------------------------------------


def test_saved_and_reloaded_pair_gives_the_same_finding(tmp_path: Path) -> None:
    """The finding is the same from loaded artifacts with no live model."""

    model_a, model_b = _decomposed_pair(bias_shift=0.5)
    x = torch.randn(2, 6)
    live_a, live_b = _traces(model_a, model_b, x)
    live = tl.debug.first_divergence(live_a, live_b)
    torch_b = tl.trace(model_b.eval(), x)
    live_a.save(str(tmp_path / "a.tlspec"))
    torch_b.save(str(tmp_path / "b.tlspec"))
    loaded_a, loaded_b = tl.load(str(tmp_path / "a.tlspec")), tl.load(str(tmp_path / "b.tlspec"))
    loaded_b._provenance_backend = "mlx"  # type: ignore[assignment]
    loaded = tl.debug.first_divergence(loaded_a, loaded_b)
    assert live is not None and loaded is not None
    assert (_address(loaded), loaded["module_b"], loaded["outcome"]) == (
        _address(live),
        live["module_b"],
        live["outcome"],
    )


def _returning(label: str, *calls: str) -> SimpleNamespace:
    """A stub op flagged as an output of ``calls`` (the recorded-passthrough spelling)."""

    return SimpleNamespace(
        label=label,
        _module_context_is_module_output=True,
        _module_context_output_of_module_calls=calls,
    )


@pytest.mark.parametrize(
    ("own", "returned", "expected"),
    [
        ((), ("linear_1_1:1",), (("linear_1_1:1",), True)),
        (("relu_1_2:1",), ("linear_1_1:1",), (("relu_1_2:1", "linear_1_1:1"), False)),
        ((), ("linear_1_1:1", "linear_2_2:1"), ((), True)),
    ],
    ids=["passthrough", "own-then-returned", "two-returned-unresolved"],
)
def test_recorded_passthrough_is_read_as_the_call_output(
    own: tuple[str, ...], returned: tuple[str, ...], expected: tuple[tuple[str, ...], bool]
) -> None:
    """A returned input reads as the call's output; beside its own output the order is unknown."""

    call = SimpleNamespace(address="drop", label="drop:1", ops=list(own), output_ops=list(own))
    trace = SimpleNamespace(
        output_layers=[], _layer_list=[_returning(label, "drop:1") for label in (*returned, *own)]
    )
    assert _xb_calls._outputs(call, trace) == expected


def test_plain_container_calls_are_dropped_and_their_children_lifted() -> None:
    """A call recorded for a plain list is no call; its children take the list's parent."""

    def entry(label: str, class_name: str) -> tuple[SimpleNamespace, SimpleNamespace]:
        """A ``(module, call)`` pair whose module has ``class_name``."""

        return SimpleNamespace(class_name=class_name), SimpleNamespace(label=label, class_name="")

    raw = [entry("self:1", "Twin"), entry("blocks:1", "list"), entry("blocks.0:1", "Block")]
    parents = {"self:1": None, "blocks:1": "self:1", "blocks.0:1": "blocks:1"}
    kept, lifted = _xb_calls._drop_containers(raw, parents)
    assert [call.label for _module, call in kept] == ["self:1", "blocks.0:1"]
    assert lifted == {"self:1": None, "blocks.0:1": "self:1"}


class _AddConst(nn.Module):
    """``x + value``."""

    def __init__(self, value: float) -> None:
        """Hold the constant."""

        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the constant."""

        return x + self.value


def _zero_then_add(value: float) -> nn.Module:
    """A zero linear layer (all-zero output) followed by ``+ value``."""

    zero = nn.Linear(4, 4)
    nn.init.zeros_(zero.weight)
    nn.init.zeros_(zero.bias)
    return nn.Sequential(zero, _AddConst(value))


@pytest.mark.parametrize(
    ("shift", "atol", "outcome"),
    [(1e-12, None, "equal"), (1e-3, None, "different"), (1e-12, 0.0, "different")],
    ids=["tiny-partner-equal", "near-zero-bug-different", "user-atol-replaces-floor"],
)
def test_atol_floor_on_an_all_zero_reference(
    shift: float, atol: float | None, outcome: str
) -> None:
    """An all-zero reference meets a 1e-12 partner under the floor; a real bug still differs."""

    trace_a, trace_b = _traces(_zero_then_add(0.0), _zero_then_add(shift), torch.randn(2, 4))
    frame = _frame(trace_a, trace_b, atol=atol)
    row = frame.loc[frame["op"] == "1:1"].iloc[0]
    assert row["outcome"] == outcome, row["reason"]
    # The module's side-a input is all zero, so the floor's scale falls back to 1.0.
    expected_atol = 500 * torch.finfo(torch.float32).eps if atol is None else atol
    assert row["atol"] == pytest.approx(expected_atol)
    assert frame.loc[frame["op"] == "0:1", "outcome"].item() == "equal"


def test_module_frames_name_the_pairing_mode_and_backends() -> None:
    """Module frames carry ``align`` and ``backends``; op-pairing frames do not change."""

    model_a, model_b = _decomposed_pair()
    x = torch.randn(2, 6)
    trace_a, trace_b = _traces(model_a, model_b, x)
    attrs = _frame(trace_a, trace_b).attrs
    assert (attrs["align"], attrs["backends"]) == ("structure", ("torch", "mlx"))
    same_a, same_b = tl.trace(model_a, x), tl.trace(model_a, x)
    structure = _frame(same_a, same_b, align="structure").attrs
    assert (structure["align"], structure["backends"]) == ("structure", ("torch", "torch"))
    for align in ("auto", "label"):
        assert not {"align", "backends"} & set(_frame(same_a, same_b, align=align).attrs)


class _EagerAttention(nn.Module):
    """Returns ``(output, weights)`` but creates the weights first (HF eager attention)."""

    def __init__(self, proj: nn.Linear, shift: float = 0.0) -> None:
        """Share ``proj``; ``shift`` injects a bug into the output."""

        super().__init__()
        self.proj, self.shift = proj, shift

    def forward(self, x: torch.Tensor) -> Any:
        """Weights (batch by batch) first, then the projected output."""

        weights = torch.softmax(x @ x.transpose(0, 1), dim=-1)
        return self.proj(x) + self.shift, weights


class _SdpaAttention(_EagerAttention):
    """The same module returning only its output (an MLX-style attention)."""

    def forward(self, x: torch.Tensor) -> Any:
        """The projected output alone."""

        return self.proj(x) + self.shift


class _AttentionNet(nn.Module):
    """``head(attn(x)[0])``, reading the first returned value when there are two."""

    def __init__(self, attn: nn.Module, head: nn.Linear) -> None:
        """Hold the attention and the head."""

        super().__init__()
        self.attn, self.head = attn, head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention, keep its output, project."""

        out = self.attn(x)
        return self.head(out[0] if isinstance(out, tuple) else out)


def _attention_pair(shift: float = 0.0) -> tuple[tl.Trace, tl.Trace]:
    """Eager two-output attention on side a, a single-output port (``shift`` bug) on side b."""

    torch.manual_seed(0)
    proj, head = nn.Linear(6, 6), nn.Linear(6, 2)
    eager = _AttentionNet(_EagerAttention(proj), head)
    port = _AttentionNet(_SdpaAttention(copy.deepcopy(proj), shift), copy.deepcopy(head))
    return _traces(eager, port, torch.randn(3, 6))


def test_two_returned_values_pair_by_return_position_not_op_order() -> None:
    """The eager output (returned first, created second) meets the port's single output."""

    trace_a, trace_b = _attention_pair()
    assert tl.debug.first_divergence(trace_a, trace_b) is None
    row = _frame(trace_a, trace_b).set_index("op").loc["attn:1"]
    assert (row["alignment"], row["outcome"]) == ("supported", "equal")
    assert "1 more on side a not compared" in row["reason"]


def test_a_bug_behind_two_returned_values_is_named_at_its_module() -> None:
    """A bug in the port's attention output reads different at ``attn``, not shape-incompatible."""

    row = tl.debug.first_divergence(*_attention_pair(shift=0.5))
    assert row is not None
    assert (row["op"], row["outcome"]) == ("attn:1", "different")


@pytest.mark.parametrize(
    ("counts", "ordered", "expected"),
    [
        ((2, 2), (True, True), (2, "")),
        (
            (2, 1),
            (True, True),
            (1, "compared the first returned value; 1 more on side a not compared"),
        ),
        ((2, 1), (False, True), "return order of 2 values not recorded (a)"),
        ((3, 2), (True, True), "return trees differ (3 vs 2 values)"),
    ],
    ids=["equal", "one-against-two", "order-unknown", "trees-differ"],
)
def test_leaf_plan(counts: tuple[int, int], ordered: tuple[bool, bool], expected: Any) -> None:
    """Leaves pair by return position; unknown order or unequal trees refuse the row."""

    plan = _cross_backend._leaf_plan(*counts, ordered)
    if isinstance(expected, str):
        assert plan.outcome == "unavailable" and plan.reason == expected  # type: ignore[union-attr]
    else:
        assert plan == expected


class _ZeroWithMask(nn.Module):
    """Returns zeros shaped like ``x`` plus ``shift``, reading a mask-like second input."""

    def __init__(self, shift: float = 0.0) -> None:
        """Hold the bug shift."""

        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x * 0 + (mask * 0).sum() + shift``: all zero on the reference."""

        return x * 0.0 + (mask * 0.0).sum() + self.shift


class _MaskNet(nn.Module):
    """Builds a causal-mask-like tensor of huge finite values and feeds it beside ``x``."""

    def __init__(self, shift: float = 0.0) -> None:
        """Hold the zero module."""

        super().__init__()
        self.zero = _ZeroWithMask(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``zero(x, mask)`` with ``mask`` full of ``-3e38``."""

        return self.zero(x, torch.full_like(x, -3e38))


def test_a_mask_like_input_cannot_inflate_the_atol_floor() -> None:
    """On an all-zero output the floor scales by the smallest input rms, never the mask's."""

    x = torch.randn(2, 4)
    trace_a, trace_b = _traces(_MaskNet(), _MaskNet(shift=1e-3), x)
    row = _frame(trace_a, trace_b).set_index("op").loc["zero:1"]
    assert row["outcome"] == "different", row["reason"]
    floor = 500 * torch.finfo(torch.float32).eps * float(x.double().square().mean().sqrt())
    assert row["atol"] == pytest.approx(floor)
    assert tl.debug.first_divergence(trace_a, trace_b)["op"] == "zero:1"  # type: ignore[index]


class _ResidueLayerNorm(nn.LayerNorm):
    """LayerNorm plus a constant: the port's rounding residue (or a bias bug) on zero rows."""

    def __init__(self, width: int, residue: float) -> None:
        """Hold the residue added to the normalized output."""

        super().__init__(width)
        self.residue = residue

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, then add the residue."""

        return super().forward(x) + self.residue


@pytest.mark.parametrize(
    ("residue", "outcome"),
    [(1e-6, "equal"), (1e-4, "different")],
    ids=["rounding-residue-equal", "bias-bug-different"],
)
def test_float32_layernorm_over_constant_rows(residue: float, outcome: str) -> None:
    """Side a normalizes constant rows to exact zeros; a 1e-6 residue is rounding, 1e-4 a bug."""

    x = torch.full((2, 4), 0.5)
    trace_a, trace_b = _traces(
        nn.Sequential(nn.LayerNorm(4)), nn.Sequential(_ResidueLayerNorm(4, residue)), x
    )
    row = _frame(trace_a, trace_b).set_index("op").loc["0:1"]
    assert row["outcome"] == outcome, (row["atol"], row["max_abs"])
    floor = 500 * torch.finfo(torch.float32).eps * 0.5
    assert row["atol"] == pytest.approx(floor)

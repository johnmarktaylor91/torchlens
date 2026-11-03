"""The README's Debug promises hold: compare, grad_health, dead_neurons, summary(where=).

Each test pins one behaviour the README states and the code used to miss:
``first_divergence`` compares runs across dtypes and through integer and
boolean ops instead of skipping them; ``grad_health`` reaches a verdict on an
ordinary unscaled run with a library loss; ``dead_neurons`` counts units on the
feature axis (last axis for a transformer block, channel axis for a
convolution); ``summary(where=...)`` finds the ops of a named module by regex
and by selector; and ``first_divergence(italy, patched)`` names the patched
site when ``patched`` is a fork.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl


class _Threshold(nn.Module):
    """A float op followed by a comparison, so a mask can flip on a tiny change."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 1.0
        mask = y > 0.5
        return mask.float()


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_first_divergence_reports_a_flipped_boolean_mask() -> None:
    """A mask that flips while its float input stays within tolerance is the divergence."""

    model = _Threshold()
    trace_a = tl.trace(model, torch.tensor([[0.5 - 1e-7]]))
    trace_b = tl.trace(model, torch.tensor([[0.5 + 1e-7]]))

    row = tl.debug.first_divergence(trace_a, trace_b)

    assert row is not None
    assert row["op"].startswith("gt"), row
    assert row["allclose"] is False
    assert row["max_abs"] == 1.0


def test_compare_names_the_dtype_alignment_on_the_card() -> None:
    """float32 against float64 is compared, and the card says how."""

    torch.manual_seed(11)
    model = _Mlp()
    twin = _Mlp().double()
    twin.load_state_dict({key: value.double() for key, value in model.state_dict().items()})
    x = torch.randn(5, 4)

    report = tl.debug.compare(tl.trace(model, x), tl.trace(twin, x.double()))
    frame = report.to_pandas()

    assert frame["allclose"].notna().all()
    assert "dtypes differ (float32 vs float64): compared in float64" in str(report)


def test_grad_health_resolves_verdicts_for_cross_entropy_without_grad_scale() -> None:
    """A plain fp32 run with a library loss gets verdicts, not 'unknown'."""

    torch.manual_seed(12)
    model = _Mlp()
    trace = tl.trace(model, torch.randn(6, 4), capture=tl.options.CaptureOptions(save_grads=True))
    logits = trace[trace.output_layers[0]].out
    trace.backward(F.cross_entropy(logits, torch.tensor([0, 1, 2, 0, 1, 2])))

    frame = tl.debug.grad_health(trace).to_pandas()
    graded = frame[frame["verdict"] != "unavailable"]

    assert len(graded) > 0
    assert set(graded["scale_provenance"]) == {"proven_unscaled"}
    assert "unknown" not in set(graded["verdict"])


def test_grad_health_empty_hint_names_save_grads() -> None:
    """A forward-only trace says exactly which option produces a report."""

    trace = tl.trace(_Mlp(), torch.randn(2, 4))

    report = tl.debug.grad_health(trace)

    assert report.status == "empty"
    assert "CaptureOptions(save_grads=True)" in str(report)


class _Block(nn.Module):
    """A transformer-style feed-forward on (batch, tokens, features).

    The first ``dead`` features get a large negative bias and the rest a large
    positive one, so which features are dead does not depend on the draw.
    """

    def __init__(self, dead: int) -> None:
        super().__init__()
        self.up = nn.Linear(6, 10)
        with torch.no_grad():
            self.up.bias[:dead] = -100.0
            self.up.bias[dead:] = 100.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.up(x))


def _row(frame, prefix: str):
    rows = frame[frame["op"].str.startswith(prefix)]
    assert len(rows) == 1, frame
    return rows.iloc[0]


def test_dead_neurons_counts_transformer_features_on_the_last_axis() -> None:
    """(batch, tokens, features): units are the 10 features, not the 7 tokens."""

    torch.manual_seed(13)
    trace = tl.trace(_Block(dead=3), torch.randn(2, 7, 6))

    relu = _row(tl.debug.dead_neurons(trace).to_pandas(), "relu")

    assert relu["total_units"] == 10
    assert relu["dead_count"] == 3
    assert list(relu["sample_dead_idx"]) == [0, 1, 2]


class _ConvNet(nn.Module):
    """Channels 0 and 1 are dead (zero weight, negative bias); 2 to 4 vary."""

    def __init__(self, spatial: int) -> None:
        super().__init__()
        conv = nn.Conv1d if spatial == 1 else nn.Conv2d
        self.conv = conv(3, 5, 3)
        with torch.no_grad():
            self.conv.weight[:2].zero_()
            self.conv.bias.copy_(torch.tensor([-1.0, -1.0, 3.0, 3.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


@pytest.mark.parametrize("shape", [(4, 3, 12), (4, 3, 9, 9)], ids=["conv1d", "conv2d"])
def test_dead_neurons_counts_conv_channels(shape: tuple[int, ...]) -> None:
    """A conv output and the activation after it count the 5 channels."""

    torch.manual_seed(14)
    trace = tl.trace(_ConvNet(len(shape) - 2), torch.randn(*shape))
    frame = tl.debug.dead_neurons(trace).to_pandas()

    for prefix in ("conv", "relu"):
        row = _row(frame, prefix)
        assert row["total_units"] == 5, (prefix, row)
    relu = _row(frame, "relu")
    assert relu["dead_count"] == 2
    assert list(relu["sample_dead_idx"]) == [0, 1]


def test_dead_neurons_explicit_dim_still_wins() -> None:
    """An int dim forces that axis on every op."""

    trace = tl.trace(_Block(dead=0), torch.randn(2, 7, 6))

    relu = _row(tl.debug.dead_neurons(trace, dim=1).to_pandas(), "relu")

    assert relu["total_units"] == 7


class _Nested(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.mlp = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block.mlp(x))


@pytest.fixture(scope="module")
def nested_trace() -> Iterator[tl.Trace]:
    torch.manual_seed(15)
    captured = tl.trace(_Nested(), torch.randn(3, 4))
    try:
        yield captured
    finally:
        captured.cleanup()


@pytest.mark.parametrize(
    "where",
    [
        pytest.param("block.mlp", id="regex-module-path"),
        pytest.param(tl.in_module("block.mlp"), id="selector-in-module"),
    ],
)
def test_summary_where_finds_the_ops_of_a_module(nested_trace: tl.Trace, where: object) -> None:
    """At op level, a module path or tl.in_module keeps that module's two ops."""

    text = str(nested_trace.summary(level="op", where=where))

    assert "showing 2/3 rows" in text
    assert "linear_1_1" in text
    assert "relu_1_2" in text
    assert "linear_2_3" not in text


def test_summary_where_selector_by_function(nested_trace: tl.Trace) -> None:
    """A function selector keeps only the ops it names."""

    text = str(nested_trace.summary(level="op", where=tl.func("relu")))

    assert "showing 1/3 rows" in text
    assert "relu_1_2" in text


def test_summary_where_callable_still_receives_rows(nested_trace: tl.Trace) -> None:
    """A plain callable still filters the typed rows."""

    text = str(nested_trace.summary(level="op", where=lambda row: row.name.startswith("linear")))

    assert "showing 2/3 rows" in text


def test_summary_filter_spelling_refuses_naming_where(nested_trace: tl.Trace) -> None:
    """The removed filter= spelling refuses and names where=."""

    with pytest.raises(tl.errors.TorchLensError, match="where="):
        nested_trace.summary(level="op", filter="block.mlp")


class _Two(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def test_first_divergence_names_the_patched_module_on_a_fork() -> None:
    """The README's shape: patched = italy.fork().do(...); italy stays untouched."""

    torch.manual_seed(16)
    model = _Two().eval()
    capture = tl.options.CaptureOptions(replayable=True)
    france = tl.trace(model, torch.randn(2, 4), capture=capture)
    italy = tl.trace(model, torch.randn(2, 4), capture=capture)

    patched = italy.fork().do(tl.module("b"), tl.patch_from(france))
    row = tl.debug.first_divergence(italy, patched)

    assert patched is not italy
    assert row is not None
    assert row["op"].startswith("linear_2"), row
    assert tl.debug.first_divergence(italy, italy) is None


class _TanhGelu(nn.Module):
    """A linear layer under a hand-written tanh GELU, as GPT-2's MLP spells it.

    The first ``dead`` units get bias -100, so their GELU output is exactly
    zero (``tanh`` saturates to -1 in float32); the rest get bias +100 and
    pass through. The composite's inner ops (``pow``, ``mul``, ``add``,
    ``tanh``) are not layer outputs.
    """

    def __init__(self, dead: int) -> None:
        super().__init__()
        self.up = nn.Linear(6, 10)
        with torch.no_grad():
            self.up.bias[:dead] = -100.0
            self.up.bias[dead:] = 100.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        return 0.5 * h * (1.0 + torch.tanh(0.7978845608 * (h + 0.044715 * h**3)))


class _ZeroedSlice(nn.Module):
    """Units 2, 3 and 4 of ``up`` have zero weights and bias; unit 5 is a nonzero constant.

    Every other unit gets bias 100, so which units are dead does not depend on the draw.
    """

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(6, 10)
        self.act = nn.ReLU()
        with torch.no_grad():
            self.up.weight[2:6].zero_()
            self.up.bias[:2] = 100.0
            self.up.bias[2:5] = 0.0
            self.up.bias[5] = 5.0
            self.up.bias[6:] = 100.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.up(x))


class _NegativeGelu(nn.Module):
    """A GELU module whose first three units see inputs near -3: negative, never zero."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(6, 10)
        self.act = nn.GELU()
        with torch.no_grad():
            self.up.weight.mul_(0.01)
            self.up.bias[:3] = -3.0
            self.up.bias[3:] = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.up(x))


def test_dead_neurons_rows_are_layer_outputs_not_composite_intermediates() -> None:
    """A tanh GELU's inner tanh, pow, mul and add ops get no row; the layer's output does."""

    torch.manual_seed(15)
    frame = tl.debug.dead_neurons(tl.trace(_TanhGelu(dead=3), torch.randn(2, 7, 6))).to_pandas()

    assert len(frame) == 2, frame
    prefixes = {str(label).split("_")[0] for label in frame["op"]}
    assert "tanh" not in prefixes and "pow" not in prefixes and "add" not in prefixes, frame
    linear = _row(frame, "linear")
    output = frame.iloc[-1]
    assert linear["dead_count"] == 0
    assert output["total_units"] == 10
    assert output["dead_count"] == 3
    assert list(output["sample_dead_idx"]) == [0, 1, 2]


def test_dead_neurons_reports_exactly_the_zeroed_slice() -> None:
    """Hand-zeroed units 2 to 4 are the dead ones; a nonzero constant unit is not dead."""

    torch.manual_seed(16)
    trace = tl.trace(_ZeroedSlice(), torch.randn(4, 6))
    frame = tl.debug.dead_neurons(trace).to_pandas()

    for prefix in ("linear", "relu"):
        row = _row(frame, prefix)
        assert row["total_units"] == 10, row
        assert row["dead_count"] == 3, row
        assert list(row["sample_dead_idx"]) == [2, 3, 4], row


def test_dead_neurons_max_activation_is_absolute_on_the_output() -> None:
    """``max_activation`` bounds the unit's output magnitude at every position."""

    torch.manual_seed(17)
    trace = tl.trace(_ZeroedSlice(), torch.randn(4, 6))

    loose = _row(tl.debug.dead_neurons(trace, max_activation=5.0).to_pandas(), "relu")
    tight = _row(tl.debug.dead_neurons(trace, max_activation=4.999).to_pandas(), "relu")

    assert list(loose["sample_dead_idx"]) == [2, 3, 4, 5]
    assert list(tight["sample_dead_idx"]) == [2, 3, 4]


def test_dead_neurons_negative_outputs_are_not_dead() -> None:
    """GELU outputs near -0.004 are small and negative, not zero: no unit is dead."""

    torch.manual_seed(18)
    frame = tl.debug.dead_neurons(tl.trace(_NegativeGelu(), torch.randn(4, 6))).to_pandas()

    gelu = _row(frame, "gelu")
    assert gelu["total_units"] == 10
    assert gelu["dead_count"] == 0


def _tiny_gpt2(activation: str) -> tuple[Any, torch.Tensor]:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(19)
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=16,
        vocab_size=64,
        n_positions=32,
        activation_function=activation,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = transformers.GPT2Model(config).eval()
    return model, torch.tensor([[1, 2, 3, 4, 5]])


@pytest.mark.parametrize("activation", ["gelu_new", "silu"])
def test_dead_neurons_on_a_gpt2_block_skips_the_activation_internals(activation: str) -> None:
    """GPT-2's tanh GELU and SiLU MLPs: the activation module's output has a row, its insides do not."""

    model, ids = _tiny_gpt2(activation)
    frame = tl.debug.dead_neurons(tl.trace(model, ids)).to_pandas()

    labels = [str(label) for label in frame["op"]]
    assert not any(label.startswith(("tanh", "sigmoid", "pow")) for label in labels), labels
    counted = frame[frame["reason"] == ""]
    assert len(counted) >= 2 * 8, labels
    assert int(counted["dead_count"].sum()) == 0, counted


class _Composite(nn.Module):
    """A linear layer under a tanh GELU or a SiLU written out of torch ops, default init."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.up = nn.Linear(6, 10)
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        if self.kind == "silu":
            return h * torch.sigmoid(h)
        return 0.5 * h * (1.0 + torch.tanh(0.7978845608 * (h + 0.044715 * h**3)))


def _grad_frame(model: nn.Module, x: torch.Tensor) -> Any:
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
    trace.log_backward(trace[trace.output_layers[0]].out.square().mean())
    frame = tl.debug.grad_health(trace).to_pandas()
    return frame[frame["verdict"] != "unavailable"]


@pytest.mark.parametrize("kind", ["gelu", "silu"])
def test_grad_health_on_composite_activations_reports_no_dead_op(kind: str) -> None:
    """Every op of an unsaturated tanh GELU or SiLU carries gradient: no 'dead' verdict."""

    torch.manual_seed(20)
    graded = _grad_frame(_Composite(kind), torch.randn(4, 6))

    assert len(graded) > 0, graded
    assert "dead" not in set(graded["verdict"]), graded


def test_grad_health_saturated_composite_keeps_the_layer_output_healthy() -> None:
    """With every tanh saturated the zero gradient inside the GELU is real and stays inside.

    The intermediates feeding ``tanh`` have exactly zero gradient (``1 - tanh**2`` is
    ``0.0`` in float32), which grad_health reports op by op; the layer's output and
    the linear layer feeding it still carry gradient through ``0.5 * h``.
    """

    torch.manual_seed(22)
    graded = _grad_frame(_TanhGelu(dead=3), torch.randn(4, 6))

    output = graded[graded["op"].str.startswith("mul")].iloc[-1]
    linear = graded[graded["op"].str.startswith("linear")].iloc[0]
    assert linear["verdict"] != "dead", graded
    assert output["verdict"] != "dead", graded
    dead_rows = graded[graded["verdict"] == "dead"]
    assert all(str(op).startswith(("pow", "mul", "add")) for op in dead_rows["op"]), dead_rows


def test_dtype_range_on_composite_activations_is_clean() -> None:
    """The intermediates of a tanh GELU at bias 100 are large but in range: no finding."""

    torch.manual_seed(21)
    report = tl.debug.dtype_range(tl.trace(_TanhGelu(dead=3), torch.randn(2, 7, 6)))

    assert report.status == "clean", report

"""Hardening tests for ``tl.report.guess_input_shape`` (round 22).

Three mandated properties:

1. NO SIDE EFFECTS: the caller's model (state_dict, training flags, lazy-init
   status) is identical after any call, across lazy, train-mode BatchNorm, and
   meta-device models.
2. NO ESCAPE: ``on_failure="return"`` never raises across the failure zoo;
   ``on_failure="raise"`` raises a typed ``ShapeInferenceError``, never a raw
   internal ``KeyError``/``ValueError``.
3. FOUND=TRUE HONESTY: a returned shape reproduces the intended computation --
   sequence models are not silently collapsed to 2D, ``batch_first`` layouts are
   respected, broadcast-escape "verifications" are rejected, claimed flexible
   dimensions actually tolerate growth, and token recipes report the real
   vocabulary.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens._errors import ShapeInferenceError

# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------


class LazyMlp(nn.Module):
    """MLP whose first layer is an un-materialized ``LazyLinear``."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.LazyLinear(8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc2(F.relu(self.fc1(x)))


class LazyConvNet(nn.Module):
    """CNN whose first layer is an un-materialized ``LazyConv2d``."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.LazyConv2d(8, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN."""

        return self.fc(torch.flatten(self.pool(F.relu(self.conv(x))), 1))


class SubclassConv2d(nn.Conv2d):
    """timm-style ``Conv2d`` subclass whose exact type is unknown to torchlens."""


class SubclassConvNet(nn.Module):
    """CNN whose entry conv is a third-party ``Conv2d`` subclass."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = SubclassConv2d(3, 8, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN."""

        return self.fc(torch.flatten(self.pool(F.relu(self.conv(x))), 1))


def make_train_bn1d() -> nn.Module:
    """Build a default-train-mode MLP with BatchNorm1d."""

    return nn.Sequential(nn.Linear(20, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2))


class ConvBn2d(nn.Module):
    """Default-train-mode Conv+BatchNorm2d image model."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN."""

        return self.fc(torch.flatten(self.pool(F.relu(self.bn(self.conv(x)))), 1))


def make_meta_mlp() -> nn.Module:
    """Build an MLP whose parameters live on the meta device."""

    return nn.Sequential(nn.Linear(20, 8), nn.ReLU(), nn.Linear(8, 2)).to("meta")


class MetaBufferModel(nn.Module):
    """CPU model with one meta-device buffer (device-mismatch stand-in)."""

    def __init__(self) -> None:
        """Initialize layers and the stale-device buffer."""

        super().__init__()
        self.fc = nn.Linear(20, 2)
        self.register_buffer("bad", torch.zeros(2, device="meta"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the stale buffer to force a device error."""

        return self.fc(x) + self.bad


class TwoInputAdd(nn.Module):
    """Two-positional-input model (documented limit)."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two inputs."""

        return self.fc(x + y)


class DictStyleInput(nn.Module):
    """Model requiring an extra tensor kwarg (documented limit)."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a mask."""

        return self.fc(x * mask)


class LstmPerStepHead(nn.Module):
    """Batch-first LSTM with a per-step linear head; input must stay rank 3."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.rnn = nn.LSTM(32, 32, batch_first=True)
        self.head = nn.Linear(32, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LSTM and per-step head."""

        out, _state = self.rnn(x)
        return self.head(out)


class EncoderWrap(nn.Module):
    """Single transformer encoder layer wrapper."""

    def __init__(self, batch_first: bool) -> None:
        """Initialize the encoder layer."""

        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, batch_first=batch_first, dim_feedforward=64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder layer."""

        return self.layer(x)


class GrayscaleRepeat(nn.Module):
    """Grayscale image model that repeats channels internally.

    Ground truth input is ``(1, 1, side, side)``; a 2D input only "runs" by
    broadcast, and a 3-channel probe hits a channel mismatch.
    """

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat to three channels and classify."""

        x = x.repeat(1, 3, 1, 1)
        return self.fc(torch.flatten(self.pool(F.relu(self.conv(x))), 1))


class DepthwiseFunctional(nn.Module):
    """Functional depthwise conv whose weight shape hides the channel count."""

    def __init__(self) -> None:
        """Initialize raw parameters."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 1, 3, 3))
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run depthwise conv, pool, and classify."""

        x = F.conv2d(x, self.weight, padding=1, groups=8)
        return self.fc(x.mean(dim=(2, 3)))


class PosIdsLm(nn.Module):
    """Token LM with an HF-style ``(1, max_len)`` position_ids buffer."""

    def __init__(self) -> None:
        """Initialize layers and buffer."""

        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.register_buffer("position_ids", torch.arange(512).unsqueeze(0))
        self.head = nn.Linear(16, 100)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run token embedding and head."""

        return self.head(self.embed(tokens))


class FuncWte(nn.Module):
    """Functional embedding with a GPT-style raw table name."""

    def __init__(self) -> None:
        """Initialize raw parameters."""

        super().__init__()
        self.wte = nn.Parameter(torch.randn(50, 16))
        self.head = nn.Linear(16, 50)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Run functional embedding and head."""

        return self.head(F.embedding(idx, self.wte))


class FixedBranchAdaptive(nn.Module):
    """Adaptive-pool branch plus a fixed-size flatten branch (side pinned at 224)."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 + 4 * 224 * 224, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches and classify."""

        x = F.relu(self.conv(x))
        pooled = torch.flatten(self.pool(x), 1)
        raw = torch.flatten(x, 1)
        return self.fc(torch.cat((pooled, raw), dim=1))


class SeqPosAdd(nn.Module):
    """Sequence model whose 2D probe only runs by broadcasting against pos."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 32, 12))
        self.proj = nn.Linear(12, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings and project."""

        x = x + self.pos[:, : x.size(1)]
        return self.proj(x)


class TinyMlp(nn.Module):
    """Plain MLP used to check the rank-ambiguity message note."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.net(x)


class AdaptiveImage(nn.Module):
    """CNN with adaptive pooling and genuinely flexible spatial dims."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN."""

        return self.net(x)


class TinyTokenLm(nn.Module):
    """Token LM with a capped positional table and flexible sequence dim."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, 16))
        self.head = nn.Linear(16, 100)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run token embedding and head."""

        x = self.embed(tokens) + self.position_embeddings[:, : tokens.shape[1]]
        return self.head(x)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

_UNINITIALIZED = (nn.parameter.UninitializedParameter, nn.parameter.UninitializedBuffer)


def _state_snapshot(model: nn.Module) -> tuple[dict, dict, tuple]:
    """Capture state_dict values, training flags, and lazy-init status."""

    entries: dict[str, tuple] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, _UNINITIALIZED):
            entries[key] = ("uninitialized",)
        elif value.is_meta:
            entries[key] = ("meta", tuple(value.shape), value.dtype)
        else:
            entries[key] = ("tensor", value.dtype, value.detach().clone())
    training = {name: module.training for name, module in model.named_modules()}
    lazy = tuple(
        name
        for name, module in model.named_modules()
        if any(
            isinstance(tensor, _UNINITIALIZED)
            for tensor in [*module.parameters(recurse=False), *module.buffers(recurse=False)]
        )
    )
    return entries, training, lazy


def _assert_model_unchanged(model: nn.Module, snapshot: tuple[dict, dict, tuple]) -> None:
    """Assert the model matches a pre-call snapshot exactly."""

    entries, training, lazy = snapshot
    new_entries, new_training, new_lazy = _state_snapshot(model)
    assert set(new_entries) == set(entries), "state_dict keys changed"
    for key, old in entries.items():
        new = new_entries[key]
        assert new[0] == old[0], f"state kind changed for {key}: {old[0]} -> {new[0]}"
        if old[0] == "tensor":
            assert new[1] == old[1], f"dtype changed for {key}"
            assert torch.equal(new[2], old[2]), f"state value changed for {key}"
        elif old[0] == "meta":
            assert new[1:] == old[1:], f"meta state changed for {key}"
    assert new_training == training, "training flags changed"
    assert new_lazy == lazy, "lazy-initialization status changed"


def _tensor_from_result(
    result: tl.report.GuessInputShapeReport, shape: tuple[int, ...]
) -> torch.Tensor:
    """Build a fresh input of ``shape`` from a result's dtype and value recipe."""

    assert result.dtype is not None
    if result.value_range is not None and result.value_range[0] == "randint":
        low, high = int(result.value_range[1]), int(result.value_range[2])
        return torch.randint(low, max(high, low + 1), shape, dtype=result.dtype)
    if result.dtype == torch.bool:
        return torch.rand(shape) > 0.5
    return torch.rand(shape, dtype=result.dtype)


# ---------------------------------------------------------------------------
# Property 1 + 2: no side effects, no escape
# ---------------------------------------------------------------------------

_ZOO: list[tuple[str, Callable[[], nn.Module]]] = [
    ("lazy_linear", LazyMlp),
    ("lazy_conv", LazyConvNet),
    ("subclass_conv", SubclassConvNet),
    ("train_bn1d", make_train_bn1d),
    ("train_bn2d_conv", ConvBn2d),
    ("meta_mlp", make_meta_mlp),
    ("meta_buffer_device", MetaBufferModel),
    ("two_input", TwoInputAdd),
    ("dict_style_input", DictStyleInput),
]


@pytest.mark.parametrize(("name", "factory"), _ZOO, ids=[name for name, _ in _ZOO])
def test_no_escape_and_no_side_effects(name: str, factory: Callable[[], nn.Module]) -> None:
    """on_failure='return' never raises and never mutates the caller's model."""

    model = factory()
    snapshot = _state_snapshot(model)
    result = tl.report.guess_input_shape(model, on_failure="return")
    assert isinstance(result, tl.report.GuessInputShapeReport)
    if result.found:
        assert result.shape is not None
        assert all(dim >= 1 for dim in result.shape), "degenerate zero/negative dim reported"
    else:
        assert result.reason is not None
    _assert_model_unchanged(model, snapshot)


@pytest.mark.parametrize(
    "factory",
    [LazyMlp, LazyConvNet, make_meta_mlp, MetaBufferModel],
    ids=["lazy_linear", "lazy_conv", "meta_mlp", "meta_buffer_device"],
)
def test_on_failure_raise_is_typed(factory: Callable[[], nn.Module]) -> None:
    """on_failure='raise' raises ShapeInferenceError, never a raw internal error."""

    with pytest.raises(ShapeInferenceError):
        tl.report.guess_input_shape(factory(), on_failure="raise")


def test_failure_reasons_are_specific() -> None:
    """Failures carry the honest, specific reason for their class."""

    lazy = tl.report.guess_input_shape(LazyMlp(), on_failure="return")
    assert not lazy.found
    assert lazy.reason == "lazy_uninitialized"
    assert "lazy" in lazy.message.lower()

    meta = tl.report.guess_input_shape(make_meta_mlp(), on_failure="return")
    assert not meta.found
    assert meta.reason == "verification_failed"

    device = tl.report.guess_input_shape(MetaBufferModel(), on_failure="return")
    assert not device.found
    assert device.reason == "device_mismatch"
    assert "device" in device.message.lower()
    assert "aspect" not in device.message.lower()


def test_lazy_model_still_works_after_refusal() -> None:
    """A refused lazy model materializes correctly on its first real input."""

    model = LazyMlp()
    result = tl.report.guess_input_shape(model, on_failure="return")
    assert not result.found
    out = model(torch.rand(2, 20))
    assert out.shape == (2, 2)
    assert model.fc1.in_features == 20


def test_subclass_conv_is_inferred_like_a_conv() -> None:
    """Conv2d subclasses dispatch by rank, not exact type, and succeed."""

    model = SubclassConvNet()
    result = tl.report.guess_input_shape(model)
    assert result.found
    assert result.shape is not None
    assert len(result.shape) == 4
    assert result.shape[:2] == (1, 3)


def test_train_mode_bn1d_succeeds_without_escape() -> None:
    """Train-mode BatchNorm1d verifies in eval regime instead of raising raw."""

    model = make_train_bn1d()
    assert model.training
    result = tl.report.guess_input_shape(model, on_failure="return")
    assert result.found
    assert result.shape == (1, 20)
    assert model.training


def test_argument_validation_raises_typed_errors() -> None:
    """Degenerate search arguments raise ShapeInferenceError up front."""

    model = TinyMlp()
    for kwargs in (
        {"batch_size": 0},
        {"batch_size": -1},
        {"max_probes": 0},
        {"min_size": 0},
        {"min_size": 8, "max_size": 4},
        {"seq_len": 0},
        {"channels": 0},
    ):
        with pytest.raises(ShapeInferenceError):
            tl.report.guess_input_shape(model, **kwargs)


# ---------------------------------------------------------------------------
# Property 3: found=True honesty
# ---------------------------------------------------------------------------


def test_lstm_per_step_head_stays_rank3() -> None:
    """A correct rank-3 RNN success is never normalized down to rank 2."""

    model = LstmPerStepHead()
    result = tl.report.guess_input_shape(model)
    assert result.found
    assert result.shape == (1, 16, 32)
    assert result.strategy != "executed_op_normalize"
    assert result.example_input is not None
    model.eval()
    with torch.no_grad():
        out = model(result.example_input)
    assert out.ndim == 3


def test_transformer_batch_first_layouts() -> None:
    """batch_size lands on the dimension the encoder layer reads as batch."""

    seq_first = tl.report.guess_input_shape(EncoderWrap(batch_first=False), batch_size=4)
    assert seq_first.found
    assert seq_first.shape is not None
    assert len(seq_first.shape) == 3
    assert seq_first.shape[1] == 4, "batch must land on dim 1 for seq-first layers"
    assert seq_first.shape[2] == 32

    batch_first = tl.report.guess_input_shape(EncoderWrap(batch_first=True), batch_size=4)
    assert batch_first.found
    assert batch_first.shape is not None
    assert len(batch_first.shape) == 3
    assert batch_first.shape[0] == 4, "batch must land on dim 0 for batch-first layers"
    assert batch_first.shape[2] == 32


def test_no_broadcast_escape_on_image_model() -> None:
    """A grayscale image model is inferred at rank 4, never as a 2D broadcast."""

    model = GrayscaleRepeat()
    with torch.no_grad():
        model(torch.rand(1, 1, 32, 32))
    result = tl.report.guess_input_shape(model)
    assert result.found
    assert result.shape is not None
    assert len(result.shape) == 4, f"broadcast escape: got rank {len(result.shape)}"
    assert result.shape[1] == 1, "channel mismatch was parsed but not corrected"


def test_depthwise_functional_conv_channels_corrected() -> None:
    """Parsed channel-mismatch facts drive a corrected retry (depthwise case)."""

    model = DepthwiseFunctional()
    result = tl.report.guess_input_shape(model)
    assert result.found
    assert result.shape is not None
    assert len(result.shape) == 4
    assert result.shape[1] == 8


def test_sequence_model_never_reported_as_2d() -> None:
    """A broadcast-only 2D 'success' is rejected instead of reported verified."""

    result = tl.report.guess_input_shape(SeqPosAdd(), on_failure="return")
    if result.found:
        assert result.shape is not None
        assert len(result.shape) >= 3
    else:
        assert result.reason == "rank_undetermined"


def test_pure_linear_win_discloses_rank_ambiguity() -> None:
    """A last-dim-only constraint success discloses rank under-determination."""

    result = tl.report.guess_input_shape(TinyMlp())
    assert result.found
    assert result.shape == (1, 20)
    assert "under-determined" in result.message


@pytest.mark.parametrize(
    "factory",
    [AdaptiveImage, TinyTokenLm, LstmPerStepHead],
    ids=["adaptive_image", "token_lm", "lstm_head"],
)
def test_claimed_flexible_dims_tolerate_growth(factory: Callable[[], nn.Module]) -> None:
    """Every claimed flexible dim actually runs when grown by 4."""

    model = factory()
    result = tl.report.guess_input_shape(model)
    assert result.found
    assert result.shape is not None
    assert result.flexible_dims, "these models genuinely have a flexible dim"
    model.eval()
    for dim in result.flexible_dims:
        grown_shape = tuple(
            size + 4 if index == dim else size for index, size in enumerate(result.shape)
        )
        with torch.no_grad():
            model(_tensor_from_result(result, grown_shape))


def test_fixed_branch_drops_false_flexible_claim() -> None:
    """A fixed-size parallel branch falsifies the adaptive-pool flexibility claim."""

    result = tl.report.guess_input_shape(FixedBranchAdaptive())
    assert result.found
    assert result.shape == (1, 3, 224, 224)
    assert result.flexible_dims == (), "spatial dims claimed flexible but a branch pins 224"


def test_position_ids_buffer_does_not_cap_sequence_to_one() -> None:
    """An HF-style (1, max_len) position_ids buffer must not cap seq len at 1."""

    result = tl.report.guess_input_shape(PosIdsLm())
    assert result.found
    assert result.shape is not None
    assert result.shape[0] == 1
    assert result.shape[1] > 1


def test_token_value_range_reports_executed_vocab() -> None:
    """dtype-corrected token models report the real embedding vocabulary."""

    result = tl.report.guess_input_shape(FuncWte())
    assert result.found
    assert result.dtype == torch.long
    assert result.value_range == ("randint", 0.0, 50.0)

"""Tests for ``tl.debug.infer_input_shape``."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import torchlens as tl


class TinyMlp(nn.Module):
    """Simple MLP with a statically readable input width."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.net(x)


class LeNetish(nn.Module):
    """CNN whose valid side is constrained by flatten-to-linear width."""

    def __init__(self, channels: int = 3, side_features: int = 5) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Linear(16 * side_features * side_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN."""

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc(torch.flatten(x, 1))


class AdaptiveCnn(nn.Module):
    """CNN with adaptive pooling and flexible spatial input."""

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


class MiniVit(nn.Module):
    """Patch-conv ViT-style model with a square positional table."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.patch = nn.Conv2d(3, 12, 8, stride=8)
        self.pos_embed = nn.Parameter(torch.zeros(1, 17, 12))
        self.head = nn.Linear(12, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch embedding and a tiny head."""

        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        return self.head(x[:, 0])


class TinyLm(nn.Module):
    """Token model constrained by embedding vocabulary and positional cap."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, 16))
        self.head = nn.Linear(16, 100)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run token embedding and language-model head."""

        x = self.embed(tokens) + self.position_embeddings[:, : tokens.shape[1]]
        return self.head(x)


class FunctionalLinearCnn(nn.Module):
    """CNN using functional flatten and functional linear."""

    def __init__(self) -> None:
        """Initialize layers and functional linear parameters."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.weight = nn.Parameter(torch.randn(5, 4 * 30 * 30))
        self.bias = nn.Parameter(torch.randn(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolution and functional linear."""

        x = F.relu(self.conv(x))
        return F.linear(torch.flatten(x, 1), self.weight, self.bias)


class FuncMlp(nn.Module):
    """Functional-only MLP with raw parameters."""

    def __init__(self) -> None:
        """Initialize raw parameters."""

        super().__init__()
        self.w1 = nn.Parameter(torch.randn(8, 20))
        self.b1 = nn.Parameter(torch.randn(8))
        self.w2 = nn.Parameter(torch.randn(2, 8))
        self.b2 = nn.Parameter(torch.randn(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run functional linear layers."""

        return F.linear(F.relu(F.linear(x, self.w1, self.b1)), self.w2, self.b2)


class FuncMatmulMlp(nn.Module):
    """Functional-only MLP using raw matmul."""

    def __init__(self) -> None:
        """Initialize raw parameters."""

        super().__init__()
        self.w1 = nn.Parameter(torch.randn(20, 8))
        self.w2 = nn.Parameter(torch.randn(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run raw matmul layers."""

        return torch.relu(x @ self.w1) @ self.w2


class FuncConvFlattenLinear(nn.Module):
    """Functional conv-to-flatten-to-linear with an exact side."""

    def __init__(self, side: int = 32) -> None:
        """Initialize raw parameters."""

        super().__init__()
        self.conv_weight = nn.Parameter(torch.randn(4, 3, 3, 3))
        self.conv_bias = nn.Parameter(torch.randn(4))
        self.fc_weight = nn.Parameter(torch.randn(5, 4 * (side - 2) * (side - 2)))
        self.fc_bias = nn.Parameter(torch.randn(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run functional conv and linear."""

        x = F.conv2d(x, self.conv_weight, self.conv_bias)
        return F.linear(torch.flatten(x, 1), self.fc_weight, self.fc_bias)


class FuncEmbedding(nn.Module):
    """Functional embedding with a raw table."""

    def __init__(self) -> None:
        """Initialize raw parameters."""

        super().__init__()
        self.token_table = nn.Parameter(torch.randn(40, 8))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 10, 8))
        self.weight = nn.Parameter(torch.randn(40, 8))
        self.bias = nn.Parameter(torch.randn(40))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run functional embedding and head."""

        x = F.embedding(tokens, self.token_table) + self.position_embeddings[:, : tokens.shape[1]]
        return F.linear(x, self.weight, self.bias)


class FunctionalOnlyRawConvMatmul(nn.Module):
    """Functional conv followed by raw matmul."""

    def __init__(self) -> None:
        """Initialize raw parameters."""

        super().__init__()
        self.conv_weight = nn.Parameter(torch.randn(4, 3, 3, 3))
        self.proj = nn.Parameter(torch.randn(4 * 30 * 30, 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run functional conv and matmul."""

        x = F.conv2d(x, self.conv_weight)
        return torch.flatten(x, 1) @ self.proj


class DecoyConvBeforeRealLinear(nn.Module):
    """Unused conv appears before the real linear input."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.decoy = nn.Conv2d(17, 33, 5)
        self.real = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the real linear layer."""

        return self.real(x)


class DecoyConvUnusedRealConv(nn.Module):
    """Unused Conv2d with wrong channels appears before the real path."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.decoy = nn.Conv2d(1, 8, 1)
        self.real = nn.Conv2d(3, 4, 3)
        self.fc = nn.Linear(4 * 30 * 30, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the real conv path."""

        x = F.relu(self.real(x))
        return self.fc(torch.flatten(x, 1))


class DefinitionOrderWrongConv(nn.Module):
    """Conv definition order is opposite of execution order."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.conv2 = nn.Conv2d(4, 5, 3)
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.fc = nn.Linear(5 * 28 * 28, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv1 before conv2."""

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(torch.flatten(x, 1))


class DecoyAdaptivePoolUnused(nn.Module):
    """Unused adaptive pool must not mark the real path flexible."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.decoy_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * 30 * 30, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ignore the adaptive pool."""

        x = F.relu(self.conv(x))
        return self.fc(torch.flatten(x, 1))


class ConvExactSide(nn.Module):
    """Conv exact-side model whose side is outside preferred sizes."""

    def __init__(self, side: int) -> None:
        """Initialize modules."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.fc = nn.Linear(4 * (side - 2) * (side - 2), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run exact-side conv model."""

        x = F.relu(self.conv(x))
        return self.fc(torch.flatten(x, 1))


class ConvViewExactNoLinear(nn.Module):
    """Exact-size model constrained by view instead of Linear."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv then exact view."""

        return self.conv(x).view(x.shape[0], 4 * 30 * 30)


class GroupNormChannelLock(nn.Module):
    """GroupNorm model with a channel-locked input."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.norm = nn.GroupNorm(2, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run group normalization."""

        return self.norm(x)


class BoolMaskPrimary(nn.Module):
    """Primary input must be bool."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Run bool-mask semantics."""

        x = torch.zeros_like(mask, dtype=torch.float32)
        return self.fc(x.masked_fill(mask, 1.0))


class ComplexLinear(nn.Module):
    """Complex-valued linear model."""

    def __init__(self) -> None:
        """Initialize module."""

        super().__init__()
        self.fc = nn.Linear(4, 2, dtype=torch.complex64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run complex linear."""

        return self.fc(x)


class VitOddPosName(nn.Module):
    """Patch model whose position table name is non-standard."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.patch = nn.Conv2d(3, 12, 8, stride=8)
        self.grid = nn.Parameter(torch.zeros(1, 17, 12))
        self.head = nn.Linear(12, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch model."""

        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        x = torch.cat((cls, x), dim=1) + self.grid
        return self.head(x[:, 0])


class VitTwoSpecialTokens(nn.Module):
    """Patch model with two special tokens in the position table."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.patch = nn.Conv2d(3, 12, 8, stride=8)
        self.pos_embed = nn.Parameter(torch.zeros(1, 18, 12))
        self.head = nn.Linear(12, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch model with two special tokens."""

        x = self.patch(x).flatten(2).transpose(1, 2)
        specials = torch.zeros(x.shape[0], 2, x.shape[2], device=x.device, dtype=x.dtype)
        x = torch.cat((specials, x), dim=1) + self.pos_embed
        return self.head(x[:, 0])


class MultiInputAdd(nn.Module):
    """Two tensor inputs are a documented limit."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two inputs."""

        return self.fc(x + y)


class DictInputMask(nn.Module):
    """Dict/kwarg-style tensor input is a documented limit."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a mask."""

        return self.fc(x * mask)


class GnnStyle(nn.Module):
    """GNN-style coupled inputs are a documented limit."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.lin = nn.Linear(6, 4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Aggregate source node features into destinations."""

        h = self.lin(x)
        out = torch.zeros_like(h)
        src, dst = edge_index
        out.index_add_(0, dst, h[src])
        return out


class NeedsKwarg(nn.Module):
    """Model requiring a non-inferred kwarg."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, *, scale: float) -> torch.Tensor:
        """Require a kwarg."""

        return self.fc(x * scale)


class NonSquareExact(nn.Module):
    """Only non-square spatial input works."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.fc = nn.Linear(4 * 10 * 20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expect input height 12 and width 22 after convolution."""

        x = F.relu(self.conv(x))
        return self.fc(torch.flatten(x, 1))


class NeedsMask(nn.Module):
    """Model that cannot run without a non-input kwarg."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Require a mask argument."""

        return self.fc(x * mask)


def _assert_result_runs(model: nn.Module, result: tl.debug.InferInputShapeResult) -> None:
    """Assert a successful inference result runs through the model."""

    assert result.found
    assert isinstance(result.example_input, torch.Tensor)
    model.eval()
    with torch.no_grad():
        model(result.example_input)


def test_infer_mlp_static_shape() -> None:
    """Infer a no-probe MLP input shape."""

    result = tl.debug.infer_input_shape(TinyMlp())
    assert result.found
    assert result.shape == (1, 20)
    assert result.dtype == torch.float32
    assert isinstance(result.example_input, torch.Tensor)


def test_infer_conv_flatten_linear_exact_shape() -> None:
    """Infer the LeNet-style conv-to-linear exact spatial side."""

    result = tl.debug.infer_input_shape(LeNetish())
    assert result.found
    assert result.shape == (1, 3, 32, 32)


def test_infer_grayscale_exact_shape_with_channel_hint() -> None:
    """Infer a grayscale exact side with an explicit channel hint."""

    result = tl.debug.infer_input_shape(LeNetish(channels=1, side_features=4), channels=1)
    assert result.found
    assert result.shape == (1, 1, 28, 28)


def test_infer_adaptive_pool_marks_flexible_dims() -> None:
    """Use a sane default and mark adaptive-pool spatial dimensions flexible."""

    result = tl.debug.infer_input_shape(AdaptiveCnn())
    assert result.found
    assert result.shape is not None
    assert result.shape[:2] == (1, 3)
    assert result.flexible_dims == (2, 3)
    assert result.strategy == "adaptive_default"


def test_infer_vit_patch_grid_from_position_embedding() -> None:
    """Infer a ViT-style 32x32 image side from patch stride and pos_embed length."""

    result = tl.debug.infer_input_shape(MiniVit())
    assert result.found
    assert result.shape == (1, 3, 32, 32)


def test_infer_transformer_lm_tokens() -> None:
    """Infer integer token inputs within embedding range."""

    result = tl.debug.infer_input_shape(TinyLm())
    assert result.found
    assert result.shape is not None
    assert result.shape[0] == 1
    assert result.shape[1] <= 64
    assert result.dtype == torch.long
    assert result.value_range == ("randint", 0.0, 100.0)


def test_infer_rnn_batch_first_and_time_first() -> None:
    """Infer RNN layouts for batch-first and time-first modules."""

    batch_first = tl.debug.infer_input_shape(nn.LSTM(input_size=7, hidden_size=3, batch_first=True))
    time_first = tl.debug.infer_input_shape(nn.GRU(input_size=5, hidden_size=3, batch_first=False))
    assert batch_first.found
    assert batch_first.shape == (1, 16, 7)
    assert time_first.found
    assert time_first.shape == (16, 1, 5)


def test_infer_conv1d_and_conv3d_adaptive_variants() -> None:
    """Infer 1D and 3D convolutional inputs."""

    audio = nn.Sequential(
        nn.Conv1d(2, 4, 3, padding=1),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    video = nn.Sequential(
        nn.Conv3d(3, 4, 3, padding=1),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    audio_result = tl.debug.infer_input_shape(audio)
    video_result = tl.debug.infer_input_shape(video)
    assert audio_result.found
    assert audio_result.shape is not None
    assert audio_result.shape[:2] == (1, 2)
    assert audio_result.flexible_dims == (2,)
    assert video_result.found
    assert video_result.shape is not None
    assert video_result.shape[:2] == (1, 3)
    assert video_result.flexible_dims == (2, 3, 4)


def test_infer_functional_linear_after_functional_flatten() -> None:
    """Handle functional flatten followed by functional linear."""

    result = tl.debug.infer_input_shape(FunctionalLinearCnn())
    assert result.found
    assert result.shape == (1, 3, 32, 32)


def test_non_shape_blocker_returns_diagnostic() -> None:
    """Return instead of hanging when a missing kwarg blocks the forward."""

    result = tl.debug.infer_input_shape(NeedsMask())
    assert not result.found
    assert result.reason == "non_shape_blocker"
    assert "mask" in result.message


def test_embedding_wrong_dtype_self_corrects() -> None:
    """Correct an explicitly wrong embedding dtype to integer tokens."""

    result = tl.debug.infer_input_shape(TinyLm(), input_dtype=torch.float32)
    assert result.found
    assert result.dtype == torch.long
    assert any("Long" in outcome or "Int" in outcome for _shape, outcome in result.attempts)


def test_functional_only_mlp_and_matmul_models() -> None:
    """Infer functional-only 2D models from raw parameter and op metadata."""

    mlp_result = tl.debug.infer_input_shape(FuncMlp())
    matmul_result = tl.debug.infer_input_shape(FuncMatmulMlp())
    assert mlp_result.shape == (1, 20)
    assert matmul_result.shape == (1, 20)
    _assert_result_runs(FuncMlp(), mlp_result)
    _assert_result_runs(FuncMatmulMlp(), matmul_result)


def test_functional_only_conv_linear_and_matmul_models() -> None:
    """Infer functional-only conv exact shapes without nn.Conv/nn.Linear modules."""

    linear_model = FuncConvFlattenLinear()
    matmul_model = FunctionalOnlyRawConvMatmul()
    linear_result = tl.debug.infer_input_shape(linear_model)
    matmul_result = tl.debug.infer_input_shape(matmul_model)
    assert linear_result.shape == (1, 3, 32, 32)
    assert matmul_result.shape == (1, 3, 32, 32)
    _assert_result_runs(linear_model, linear_result)
    _assert_result_runs(matmul_model, matmul_result)


def test_functional_embedding_uses_integer_tokens() -> None:
    """Infer functional embedding inputs as long token IDs."""

    model = FuncEmbedding()
    result = tl.debug.infer_input_shape(model)
    assert result.found
    assert result.shape == (1, 10)
    assert result.dtype == torch.long
    assert result.value_range == ("randint", 0.0, 40.0)
    _assert_result_runs(model, result)


def test_decoys_do_not_determine_input_shape() -> None:
    """Ignore off-path decoys and definition-order traps."""

    linear_model = DecoyConvBeforeRealLinear()
    conv_model = DecoyConvUnusedRealConv()
    order_model = DefinitionOrderWrongConv()
    pool_model = DecoyAdaptivePoolUnused()

    linear_result = tl.debug.infer_input_shape(linear_model)
    conv_result = tl.debug.infer_input_shape(conv_model)
    order_result = tl.debug.infer_input_shape(order_model)
    pool_result = tl.debug.infer_input_shape(pool_model)

    assert linear_result.shape == (1, 20)
    assert conv_result.shape == (1, 3, 32, 32)
    assert order_result.shape == (1, 3, 32, 32)
    assert pool_result.shape == (1, 3, 32, 32)
    assert pool_result.flexible_dims == ()
    assert linear_result.shape != (1, 17, 20, 20)
    _assert_result_runs(linear_model, linear_result)
    _assert_result_runs(conv_model, conv_result)
    _assert_result_runs(order_model, order_result)
    _assert_result_runs(pool_model, pool_result)


def test_conv_exact_side_outside_preferred_sizes() -> None:
    """Solve exact conv sides even when defaults first hit kernel/linear shape errors."""

    side_77 = ConvExactSide(77)
    side_150 = ConvExactSide(150)
    result_77 = tl.debug.infer_input_shape(side_77)
    result_150 = tl.debug.infer_input_shape(side_150)
    assert result_77.shape == (1, 3, 77, 77)
    assert result_150.shape == (1, 3, 150, 150)
    _assert_result_runs(side_77, result_77)
    _assert_result_runs(side_150, result_150)


def test_no_linear_and_groupnorm_channel_cases() -> None:
    """Handle exact view-only and channel-locked normalization cases."""

    view_model = ConvViewExactNoLinear()
    norm_model = GroupNormChannelLock()
    view_result = tl.debug.infer_input_shape(view_model)
    norm_result = tl.debug.infer_input_shape(norm_model)
    assert view_result.shape == (1, 3, 32, 32)
    assert norm_result.shape == (1, 6, 32, 32)
    _assert_result_runs(view_model, view_result)
    _assert_result_runs(norm_model, norm_result)


def test_bool_and_complex_dtype_corrections() -> None:
    """Create bool and complex tensors without crashing."""

    bool_model = BoolMaskPrimary()
    complex_model = ComplexLinear()
    bool_result = tl.debug.infer_input_shape(bool_model)
    complex_result = tl.debug.infer_input_shape(complex_model)
    assert bool_result.shape == (1, 4)
    assert bool_result.dtype == torch.bool
    assert complex_result.shape == (1, 4)
    assert complex_result.dtype == torch.complex64
    _assert_result_runs(bool_model, bool_result)
    _assert_result_runs(complex_model, complex_result)


def test_vit_nonstandard_position_tables() -> None:
    """Infer patch grids with non-standard names and two special tokens."""

    odd_model = VitOddPosName()
    two_token_model = VitTwoSpecialTokens()
    odd_result = tl.debug.infer_input_shape(odd_model)
    two_token_result = tl.debug.infer_input_shape(two_token_model)
    assert odd_result.shape == (1, 3, 32, 32)
    assert two_token_result.shape == (1, 3, 32, 32)
    _assert_result_runs(odd_model, odd_result)
    _assert_result_runs(two_token_model, two_token_result)


def test_genuine_limits_return_clean_failures() -> None:
    """Return structured failures for documented unsupported signatures."""

    for model in (MultiInputAdd(), DictInputMask(), GnnStyle(), NeedsKwarg()):
        result = tl.debug.infer_input_shape(model)
        assert not result.found
        assert result.reason == "non_shape_blocker"
        assert result.message

    rectangular = tl.debug.infer_input_shape(NonSquareExact())
    assert not rectangular.found
    assert rectangular.reason == "exact_size_unreachable"
    assert "aspect" in rectangular.message

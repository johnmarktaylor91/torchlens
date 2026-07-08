"""Determinism and two-hash behavior tests for TorchLens graph digests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.utils.hashing import compute_graph_shape_hash


HashPair = tuple[str, str]
ModelCase = tuple[str, Callable[[], tuple[nn.Module, Any]]]


class TinyMlp(nn.Module):
    """Small MLP with two affine layers."""

    def __init__(self, hidden_width: int = 8) -> None:
        """Initialize the MLP.

        Parameters
        ----------
        hidden_width:
            Hidden layer width.
        """

        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.net(x)


class SmallConvNet(nn.Module):
    """Small convolutional network."""

    def __init__(self) -> None:
        """Initialize the convolutional network."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.head = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional network.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.head(self.features(x))


class TinyTransformerBlock(nn.Module):
    """Two-layer transformer-style block."""

    def __init__(self) -> None:
        """Initialize the transformer block."""

        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(8)
        self.ff = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.norm2 = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention and a feed-forward block.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Block output.
        """

        attended, _weights = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ff(x))


class TupleOutputModel(nn.Module):
    """Model returning a nested tuple output."""

    def __init__(self) -> None:
        """Initialize the tuple-output model."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the model and return structured outputs.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            Structured output tensors.
        """

        left = torch.relu(self.left(x))
        right = self.right(x)
        return left, (right, left.mean(dim=-1))


class BufferModel(nn.Module):
    """Model with a registered buffer participating in the graph."""

    def __init__(self) -> None:
        """Initialize the buffer model."""

        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("offset", torch.arange(4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model with a buffer addend.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.proj(x + self.offset)


class ModuleListModel(nn.Module):
    """Model using a ModuleList."""

    def __init__(self) -> None:
        """Initialize the ModuleList model."""

        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run each layer in sequence.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class BranchModel(nn.Module):
    """Model with a deterministic branch."""

    def __init__(self) -> None:
        """Initialize the branch model."""

        super().__init__()
        self.positive = nn.Linear(4, 4)
        self.negative = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one deterministic branch.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        if bool(x.sum() > 0):
            return self.positive(x)
        return self.negative(x)


class EmbeddingModel(nn.Module):
    """Embedding model with integer inputs."""

    def __init__(self) -> None:
        """Initialize the embedding model."""

        super().__init__()
        self.embed = nn.Embedding(12, 6)
        self.proj = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run embedding lookup and projection.

        Parameters
        ----------
        x:
            Token IDs.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.proj(self.embed(x)).mean(dim=1)


class MultiInputModel(nn.Module):
    """Model accepting two tensor inputs."""

    def __init__(self) -> None:
        """Initialize the multi-input model."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run the model on two inputs.

        Parameters
        ----------
        inputs:
            Pair of input tensors.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        left, right = inputs
        return self.proj(left + right)


class ConvStrideModel(nn.Module):
    """Convolutional model with stride and pooling."""

    def __init__(self) -> None:
        """Initialize the strided convolutional model."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(48, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.net(x)


class ResidualModel(nn.Module):
    """Small residual model."""

    def __init__(self) -> None:
        """Initialize the residual model."""

        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return x + self.b(torch.relu(self.a(x)))


class ParameterReuseModel(nn.Module):
    """Model that reuses the same module twice."""

    def __init__(self) -> None:
        """Initialize the parameter-reuse model."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared projection twice.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.proj(torch.relu(self.proj(x)))


class DirectPackingModel(nn.Module):
    """Model whose operations live under direct child modules."""

    def __init__(self) -> None:
        """Initialize the direct module packing."""

        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the directly packed operations.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.activation(self.proj(x))


class NestedPackingModel(nn.Module):
    """Model whose identical operations live under a nested module."""

    def __init__(self) -> None:
        """Initialize the nested module packing."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested operations.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.block(x)


def _trace_model(model: nn.Module, example_input: Any) -> Any:
    """Trace a model under deterministic inference settings.

    Parameters
    ----------
    model:
        Model to trace.
    example_input:
        Example input passed to ``tl.trace``.

    Returns
    -------
    Any
        TorchLens trace.
    """

    model.eval()
    with torch.no_grad():
        return tl.trace(model, example_input, capture=CaptureOptions(inference_only=True))


def _trace_hash_pair(model: nn.Module, example_input: Any) -> HashPair:
    """Trace a model and return both graph-shape hashes.

    Parameters
    ----------
    model:
        Model to trace.
    example_input:
        Example input passed to ``tl.trace``.

    Returns
    -------
    HashPair
        Public graph-shape hash and raw-event shape hash.
    """

    trace = _trace_model(model, example_input)
    return str(trace.graph_shape_hash), str(trace._raw_event_shape_hash)  # noqa: SLF001


def _graph_hash_for_tiny_mlp(seed: int, device: str = "cpu") -> str:
    """Return the graph-shape hash for a seeded tiny MLP.

    Parameters
    ----------
    seed:
        Torch random seed used for model/input construction.
    device:
        Device for the model and input.

    Returns
    -------
    str
        Graph-shape hash.
    """

    torch.manual_seed(seed)
    model = TinyMlp().to(device)
    example_input = torch.randn(2, 4, device=device)
    trace = _trace_model(model, example_input)
    return str(compute_graph_shape_hash(trace))


def _hashes_for_repeated_traces(factory: Callable[[], tuple[nn.Module, Any]]) -> list[HashPair]:
    """Build one sample and hash three repeated traces.

    Parameters
    ----------
    factory:
        Callable returning a model and example input.

    Returns
    -------
    list[HashPair]
        Hashes from three repeated traces.
    """

    torch.manual_seed(0)
    model, example_input = factory()
    return [_trace_hash_pair(model, example_input) for _ in range(3)]


def _model_cases() -> list[ModelCase]:
    """Return the deterministic model sample.

    Returns
    -------
    list[ModelCase]
        Named factories for small model/input pairs.
    """

    return [
        ("tiny_mlp", lambda: (TinyMlp(), torch.randn(2, 4))),
        ("small_conv", lambda: (SmallConvNet(), torch.randn(2, 3, 8, 8))),
        ("transformer_block", lambda: (TinyTransformerBlock(), torch.randn(2, 4, 8))),
        ("tuple_output", lambda: (TupleOutputModel(), torch.randn(2, 4))),
        ("buffer", lambda: (BufferModel(), torch.randn(2, 4))),
        ("module_list", lambda: (ModuleListModel(), torch.randn(2, 4))),
        ("deterministic_branch", lambda: (BranchModel(), torch.ones(2, 4))),
        ("embedding", lambda: (EmbeddingModel(), torch.tensor([[1, 2, 3], [3, 4, 5]]))),
        ("multi_input", lambda: (MultiInputModel(), (torch.randn(2, 4), torch.randn(2, 4)))),
        ("conv_stride", lambda: (ConvStrideModel(), torch.randn(2, 1, 8, 8))),
        ("residual", lambda: (ResidualModel(), torch.randn(2, 4))),
        ("parameter_reuse", lambda: (ParameterReuseModel(), torch.randn(2, 4))),
    ]


@pytest.mark.parametrize(("case_name", "factory"), _model_cases())
def test_graph_hashes_are_deterministic_across_repeated_traces(
    case_name: str, factory: Callable[[], tuple[nn.Module, Any]]
) -> None:
    """Both graph hashes remain identical across repeated traces.

    Parameters
    ----------
    case_name:
        Human-readable case name.
    factory:
        Callable returning a model and input.
    """

    hash_pairs = _hashes_for_repeated_traces(factory)
    graph_hashes = {graph_hash for graph_hash, _raw_hash in hash_pairs}
    raw_hashes = {raw_hash for _graph_hash, raw_hash in hash_pairs}

    assert len(graph_hashes) == 1, case_name
    assert len(raw_hashes) == 1, case_name


def test_graph_hash_is_shape_blind_but_raw_event_hash_catches_width_change() -> None:
    """Graph hash collides on width changes while raw-event hash differs."""

    torch.manual_seed(0)
    input_value = torch.randn(2, 4)
    graph_width_8, raw_width_8 = _trace_hash_pair(TinyMlp(hidden_width=8), input_value)
    graph_width_16, raw_width_16 = _trace_hash_pair(TinyMlp(hidden_width=16), input_value)

    assert graph_width_8 == graph_width_16
    assert raw_width_8 != raw_width_16


def test_graph_shape_hash_matches_across_fresh_python_process() -> None:
    """Graph-shape hash is deterministic across a fresh Python process."""

    expected_hash = _graph_hash_for_tiny_mlp(seed=123)
    script = textwrap.dedent(
        """
        import torch
        from torch import nn

        import torchlens as tl
        from torchlens.options import CaptureOptions
        from torchlens.utils.hashing import compute_graph_shape_hash


        class TinyMlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

            def forward(self, x):
                return self.net(x)


        torch.manual_seed(123)
        model = TinyMlp()
        example_input = torch.randn(2, 4)
        model.eval()
        with torch.no_grad():
            trace = tl.trace(model, example_input, capture=CaptureOptions(inference_only=True))
        print(compute_graph_shape_hash(trace))
        """
    )

    actual_hash = subprocess.check_output(
        [sys.executable, "-c", script],
        text=True,
    ).strip()

    assert actual_hash == expected_hash


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_graph_shape_hash_matches_between_cpu_and_cuda() -> None:
    """Graph-shape hash is device-blind for the same model topology."""

    assert _graph_hash_for_tiny_mlp(seed=456, device="cpu") == _graph_hash_for_tiny_mlp(
        seed=456, device="cuda"
    )


def test_graph_shape_hash_distinguishes_identical_ops_under_different_module_packings() -> None:
    """Identical op sequences under different module addresses hash differently."""

    torch.manual_seed(0)
    example_input = torch.randn(2, 4)
    direct_trace = _trace_model(DirectPackingModel(), example_input)
    nested_trace = _trace_model(NestedPackingModel(), example_input)

    direct_ops = [layer.func_name for layer in direct_trace.layer_list]
    nested_ops = [layer.func_name for layer in nested_trace.layer_list]

    assert direct_ops == nested_ops
    assert compute_graph_shape_hash(direct_trace) != compute_graph_shape_hash(nested_trace)


# Torch-version-matrix determinism is a manual CI-matrix gate: one test
# environment cannot import and compare two independent torch versions.

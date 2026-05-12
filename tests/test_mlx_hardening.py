"""Hardening tests for the technical-preview MLX backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402


class TinyMLP(nn.Module):
    """Small MLX MLP used by backend hardening tests."""

    def __init__(self) -> None:
        """Initialize the MLP layers."""

        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass."""

        hidden = nn.relu(self.l1(x))
        return self.l2(hidden)


class TinyCNN(nn.Module):
    """Small MLX CNN used to verify Conv2d capture."""

    def __init__(self) -> None:
        """Initialize convolution and projection layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.proj = nn.Linear(4, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run a tiny CNN forward pass."""

        hidden = nn.relu(self.conv(x))
        pooled = mx.mean(hidden, axis=(1, 2))
        return self.proj(pooled)


class TinyNorm(nn.Module):
    """Small MLX model used to verify normalization capture."""

    def __init__(self) -> None:
        """Initialize normalization and projection layers."""

        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.proj = nn.Linear(4, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run a normalized forward pass."""

        return self.proj(self.norm(x))


class TinyAttention(nn.Module):
    """Small MLX model used to verify attention capture."""

    def __init__(self) -> None:
        """Initialize attention and projection layers."""

        super().__init__()
        self.attn = nn.MultiHeadAttention(dims=8, num_heads=2)
        self.proj = nn.Linear(8, 4)

    def __call__(self, x: mx.array) -> mx.array:
        """Run a self-attention forward pass."""

        return self.proj(self.attn(x, x, x))


def _tiny_mlp_input() -> mx.array:
    """Return a deterministic-shape MLX input array for hardening tests."""

    return mx.random.normal((2, 4))


def _assert_captured(trace: tl.Trace, expected_fragment: str) -> None:
    """Assert that an MLX trace captured an operation label fragment."""

    assert any(expected_fragment in label.lower() for label in trace.layer_labels)


@pytest.mark.optional
def test_mlx_intervention_ready_raises() -> None:
    """MLX capture rejects intervention metadata requests explicitly."""

    with pytest.raises(NotImplementedError, match="intervention_ready"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), intervention_ready=True, vis_opt="none")


@pytest.mark.optional
def test_mlx_save_grads_raises() -> None:
    """MLX capture rejects backward-gradient capture explicitly."""

    with pytest.raises(NotImplementedError, match="backward capture"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), save_grads=True, vis_opt="none")


@pytest.mark.optional
def test_mlx_hooks_raise() -> None:
    """MLX capture rejects pre-attached live hook plans explicitly."""

    hooks: list[dict[str, Any]] = [{"target": "linear_1_1", "action": lambda x: x}]
    with pytest.raises(NotImplementedError, match="hooks"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), hooks=hooks, vis_opt="none")


@pytest.mark.optional
def test_mlx_trace_draw_smokes(tmp_path: Path) -> None:
    """MLX traces draw without dangling synthetic input-parent labels."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input(), vis_opt="none")
    rendered = trace.draw(
        vis_outpath=str(tmp_path / "mlx_trace"),
        vis_fileformat="pdf",
        vis_save_only=True,
    )

    assert rendered is not None
    assert (tmp_path / "mlx_trace.pdf").exists()


@pytest.mark.optional
def test_mlx_parents_resolve() -> None:
    """Every MLX parent label resolves to a captured layer."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input(), vis_opt="none")

    for op in trace.layer_list:
        for parent in op.parents:
            assert parent in trace.layer_dict_all_keys


@pytest.mark.optional
def test_mlx_repeated_trace_no_wrapper_leak() -> None:
    """Repeated MLX captures install fresh wrappers and clean them up."""

    first = tl.trace(TinyMLP(), _tiny_mlp_input(), vis_opt="none")
    second = tl.trace(TinyMLP(), _tiny_mlp_input(), vis_opt="none")

    assert first.num_ops > 0
    assert second.num_ops > 0


@pytest.mark.optional
def test_mlx_save_load_audit_only(tmp_path: Path) -> None:
    """MLX traces save/load with array payloads nulled as unsupported records."""

    trace = tl.trace(TinyMLP(), _tiny_mlp_input(), vis_opt="none")
    bundle_path = tmp_path / "mlx-audit.tlspec"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)
    manifest = tl.io.inspect_tlspec(bundle_path)

    assert loaded._backend_name == "mlx"
    assert manifest["unsupported_tensors"]
    assert all(
        record["reason"] == "mlx_array_audit_null" for record in manifest["unsupported_tensors"]
    )
    assert all(op.out is None for op in loaded.layer_list)


@pytest.mark.optional
def test_mlx_cnn_captures_conv2d() -> None:
    """MLX Conv2d calls are wrapped and appear in the captured trace."""

    trace = tl.trace(TinyCNN(), mx.random.normal((1, 8, 8, 3)), vis_opt="none")

    _assert_captured(trace, "conv2d")


@pytest.mark.optional
def test_mlx_layernorm_wraps_and_captures() -> None:
    """MLX LayerNorm calls are wrapped and appear in the captured trace."""

    trace = tl.trace(TinyNorm(), _tiny_mlp_input(), vis_opt="none")

    _assert_captured(trace, "layernorm")


@pytest.mark.optional
def test_mlx_attention_wraps_and_captures() -> None:
    """MLX attention calls are wrapped and appear in the captured trace."""

    trace = tl.trace(TinyAttention(), mx.random.normal((1, 3, 8)), vis_opt="none")

    _assert_captured(trace, "multiheadattention")

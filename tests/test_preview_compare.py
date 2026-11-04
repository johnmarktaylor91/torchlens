"""Saved preview arrays participate in run comparison and dead-unit analysis."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import torchlens as tl

pytestmark = pytest.mark.rare


def _inputs(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, distinct inputs for the small convolutional subjects."""

    rng = np.random.default_rng(0)
    return tuple(rng.standard_normal(shape).astype(np.float32) for _ in range(2))


def _check_comparison(backend: str, model: Any, x: Any, alt: Any, tmp_path: Path) -> None:
    """Exercise each comparison surface on different and repeated inputs."""

    first = tl.trace(model, x, backend=backend)
    second = tl.trace(model, alt, backend=backend)
    repeated = tl.trace(model, x, backend=backend)
    bundle = tl.Bundle({"a": first, "b": second})
    ranked = bundle.rank_changes("a", k=3)
    assert ranked and any(value > 0 for _, value in ranked)
    differences = bundle.compare("a", "b", metric="relative_l2")
    assert differences and any(value > 0 for _, value in differences)
    assert all(value == 0 for _, value in tl.Bundle({"a": first, "b": repeated}).compare("a", "b"))
    assert tl.debug.first_divergence(first, second) is not None
    assert tl.debug.first_divergence(first, repeated) is None
    compared = tl.debug.compare(first, second).to_pandas()
    assert compared.attrs["value_diverged"] > 0
    assert compared["max_abs"].dropna().gt(0).any()
    source = bundle.draw(str(tmp_path / "diff.svg"), graph="diff", members=("a", "b")).dot
    assert any(float(value) > 0 for value in re.findall(r"delta=([0-9.eE+\-]+)", source))
    dead = tl.debug.dead_neurons(first).to_pandas()
    assert not dead.empty
    counted = dead[dead["total_units"].notna()]
    assert not counted.empty and counted["dead_count"].notna().all()
    assert counted["dead_count"].between(0, counted["total_units"]).all()
    # The dense model head has two output units on every backend.
    assert counted["total_units"].iloc[-1] == 2
    spatial = dead[dead["reason"].str.contains("unit axis unavailable", na=False)]
    assert spatial["total_units"].isna().all()
    explicit_dim = -1 if backend in {"jax", "tf", "mlx"} else 1
    explicit = tl.debug.dead_neurons(first, dim=explicit_dim).to_pandas()
    # Preview layer-output flags identify the two-unit model head.
    assert explicit["total_units"].iloc[-1] == 2


@pytest.mark.backend_jax
def test_jax_preview_comparison(tmp_path: Path) -> None:
    """JAX arrays compare through the JAX payload codec."""

    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    nnx = pytest.importorskip("flax.nnx")

    class Net(nnx.Module):
        def __init__(self, rngs: Any) -> None:
            self.conv = nnx.Conv(1, 4, (3, 3), padding="VALID", rngs=rngs)
            self.head = nnx.Linear(64, 2, rngs=rngs)

        def __call__(self, x: Any) -> Any:
            h = nnx.relu(self.conv(x))
            return self.head(h.reshape((h.shape[0], -1)))

    x, alt = _inputs((1, 6, 6, 1))
    _check_comparison("jax", Net(nnx.Rngs(0)), jnp.asarray(x), jnp.asarray(alt), tmp_path)


@pytest.mark.tf_backend
def test_tf_preview_comparison(tmp_path: Path) -> None:
    """TensorFlow eager arrays compare through the TF payload codec."""

    tf = pytest.importorskip("tensorflow")
    tf.random.set_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.Input((6, 6, 1)),
            tf.keras.layers.Conv2D(4, 3, name="conv"),
            tf.keras.layers.ReLU(name="act"),
            tf.keras.layers.Flatten(name="flat"),
            tf.keras.layers.Dense(2, name="head"),
        ]
    )
    x, alt = _inputs((1, 6, 6, 1))
    _check_comparison("tf", model, tf.constant(x), tf.constant(alt), tmp_path)


@pytest.mark.backend_mlx
def test_mlx_preview_comparison(tmp_path: Path) -> None:
    """MLX arrays compare through the MLX payload codec."""

    if sys.platform != "darwin" and os.environ.get("TORCHLENS_RUN_UNSTABLE_MLX_TESTS") != "1":
        pytest.skip("MLX off macOS runs only with TORCHLENS_RUN_UNSTABLE_MLX_TESTS=1")
    mx = pytest.importorskip("mlx.core")
    mlx_nn = pytest.importorskip("mlx.nn")

    class Net(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = mlx_nn.Conv2d(1, 4, 3)
            self.head = mlx_nn.Linear(64, 2)

        def __call__(self, x: Any) -> Any:
            h = mlx_nn.relu(self.conv(x))
            return self.head(h.reshape(h.shape[0], -1))

    mx.random.seed(0)
    model = Net()
    mx.eval(model.parameters())
    x, alt = _inputs((1, 6, 6, 1))
    _check_comparison("mlx", model, mx.array(x), mx.array(alt), tmp_path)


@pytest.mark.backend_paddle
def test_paddle_preview_comparison(tmp_path: Path) -> None:
    """Paddle arrays compare through the Paddle payload codec."""

    paddle = pytest.importorskip("paddle")
    paddle.seed(0)
    model = paddle.nn.Sequential(
        paddle.nn.Conv2D(1, 4, 3), paddle.nn.ReLU(), paddle.nn.Flatten(), paddle.nn.Linear(64, 2)
    )
    model.eval()
    x, alt = _inputs((1, 1, 6, 6))
    _check_comparison("paddle", model, paddle.to_tensor(x), paddle.to_tensor(alt), tmp_path)


@pytest.mark.backend_tinygrad
def test_tinygrad_preview_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Realized tinygrad arrays compare through the tinygrad payload codec."""

    monkeypatch.setenv("DEV", "PYTHON")
    tinygrad = pytest.importorskip("tinygrad")
    tiny_nn = pytest.importorskip("tinygrad.nn")
    tensor = tinygrad.Tensor
    tensor.manual_seed(0)

    class Net:
        def __init__(self) -> None:
            self.conv = tiny_nn.Conv2d(1, 4, 3)
            self.head = tiny_nn.Linear(64, 2)

        def __call__(self, x: Any) -> Any:
            return self.head(self.conv(x).relu().flatten(1))

    model = Net()
    for param in tiny_nn.state.get_parameters(model):
        param.realize()
    x, alt = _inputs((1, 1, 6, 6))
    _check_comparison("tinygrad", model, tensor(x).realize(), tensor(alt).realize(), tmp_path)


class _HostArrayCodec:
    """Stand-in codec requiring no optional framework runtime."""

    def to_numpy(self, value: Any) -> Any:
        """Use NumPy's conversion failures for unsupported payloads."""

        return SimpleNamespace(array=np.asarray(value), logical_dtype="float32", codec_metadata={})


def test_comparison_tensor_reads_preview_arrays_through_the_codec() -> None:
    """Convert arrays independently and preserve torch payload identity and metadata."""

    from torchlens._comparison_tensor import _comparison_tensor

    codec = _HostArrayCodec()
    array = np.arange(6, dtype=np.float32).reshape(2, 3)[:, ::-1]
    result = _comparison_tensor(array, codec)
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu" and result.dtype == torch.float32
    np.testing.assert_array_equal(result.numpy(), array)
    result.zero_()
    assert np.count_nonzero(array) == 5
    tensor = torch.ones(2, requires_grad=True)
    assert _comparison_tensor(tensor, codec) is tensor
    assert _comparison_tensor({"container": array}, codec) is None
    assert _comparison_tensor(None, codec) is None
    scalar_codec = SimpleNamespace(
        to_numpy=lambda value: SimpleNamespace(
            array=np.asarray([value], dtype=np.float32),
            logical_dtype="float32",
            codec_metadata={"logical_shape": []},
        )
    )
    scalar = _comparison_tensor(3.0, scalar_codec)
    assert scalar.ndim == 0 and scalar.item() == 3.0
    assert _comparison_tensor([1.0, 2.0], scalar_codec) is None


def test_comparison_tensor_decodes_bfloat16_storage_bits() -> None:
    """TF and Paddle bf16 transport bits must compare as floating values."""

    from torchlens._comparison_tensor import _comparison_tensor

    expected = torch.tensor([0.0, -2.0, 1.5], dtype=torch.bfloat16)
    for backend in ("tf", "paddle"):
        codec = SimpleNamespace(
            to_numpy=lambda value: SimpleNamespace(
                array=value, logical_dtype=f"{backend}.bfloat16", codec_metadata={}
            )
        )
        actual = _comparison_tensor(expected.view(torch.uint16).numpy(), codec)
        assert torch.equal(actual, expected)


def test_comparison_tensor_spatial_units_need_explicit_dim() -> None:
    """Spatial preview counts require layout evidence; torch keeps its channel axis."""

    from torchlens.debug._graph import _feature_dim, _unit_axis_reason

    out, op = torch.zeros(2, 3, 4, 5), SimpleNamespace()
    for backend in ("jax", "tf", "mlx", "paddle", "tinygrad"):
        trace = SimpleNamespace(_provenance_backend=backend)
        assert _feature_dim(trace, op, out, None, {}) is None
        assert "unit axis unavailable" in _unit_axis_reason(trace, out, None)
        assert _feature_dim(trace, op, out, -1, {}) == 3
    assert _feature_dim(SimpleNamespace(_provenance_backend="torch"), op, out, None, {}) == 1

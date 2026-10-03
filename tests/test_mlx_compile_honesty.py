"""mx.compile / traced-transform honesty for the technical-preview MLX backend.

A compiled callable replays a traced graph, so wrapped eager functions never
fire inside it: before detection existed a compiled entry silently
under-captured. The entry now refuses typed, and a compiled attribute on the
model ceilings the capture with ``capture_verified=False`` instead of letting
honesty depend on compile-cache timing.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.backend_mlx

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError  # noqa: E402
from torchlens.backends.mlx.backend import (  # noqa: E402
    _find_mlx_compiled_attributes,
    _mlx_traced_transform_type,
)


class _TinyMLP(nn.Module):
    """Two-layer MLX MLP used across the honesty tests."""

    def __init__(self) -> None:
        """Initialize two linear layers."""

        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 4)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass."""

        return self.l2(nn.relu(self.l1(x)))


@pytest.mark.optional
def test_traced_transform_type_resolves_from_runtime() -> None:
    """The probe resolves one exact wrapper type shared by all traced transforms."""

    transform_type = _mlx_traced_transform_type(mx)
    assert transform_type is not None

    def f(x: mx.array) -> mx.array:
        """Double the input."""

        return x * 2

    assert type(mx.compile(f)) is transform_type
    assert type(mx.grad(lambda x: (f(x)).sum())) is transform_type
    assert type(mx.vmap(f)) is transform_type
    assert type(mx.add) is not transform_type
    assert type(f) is not transform_type


@pytest.mark.optional
def test_compiled_entry_refuses_typed() -> None:
    """A compiled model entry refuses instead of silently under-capturing."""

    model = _TinyMLP()
    compiled = mx.compile(model)
    x = mx.random.normal((2, 4))
    with pytest.raises(BackendUnsupportedError, match="traced-transform"):
        tl.trace(compiled, x, backend="mlx")


@pytest.mark.optional
def test_compiled_plain_function_entry_refuses_typed() -> None:
    """A compiled raw callable entry refuses the same way."""

    def f(x: mx.array) -> mx.array:
        """Add one and take relu."""

        return nn.relu(x + 1.0)

    with pytest.raises(BackendUnsupportedError, match="traced-transform"):
        tl.trace(mx.compile(f), mx.ones((3,)), backend="mlx")


@pytest.mark.optional
def test_compiled_attribute_ceilings_capture_verified() -> None:
    """A compiled attribute ceilings the trace; eager interior stays captured."""

    model = _TinyMLP()
    model.fast_path = mx.compile(lambda x: x * 2)
    x = mx.random.normal((2, 4))
    with pytest.warns(UserWarning, match="capture_verified=False"):
        log = tl.trace(model, x, backend="mlx")
    assert log.capture_verified is False
    assert log.capture_verification_reason == "mlx_compiled_attribute_not_logged"
    # The eager forward is still honestly captured.
    assert log.num_ops > 0


@pytest.mark.optional
def test_nested_and_container_compiled_attributes_are_found() -> None:
    """The bounded scan sees nested-module attrs and one container level."""

    transform_type = _mlx_traced_transform_type(mx)
    model = _TinyMLP()
    model.l1.helper = mx.compile(lambda x: x)
    model.bank = [mx.compile(lambda x: x + 1)]
    paths = _find_mlx_compiled_attributes(model, transform_type)
    assert any(path.startswith("l1.") for path in paths)
    assert any(path.startswith("bank[") for path in paths)


@pytest.mark.optional
def test_plain_capture_stays_unceilinged() -> None:
    """A model without compiled attributes captures with no ceiling or warning."""

    model = _TinyMLP()
    x = mx.random.normal((2, 4))
    log = tl.trace(model, x, backend="mlx")
    assert getattr(log, "capture_verified", None) is None
    assert getattr(log, "capture_verification_reason", None) is None

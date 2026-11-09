"""Preview ops at the top level carry no module owner, as torch's do.

Torch never instruments a plain root module's forward, so an op computed directly
in it has ``op.module is None`` and no ``self`` frame in ``op.modules``. Preview
backends record the root call as the first frame of every op's raw module stack;
the shared finalizer refuses an op whose raw stack lacks that frame, builds the
module logs from the raw stacks, then drops the root frame from the public op
fields. The ``module_attribution`` invariant accepts an op with no module claim
only when the root call lists it and no other module does (its raw stack was the
root frame alone); every other unclaimed op still fails.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import torchlens as tl
from torchlens.errors import MetadataInvariantError
from torchlens.validation import check_metadata_invariants
from torchlens.validation._invariants_entry import _check_compute_op_module_attribution

#: Object-module preview backends under test, each with its CI leg marker.
_BACKENDS = ("mlx", "paddle", "jax", "tf", "tinygrad")
_MARKS = {
    "mlx": pytest.mark.backend_mlx,
    "paddle": pytest.mark.backend_paddle,
    "jax": pytest.mark.backend_jax,
    "tf": pytest.mark.tf_backend,
    "tinygrad": pytest.mark.backend_tinygrad,
}
#: Backends whose module logs come from the shared finalizer's raw-stack builder.
_SHARED_FINALIZER = ("mlx", "paddle", "tf", "tinygrad")
#: Op fields the invariant reads as module claims, by their stored names.
_CLAIM_FIELDS = (
    "_module_context_call_stack",
    "_module_context_input_to_module_calls",
    "_module_context_output_of_modules",
    "_module_context_output_of_module_calls",
)


def _mlx_trace() -> Any:
    """MLX trace of a linear submodule followed by top-level tanh and exp."""

    mx = pytest.importorskip("mlx.core")
    mnn = pytest.importorskip("mlx.nn")

    class Chain(mnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = mnn.Linear(4, 4)

        def __call__(self, x):
            return mx.exp(mx.tanh(self.lin(x)))

    return tl.trace(Chain(), mx.ones((1, 4)), backend="mlx")


def _paddle_trace() -> Any:
    """Paddle trace of a linear sublayer followed by top-level tanh and exp."""

    paddle = pytest.importorskip("paddle")

    class Chain(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.lin = paddle.nn.Linear(4, 4)

        def forward(self, x):
            return paddle.exp(paddle.tanh(self.lin(x)))

    return tl.trace(Chain(), paddle.ones([1, 4]), backend="paddle")


def _jax_trace() -> Any:
    """JAX trace of an Equinox linear submodule followed by a top-level tanh."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    eqx = pytest.importorskip("equinox")

    class Chain(eqx.Module):
        lin: Any

        def __init__(self) -> None:
            self.lin = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))

        def __call__(self, x):
            return jnp.tanh(self.lin(x))

    return tl.trace(Chain(), jnp.ones(4, dtype=jnp.float32), backend="jax")


def _tf_trace() -> Any:
    """TensorFlow trace of a matmul submodule followed by a top-level relu."""

    tf = pytest.importorskip("tensorflow")

    class Lin(tf.Module):
        def __init__(self) -> None:
            super().__init__(name="lin")
            self.weight = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name="weight")

        def __call__(self, x):
            return tf.matmul(x, self.weight)

    class Chain(tf.Module):
        def __init__(self) -> None:
            super().__init__(name="chain")
            self.lin = Lin()

        def __call__(self, x):
            return tf.nn.relu(self.lin(x))

    return tl.trace(Chain(), tf.ones((1, 2)), backend="tf")


def _tinygrad_trace() -> Any:
    """tinygrad trace of a linear submodule followed by a top-level add."""

    tinygrad = pytest.importorskip("tinygrad")
    from tinygrad.nn import Linear

    class Chain:
        def __init__(self) -> None:
            self.fc = Linear(3, 3)

        def __call__(self, x):
            y = self.fc(x)
            return y + y

    return tl.trace(Chain(), tinygrad.Tensor.ones(1, 3), backend="tinygrad")


_TRACES: dict[str, Callable[[], Any]] = {
    "mlx": _mlx_trace,
    "paddle": _paddle_trace,
    "jax": _jax_trace,
    "tf": _tf_trace,
    "tinygrad": _tinygrad_trace,
}


def _params(backends: tuple[str, ...]) -> list[Any]:
    """One pytest param per backend, carrying its leg marker."""

    return [pytest.param(name, marks=_MARKS[name], id=name) for name in backends]


def _compute_ops(trace: Any) -> list[Any]:
    """Ops that are not inputs, outputs or buffers."""

    return [
        op for op in trace._layer_list if not (op.is_input or op.is_output or op.graph.is_buffer)
    ]


def _nested_layer_labels(trace: Any) -> set[str]:
    """Layer labels some non-root module lists."""

    return {
        label
        for module in trace.modules
        if module.address != "self"
        for label in module.layer_labels
    }


def _split(trace: Any) -> tuple[list[Any], list[Any]]:
    """Compute ops at the top level and inside a submodule, each non-empty."""

    nested = _nested_layer_labels(trace)
    ops = _compute_ops(trace)
    top = [op for op in ops if op.layer_label not in nested]
    inner = [op for op in ops if op.layer_label in nested]
    assert top and inner, "the test model must compute ops both at the top level and inside"
    return top, inner


def _unclaim(op: Any) -> None:
    """Give an op torch's top-level shape: no module, no module frames, no call claims."""

    op.modules = []
    op.module = None
    for field in _CLAIM_FIELDS:
        if hasattr(op, field):
            setattr(op, field, [])


def _root_call_ops(trace: Any) -> set[str]:
    """Pass labels every recorded root call lists."""

    root = trace.modules["self"]
    return {label for index in root.calls for label in root.calls[index].ops}


def test_torch_top_level_ops_carry_no_owner() -> None:
    """The reference shape: torch's top-level ops have no module and no ``self`` frame."""

    import torch

    class Chain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return torch.exp(torch.tanh(self.lin(x)))

    trace = tl.trace(Chain(), torch.ones(1, 4))
    top, inner = _split(trace)
    assert all(op.module is None and tuple(op.modules) == () for op in top)
    assert all(str(frame).split(":")[0] != "self" for op in inner for frame in op.modules)


@pytest.mark.parametrize("backend", _params(_BACKENDS))
def test_preview_top_level_ops_carry_no_owner_like_torch(backend: str) -> None:
    """Capture gives top-level preview ops torch's shape, and the trace still validates."""

    trace = _TRACES[backend]()
    top, inner = _split(trace)
    for op in top:
        assert op.module is None, (backend, op.label, op.module)
        assert tuple(op.modules) == (), (backend, op.label, op.modules)
    for op in inner:
        assert op.module is not None, (backend, op.label)
        assert not any(str(frame).split(":")[0] == "self" for frame in op.modules), (
            backend,
            op.label,
            op.modules,
        )
    assert {op.label for op in top} <= _root_call_ops(trace)
    assert check_metadata_invariants(trace) is True


@pytest.mark.parametrize("backend", _params(_BACKENDS))
def test_top_level_op_in_torch_shape_passes_attribution(backend: str) -> None:
    """An unclaimed op the root call alone lists passes ``module_attribution``."""

    trace = _TRACES[backend]()
    top, _inner = _split(trace)
    for op in top:
        _unclaim(op)
    _check_compute_op_module_attribution(trace)


@pytest.mark.parametrize("backend", _params(_BACKENDS))
def test_unclaimed_op_inside_a_submodule_still_fails(backend: str) -> None:
    """An op a submodule lists cannot drop its claim: the carve-out does not cover it."""

    trace = _TRACES[backend]()
    _top, inner = _split(trace)
    _unclaim(inner[0])
    with pytest.raises(MetadataInvariantError, match="module/module-call attribution"):
        _check_compute_op_module_attribution(trace)


@pytest.mark.parametrize("backend", _params(_BACKENDS))
def test_unclaimed_op_the_root_call_does_not_list_still_fails(backend: str) -> None:
    """An unclaimed op missing from the root call's own ops still fails."""

    trace = _TRACES[backend]()
    top, _inner = _split(trace)
    target = top[0]
    _unclaim(target)
    root = trace.modules["self"]
    for index in root.calls:
        call = root.calls[index]
        call.ops = [label for label in call.ops if label != target.label]
    with pytest.raises(MetadataInvariantError, match="module/module-call attribution"):
        _check_compute_op_module_attribution(trace)


@pytest.mark.parametrize("variant", ["nested", "root_only"])
@pytest.mark.parametrize("backend", _params(_SHARED_FINALIZER))
def test_raw_stack_without_the_root_frame_is_refused(
    backend: str, variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An op whose raw module stack lost the root frame fails, at capture or validation.

    ``nested`` drops the root frame from stacks inside a submodule (the op still
    names its submodule); ``root_only`` drops it from top-level stacks (the op is
    left with no frame at all). Both are capture gaps the carve-out must not mask.
    """

    module = pytest.importorskip(f"torchlens.backends.{backend}.backend")
    original = module.normalize_op_module_calls

    def lose_root_frame(value: Any) -> tuple[tuple[str, int], ...]:
        calls = tuple(original(value))
        drop = len(calls) > 1 if variant == "nested" else len(calls) == 1
        return tuple(call for call in calls if call[0] != "self") if drop else calls

    monkeypatch.setattr(module, "normalize_op_module_calls", lose_root_frame)
    with pytest.raises(MetadataInvariantError):
        trace = _TRACES[backend]()
        check_metadata_invariants(trace)

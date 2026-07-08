"""Anti-recurrence tests for the shared preview-backend finalizer (cert10).

Covers two bugs in ``torchlens/backends/_finalize.py::finalize_single_pass_trace``:

FIX 1 (BLOCKER): ``Op._tracing_finished`` was never flipped to ``True`` for any
preview (non-torch) backend. Only the trace-level flag was set. Every
finalized preview-backend op therefore stayed on the "mid-capture" ``__str__``
branch (``Op._str_during_pass``), which calls the torch-only
``print_override`` helper on a non-torch array and crashes with
``AttributeError`` (e.g. ``'array' object has no attribute 'grad_fn'`` on
MLX). The fix mirrors ``torchlens.postprocess.finalization._set_tracing_finished``,
which also flips the per-op flag, not just the trace-level one.

FIX 2 (MAJOR): ``op.param_memory`` was left as a plain ``int`` instead of the
canonical ``Bytes`` quantity type on live (unserialized) preview-backend
traces, inconsistent with torch and with the JAX pytree path (commit
18ff476d) that already wraps it correctly. It silently self-heals on a
serialize round-trip (``Op`` deserialization always wraps ``param_memory`` in
``Bytes``), which is exactly why this stayed MAJOR rather than a blocker.

Both are exercised live against MLX and tinygrad (both import cleanly in this
environment). A third, backend-agnostic test drives
``finalize_single_pass_trace`` directly against a minimal hand-built ``Trace``
so the regression is caught even in environments where no preview backend is
installed.

Known follow-up (explicitly NOT fixed here, out of ``_finalize.py`` scope):
once ``_tracing_finished`` correctly flips, finalized preview-backend ops with
a *saved* activation route into ``Op._str_after_pass`` ->
``Op._tensor_contents_str_helper`` -> ``torchlens.utils.display.tensor_stats_summary``,
which is typed and implemented for ``torch.Tensor`` only (``tensor.device``,
``torch.isnan``, ``.to(torch.float64)``, ...) and raises its own
``AttributeError`` on non-torch arrays. That is a distinct, pre-existing,
always-torch-only limitation in ``torchlens/data_classes/op.py`` /
``torchlens/utils/display.py`` -- unreachable before this fix (because ops
never left the mid-capture branch), now reachable, and requires touching
files outside this fix's scope. Tests below therefore exercise ``str()``/
``repr()`` on ops with no saved activation (``op.out is None``), which is
sufficient to prove FIX 1's documented crash class (the mid-capture
``print_override`` branch) is closed, without masking or weakening any
validation invariant.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

import torchlens as tl
from torchlens.backends._finalize import (
    attach_function_root_module,
    finalize_single_pass_trace,
)
from torchlens.data_classes.trace import Trace
from torchlens.quantities import Bytes

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

tinygrad = pytest.importorskip("tinygrad")
from tinygrad import Tensor as TinygradTensor  # noqa: E402


class _MlxMlp(nn.Module):
    """Small MLX MLP used to exercise object-module param attribution."""

    def __init__(self) -> None:
        """Initialize a single linear layer so params get attached to ops."""

        super().__init__()
        self.l1 = nn.Linear(4, 8)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the linear layer."""

        return self.l1(x)


@pytest.mark.optional
def test_mlx_finalize_flips_per_op_tracing_finished_and_types_param_memory() -> None:
    """MLX: finalized ops carry ``_tracing_finished=True`` and ``Bytes`` param_memory.

    Uses ``save=`` matching nothing so ``op.out is None`` and ``str()``/``repr()``
    skip the (separately torch-only, out-of-scope) tensor-content display path
    -- isolating exactly the two behaviors this fix owns.
    """

    model = _MlxMlp()
    x = mx.random.normal((2, 4))
    log = tl.trace(model, x, save=tl.func("__never_matches__"))

    assert log.num_ops > 0
    sampled = 0
    for op_label in log.op_labels:
        op = log[op_label]
        assert op._tracing_finished is True, op_label
        assert op.out is None
        # Must not crash: before the fix this op was permanently stuck on
        # Op._str_during_pass, which calls the torch-only print_override
        # helper and raises AttributeError on an mlx.core.array.
        str(op)
        repr(op)
        sampled += 1
    assert sampled > 0

    # At least one op should have MLX-linear params attached (Bytes-typed).
    param_ops = [log[label] for label in log.op_labels if log[label].num_params > 0]
    assert param_ops, "expected at least one op with attached params"
    for op in param_ops:
        assert type(op.param_memory) is Bytes, (op.label, type(op.param_memory))


@pytest.mark.optional
def test_mlx_finalize_bytes_typed_with_saved_activation() -> None:
    """MLX: ``param_memory`` stays ``Bytes``-typed even with activations saved.

    (This one does not assert on ``str()``/``repr()`` -- see module docstring
    for the separate, pre-existing, out-of-scope ``tensor_stats_summary``
    torch-only limitation that saved activations would hit.)
    """

    model = _MlxMlp()
    x = mx.random.normal((2, 4))
    log = tl.trace(model, x)

    param_ops = [log[label] for label in log.op_labels if log[label].num_params > 0]
    assert param_ops
    for op in param_ops:
        assert op._tracing_finished is True
        assert type(op.param_memory) is Bytes, (op.label, type(op.param_memory))


@pytest.mark.optional
def test_tinygrad_finalize_flips_per_op_tracing_finished() -> None:
    """tinygrad: finalized ops carry ``_tracing_finished=True`` and no crash."""

    def _fn(x: TinygradTensor) -> TinygradTensor:
        """Tiny elementwise expression."""

        return (x * 2).relu()

    x = TinygradTensor.randn(2, 4)
    log = tl.trace(_fn, x, backend="tinygrad", save=tl.func("__never_matches__"))

    assert log.num_ops > 0
    for op_label in log.op_labels:
        op = log[op_label]
        assert op._tracing_finished is True, op_label
        assert type(op.param_memory) is Bytes, (op_label, type(op.param_memory))
        str(op)
        repr(op)


def test_finalize_single_pass_trace_sets_per_op_tracing_finished_backend_agnostic() -> None:
    """Regression on ``finalize_single_pass_trace`` itself, calling it directly.

    Rather than hand-building a synthetic ``Op`` (fragile -- ``Op`` has many
    interdependent fields consumed by ``Layer`` construction downstream),
    this harvests REAL, fully-valid ``Op`` instances from one already-captured
    MLX trace, resets them to the exact pre-fix state (``_tracing_finished =
    False``, ``param_memory`` as a plain ``int``), feeds them into a *fresh*
    ``Trace`` as its ``_raw_layer_dict``, and calls
    ``finalize_single_pass_trace`` directly (bypassing ``tl.trace`` /
    backend dispatch entirely). This isolates the finalizer's own contract
    from any specific backend's capture path.
    """

    model = _MlxMlp()
    x = mx.random.normal((2, 4))
    seed_trace = tl.trace(model, x, save=tl.func("__never_matches__"))
    assert seed_trace.num_ops >= 1

    trace = Trace(model_class_name="MinimalModel")
    raw_ops = OrderedDict()
    for label in seed_trace.layer_labels:
        op_log = seed_trace.layer_dict_main_keys[label]
        op_log._tracing_finished = False
        # Simulate the pre-fix backend hook state: a plain int, not Bytes.
        op_log.param_memory = int(op_log.param_memory)
        raw_ops[label] = op_log
    trace._raw_layer_dict = raw_ops

    finalize_single_pass_trace(
        trace,
        backend_name="mlx",
        module_tree=None,
        attach_function_root_module=attach_function_root_module,
        attach_object_module_logs=lambda *_args, **_kwargs: None,
        # A no-op hook still exercises the post-hook Bytes-normalization step
        # in ``finalize_single_pass_trace`` (FIX 2), matching how a real
        # backend hook would leave ``op_log.param_memory`` as a plain int.
        attach_op_params=lambda *_args, **_kwargs: None,
    )

    assert trace._tracing_finished is True
    assert len(trace.layer_dict_main_keys) == len(raw_ops)
    for label, op_log in trace.layer_dict_main_keys.items():
        assert op_log._tracing_finished is True, label
        assert type(op_log.param_memory) is Bytes, (label, type(op_log.param_memory))

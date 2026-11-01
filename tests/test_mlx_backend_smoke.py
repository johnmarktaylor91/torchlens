"""Optional smoke coverage for the technical-preview MLX backend."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.backend_mlx

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.errors import BackendUnsupportedError  # noqa: E402


@pytest.mark.optional
def test_mlx_linear_mlp_smoke() -> None:
    """Capture an MLX linear MLP; assert structural Trace parity with torch equivalent."""

    class MLP(nn.Module):
        """Small MLX MLP used for backend smoke testing."""

        def __init__(self) -> None:
            """Initialize two linear layers."""

            super().__init__()
            self.l1 = nn.Linear(4, 8)
            self.l2 = nn.Linear(8, 4)

        def __call__(self, x: mx.array) -> mx.array:
            """Run the MLP forward pass."""

            h = self.l1(x)
            h = nn.relu(h)
            return self.l2(h)

    model = MLP()
    x = mx.random.normal((2, 4))
    log = tl.trace(model, x)

    assert log._num_ops > 0
    assert log.autograd.has_backward_pass is False
    assert any("linear" in label.lower() for label in log.layer_labels)
    assert log._num_ops in (3, 4, 5)
    for op_label in log.op_labels:
        op = log[op_label]
        assert op.shape is not None
        assert op.dtype is not None


@pytest.mark.optional
def test_mlx_unwrapped_output_marks_capture_unverified() -> None:
    """An output no wrapped op produced flags the trace instead of passing silently.

    Array operators and methods are captured now; a function outside the
    wrapped surface (``mx.linalg.norm``) still is not, and a forward ending in
    one returns a trace flagged ``unwrapped_op_in_capture``, never a silent
    trace with zero output ops.
    """

    with pytest.warns(UserWarning, match="not produced by a wrapped operation"):
        trace = tl.trace(lambda v: mx.linalg.norm(v), mx.array([1.0, 2.0]), backend="mlx")
    assert trace.provenance.verified is False
    assert trace.provenance.verification_reason == "unwrapped_op_in_capture"
    assert not list(trace.output_ops)

    wrapped = tl.trace(lambda v: v + 1.0, mx.array([1.0, 2.0]), backend="mlx")
    # A clean capture leaves the field unset (the missing-attribute guard raises).
    assert getattr(wrapped, "_provenance_verified", True) is True
    assert getattr(wrapped, "_provenance_verification_reason", None) is None
    assert len(list(wrapped.output_ops)) == 1


@pytest.mark.optional
def test_mlx_preview_rejects_random_seed() -> None:
    """MLX preview refuses an explicit seed instead of storing inert metadata."""

    with pytest.raises(BackendUnsupportedError, match="explicit seed is refused"):
        tl.trace(
            lambda x: x + 1,
            mx.array([1.0]),
            backend="mlx",
            capture=tl.options.CaptureOptions(seed=123),
        )


@pytest.mark.optional
def test_mlx_save_raw_activations_false_drops_payloads_keeps_metadata() -> None:
    """MLX's declared save_raw_activations=False capability is real (R17-5)."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(4, 4)

        def __call__(self, x: mx.array) -> mx.array:
            return nn.relu(self.l1(x))

    model = Tiny()
    x = mx.random.normal((2, 4))
    log = tl.trace(model, x, backend="mlx", save=tl.options.SaveOptions(save_raw_activations=False))

    assert log._num_ops > 0
    for op_label in log.op_labels:
        op = log[op_label]
        assert op.out is None
        assert op.shape is not None
        assert op.dtype is not None


@pytest.mark.optional
def test_mlx_capture_options_does_not_keyerror() -> None:
    """N5: any ``capture=CaptureOptions(...)`` call must not ``KeyError``.

    ``trace()``'s own public signature dropped every individual flat capture
    kwarg (``layers_to_save``, ``activation_transform``, ``keep_orphans``,
    ...) in favor of the single grouped ``capture=`` spelling, but
    ``_trace_mlx_model_from_public_kwargs`` kept reading them with
    ``kwargs["activation_transform"]`` etc. -- a plain dict subscript that
    raised ``KeyError: 'activation_transform'`` on essentially every call
    that supplied ``capture=`` (the key is never present in the registry
    dispatch's keyword bundle anymore). This is the first such flat name
    alphabetically reached, so it masked every other missing key behind it.
    """

    model = nn.Linear(4, 4)
    x = mx.random.normal((2, 4))
    log = tl.trace(
        model,
        x,
        backend="mlx",
        capture=tl.options.CaptureOptions(keep_orphans=True),
    )

    assert log._num_ops > 0


@pytest.mark.optional
def test_realize_once_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture evaluates exactly once: one ``mx.eval`` over payloads and outputs."""

    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = mx.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]])
    calls: list[int] = []
    real_eval = mx.eval

    def counting_eval(*values: object, **kwargs: object) -> object:
        calls.append(len(values))
        return real_eval(*values, **kwargs)

    monkeypatch.setattr(mx, "eval", counting_eval)
    trace = tl.trace(model, x, backend="mlx")
    assert len(calls) == 1, f"expected one evaluation, saw {len(calls)}"
    assert calls[0] >= trace._num_ops
    assert all(op.out is not None for op in trace._layer_list)

"""Optional smoke coverage for the technical-preview MLX backend."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.backend_mlx

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError  # noqa: E402


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

    assert log.num_ops > 0
    assert log.has_backward_pass is False
    assert any("linear" in label.lower() for label in log.layer_labels)
    assert log.num_ops in (3, 4, 5)
    for op_label in log.op_labels:
        op = log[op_label]
        assert op.shape is not None
        assert op.dtype is not None


@pytest.mark.optional
def test_mlx_preview_rejects_random_seed() -> None:
    """MLX preview rejects random_seed instead of storing inert metadata."""

    with pytest.raises(BackendUnsupportedError, match="random_seed"):
        tl.trace(
            lambda x: x + 1,
            mx.array([1.0]),
            backend="mlx",
            capture=tl.options.CaptureOptions(random_seed=123),
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

    assert log.num_ops > 0
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

    assert log.num_ops > 0

"""Optional smoke coverage for the technical-preview MLX backend."""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402


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

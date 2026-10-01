"""Preview parity for compute_input_output_distances (torch Step 4).

Every preview backend must honor the same public surface torch honors — the
canonical ``compute_input_output_distances`` AND the deprecated public
``mark_layer_depths`` alias — with the SAME default (torch's
``CaptureOptions`` default is ``True``) and a working ``=False`` off switch
on every backend, with input/output hop distances plus ancestor/descendant
lineage sets and the effective value stored on ``trace.mark_layer_depths``.
Distances are asserted EXACTLY, including min/max splits on branch/merge
graphs, not just non-emptiness. Previews must also store the EFFECTIVE
``recurrence_detection``: True where the neutral grouper actually ran
(JAX, plus each backend ported in parity wave 2), False where finalize
never grouped.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.backend_parity


def _assert_depths(trace, expect_recurrence: bool, expected_max_depth: int) -> None:
    ops = list(trace.layer_list)
    depths = {
        op._label_raw: (op.min_distance_from_input, op.max_distance_from_input)
        for op in ops
        if getattr(op, "min_distance_from_input", None) is not None
    }
    assert depths, "depth flood populated no op"
    assert trace.mark_layer_depths is True
    assert trace.recurrence_detection is expect_recurrence
    assert max(min_d for min_d, _max_d in depths.values()) == expected_max_depth
    assert any(getattr(op, "input_ancestors", None) for op in ops)
    assert any(getattr(op, "output_descendants", None) for op in ops)


def _depth_by_prefix(trace, prefix: str) -> tuple[int, int]:
    op = next(op for op in trace.layer_list if op._label_raw.startswith(prefix))
    return op.min_distance_from_input, op.max_distance_from_input


def _assert_alias_matches(trace_fn) -> None:
    """The deprecated public mark_layer_depths alias must behave identically."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alias_trace = trace_fn()
    assert alias_trace.mark_layer_depths is True
    assert any(
        getattr(op, "min_distance_from_input", None) is not None for op in alias_trace.layer_list
    )


def _assert_default_on_and_off_switch(trace_fn) -> None:
    """Torch-parity default is True on every preview and =False disables it.

    N5: ``compute_input_output_distances`` is a ``CaptureOptions`` field now,
    not a flat top-level ``trace()`` kwarg; passing it bare lands in
    ``**forward_kwargs`` and raises an unrelated "keyword(s) it does not
    route" error instead of exercising the on/off switch under test.
    """

    from torchlens.options import CaptureOptions

    default_trace = trace_fn(CaptureOptions())
    assert default_trace.mark_layer_depths is True
    assert any(
        getattr(op, "min_distance_from_input", None) is not None for op in default_trace.layer_list
    )
    off_trace = trace_fn(CaptureOptions(compute_input_output_distances=False))
    assert off_trace.mark_layer_depths is False
    assert all(getattr(op, "min_distance_from_input", None) is None for op in off_trace.layer_list)


@pytest.mark.backend_paddle
def test_paddle_depth_parity() -> None:
    paddle = pytest.importorskip("paddle")
    import torchlens as tl

    class M(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = paddle.nn.Linear(4, 3)
            self.l2 = paddle.nn.Linear(3, 2)

        def forward(self, x):
            return self.l2(paddle.nn.functional.relu(self.l1(x)))

    trace = tl.trace(
        M(),
        paddle.ones([1, 4]),
        backend="paddle",
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )
    # Paddle runs the neutral recurrence grouper by default (parity wave 2), so
    # the stored EFFECTIVE value is True; this model has no recurrence and all
    # layers stay single-pass.
    _assert_depths(trace, expect_recurrence=True, expected_max_depth=3)
    assert _depth_by_prefix(trace, "functional.relu") == (2, 2)
    _assert_alias_matches(
        lambda: tl.trace(
            M(),
            paddle.ones([1, 4]),
            backend="paddle",
            capture=tl.options.CaptureOptions(compute_input_output_distances=True),
        )
    )
    _assert_default_on_and_off_switch(
        lambda capture: tl.trace(M(), paddle.ones([1, 4]), backend="paddle", capture=capture)
    )
    base = tl.trace(M(), paddle.ones([1, 4]), backend="paddle")
    assert base.recurrence_detection is True
    ungrouped = tl.trace(
        M(),
        paddle.ones([1, 4]),
        backend="paddle",
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )
    assert ungrouped.recurrence_detection is False


@pytest.mark.backend_tinygrad
def test_tinygrad_depth_parity() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    import torchlens as tl

    def model(x):
        return ((x + 1.0).relu() * 2.0).sum()

    trace = tl.trace(
        model,
        Tensor([1.0, -2.0, 3.0]),
        backend="tinygrad",
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )
    # tinygrad runs the neutral recurrence grouper by default (parity wave 2);
    # this model has no recurrence and all layers stay single-pass.
    _assert_depths(trace, expect_recurrence=True, expected_max_depth=5)
    # relu decomposes through where; mul merges the where branch (depth 3)
    # with the broadcast constant path, so its min/max split is exact.
    assert _depth_by_prefix(trace, "mul_1") == (3, 4)
    _assert_alias_matches(
        lambda: tl.trace(
            model,
            Tensor([1.0, -2.0, 3.0]),
            backend="tinygrad",
            capture=tl.options.CaptureOptions(compute_input_output_distances=True),
        )
    )
    _assert_default_on_and_off_switch(
        lambda capture: tl.trace(
            model, Tensor([1.0, -2.0, 3.0]), backend="tinygrad", capture=capture
        )
    )


@pytest.mark.backend_jax
def test_jax_depth_parity() -> None:
    jnp = pytest.importorskip("jax.numpy")
    import torchlens as tl

    def model(x):
        hidden = jnp.tanh(x @ jnp.ones((4, 3)))
        return hidden + jnp.tanh(hidden)

    trace = tl.trace(
        model,
        jnp.ones((1, 4)),
        backend="jax",
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )
    _assert_depths(trace, expect_recurrence=True, expected_max_depth=3)
    # Branch/merge exactness: dot(1) -> tanh(2) -> tanh(3); the merge add sees
    # the short path (tanh#1 + 1 = 3) and the long path (tanh#2 + 1 = 4).
    assert _depth_by_prefix(trace, "dot_general_1") == (1, 1)
    assert _depth_by_prefix(trace, "tanh_1") == (2, 2)
    assert _depth_by_prefix(trace, "tanh_2") == (3, 3)
    assert _depth_by_prefix(trace, "add_1") == (3, 4)
    _assert_alias_matches(
        lambda: tl.trace(
            model,
            jnp.ones((1, 4)),
            backend="jax",
            capture=tl.options.CaptureOptions(compute_input_output_distances=True),
        )
    )
    _assert_default_on_and_off_switch(
        lambda capture: tl.trace(model, jnp.ones((1, 4)), backend="jax", capture=capture)
    )


@pytest.mark.backend_mlx
def test_mlx_depth_parity() -> None:
    mx = pytest.importorskip("mlx.core")
    mnn = pytest.importorskip("mlx.nn")
    import torchlens as tl

    class M(mnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = mnn.Linear(4, 3)
            self.l2 = mnn.Linear(3, 2)

        def __call__(self, x):
            return self.l2(mnn.relu(self.l1(x)))

    trace = tl.trace(
        M(),
        mx.ones((1, 4)),
        backend="mlx",
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )
    # MLX runs the neutral recurrence grouper by default (parity wave 2); this
    # model has no recurrence and all layers stay single-pass.
    _assert_depths(trace, expect_recurrence=True, expected_max_depth=3)
    assert _depth_by_prefix(trace, "linear_1") == (1, 1)
    assert _depth_by_prefix(trace, "relu_1") == (2, 2)
    assert _depth_by_prefix(trace, "linear_2") == (3, 3)
    _assert_alias_matches(
        lambda: tl.trace(
            M(),
            mx.ones((1, 4)),
            backend="mlx",
            capture=tl.options.CaptureOptions(compute_input_output_distances=True),
        )
    )
    # F1 regression: the public kwarg is threaded through the MLX dispatch, so
    # =False actually disables the flood instead of being silently dropped.
    _assert_default_on_and_off_switch(
        lambda capture: tl.trace(M(), mx.ones((1, 4)), backend="mlx", capture=capture)
    )


@pytest.mark.backend_mlx
def test_mlx_depth_branch_merge_exact() -> None:
    """Merge nodes carry an exact min/max split, not just any value."""

    mx = pytest.importorskip("mlx.core")
    mnn = pytest.importorskip("mlx.nn")
    import torchlens as tl

    class M(mnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = mnn.Linear(4, 4)

        def __call__(self, x):
            hidden = self.l1(x)
            return mx.add(mnn.relu(hidden), hidden)

    trace = tl.trace(
        M(),
        mx.ones((1, 4)),
        backend="mlx",
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )
    assert _depth_by_prefix(trace, "linear_1") == (1, 1)
    assert _depth_by_prefix(trace, "relu_1") == (2, 2)
    assert _depth_by_prefix(trace, "add_1") == (2, 3)


@pytest.mark.tf_backend
def test_tf_depth_parity() -> None:
    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    import torchlens as tl

    model = keras.Sequential([keras.layers.Dense(3, activation="relu"), keras.layers.Dense(2)])
    inputs = tf.ones((1, 4))
    model(inputs)
    trace = tl.trace(
        model,
        inputs,
        backend="tf",
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )
    # TF eager runs the neutral recurrence grouper by default (parity wave
    # 2); this model has no recurrence and all layers stay single-pass.
    # matmul(1) -> biasadd(2) -> relu(3) -> matmul(4) -> biasadd(5)
    _assert_depths(trace, expect_recurrence=True, expected_max_depth=5)
    assert _depth_by_prefix(trace, "relu_1") == (3, 3)
    _assert_alias_matches(
        lambda: tl.trace(
            model,
            inputs,
            backend="tf",
            capture=tl.options.CaptureOptions(compute_input_output_distances=True),
        )
    )
    # F2 regression: compute_input_output_distances joins tf's
    # default_if_missing block, so the flood has a real off switch instead of
    # a truthy MISSING sentinel keeping it unconditionally on.
    _assert_default_on_and_off_switch(
        lambda capture: tl.trace(model, inputs, backend="tf", capture=capture)
    )

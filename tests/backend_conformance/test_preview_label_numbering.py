"""Preview backends number op labels by torch's rule.

Torch (``_postprocess/labeling.py``) gives input ops no step number (``input_1``),
numbers every other layer's step from 1 in execution order (``linear_1_1``), counts
type numbers per layer, and lets later passes of a recurrent layer inherit the first
pass's numbers. Preview traces follow the same rule
(``torchlens/backends/_finalize_labels.py``), so where a preview graph matches torch's
the labels of matching ops are equal. Artifacts saved before the rule changed keep
their saved labels and still load.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

#: Preview backends under test, each with its CI leg marker.
_BACKENDS = ("mlx", "paddle", "jax", "tf", "tinygrad")
_MARKS = {
    "mlx": pytest.mark.backend_mlx,
    "paddle": pytest.mark.backend_paddle,
    "jax": pytest.mark.backend_jax,
    "tf": pytest.mark.tf_backend,
    "tinygrad": pytest.mark.backend_tinygrad,
}
#: Backends whose chain and loop graphs match torch's op for op, so labels must be equal.
_GRAPH_MATCHES_TORCH = frozenset({"mlx"})


def _torch_traces() -> tuple[Any, Any]:
    """Torch traces of the chain (linear, tanh, exp) and the three-pass loop."""

    import torch

    import torchlens as tl

    class Chain(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return torch.exp(torch.tanh(self.lin(x)))

    class Loop(Chain):
        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.lin(x))
            return x

    x = torch.ones(1, 4)
    return tl.trace(Chain(), x), tl.trace(Loop(), x)


def _mlx_traces() -> tuple[Any, Any]:
    """MLX traces of the chain and the loop."""

    mx = pytest.importorskip("mlx.core")
    mnn = pytest.importorskip("mlx.nn")
    import torchlens as tl

    class Chain(mnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = mnn.Linear(4, 4)

        def __call__(self, x):
            return mx.exp(mx.tanh(self.lin(x)))

    class Loop(Chain):
        def __call__(self, x):
            for _ in range(3):
                x = mx.tanh(self.lin(x))
            return x

    x = mx.ones((1, 4))
    return tl.trace(Chain(), x, backend="mlx"), tl.trace(Loop(), x, backend="mlx")


def _paddle_traces() -> tuple[Any, Any]:
    """Paddle traces of the chain and the loop."""

    paddle = pytest.importorskip("paddle")
    import torchlens as tl

    class Chain(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.lin = paddle.nn.Linear(4, 4)

        def forward(self, x):
            return paddle.exp(paddle.tanh(self.lin(x)))

    class Loop(Chain):
        def forward(self, x):
            for _ in range(3):
                x = paddle.tanh(self.lin(x))
            return x

    x = paddle.ones([1, 4])
    return tl.trace(Chain(), x, backend="paddle"), tl.trace(Loop(), x, backend="paddle")


def _jax_traces() -> tuple[Any, Any]:
    """JAX traces of a chain and a loop written as functions."""

    jnp = pytest.importorskip("jax.numpy")
    import torchlens as tl

    w = jnp.ones((4, 4)) / 8.0

    def chain(x):
        return jnp.exp(jnp.tanh(x @ w))

    def loop(x):
        for _ in range(3):
            x = jnp.tanh(x @ w)
        return x

    x = jnp.ones((1, 4))
    return tl.trace(chain, x, backend="jax"), tl.trace(loop, x, backend="jax")


def _tf_traces() -> tuple[Any, Any]:
    """TensorFlow traces of the chain and the loop."""

    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    import torchlens as tl

    class Chain(keras.Model):
        def __init__(self) -> None:
            super().__init__()
            self.dense = keras.layers.Dense(4)

        def call(self, x):
            return tf.math.exp(tf.math.tanh(self.dense(x)))

    class Loop(Chain):
        def call(self, x):
            for _ in range(3):
                x = tf.math.tanh(self.dense(x))
            return x

    x = tf.ones((1, 4))
    chain, loop = Chain(), Loop()
    chain(x)
    loop(x)
    return tl.trace(chain, x, backend="tf"), tl.trace(loop, x, backend="tf")


def _tinygrad_traces() -> tuple[Any, Any]:
    """tinygrad traces of a chain and a loop written as functions."""

    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    import torchlens as tl

    w = Tensor.arange(16, dtype="float").reshape(4, 4) / 20.0

    def chain(x, w):
        return (x @ w).tanh().exp()

    def loop(x, w):
        for _ in range(3):
            x = (x @ w).tanh()
        return x

    x = Tensor.ones(1, 4)
    return tl.trace(chain, (x, w), backend="tinygrad"), tl.trace(loop, (x, w), backend="tinygrad")


_BUILDERS: dict[str, Callable[[], tuple[Any, Any]]] = {
    "mlx": _mlx_traces,
    "paddle": _paddle_traces,
    "jax": _jax_traces,
    "tf": _tf_traces,
    "tinygrad": _tinygrad_traces,
}


def _backend_params() -> list[Any]:
    """One parameter per backend, carrying that backend's marker."""

    return [pytest.param(name, marks=_MARKS[name], id=name) for name in _BACKENDS]


def _torch_rule_labels(trace: Any) -> list[str]:
    """Recompute every op's label from op types and passes alone, by torch's rule."""

    type_counts: defaultdict[str, int] = defaultdict(int)
    step = 0
    by_layer: dict[str, str] = {}
    labels = []
    for op in trace._layer_list:
        if op.pass_number == 1:
            type_counts[op.type] += 1
            short = f"{op.type}_{type_counts[op.type]}"
            if op.is_input or op.graph.is_buffer or op.type == "output":
                by_layer[op.layer_label] = short
            else:
                step += 1
                by_layer[op.layer_label] = f"{short}_{step}"
        labels.append(f"{by_layer[op.layer_label]}:{op.pass_number}")
    return labels


def _compute_view(trace: Any) -> list[tuple[str, str]]:
    """``(type, label)`` of every op except torch's output pseudo-op."""

    return [(op.type, op.label) for op in trace._layer_list if not op.is_output]


@pytest.mark.parametrize("backend", _backend_params())
def test_preview_labels_follow_torch_rule(backend: str) -> None:
    """Inputs carry no step, compute steps count from 1, passes share their layer's numbers."""

    for trace in _BUILDERS[backend]():
        labels = [op.label for op in trace._layer_list]
        assert labels == _torch_rule_labels(trace)
        assert all(op.is_input for op in trace._layer_list if op.type == "input")
        inputs = [op for op in trace._layer_list if op.is_input]
        assert inputs and inputs[0].label == "input_1:1"
        assert all(op.step_index == 0 for op in inputs)
        steps = [op.step_index for op in trace._layer_list if not op.is_input]
        assert steps == list(range(1, len(steps) + 1))
        for op in trace._layer_list:
            assert op.short_label == f"{op.type}_{op.type_index}:{op.pass_number}"
            assert trace[op.label] is op


@pytest.mark.parametrize("backend", _backend_params())
def test_preview_labels_equal_torch_where_graphs_match(backend: str) -> None:
    """Where the preview graph has torch's op types in torch's order, the labels are equal."""

    previews = _BUILDERS[backend]()
    torch_traces = _torch_traces()
    matched = 0
    for preview, reference in zip(previews, torch_traces, strict=True):
        preview_view, torch_view = _compute_view(preview), _compute_view(reference)
        if [t for t, _ in preview_view] != [t for t, _ in torch_view]:
            continue
        matched += 1
        assert [label for _, label in preview_view] == [label for _, label in torch_view]
    if backend in _GRAPH_MATCHES_TORCH:
        assert matched == len(previews)
    print(f"label-equality {backend}: {matched} of {len(previews)} graphs match torch")


def _old_rule(raw_ops: Any, assignments: Any = None) -> dict[str, Any]:
    """The numbering preview traces had before torch's rule: the raw label, suffix stripped."""

    from torchlens._capture._nonfinite_prefix import strip_raw_label_suffix
    from torchlens.backends._finalize_labels import PreviewFinalLabel

    finals = {}
    for raw_label, op_log in raw_ops.items():
        assignment = assignments.get(raw_label) if assignments is not None else None
        layer = strip_raw_label_suffix(assignment.layer_label if assignment else raw_label)
        pass_index = assignment.pass_index if assignment else 1
        finals[raw_label] = PreviewFinalLabel(
            layer, layer, op_log.type_index, op_log.step_index, pass_index
        )
    return finals


@pytest.mark.parametrize("backend", _backend_params())
def test_artifact_saved_with_old_numbering_loads(
    backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A preview artifact written before the renumbering loads with its saved labels."""

    import torchlens as tl
    import torchlens.backends._finalize as finalize

    monkeypatch.setattr(finalize, "torch_rule_final_labels", _old_rule)
    if backend == "jax":
        import torchlens.backends.jax.backend as jax_backend

        monkeypatch.setattr(jax_backend, "torch_rule_final_labels", _old_rule)
    old_trace = _BUILDERS[backend]()[0]
    old_labels = [op.label for op in old_trace._layer_list]
    assert old_labels != _torch_rule_labels(old_trace), "the old rule must differ from torch's"
    path = tmp_path / f"{backend}-old-numbering"
    old_trace.save(path)
    monkeypatch.undo()

    loaded = tl.load(path)
    assert [op.label for op in loaded._layer_list] == old_labels
    for label in old_labels:
        op = loaded[label]
        assert op.label == label
        if op.has_saved_activation:
            assert op.out is not None
    assert loaded.output_layers == old_trace.output_layers

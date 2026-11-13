"""The "What works on each framework" table in ``docs/backends.md``, rerun.

The table's cells say Yes, Partial or No for each framework and task. This file parses the
table and, for every Yes or Partial cell, runs that task's representative call on a small
convolutional network built from the framework's own library, so a backend regression fails
here instead of leaving the table stale. No cells are not run. The preview parameters carry
their framework marker, so each backend's CI job runs its own row on the pinned version.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

pytestmark = pytest.mark.rare

_DOC = Path(__file__).resolve().parents[1] / "docs" / "backends.md"
_HEADING = "## What works on each framework"

#: Table row name to backend key.
_FRAMEWORKS: dict[str, str] = {
    "PyTorch": "torch",
    "JAX": "jax",
    "TensorFlow": "tf",
    "MLX": "mlx",
    "PaddlePaddle": "paddle",
    "tinygrad": "tinygrad",
}
_COLUMNS = (
    "Select",
    "Extract",
    "Inspect",
    "Visualize",
    "Intervene",
    "Compare",
    "Debug",
    "Save and load",
    "Gradients",
)
_VERDICTS = ("Yes", "Partial", "No")
#: The pytest marker each preview's framework-dependent tests carry (``pyproject.toml``).
_MARKERS: dict[str, str] = {
    "jax": "backend_jax",
    "tf": "tf_backend",
    "mlx": "backend_mlx",
    "paddle": "backend_paddle",
    "tinygrad": "backend_tinygrad",
}
_CELL_RE = re.compile(r"^(?P<verdict>Yes|Partial|No)(?: \((?P<note>\d+)\))?$")


def _section() -> list[str]:
    """The lines of the table section, up to the next ``## `` heading."""

    lines = _DOC.read_text(encoding="utf-8").splitlines()
    assert _HEADING in lines, f"docs/backends.md lacks {_HEADING!r}"
    body: list[str] = []
    for line in lines[lines.index(_HEADING) + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return body


def _rows() -> list[list[str]]:
    """The table's rows (header first), separator row dropped."""

    rows = []
    for line in _section():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _table() -> dict[tuple[str, str], tuple[str, str | None]]:
    """``{(backend, column): (verdict, note number)}`` read from the docs table."""

    rows = _rows()
    assert rows and tuple(rows[0]) == ("Framework", *_COLUMNS), f"unexpected header {rows[:1]}"
    cells: dict[tuple[str, str], tuple[str, str | None]] = {}
    for row in rows[1:]:
        assert row[0] in _FRAMEWORKS, f"unknown framework row {row[0]!r}"
        assert len(row) == len(_COLUMNS) + 1, f"{row[0]} row has {len(row) - 1} cells"
        for column, cell in zip(_COLUMNS, row[1:], strict=True):
            match = _CELL_RE.match(cell)
            assert match is not None, (
                f"{row[0]} / {column}: cell {cell!r} is not Yes, Partial or No"
            )
            cells[(_FRAMEWORKS[row[0]], column)] = (match["verdict"], match["note"])
    return cells


def test_table_lists_every_framework_and_explains_every_gap() -> None:
    """Six framework rows, nine columns, and a note for every Partial or No cell."""

    cells = _table()
    assert {backend for backend, _ in cells} == set(_FRAMEWORKS.values())
    notes = {match[1] for line in _section() if (match := re.match(r"^(\d+)\. \S", line))}
    for (backend, column), (verdict, note) in cells.items():
        if verdict == "Yes":
            assert note is None, f"{backend} / {column}: a Yes cell carries no note"
            continue
        assert note is not None, f"{backend} / {column}: a {verdict} cell needs a note"
        assert note in notes, f"{backend} / {column}: note ({note}) is missing below the table"


@dataclass
class _Subject:
    """One framework's small model, inputs and the hooks the representative calls need."""

    backend: str | None
    model: Any
    x: Any
    x_alt: Any
    x_nan: Any
    act_func: str
    gradients: Callable[[], None]
    # Builds a trace after one native ``trace.backward``; set only where the backend
    # records backward, and required for a non-PyTorch Visualize cell that says Yes.
    backward: Callable[[], Any] | None = None

    def trace(self, inputs: Any = None, **kwargs: Any) -> Any:
        """Capture the model on ``inputs`` (default ``x``) with this subject's backend."""

        if self.backend is not None:
            kwargs["backend"] = self.backend
        return tl.trace(self.model, self.x if inputs is None else inputs, **kwargs)


def _np(value: Any) -> np.ndarray:
    """Any framework array as a NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _output(trace: Any) -> Any:
    return trace[trace.output_layers[0]].out


def _act_op(trace: Any, func_name: str) -> Any:
    return next(trace[label] for label in trace.op_labels if trace[label].func_name == func_name)


def _grad_options(backend: str) -> Any:
    """The preview's ``GradOptions`` class, through ``tl.backends.<name>``.

    The backend packages resolve lazily, so this works in a fresh process before any capture.
    """

    return getattr(tl.backends, backend).GradOptions


def _with_nan(array: np.ndarray, index: tuple[int, ...]) -> np.ndarray:
    poisoned = array.copy()
    poisoned[index] = np.nan
    return poisoned


def _torch_subject() -> _Subject:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(64, 2)).eval()
    x = torch.randn(1, 1, 6, 6)
    x_nan = torch.from_numpy(_with_nan(x.numpy(), (0, 0, 2, 2)))

    def gradients() -> None:
        trace = tl.trace(model, x, capture=CaptureOptions(save_grads="all"))
        _output(trace).sum().backward()
        assert model[0].weight.grad is not None
        assert any(trace[label].grad is not None for label in trace.op_labels)

    return _Subject(None, model, x, torch.randn(1, 1, 6, 6), x_nan, "relu", gradients)


def _jax_subject() -> _Subject:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    nnx = pytest.importorskip("flax.nnx")

    class Net(nnx.Module):
        def __init__(self, rngs: Any) -> None:
            self.conv = nnx.Conv(1, 4, (3, 3), padding="VALID", rngs=rngs)
            self.head = nnx.Linear(64, 2, rngs=rngs)

        def __call__(self, x: Any) -> Any:
            h = nnx.relu(self.conv(x))
            return self.head(h.reshape((h.shape[0], -1)))

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 6, 6, 1)).astype(np.float32)
    alt = rng.standard_normal((1, 6, 6, 1)).astype(np.float32)

    def gradients() -> None:
        # Table note 13: gradients work on a plain function of its weights.
        params = {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))}

        def fn(p: Any, inp: Any) -> Any:
            return jax.nn.relu(inp @ p["w"] + p["b"])

        options = _grad_options("jax")(
            params=params,
            loss_fn=lambda out: jnp.sum(out * out),
            input_grad_argnums=(0,),
            intermediate_grads=True,
        )
        trace = tl.trace(
            fn,
            (params, jnp.ones((4, 3))),
            backend="jax",
            capture=CaptureOptions(grad_options=options),
        )
        assert len(trace.autograd.derived_grads) > 0
        assert len(trace.autograd.intermediate_derived_grads) > 0

    subject = _Subject(
        "jax",
        Net(nnx.Rngs(0)),
        jnp.asarray(x),
        jnp.asarray(alt),
        jnp.asarray(_with_nan(x, (0, 2, 2, 0))),
        "max",
        gradients,
    )
    subject.backward = lambda: subject.trace().backward(lambda out: jnp.sum(out * out))
    return subject


def _tf_subject() -> _Subject:
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
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 6, 6, 1)).astype(np.float32)
    alt = rng.standard_normal((1, 6, 6, 1)).astype(np.float32)
    subject = _Subject(
        "tf",
        model,
        tf.constant(x),
        tf.constant(alt),
        tf.constant(_with_nan(x, (0, 2, 2, 0))),
        "Relu",
        lambda: None,
    )
    subject.gradients = lambda: _derived_gradients(
        subject,
        _grad_options("tf")(
            loss_fn=lambda out: tf.reduce_sum(out * out),
            input_grad_argnums=(0,),
            intermediate_grads=True,
        ),
    )
    subject.backward = lambda: subject.trace(capture=CaptureOptions(backward_ready=True)).backward(
        lambda out: tf.reduce_sum(out * out)
    )
    return subject


def _mlx_subject() -> _Subject:
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
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 6, 6, 1)).astype(np.float32)
    alt = rng.standard_normal((1, 6, 6, 1)).astype(np.float32)
    subject = _Subject(
        "mlx",
        model,
        mx.array(x),
        mx.array(alt),
        mx.array(_with_nan(x, (0, 2, 2, 0))),
        "relu",
        lambda: None,
    )
    subject.gradients = lambda: _derived_gradients(
        subject,
        _grad_options("mlx")(
            loss_fn=lambda out: mx.sum(out * out), input_grad_argnums=(0,), intermediate_grads=True
        ),
    )
    return subject


def _paddle_subject() -> _Subject:
    paddle = pytest.importorskip("paddle")
    paddle.seed(0)
    model = paddle.nn.Sequential(
        paddle.nn.Conv2D(1, 4, 3), paddle.nn.ReLU(), paddle.nn.Flatten(), paddle.nn.Linear(64, 2)
    )
    model.eval()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 6, 6)).astype(np.float32)
    alt = rng.standard_normal((1, 1, 6, 6)).astype(np.float32)
    subject = _Subject(
        "paddle",
        model,
        paddle.to_tensor(x),
        paddle.to_tensor(alt),
        paddle.to_tensor(_with_nan(x, (0, 0, 2, 2))),
        "functional.relu",
        lambda: None,
    )
    subject.gradients = lambda: _derived_gradients(
        subject,
        _grad_options("paddle")(
            loss_fn=lambda out: paddle.sum(out * out),
            input_grad_argnums=(0,),
            intermediate_grads=True,
        ),
    )
    subject.backward = lambda: subject.trace(capture=CaptureOptions(backward_ready=True)).backward(
        lambda out: paddle.sum(out * out)
    )
    return subject


def _tinygrad_subject() -> _Subject:
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
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 6, 6)).astype(np.float32)
    alt = rng.standard_normal((1, 1, 6, 6)).astype(np.float32)
    subject = _Subject(
        "tinygrad",
        model,
        tensor(x).realize(),
        tensor(alt).realize(),
        tensor(_with_nan(x, (0, 0, 2, 2))).realize(),
        "where",
        lambda: None,
    )
    subject.gradients = lambda: _derived_gradients(
        subject,
        _grad_options("tinygrad")(
            loss_fn=lambda out: (out * out).sum(), input_grad_argnums=(0,), intermediate_grads=True
        ),
    )
    return subject


def _derived_gradients(subject: _Subject, options: Any) -> None:
    """Loss gradients reach the inputs, the weights and each saved op."""

    trace = subject.trace(capture=CaptureOptions(grad_options=options))
    assert len(trace.autograd.derived_grads) > 0
    assert len(trace.autograd.intermediate_derived_grads) > 0, "no per-op gradients"


_SUBJECTS: dict[str, Callable[[], _Subject]] = {
    "torch": _torch_subject,
    "jax": _jax_subject,
    "tf": _tf_subject,
    "mlx": _mlx_subject,
    "paddle": _paddle_subject,
    "tinygrad": _tinygrad_subject,
}


def _select(subject: _Subject, tmp_path: Path) -> None:
    del tmp_path
    assert len(subject.trace()[tl.func(subject.act_func)]) > 0


def _extract(subject: _Subject, tmp_path: Path) -> None:
    del tmp_path
    # Hold each Trace while reading its ops: an op does not keep its trace alive.
    full_trace = subject.trace()
    kept_trace = subject.trace(save=tl.func(subject.act_func))
    full = _act_op(full_trace, subject.act_func)
    kept = _act_op(kept_trace, subject.act_func)
    np.testing.assert_allclose(_np(kept.out), _np(full.out))


def _inspect(subject: _Subject, tmp_path: Path) -> None:
    del tmp_path
    trace = subject.trace()
    op = _act_op(trace, subject.act_func)
    assert op.short_layer_label in str(op)
    assert tuple(op.shape) == _np(op.out).shape


def _visualize(subject: _Subject, tmp_path: Path) -> None:
    subject.trace().draw(str(tmp_path / "graph"))
    assert any(path.name.startswith("graph") for path in tmp_path.iterdir())
    if subject.backend is None or _table()[(subject.backend, "Visualize")][0] != "Yes":
        return
    # Yes on a non-PyTorch row means the backward pass draws too, alone and with the forward.
    assert subject.backward is not None, f"{subject.backend}: a Yes Visualize cell needs backward"
    trace = subject.backward()
    assert trace.autograd.has_backward_pass
    for graph in ("backward", "combined"):
        drawn = trace.draw(str(tmp_path / graph), graph=graph, file_format="svg", rolled=False)
        assert "digraph" in drawn.dot, f"{subject.backend}: no {graph} graph"


def _intervene(subject: _Subject, tmp_path: Path) -> None:
    del tmp_path
    plain = _output(subject.trace())
    edited = _output(subject.trace(intervene=tl.when(tl.func(subject.act_func), tl.zero_ablate())))
    assert not np.allclose(_np(edited), _np(plain)), "the edit did not change the output"


def _compare(subject: _Subject, tmp_path: Path) -> None:
    del tmp_path
    runs = tl.Bundle({"a": subject.trace(), "b": subject.trace(subject.x_alt)})
    assert runs.rank_changes("a", k=3), "two different inputs ranked no changes"


def _debug(subject: _Subject, tmp_path: Path) -> None:
    del tmp_path
    kwargs = {} if subject.backend is None else {"backend": subject.backend}
    report = tl.debug.find_nan(subject.model, subject.x_nan, **kwargs)
    assert str(report.status) == "found", str(report)


def _save_and_load(subject: _Subject, tmp_path: Path) -> None:
    trace = subject.trace()
    op = _act_op(trace, subject.act_func)
    path = tmp_path / "trace.tlspec"
    trace.save(str(path))
    # Hold the loaded Trace while reading its op: an op does not keep its trace alive.
    loaded = tl.load(str(path))
    np.testing.assert_allclose(_np(loaded[op.label].out), _np(op.out))


def _gradients(subject: _Subject, tmp_path: Path) -> None:
    del tmp_path
    subject.gradients()


_CALLS: dict[str, Callable[[_Subject, Path], None]] = {
    "Select": _select,
    "Extract": _extract,
    "Inspect": _inspect,
    "Visualize": _visualize,
    "Intervene": _intervene,
    "Compare": _compare,
    "Debug": _debug,
    "Save and load": _save_and_load,
    "Gradients": _gradients,
}


def _cell_params() -> list[Any]:
    """One parameter per Yes or Partial cell, carrying its framework's marker."""

    params = []
    for (backend, column), (verdict, _) in sorted(_table().items()):
        if verdict == "No":
            continue
        marker = _MARKERS.get(backend)
        marks = [getattr(pytest.mark, marker)] if marker else []
        ident = f"{backend}-{column.lower().replace(' ', '_')}"
        params.append(pytest.param(backend, column, marks=marks, id=ident))
    return params


@pytest.mark.parametrize(("backend", "column"), _cell_params())
def test_yes_and_partial_cells_still_work(backend: str, column: str, tmp_path: Path) -> None:
    """The representative call behind a Yes or Partial cell runs and gives a real answer."""

    _CALLS[column](_SUBJECTS[backend](), tmp_path)

"""Every preview-backend ``Trace`` carries the full declared Trace field set.

The v11 respell made shared readers (save's scrub, the summary report,
capture-honesty) read declared fields such as ``Trace._provenance_verified``
directly. Only torch capture stages stamp that field and its siblings (the
completeness, escape and thread-provenance stamps), so every tf, jax, mlx,
paddle and tinygrad save or summary raised ``AttributeError``. Every preview
``Trace(...)`` construction is now followed by one shared fill,
``_trace_state_io.fill_unset_declared_fields``, which gives each
declared-but-unset field the neutral value a reloaded trace receives.

The first three tests need no preview framework: they check the fill itself
and statically require every preview ``Trace(...)`` construction to be
followed by it. The marked tests capture a tiny function root on each installed preview
backend and check the field set plus a save and reload.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import torchlens as tl
from torchlens._constants import MODEL_LOG_FIELD_ORDER
from torchlens._data_classes._trace_state_io import fill_unset_declared_fields
from torchlens._data_classes.trace import Trace
from torchlens.backends._finalize import new_preview_function_trace

_BACKENDS_DIR = Path(tl.__file__).parent / "backends"


def _missing_declared_fields(trace: Trace) -> list[str]:
    """Declared Trace fields that neither the instance nor its class provides."""

    class_names = set(dir(type(trace)))
    return sorted(
        name
        for name in MODEL_LOG_FIELD_ORDER
        if name not in trace.__dict__ and name not in class_names
    )


def _plain_root(x: Any) -> Any:
    """A framework-free root callable for the shared preview constructor."""

    return x


def test_shared_preview_constructor_yields_every_declared_field() -> None:
    """``new_preview_function_trace`` returns a trace with no declared field absent."""

    trace = new_preview_function_trace(
        backend_name="jax",
        model=_plain_root,
        keep_orphans=False,
        num_context_lines=7,
        recurrence_detection=True,
        verbose=False,
        name=None,
        raw_input=None,
        save_raw_input="small",
        batch_render="auto",
        output_transform=None,
        save_transformed_output="small",
        param_source="none",
    )

    assert _missing_declared_fields(trace) == []
    # The field whose absence broke every preview save: unverified, not ceilinged.
    assert trace._provenance_verified is None
    assert trace._provenance_verification_reason is None


def test_declared_field_fill_sets_absent_fields_only_with_fresh_values() -> None:
    """The fill never overwrites a stamped field and never shares a mutable default."""

    first = Trace("first")
    first._provenance_verified = False
    first._provenance_verification_reason = "compiled_attribute"
    second = Trace("second")
    fill_unset_declared_fields(first)
    fill_unset_declared_fields(second)

    assert _missing_declared_fields(first) == []
    assert _missing_declared_fields(second) == []
    assert first._provenance_verified is False
    assert first._provenance_verification_reason == "compiled_attribute"
    assert second._provenance_verified is None
    assert first.capture_guard_passes == [] == second.capture_guard_passes
    assert first.capture_guard_passes is not second.capture_guard_passes
    assert first.completeness_diagnostics is not second.completeness_diagnostics


def _preview_backend_sources() -> list[Path]:
    """Every non-torch backend module plus the shared preview finalizer."""

    sources = [_BACKENDS_DIR / "_finalize.py"]
    for package in sorted(p for p in _BACKENDS_DIR.iterdir() if p.is_dir()):
        if package.name in {"torch", "__pycache__"} or package.name.startswith("_"):
            continue
        sources.extend(sorted(package.rglob("*.py")))
    return sources


def _is_fill_of(statement: ast.stmt, name: str) -> bool:
    """Whether ``statement`` is ``fill_unset_declared_fields(<name>)``."""

    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "fill_unset_declared_fields"
        and [ast.unparse(arg) for arg in statement.value.args] == [name]
    )


def _unfilled_trace_constructions(path: Path) -> list[str]:
    """``Trace(...)`` calls in ``path`` not assigned and then filled at once."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    filled: set[int] = set()
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for statement, following in zip(body, body[1:], strict=False):
                if (
                    isinstance(statement, ast.Assign)
                    and isinstance(statement.value, ast.Call)
                    and len(statement.targets) == 1
                    and _is_fill_of(following, ast.unparse(statement.targets[0]))
                ):
                    filled.add(id(statement.value))
    return [
        f"{path.relative_to(_BACKENDS_DIR.parent.parent)}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Trace"
        and id(node) not in filled
    ]


def test_every_preview_trace_construction_is_filled() -> None:
    """A preview backend that builds ``Trace(...)`` without the fill skips declared fields."""

    sources = _preview_backend_sources()
    packages = {path.parent.name for path in sources if path.parent != _BACKENDS_DIR}
    assert {"jax", "mlx", "paddle", "tf", "tinygrad"} <= packages

    offenders = [site for path in sources for site in _unfilled_trace_constructions(path)]

    assert offenders == [], (
        "follow these `trace = Trace(...)` constructions with "
        f"fill_unset_declared_fields(trace) so the trace gets every declared field: {offenders}"
    )


def _jax_capture() -> Trace:
    """A two-op jax function root."""

    jnp = pytest.importorskip("jax.numpy")

    def jax_root(x: Any) -> Any:
        """Matmul then tanh."""

        return jnp.tanh(x @ x.T)

    return tl.trace(jax_root, jnp.ones((2, 3)), backend="jax")


def _mlx_capture() -> Trace:
    """A two-op mlx function root."""

    mx = pytest.importorskip("mlx.core")

    def mlx_root(x: Any) -> Any:
        """Matmul then a ReLU."""

        return mx.maximum(mx.matmul(x, x.T), 0.0)

    return tl.trace(mlx_root, mx.ones((2, 3)), backend="mlx")


def _paddle_capture() -> Trace:
    """A two-op paddle function root."""

    paddle = pytest.importorskip("paddle")

    def paddle_root(x: Any) -> Any:
        """Matmul then a ReLU."""

        return paddle.nn.functional.relu(paddle.matmul(x, x.T))

    return tl.trace(paddle_root, paddle.ones([2, 3], dtype="float32"), backend="paddle")


def _tinygrad_capture() -> Trace:
    """A two-op tinygrad function root."""

    tinygrad = pytest.importorskip("tinygrad")

    def tinygrad_root(x: Any) -> Any:
        """Matmul then a ReLU."""

        return (x @ x.T).relu()

    return tl.trace(tinygrad_root, tinygrad.Tensor.ones(2, 3), backend="tinygrad")


def _tf_capture() -> Trace:
    """A two-op tensorflow function root."""

    tf = pytest.importorskip("tensorflow")

    def tf_root(x: Any) -> Any:
        """Matmul then a ReLU."""

        return tf.nn.relu(tf.matmul(x, x, transpose_b=True))

    return tl.trace(tf_root, tf.ones((2, 3)), backend="tf")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "capture",
    [
        pytest.param(_jax_capture, id="jax", marks=pytest.mark.backend_jax),
        pytest.param(_mlx_capture, id="mlx", marks=pytest.mark.backend_mlx),
        pytest.param(_paddle_capture, id="paddle", marks=pytest.mark.backend_paddle),
        pytest.param(_tinygrad_capture, id="tinygrad", marks=pytest.mark.backend_tinygrad),
        pytest.param(_tf_capture, id="tf", marks=pytest.mark.tf_backend),
    ],
)
def test_preview_capture_has_every_declared_field_and_saves(
    capture: Callable[[], Trace], tmp_path: Path
) -> None:
    """A real preview capture exposes every declared field, saves and reloads."""

    trace = capture()

    assert _missing_declared_fields(trace) == []
    assert trace._provenance_verified is not True
    path = tmp_path / "trace.tlspec"
    trace.save(path)
    loaded = tl.load(str(path))
    assert list(loaded.layer_labels) == list(trace.layer_labels)

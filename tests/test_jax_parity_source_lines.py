"""JAX parity: equation source lines as torch-style ``FuncCallLocation`` frames."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import torchlens as tl

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]

_X = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=np.float32)


def _inner(h: Any) -> Any:
    """Apply the nonlinearity (a nested user frame)."""

    return jnp.tanh(h)


def _model(x: Any) -> Any:
    """Return a model whose tanh runs one call below the entry."""

    return _inner(x * 2.0) + 1.0


def _op(trace: Any, func_name: str) -> Any:
    """Return the first op of ``func_name``."""

    return next(op for op in trace.ops if op.func_name == func_name)


def test_jax_source_frames_call_site_entry_innermost() -> None:
    """Frames run outermost first: the tl.trace call site, the entry, then the op's line."""

    trace = tl.trace(cast(Any, _model), jnp.asarray(_X), backend="jax")
    frames = list(_op(trace, "tanh").source.frames)
    names = [frame.func_name for frame in frames]
    assert names[-2:] == ["_model", "_inner"], names
    assert names[0] == "test_jax_source_frames_call_site_entry_innermost", names
    assert all(frame.file == __file__ for frame in frames)
    assert frames[-1].line_number == _inner.__code__.co_firstlineno + 3
    mul = list(_op(trace, "mul").source.frames)
    assert [frame.func_name for frame in mul][-1] == "_model"


def test_jax_source_context_snapshots_text() -> None:
    """``save_source_context=True`` is accepted and snapshots the source line text."""

    trace = tl.trace(
        cast(Any, _model),
        jnp.asarray(_X),
        backend="jax",
        capture=tl.options.CaptureOptions(save_source_context=True),
    )
    innermost = list(_op(trace, "tanh").source.frames)[-1]
    assert "jnp.tanh(h)" in innermost.call_line


def test_jax_source_frames_module_root_record_call_site() -> None:
    """An Equinox module root's frames start at the call site and include ``__call__``."""

    eqx = pytest.importorskip("equinox")

    class Net(eqx.Module):
        """One-layer Equinox net."""

        fc: Any

        def __init__(self) -> None:
            """Initialize a deterministic layer."""

            self.fc = eqx.nn.Linear(3, 2, key=jax.random.PRNGKey(0))

        def __call__(self, x: Any) -> Any:
            """Return the activated layer output."""

            return jax.nn.relu(self.fc(x))

    trace = tl.trace(Net(), jnp.asarray(_X[0]), backend="jax")
    contexts = [[frame.func_name for frame in op.source.frames] for op in trace.ops]
    in_call = [names for names in contexts if "__call__" in names]
    assert in_call, contexts
    assert all(
        names[0] == "test_jax_source_frames_module_root_record_call_site" for names in in_call
    ), in_call

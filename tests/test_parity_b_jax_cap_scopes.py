"""JAX capture scopes and retention: plain captures keep every value, hooks are optional."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.jax import (  # noqa: E402
    _capture_scopes,
    _retention,
    backend as jax_backend,  # noqa: E402
)


def _fn(x: Any) -> Any:
    return jnp.tanh(x * 2.0) + jnp.sin(x)


def _x() -> Any:
    return jnp.asarray(np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(2, 3))


def test_retention_defaults_to_keep_all() -> None:
    """A session with no policy, or one that cannot hold a policy, keeps every value."""

    class Session:
        pass

    assert _retention.retention_of(Session()) is _retention.KEEP_ALL
    assert _retention.retention_of(1) is _retention.KEEP_ALL
    assert _retention.retention_policy(Session())(123)


def test_keep_only_selects_capture_indexes() -> None:
    """``keep_only`` keeps exactly the listed indexes; ``KEEP_NONE`` keeps none."""

    class Session:
        jax_retention = _retention.keep_only([0, 3])

    session = Session()
    keep = _retention.retention_policy(session)
    assert [keep(i) for i in range(5)] == [True, False, False, True, False]
    assert not _retention.KEEP_NONE.keeps(0)


def test_plain_capture_keeps_every_value_with_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capture passes interpreter hooks, and a plain capture releases nothing."""

    results: list[Any] = []
    hooks_seen: list[Any] = []
    interpret = jax_backend.interpret_closed_jaxpr_with_inlining

    def spying_interpret(*args: Any, **kwargs: Any) -> Any:
        hooks_seen.append(kwargs.get("hooks"))
        result = interpret(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(jax_backend, "interpret_closed_jaxpr_with_inlining", spying_interpret)
    trace = tl.trace(_fn, _x(), backend="jax")

    assert hooks_seen and hooks_seen[-1] is not None
    assert not results[-1].released_capture_indices
    for capture in trace.jax_equation_captures:
        for value in capture.output_values:
            assert type(value).__name__ != "ShapeDtypeStruct"
    assert trace.validate_forward_pass([]) is True


def test_absent_optional_modules_leave_the_forward_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither optional module installed the scopes only yield the hooks."""

    monkeypatch.setattr(_capture_scopes, "_optional_module", lambda _name: None)

    class Session:
        pass

    session = Session()
    with _capture_scopes.forward_scopes(session, backward_ready=True) as scope:
        assert scope.hooks.session is session
        assert scope.release is None
    trace = type("T", (), {})()
    _capture_scopes.retain_backward_state(trace, session, object(), backward_ready=True)
    assert not hasattr(trace, "capture_events")


def test_present_optional_modules_are_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed module's scope is entered and its retain hook sees the event stream."""

    calls: list[str] = []

    class Module:
        def rng_scope(self, session: Any) -> Any:
            return self._scope("rng")

        def backward_ready_scope(self, session: Any) -> Any:
            return self._scope("backward")

        def retain_backward_state(self, trace: Any, session: Any) -> None:
            calls.append(f"retain:{trace.capture_events}")

        def _scope(self, name: str) -> Any:
            from contextlib import contextmanager

            @contextmanager
            def scope() -> Any:
                calls.append(f"enter:{name}")
                yield
                calls.append(f"exit:{name}")

            return scope()

    monkeypatch.setattr(_capture_scopes, "_optional_module", lambda _name: Module())
    with _capture_scopes.forward_scopes(object(), backward_ready=False):
        pass
    assert calls == ["enter:rng", "exit:rng"]
    calls.clear()
    trace = type("T", (), {})()
    with _capture_scopes.forward_scopes(object(), backward_ready=True):
        pass
    _capture_scopes.retain_backward_state(trace, object(), "events", backward_ready=True)
    assert calls == ["enter:rng", "enter:backward", "exit:backward", "exit:rng", "retain:events"]
    assert not hasattr(trace, "capture_events")

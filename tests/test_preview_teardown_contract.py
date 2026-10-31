"""Preview-backend teardown contract: guaranteed cleanup + settle-after-cleanup.

R17-1/R50-1 (b5 parity lane): MLX and Paddle previously stamped COMPLETE/HALTED
inside the capture ``try`` and ran fallible teardown in ``finally`` -- a
teardown raise propagated with the stamp intact, and a raising session cleanup
skipped ``unwrap``, leaking process-global wrappers. The contract pinned here
(mirroring torch's path-7 no-undemoted-claim guarantee, via the stronger
settle-after-cleanup form the path-20 stamp docstring prescribes):

- registry ``unwrap()`` restores EVERY slot even when one setattr raises
  (first failure re-raises after the sweep; failed slots stay registered);
- paddle hook cleanup removes EVERY handle even when one ``remove()`` raises;
- capture epilogues settle AFTER teardown, so a teardown raise escapes
  productless (no COMPLETE stamp) with the wrappers uninstalled.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.backends.mlx.wrappers import _MLXWrapperRegistry
from torchlens.backends.paddle.wrappers import _PaddleWrapperRegistry


class _RefusingOwner:
    """Owner whose attribute restore always raises."""

    def __setattr__(self, name: str, value: object) -> None:
        raise RuntimeError("setattr refused")


@pytest.mark.smoke
@pytest.mark.parametrize("registry_cls", [_MLXWrapperRegistry, _PaddleWrapperRegistry])
def test_registry_unwrap_sweeps_past_failing_setattr(registry_cls: type) -> None:
    """One fallible setattr cannot strand the remaining wrapper restores."""

    class _Owner:
        fn: Any = staticmethod(lambda: "wrapped")

    registry = registry_cls()
    refusing = _RefusingOwner()
    good_a = _Owner()
    good_b = _Owner()
    original_a = lambda: "a"  # noqa: E731
    original_b = lambda: "b"  # noqa: E731
    registry._originals = {
        (good_a, "fn"): original_a,
        (refusing, "fn"): lambda: "refused",
        (good_b, "fn"): original_b,
    }
    registry._wrapped = True

    with pytest.raises(RuntimeError, match="setattr refused"):
        registry.unwrap()

    # Both healthy slots restored despite the mid-sweep failure.
    assert good_a.fn is original_a
    assert good_b.fn is original_b
    # The failed slot stays registered so a retry can restore it.
    assert list(registry._originals) == [(refusing, "fn")]
    assert registry.is_wrapped() is True


@pytest.mark.smoke
def test_paddle_hook_cleanup_sweeps_past_failing_remove() -> None:
    """One raising ``remove()`` cannot strand later hook handles."""

    from torchlens.backends.paddle.model_prep import cleanup_model_session

    removed: list[str] = []

    def _make_handle(tag: str) -> SimpleNamespace:
        return SimpleNamespace(remove=lambda tag=tag: removed.append(tag))

    def _raise() -> None:
        raise RuntimeError("hook remove refused")

    tree = SimpleNamespace(
        hook_handles=[
            _make_handle("first"),
            SimpleNamespace(remove=_raise),
            _make_handle("last"),
        ]
    )

    with pytest.raises(RuntimeError, match="hook remove refused"):
        cleanup_model_session(None, None, tree)  # type: ignore[arg-type]

    assert removed == ["first", "last"]
    assert tree.hook_handles == []


@pytest.mark.optional
@pytest.mark.backend_mlx
def test_mlx_teardown_failure_escapes_productless(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising MLX session cleanup propagates with NO settled stamp and no leak."""

    pytest.importorskip("mlx")
    import mlx.core as mx

    import torchlens as tl
    from torchlens._capture.outcome import outcome_for
    from torchlens.backends.mlx import backend as mlx_backend_mod
    from torchlens.backends.mlx.wrappers import is_mlx_wrapped

    seen: dict[str, Any] = {}

    def _raising_cleanup(self: Any, session: Any, prepared_model: Any) -> None:
        seen["trace"] = session
        raise RuntimeError("injected teardown failure")

    monkeypatch.setattr(mlx_backend_mod.MLXBackend, "cleanup_model_session", _raising_cleanup)

    with pytest.raises(RuntimeError, match="injected teardown failure"):
        tl.trace(lambda x: x * 2 + 1, mx.array([1.0, 2.0]), backend="mlx")

    trace = seen["trace"]
    # Settlement is the LAST act: the teardown raise preempted the stamp, so
    # the escaped object carries NO COMPLETE claim (derives UNATTESTED).
    assert outcome_for(trace) is None
    # And the process-global wrappers still came off (unwrap ran in finally).
    assert is_mlx_wrapped() is False


@pytest.mark.optional
@pytest.mark.backend_paddle
def test_paddle_teardown_failure_escapes_productless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising Paddle hook cleanup propagates with NO settled stamp and no leak."""

    paddle = pytest.importorskip("paddle")

    import torchlens as tl
    from torchlens._capture.outcome import outcome_for
    from torchlens.backends.paddle import backend as paddle_backend_mod
    from torchlens.backends.paddle.wrappers import is_paddle_wrapped

    seen: dict[str, Any] = {}

    def _raising_cleanup(session: Any, prepared_model: Any, tree: Any) -> None:
        seen["trace"] = session
        raise RuntimeError("injected teardown failure")

    monkeypatch.setattr(paddle_backend_mod, "cleanup_model_session", _raising_cleanup)

    with pytest.raises(RuntimeError, match="injected teardown failure"):
        tl.trace(
            lambda x: paddle.nn.functional.relu(x + 1),
            paddle.to_tensor([1.0, -2.0]),
            backend="paddle",
        )

    trace = seen["trace"]
    assert outcome_for(trace) is None
    assert is_paddle_wrapped() is False


# ---------------------------------------------------------------------------
# R07 install fences: a BaseException escaping the wrapper INSTALL phase must
# unwind every process-global patch already landed (Python never calls
# ``__exit__`` when ``__enter__`` raises, so the pre-fix pre-try installs
# stranded class/module wrappers for the life of the process).
# ---------------------------------------------------------------------------


class _InstallInterrupt(KeyboardInterrupt):
    """BaseException-not-Exception used to prove the fence catches BaseException."""


class _ExplodingMapping:
    """Mapping stand-in whose iteration raises after yielding its items."""

    def __init__(self, items: dict[Any, Any]) -> None:
        self._items = items

    def __iter__(self) -> Any:
        yield from self._items
        raise _InstallInterrupt("install interrupted")

    def items(self) -> Any:
        yield from self._items.items()
        raise _InstallInterrupt("install interrupted")


@pytest.mark.smoke
def test_tf_module_stack_install_raise_unwinds_installed_patches() -> None:
    """A BaseException mid-install restores every tf class ``__call__`` landed."""

    from torchlens.backends.tf.modules import patched_tf_module_stack

    class _Module:
        def __call__(self) -> str:
            return "original"

    original_call = _Module.__dict__["__call__"]
    tree = SimpleNamespace(
        modules_by_class=_ExplodingMapping({_Module: {}}),
        address_by_id={},
        call_counts={},
        forward_args_by_call={},
        metadata={},
    )

    with pytest.raises(_InstallInterrupt):
        with patched_tf_module_stack(tree, SimpleNamespace(), []):
            pytest.fail("body must not run when install raises")

    assert _Module.__dict__["__call__"] is original_call


@pytest.mark.smoke
@pytest.mark.parametrize("patcher_name", ["scoped_equinox_module_calls", "scoped_nnx_module_calls"])
def test_jax_module_calls_install_raise_unwinds_installed_patches(
    patcher_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BaseException mid-install restores every jax class ``__call__`` landed."""

    import sys
    import types

    from torchlens.backends.jax import modules as jax_modules

    jax_stub = types.ModuleType("jax")
    jax_stub.named_scope = lambda name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jax", jax_stub)

    class _Module:
        def __call__(self) -> str:
            return "original"

    original_call = _Module.__dict__["__call__"]
    tree = SimpleNamespace(
        modules_by_class=_ExplodingMapping({_Module: {}}),
        call_counts={},
        forward_args_by_call={},
    )
    patcher = getattr(jax_modules, patcher_name)

    with pytest.raises(_InstallInterrupt):
        with patcher(tree):
            pytest.fail("body must not run when install raises")

    assert _Module.__dict__["__call__"] is original_call


@pytest.mark.smoke
def test_tinygrad_module_calls_install_raise_unwinds_installed_patches() -> None:
    """A BaseException mid-install restores every tinygrad class ``__call__`` landed."""

    from torchlens.backends.tinygrad.backend import scoped_tinygrad_module_calls

    class _Module:
        def __call__(self) -> str:
            return "original"

    original_call = _Module.__dict__["__call__"]
    tree = SimpleNamespace(
        modules_by_class=_ExplodingMapping({_Module: {}}),
        call_counts={},
        forward_args_by_call={},
    )

    with pytest.raises(_InstallInterrupt):
        with scoped_tinygrad_module_calls(tree, {}):
            pytest.fail("body must not run when install raises")

    assert _Module.__dict__["__call__"] is original_call


@pytest.mark.smoke
def test_mlx_registry_wrap_install_raise_unwinds_installed_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BaseException mid-``wrap`` restores every MLX module attr landed."""

    import types

    from torchlens.backends.mlx import wrappers as mlx_wrappers

    original_add = lambda *args: "add"  # noqa: E731
    original_relu = lambda *args: "relu"  # noqa: E731
    stub_mx = types.ModuleType("stub_mx")
    stub_mx.add = original_add  # type: ignore[attr-defined]
    stub_nn = types.ModuleType("stub_nn")
    stub_nn.relu = original_relu  # type: ignore[attr-defined]
    monkeypatch.setattr(mlx_wrappers, "_import_mlx", lambda: (stub_mx, stub_nn))

    registry = mlx_wrappers._MLXWrapperRegistry()
    hostile_tree = SimpleNamespace(modules_by_class=_ExplodingMapping({}))

    with pytest.raises(_InstallInterrupt):
        registry.wrap(SimpleNamespace(), module_tree=hostile_tree)  # type: ignore[arg-type]

    # Every wrapper already landed (mx.add, nn.relu) came back off.
    assert stub_mx.add is original_add
    assert stub_nn.relu is original_relu
    assert registry.is_wrapped() is False
    assert registry._originals == {}


@pytest.mark.smoke
def test_paddle_registry_wrap_install_raise_unwinds_installed_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BaseException mid-``wrap`` restores every Paddle module attr landed."""

    import types

    from torchlens.backends.paddle import wrappers as paddle_wrappers

    original_fn = lambda *args: "matmul"  # noqa: E731
    owner = types.ModuleType("stub_paddle_owner")
    owner.matmul = original_fn  # type: ignore[attr-defined]
    monkeypatch.setattr(
        paddle_wrappers,
        "_import_paddle",
        lambda: (SimpleNamespace(), SimpleNamespace(), type("T", (), {})),
    )

    def _candidates(paddle: Any, functional: Any, tensor_cls: Any) -> Any:
        yield (owner, "paddle", "matmul", original_fn, "capture")
        raise _InstallInterrupt("install interrupted")

    monkeypatch.setattr(paddle_wrappers, "_iter_inventory_candidates", _candidates)

    registry = paddle_wrappers._PaddleWrapperRegistry()
    with pytest.raises(_InstallInterrupt):
        registry.wrap(SimpleNamespace())

    assert owner.matmul is original_fn
    assert registry.is_wrapped() is False
    assert registry._originals == {}
    assert registry.inventory() == paddle_wrappers.PaddleInventory((), ())


@pytest.mark.smoke
def test_tf_intervention_wrap_install_raise_unwinds_installed_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BaseException mid-install restores every curated tf entry point landed."""

    import types

    from torchlens.backends.tf import interventions as tf_interventions

    original_fn = lambda *args: "nn_op"  # noqa: E731
    stub_tf = types.ModuleType("stub_tf")
    stub_tf.nn_op = original_fn  # type: ignore[attr-defined]
    monkeypatch.setattr(
        tf_interventions,
        "_CURATED_WRAP_ENTRIES",
        _ExplodingMapping({("", "nn_op"): None}),
    )
    plan = SimpleNamespace(op_sites=[object()])

    with pytest.raises(_InstallInterrupt):
        with tf_interventions.tf_intervention_wrap(stub_tf, plan, None):  # type: ignore[arg-type]
            pytest.fail("body must not run when install raises")

    assert stub_tf.nn_op is original_fn

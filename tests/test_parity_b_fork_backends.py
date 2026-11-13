"""Finished-trace edits on each non-PyTorch backend, through its real replay adapter.

For every backend whose framework is installed and whose adapter module has
landed, a small branching function (a repeated parent ``a * a`` and a fan-out
of ``a``) is captured once clean. Editing the rectifier's output on a fork must
then equal a fresh capture of the same function with the edit written into its
code, a no-op edit must reproduce every payload exactly, the edited trace must
pass the replay check, and corrupting a downstream payload must fail it. A
donor value from a second capture on the same backend patches in; a PyTorch
tensor (another framework) is refused. On Paddle, which supports live
``intervene=``, the edited fork must also equal a live capture with the same
edit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends._replay_adapters import has_adapter
from torchlens.intervention._backend_adapter_lookup import trace_backend
from torchlens.intervention._backend_replay import verify_backend_replay
from torchlens.intervention.errors import HookValueError

_X = np.asarray([[-1.5, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]], dtype=np.float32)

Builder = Callable[[Callable[[Any], Any]], Callable[[Any], Any]]


def _jax() -> tuple[Builder, Any]:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def build(edit: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def chain(v: Any) -> Any:
            a = v + 1.0
            return edit(jax.nn.relu(a * a - 2.0)) + a

        return chain

    return build, jnp.asarray(_X)


def _tf() -> tuple[Builder, Any]:
    tf = pytest.importorskip("tensorflow")

    def build(edit: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def chain(v: Any) -> Any:
            a = v + 1.0
            return edit(tf.nn.relu(a * a - 2.0)) + a

        return chain

    return build, tf.constant(_X)


def _mlx() -> tuple[Builder, Any]:
    mx = pytest.importorskip("mlx.core")

    def build(edit: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def chain(v: Any) -> Any:
            a = v + 1.0
            return edit(mx.maximum(a * a - 2.0, 0.0)) + a

        return chain

    return build, mx.array(_X)


def _paddle() -> tuple[Builder, Any]:
    paddle = pytest.importorskip("paddle")

    def build(edit: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def chain(v: Any) -> Any:
            a = v + 1.0
            return edit(paddle.nn.functional.relu(a * a - 2.0)) + a

        return chain

    return build, paddle.to_tensor(_X)


def _tinygrad() -> tuple[Builder, Any]:
    Tensor = pytest.importorskip("tinygrad").Tensor

    def build(edit: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def chain(v: Any) -> Any:
            a = v + 1.0
            return edit((a * a - 2.0).relu()) + a

        return chain

    return build, Tensor(_X)


_BACKENDS = {"jax": _jax, "tf": _tf, "mlx": _mlx, "paddle": _paddle, "tinygrad": _tinygrad}

_MARKS = {
    "jax": pytest.mark.backend_jax,
    "tf": pytest.mark.tf_backend,
    "mlx": pytest.mark.backend_mlx,
    "paddle": pytest.mark.backend_paddle,
    "tinygrad": pytest.mark.backend_tinygrad,
}

pytestmark = pytest.mark.parametrize(
    "backend", [pytest.param(name, marks=_MARKS[name]) for name in sorted(_BACKENDS)]
)


def _setup(backend: str) -> tuple[Builder, Any, Any]:
    build, value = _BACKENDS[backend]()
    if not has_adapter(backend):
        pytest.skip(f"the {backend} replay adapter has not landed on this branch")
    from torchlens.backends._replay_adapters import get_adapter

    return build, value, get_adapter(backend)


def _identity(value: Any) -> Any:
    return value


def _capture(backend: str, build: Builder, value: Any, edit: Callable[[Any], Any]) -> Any:
    return tl.trace(build(edit), value, backend=backend)


def _rectifier(trace: Any) -> Any:
    """The rectifier's output: the later-made parent of the final add (the other is ``a``).

    Found by graph position, not by name, because backends record the
    rectifier under different primitives (``relu``, ``max``, ``maximum``, ...).
    """

    ops = list(trace._layer_list)
    order = {op.label: n for n, op in enumerate(ops)}
    final = ([op for op in ops if getattr(op, "is_output", False)] or ops)[-1]
    while getattr(final, "is_output", False) and len(tuple(final.parents)) == 1:
        final = trace._layer_dict_all_keys[tuple(final.parents)[0]]
    by_key = trace._layer_dict_all_keys
    parents = [
        by_key[label] if label in by_key else by_key[f"{label}:1"] for label in final.parents
    ]
    return max(parents, key=lambda op: order[op.label])


def _output(trace: Any) -> Any:
    outputs = [op for op in trace._layer_list if getattr(op, "is_output", False)]
    return (outputs or list(trace._layer_list))[-1].out


def _host(adapter: Any, value: Any) -> np.ndarray:
    return np.asarray(adapter.to_host(value))


def _double(out: Any, *, ctx: Any) -> Any:
    return out * 2.0


def test_noop_edit_reproduces_every_payload(backend: str) -> None:
    build, value, adapter = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    before = {op.label: _host(adapter, op.out) for op in clean._layer_list}
    edited = clean.fork().do(_rectifier(clean).label, lambda out, *, ctx: out)
    for op in edited._layer_list:
        assert np.array_equal(_host(adapter, op.out), before[op.label]), op.label


def test_edited_fork_equals_capture_of_edited_code(backend: str) -> None:
    build, value, adapter = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    edited = clean.fork().do(_rectifier(clean).label, _double)
    reference = _capture(backend, build, value, lambda r: r * 2.0)
    assert trace_backend(edited) == backend
    assert np.array_equal(_host(adapter, _output(edited)), _host(adapter, _output(reference)))
    assert not np.array_equal(_host(adapter, _output(edited)), _host(adapter, _output(clean)))


def test_zero_ablate_equals_capture_of_zeroed_code(backend: str) -> None:
    build, value, adapter = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    edited = clean.fork().do(_rectifier(clean).label, tl.zero_ablate())
    reference = _capture(backend, build, value, lambda r: r * 0.0)
    assert np.array_equal(_host(adapter, _output(edited)), _host(adapter, _output(reference)))


def test_edited_fork_passes_the_replay_check(backend: str) -> None:
    build, value, _ = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    edited = clean.fork().do(_rectifier(clean).label, _double)
    assert verify_backend_replay(edited) == []
    assert edited.is_pristine is False


def test_corrupted_downstream_payload_fails_the_replay_check(backend: str) -> None:
    build, value, adapter = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    edited = clean.fork().do(_rectifier(clean).label, _double)
    site = _rectifier(edited)
    child = edited._layer_dict_all_keys[next(iter(site.children))]
    child._internal_set("out", child.out + 1.0)
    assert child.label in verify_backend_replay(edited)


def test_donor_patch_from_a_trace_of_the_same_backend(backend: str) -> None:
    build, value, adapter = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    donor = _capture(backend, build, value * 2.0, _identity)
    label = _rectifier(clean).label
    donor_value = _rectifier(donor).out
    edited = clean.fork().do(label, lambda out, *, ctx: donor_value)
    site = edited._layer_dict_all_keys[label]
    assert np.array_equal(_host(adapter, site.out), _host(adapter, donor_value))
    assert verify_backend_replay(edited) == []


def test_donor_from_another_framework_is_refused(backend: str) -> None:
    build, value, adapter = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    site = _rectifier(clean)
    import torch

    foreign = torch.zeros(_host(adapter, site.out).shape)
    with pytest.raises(HookValueError) as caught:
        clean.fork().do(site.label, lambda out, *, ctx: foreign)
    assert caught.value.fields["code"] == "intervention_replacement_invalid"


def test_live_intervene_matches_where_supported(backend: str) -> None:
    if backend != "paddle":
        pytest.skip(f"{backend} has no live intervene= to compare against")
    build, value, adapter = _setup(backend)
    clean = _capture(backend, build, value, _identity)
    func = str(_rectifier(clean).func_name)
    live = tl.trace(
        build(_identity), value, backend=backend, intervene=tl.when(tl.func(func), tl.zero_ablate())
    )
    edited = clean.fork().do(tl.func(func), tl.zero_ablate())
    assert np.array_equal(_host(adapter, _output(edited)), _host(adapter, _output(live)))

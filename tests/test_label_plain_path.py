"""The plain-tensor label path and the cheaper per-call wrapper guards.

A plain tensor (exact ``torch.Tensor`` with no metadata or TorchLens ``TensorMeta``) is
stamped and read through a shorter path in ``_utils/_tensor_label.py``. Each check that
path leaves out has a reason it cannot fail for such a tensor; every test here pins that
the short path gives exactly what the general path gives, or that the case it does not
cover still reaches the general path. The wrapper guards made cheaper alongside it
(the tracing-pass argument scan, the cached functorch probe, the autocast snapshot and
the payload-window slot) are pinned the same way.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from torchlens import _state
from torchlens._utils import _tensor_label as tl_mod
from torchlens._utils._torch_compat import (
    autocast_get_dtype,
    autocast_is_enabled,
    dynamo_is_compiling,
)
from torchlens._utils.rng import autocast as autocast_mod
from torchlens.backends.torch import _tracing_pass, _wrapper_session, wrappers
from torchlens.backends.torch._completeness_origins import internal_scalar_read


def _new_session() -> Any:
    """Return a fresh label session object, not yet installed."""
    return tl_mod._LabelSession(next(tl_mod._LABEL_SESSION_COUNTER))


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fresh label session for one test; monkeypatch restores the previous one."""
    installed = _new_session()
    monkeypatch.setattr(tl_mod, "_ACTIVE_LABEL_SESSION", installed)
    return installed


def _force_general(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every stamp and read down the general path."""
    monkeypatch.setattr(tl_mod, "_plain_reads_allowed", lambda: False)


def _cdata(storage: Any) -> int | None:
    """Return a storage's ``_cdata`` (its StorageImpl identity), or ``None``."""
    return None if storage is None else int(storage._cdata)


def _alias_index(session: Any) -> dict[int, tuple[str, frozenset[int]]]:
    """Return the session's storage-alias index as bucket kind and member ids per key."""
    index: dict[int, tuple[str, frozenset[int]]] = {}
    for key, bucket in session.by_storage_ptr.items():
        if isinstance(bucket, weakref.ref):
            members = frozenset({id(bucket())})
        else:
            members = frozenset(id(item) for item in bucket.keys())
        index[key] = (type(bucket).__name__, members)
    return index


def _scenario(kind: str) -> list[torch.Tensor]:
    """Return the tensors one stamp scenario labels, in order."""
    if kind == "single":
        return [torch.randn(4)]
    if kind == "views":
        base = torch.randn(4, 2)
        return [base, base.view(-1), base[1:]]
    if kind == "relabel":
        same = torch.randn(3)
        return [same, same]
    if kind == "meta":
        meta = torch.empty(3, device="meta")
        return [meta, meta.view(3, 1)]
    if kind == "empty":
        return [torch.empty(0), torch.empty(0)]
    return [torch.randn(2), torch.randn(2)]


def _stamp_and_record(
    monkeypatch: pytest.MonkeyPatch, tensors: list[torch.Tensor]
) -> tuple[list[tuple[Any, ...]], dict[int, tuple[str, frozenset[int]]], frozenset[int]]:
    """Stamp ``tensors`` in a fresh session and return the fields, alias index and inventory."""
    stamped = _new_session()
    with monkeypatch.context() as patch:
        patch.setattr(tl_mod, "_ACTIVE_LABEL_SESSION", stamped)
        for tensor in tensors:
            tl_mod.clear_meta(tensor)
        for position, tensor in enumerate(tensors):
            tl_mod.set_tensor_label(tensor, f"op_{position}")
        fields = [
            (
                tensor._tl.label_raw,
                tensor._tl.label_session == stamped.token,
                _cdata(tensor._tl.label_storage),
            )
            for tensor in tensors
        ]
        inventory = frozenset(id(item) for item in stamped.stamped.keys())
    return fields, _alias_index(stamped), inventory


@pytest.mark.parametrize("kind", ["single", "views", "relabel", "meta", "empty", "distinct"])
def test_plain_stamp_writes_what_the_general_stamp_writes(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Fields, alias index (single and upgraded buckets, meta keys) and inventory match."""
    tensors = _scenario(kind)
    plain_calls: list[int] = []
    real_plain = tl_mod._stamp_plain_tensor

    def counting_plain(*args: Any) -> None:
        plain_calls.append(1)
        real_plain(*args)

    with monkeypatch.context() as patch:
        patch.setattr(tl_mod, "_stamp_plain_tensor", counting_plain)
        plain = _stamp_and_record(monkeypatch, tensors)
    assert len(plain_calls) == len(tensors)
    with monkeypatch.context() as patch:
        _force_general(patch)
        general = _stamp_and_record(monkeypatch, tensors)
    assert plain == general


def test_plain_stamp_outside_a_session_is_unanchored(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no session the plain stamp pins storage and leaves the label unanchored."""
    monkeypatch.setattr(tl_mod, "_ACTIVE_LABEL_SESSION", None)
    tensor = torch.randn(3)
    tl_mod.set_tensor_label(tensor, "op_0")
    assert tensor._tl.label_session is None
    assert _cdata(tensor._tl.label_storage) == _cdata(tensor.untyped_storage())
    assert tl_mod.get_tensor_label(tensor) == "op_0"


def _rebind_data(tensor: torch.Tensor) -> None:
    tensor.data = torch.randn(3)


def _set_storage(tensor: torch.Tensor) -> None:
    tensor.set_(torch.randn(3).untyped_storage())


def _grow(tensor: torch.Tensor) -> None:
    tensor.resize_(4096)


def _add_in_place(tensor: torch.Tensor) -> None:
    tensor.add_(1.0)


_READ_CASES: list[tuple[str, Callable[[torch.Tensor], None], bool]] = [
    ("untouched", lambda tensor: None, True),
    ("in_place_write", _add_in_place, True),
    ("resize_in_same_storage", _grow, True),
    ("data_rebind", _rebind_data, False),
    ("set_storage", _set_storage, False),
]


@pytest.mark.parametrize(("name", "mutate", "kept"), _READ_CASES, ids=[c[0] for c in _READ_CASES])
def test_plain_read_matches_the_general_read(
    monkeypatch: pytest.MonkeyPatch,
    session: Any,
    name: str,
    mutate: Callable[[torch.Tensor], None],
    kept: bool,
) -> None:
    """A rebound storage still rejects the label; a write into the same storage keeps it."""
    tensor = torch.randn(3)
    tl_mod.set_tensor_label(tensor, "op_7")
    mutate(tensor)
    plain = tl_mod.get_tensor_label(tensor)
    general = tl_mod._general_tensor_label(tensor)
    with monkeypatch.context() as patch:
        _force_general(patch)
        forced = tl_mod.get_tensor_label(tensor)
    assert plain == general == forced == ("op_7" if kept else None), name


def test_plain_read_session_gates(monkeypatch: pytest.MonkeyPatch, session: Any) -> None:
    """Unlabelled, earlier-session and no-session reads give the general answers."""
    unlabelled = torch.randn(2)
    assert tl_mod.get_tensor_label(unlabelled) is None
    stale = torch.randn(2)
    tl_mod.set_tensor_label(stale, "op_1")
    stale._tl.label_session = session.token - 1
    assert tl_mod.get_tensor_label(stale) is None
    assert tl_mod._general_tensor_label(stale) is None
    monkeypatch.setattr(tl_mod, "_ACTIVE_LABEL_SESSION", None)
    assert tl_mod.get_tensor_label(stale) == "op_1"
    assert tl_mod._general_tensor_label(stale) == "op_1"


@pytest.mark.parametrize("foreign", [object(), tl_mod.ParamMeta()])
def test_foreign_metadata_still_raises(session: Any, foreign: Any) -> None:
    """A foreign or wrong-kind ``_tl`` leaves the plain path and raises as before."""
    tensor = torch.randn(2)
    tensor._tl = foreign
    with pytest.raises(tl_mod.TorchLensTLCollisionError):
        tl_mod.set_tensor_label(tensor, "op_0")
    with pytest.raises(tl_mod.TorchLensTLCollisionError):
        tl_mod.get_tensor_label(tensor)


class _Sub(torch.Tensor):
    """A trivial tensor subclass: never a plain tensor."""


@pytest.mark.parametrize(
    "arm",
    [
        ("_escape_detector_mode", "shadow"),
        ("_completeness_witness_mode", "shadow"),
        ("_active_fast_run_collector", object()),
    ],
    ids=["escape_detector", "completeness_witness", "fast_collector"],
)
def test_observers_and_fast_collector_keep_the_general_stamp(
    monkeypatch: pytest.MonkeyPatch, session: Any, arm: tuple[str, Any]
) -> None:
    """An armed observer or a live fast collector routes the stamp to the general path."""

    def refuse(*args: Any) -> None:
        raise AssertionError("plain stamp taken")

    tensor = torch.randn(2)
    monkeypatch.setattr(tl_mod, "_stamp_plain_tensor", refuse)
    with monkeypatch.context() as patch:
        patch.setattr(_state, arm[0], arm[1])
        assert not tl_mod._plain_reads_allowed()
        tl_mod.set_tensor_label(tensor, "op_0")
    assert tensor._tl.label_session == session.token


def test_subclass_keeps_the_general_stamp(monkeypatch: pytest.MonkeyPatch, session: Any) -> None:
    """A tensor subclass never takes the plain stamp."""

    def refuse(*args: Any) -> None:
        raise AssertionError("plain stamp taken")

    monkeypatch.setattr(tl_mod, "_stamp_plain_tensor", refuse)
    sub = torch.randn(2).as_subclass(_Sub)
    tl_mod.set_tensor_label(sub, "op_0")
    assert tl_mod.get_tensor_label(sub) == "op_0"


def test_async_collective_tensor_is_a_strict_subclass_and_still_delegates(session: Any) -> None:
    """The delegation the plain read skips applies only to a strict tensor subclass."""
    pytest.importorskip("torch.distributed")
    import torch.distributed._functional_collectives as funcol

    act_cls = funcol.AsyncCollectiveTensor
    assert act_cls is not torch.Tensor and issubclass(act_cls, torch.Tensor)
    inner = torch.randn(3)
    tl_mod.set_tensor_label(inner, "op_3")
    wrapper = act_cls(inner)
    assert tl_mod.get_tensor_label(wrapper) == "op_3"


def _shim_autocast_states() -> dict[str, dict[str, Any]]:
    """Return the per-device autocast state through the version-neutral shims."""
    states: dict[str, dict[str, Any]] = {}
    for device in ("cpu", "cuda"):
        try:
            states[device] = {
                "enabled": autocast_is_enabled(device),
                "dtype": autocast_get_dtype(device),
            }
        except (RuntimeError, TypeError):
            pass
    return states


def _device_states(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key in ("cpu", "cuda")}


@pytest.mark.parametrize("legacy", [False, True], ids=["direct", "legacy_shims"])
def test_autocast_snapshot_equals_the_shim_values(
    monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    """Outside and inside an autocast region, each snapshot is a fresh dict of shim values."""
    if legacy:
        monkeypatch.setattr(autocast_mod, "_IS_ENABLED", None)
    first = autocast_mod.log_current_autocast_state()
    second = autocast_mod.log_current_autocast_state()
    assert _device_states(first) == _shim_autocast_states()
    assert first == second and first is not second and first["cpu"] is not second["cpu"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = autocast_mod.log_current_autocast_state()
        assert _device_states(inside) == _shim_autocast_states()
    assert inside["cpu"]["enabled"] is True


def test_functorch_probe_is_cached_and_still_sees_vmap() -> None:
    """The transform guard resolves its probe once and still reports a vmap region."""
    assert _tracing_pass.inside_functorch_transform() is False
    seen: list[bool] = []

    def body(row: torch.Tensor) -> torch.Tensor:
        seen.append(_tracing_pass.inside_functorch_transform())
        return row * 2.0

    torch.vmap(body)(torch.randn(3, 2))
    if _tracing_pass._FUNCTORCH_PROBE:
        assert seen and all(seen)
    assert wrappers._is_inside_functorch_transform is _tracing_pass.inside_functorch_transform
    assert wrappers._is_inside_dynamo_compilation is dynamo_is_compiling


class _PassThroughMode(torch.utils._python_dispatch.TorchDispatchMode):
    """A dispatch mode that is not a fake-tensor mode."""

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        return func(*args, **(kwargs or {}))


def test_tracing_guard_still_sees_fake_tensors_and_fake_mode() -> None:
    """Plain values are skipped by class; fake tensors and an active FakeTensorMode are not."""
    fake_module = pytest.importorskip("torch._subclasses.fake_tensor")
    plain = torch.randn(2)
    with fake_module.FakeTensorMode() as mode:
        fake = mode.from_tensor(plain)
        assert _tracing_pass.inside_tracing_tensor_pass((plain, 1, 2.0, None), {})
    guard = _tracing_pass.inside_tracing_tensor_pass
    assert not guard((plain, 3, "x", torch.float32, torch.device("cpu")), {})
    assert not guard((plain,), {"alpha": 1.0})
    assert guard((plain, fake), {})
    assert guard((plain,), {"other": fake})
    assert guard((plain, [plain, fake]), {})
    with _PassThroughMode():
        assert not guard((plain,), {})


class _Owner(nn.Module):
    """A model whose one parameter gives a distinct payload-window pointer set."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2))


class _TraceStub:
    """The trace attributes the payload-window state reads."""

    is_replayable = False
    _options_backward_ready = False
    _options_save_grads = None
    _options_activation_transform = None
    _options_mode = "exhaustive"
    _options_save_mode = "copy"
    _out_writer = None

    def __init__(self, model: nn.Module) -> None:
        self._source_model_ref = weakref.ref(model)


def test_payload_window_slot_returns_each_traces_own_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternating traces each get their cached state; a dead trace's slot is never reused."""
    monkeypatch.setattr(_wrapper_session, "_COW_ENABLED", True)
    model_a, model_b, model_c = _Owner(), _Owner(), _Owner()
    trace_a, trace_b = _TraceStub(model_a), _TraceStub(model_b)
    first_a = _wrapper_session._cow_payload_state_ptrs(trace_a, internal_scalar_read)
    first_b = _wrapper_session._cow_payload_state_ptrs(trace_b, internal_scalar_read)
    for _ in range(2):
        assert _wrapper_session._cow_payload_state_ptrs(trace_a, internal_scalar_read) is first_a
        assert _wrapper_session._cow_payload_state_ptrs(trace_b, internal_scalar_read) is first_b
    assert first_a == _wrapper_session._COW_STATE_PTRS_CACHE[trace_a]
    assert first_b == _wrapper_session._COW_STATE_PTRS_CACHE[trace_b]
    _wrapper_session._cow_payload_state_ptrs(trace_a, internal_scalar_read)
    del trace_a
    gc.collect()
    trace_c = _TraceStub(model_c)
    fresh = _wrapper_session._cow_payload_state_ptrs(trace_c, internal_scalar_read)
    assert fresh == _wrapper_session._COW_STATE_PTRS_CACHE[trace_c]
    if first_a is not None:
        assert fresh != first_a

"""Round-49 whole-class immunizer: SafeBundleUnpickler is a STRICT SUBSET of torch's
``weights_only`` baseline (secA_1).

A default (trust=False) bundle-metadata unpickle must NOT be able to
    (a) CONSTRUCT a torch storage -- ``UntypedStorage(N)`` allocates N raw bytes at
        ``tl.load()`` time (a multi-GiB out-of-memory DoS) and is the sole source of an
        uninitialized-heap storage; nor
    (b) REBIND tensor storage -- reach ``Tensor.__setstate__`` / ``Tensor.set_`` via the
        pickle BUILD opcode (an uninitialized-heap tensor), which is INVISIBLE to
        ``find_class`` because BUILD acts on an object already on the stack.

Both vectors are closed by two layers in ``torchlens/_io/_safe_unpickle.py``:
  Layer 1: torch storage constructors are denied by IDENTITY at ``find_class``
           (``_is_torch_storage_type``), including the two real
           ``torch.storage.{Typed,Untyped}Storage`` classes that torch's OWN baseline
           (``_weights_only_unpickler._get_allowed_globals()``) admits.
  Layer 2: ``SafeBundleUnpickler`` is a pure-Python ``pickle._Unpickler`` whose REBUILT
           opcode ``dispatch`` gates BUILD / REDUCE / NEWOBJ / NEWOBJ_EX. (Overriding
           the methods on a C ``pickle.Unpickler`` -- or even on ``pickle._Unpickler``
           WITHOUT rebuilding ``dispatch`` -- is INERT; that trap is pinned below.)

The whole-class invariant is CONSTRUCTOR-SEMANTICS, not a string subset: NO ``(module,
name)`` the unpickler admits resolves to a CONSTRUCTABLE torch storage, and no BUILD can
rebind a tensor/storage. A string-subset-of-baseline assertion is proven too weak -- the
baseline set itself lists the two real storage classes -- so a future permissive drift
toward a constructable storage/tensor ctor makes this immunizer go RED.
"""

from __future__ import annotations

import io
import pickle
import struct
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._safe_unpickle import (
    _SAFE_EXPLICIT_GLOBALS,
    _SAFE_TORCHLENS_TYPES,
    _TORCH_STORAGE_BASE_CLASSES,
    _TORCH_STORAGE_CLASSES,
    SafeBundleUnpickler,
    _is_torch_storage_instance,
    _is_torch_storage_type,
)
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _reduce_pickle_int_arg(module: str, name: str, int_arg: int) -> bytes:
    """Hand-assemble ``GLOBAL(module,name) + REDUCE((int_arg,))`` -- the storage DoS gadget.

    The int is LONG1-encoded so a size well beyond 2**31 can be expressed (the multi-GiB
    allocation a storage ctor would perform if it were not denied FIRST).
    """

    body = (
        int_arg.to_bytes((int_arg.bit_length() + 8) // 8, "little", signed=True) if int_arg else b""
    )
    out = pickle.PROTO + bytes([2]) + pickle.GLOBAL + (module + "\n" + name + "\n").encode()
    out += pickle.MARK + pickle.LONG1 + bytes([len(body)]) + body
    return out + pickle.TUPLE + pickle.REDUCE + pickle.STOP


# --------------------------------------------------------------------------- #
# 1. Storage-family construction is denied (exhaustive over the storage surface).
# --------------------------------------------------------------------------- #


def _all_storage_classes() -> list[type]:
    classes = set(_TORCH_STORAGE_CLASSES)
    classes |= set(_TORCH_STORAGE_BASE_CLASSES)
    classes |= {torch.storage.TypedStorage, torch.UntypedStorage}
    return sorted(classes, key=lambda c: f"{c.__module__}.{c.__qualname__}")


@pytest.mark.smoke
def test_every_storage_class_is_identity_denied() -> None:
    """Every torch storage class is recognized by the identity helper (surface-complete)."""

    storage_classes = _all_storage_classes()
    assert len(storage_classes) >= 31, "storage-class surface unexpectedly small"
    for cls in storage_classes:
        assert _is_torch_storage_type(cls), f"{cls!r} not recognized as a storage type"


@pytest.mark.smoke
def test_storage_construction_denied_end_to_end() -> None:
    """A REDUCE pickle of each resolvable storage class RAISES (never allocates)."""

    checked = 0
    for cls in _all_storage_classes():
        module, name = cls.__module__, cls.__qualname__
        # Only exercise classes actually addressable via their (module, name) so the
        # deny path is the storage guard, not an unrelated resolution error.
        if getattr(__import__(module, fromlist=[name]), name, None) is not cls:
            continue
        payload = _reduce_pickle_int_arg(module, name, 8_000_000_000)
        with pytest.raises(pickle.UnpicklingError, match="storage"):
            SafeBundleUnpickler(io.BytesIO(payload)).load()
        checked += 1
    assert checked >= 2, "expected at least Typed/Untyped storage to be exercised"


@pytest.mark.smoke
def test_canonical_storage_ctors_denied_at_find_class() -> None:
    """The two real baseline storage classes are denied at ``find_class`` (Layer 1)."""

    for module, name in (
        ("torch.storage", "TypedStorage"),
        ("torch.storage", "UntypedStorage"),
        ("torch", "UntypedStorage"),
        ("torch", "FloatStorage"),
    ):
        with pytest.raises(pickle.UnpicklingError, match="torch storage constructor"):
            SafeBundleUnpickler(io.BytesIO(b"")).find_class(module, name)


# --------------------------------------------------------------------------- #
# 2. Layer-2 opcode belts fire even when a storage type reaches the stack by a
#    route OTHER than find_class (defense-in-depth, white-box).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_reduce_belt_denies_storage_ctor_bypassing_find_class() -> None:
    """``load_reduce`` refuses constructing a storage placed directly on the stack."""

    u = SafeBundleUnpickler(io.BytesIO(b""))
    u.stack = [torch.UntypedStorage, (16,)]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="via REDUCE"):
        u.load_reduce()


@pytest.mark.smoke
def test_newobj_belts_deny_storage_ctor_bypassing_find_class() -> None:
    """``load_newobj`` / ``load_newobj_ex`` refuse ``storage.__new__`` on the stack."""

    u = SafeBundleUnpickler(io.BytesIO(b""))
    u.stack = [torch.UntypedStorage, (16,)]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="via NEWOBJ"):
        u.load_newobj()
    u.stack = [torch.UntypedStorage, (16,), {}]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="via NEWOBJ_EX"):
        u.load_newobj_ex()


# --------------------------------------------------------------------------- #
# 3. BUILD cannot rebind tensor/storage state (the __setstate__/set_ vector).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_build_belt_denies_tensor_rebind_white_box() -> None:
    """``load_build`` refuses a BUILD whose target is a torch Tensor / Storage."""

    u = SafeBundleUnpickler(io.BytesIO(b""))
    # BUILD stack layout: inst = stack[-2], state = stack[-1].
    u.stack = [torch.zeros(2), {"malicious": "state"}]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="BUILD applied to a torch"):
        u.load_build()

    storage = torch.zeros(2).untyped_storage()
    assert _is_torch_storage_instance(storage)
    u.stack = [storage, {"x": 1}]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="BUILD applied to a torch"):
        u.load_build()


@pytest.mark.smoke
def test_build_rebind_denied_end_to_end_no_setstate() -> None:
    """A crafted ``_load_from_bytes -> Tensor -> BUILD`` load RAISES; ``__setstate__`` never runs."""

    saved = io.BytesIO()
    torch.save(torch.zeros(3), saved)
    tensor_bytes = saved.getvalue()

    buf = pickle.PROTO + bytes([2])
    buf += pickle.GLOBAL + b"torch.storage\n_load_from_bytes\n"
    buf += pickle.MARK
    buf += pickle.BINBYTES + struct.pack("<I", len(tensor_bytes)) + tensor_bytes
    buf += pickle.TUPLE + pickle.REDUCE  # -> a real Tensor on the stack
    buf += pickle.EMPTY_DICT + pickle.BUILD + pickle.STOP  # attacker rebind BUILD

    setstate_calls = 0
    original_setstate = torch.Tensor.__setstate__

    def _spy(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        nonlocal setstate_calls
        setstate_calls += 1
        return original_setstate(self, *args, **kwargs)  # type: ignore[arg-type]

    torch.Tensor.__setstate__ = _spy  # type: ignore[method-assign, assignment]
    try:
        with pytest.raises(pickle.UnpicklingError, match="BUILD applied to a torch"):
            SafeBundleUnpickler(io.BytesIO(buf)).load()
    finally:
        torch.Tensor.__setstate__ = original_setstate  # type: ignore[method-assign]
    assert setstate_calls == 0, "Tensor.__setstate__ ran despite the BUILD gate"


# --------------------------------------------------------------------------- #
# 4. Interposition non-vacuity: the rebuilt dispatch actually points at the
#    subclass overrides (guards the "method-override-alone-is-inert" trap).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_dispatch_table_repointed_to_subclass_overrides() -> None:
    """The subclass ``dispatch`` targets its OWN opcode handlers, not the base ones."""

    for opcode, method_name in (
        (pickle.BUILD[0], "load_build"),
        (pickle.REDUCE[0], "load_reduce"),
        (pickle.NEWOBJ[0], "load_newobj"),
        (pickle.NEWOBJ_EX[0], "load_newobj_ex"),
    ):
        subclass_handler = SafeBundleUnpickler.dispatch[opcode]
        assert subclass_handler is getattr(SafeBundleUnpickler, method_name)
        assert subclass_handler is not pickle._Unpickler.dispatch[opcode], (
            f"dispatch[{opcode!r}] still points at the inert base handler"
        )


def test_base_swap_is_pure_python_unpickler() -> None:
    """The base class is the pure-Python VM (a C base makes Layer-2 inert)."""

    assert issubclass(SafeBundleUnpickler, pickle._Unpickler)


# --------------------------------------------------------------------------- #
# 5. Constructor-semantics drift guard (the anti-drift immunizer). NOT a string
#    subset of the baseline (which itself lists the storage classes).
# --------------------------------------------------------------------------- #


def _resolve_or_none(module: str, name: str) -> object:
    try:
        return SafeBundleUnpickler(io.BytesIO(b"")).find_class(module, name)
    except Exception:  # noqa: BLE001 - a denied/unresolvable name resolves to nothing
        return None


@pytest.mark.smoke
def test_admit_set_constructs_no_storage_strict_subset_of_baseline() -> None:
    """No (module,name) the unpickler admits resolves to a CONSTRUCTABLE storage.

    Enumerated over BOTH torch's ``weights_only`` baseline global set (which DOES admit
    the real ``torch.storage.{Typed,Untyped}Storage`` classes) AND TorchLens's own
    explicit allowlists. The admitted-constructable-storage set must be EMPTY -- proving
    a STRICT SUBSET of the baseline by constructor semantics, and making any future
    re-admission of a storage ctor a RED test.
    """

    from torch import _weights_only_unpickler as wou

    baseline = wou._get_allowed_globals()
    baseline_storage = {k: v for k, v in baseline.items() if _is_torch_storage_type(v)}
    # Sanity: the baseline really does admit storages (else the subset claim is vacuous).
    assert baseline_storage, "torch baseline unexpectedly lists no constructable storage"

    admitted_storage: list[str] = []
    for key in baseline_storage:
        module, _, name = key.rpartition(".")
        resolved = _resolve_or_none(module, name)
        if _is_torch_storage_type(resolved):
            admitted_storage.append(key)

    # TorchLens's own admit allowlists must never carry a storage ctor either.
    for module, name in set(_SAFE_EXPLICIT_GLOBALS) | set(_SAFE_TORCHLENS_TYPES):
        resolved = _resolve_or_none(module, name)
        if _is_torch_storage_type(resolved):
            admitted_storage.append(f"{module}.{name}")

    assert admitted_storage == [], (
        "SafeBundleUnpickler admits a CONSTRUCTABLE storage ctor -- it is no longer a "
        f"strict subset of the weights_only baseline: {admitted_storage}"
    )


@pytest.mark.smoke
def test_intentional_inert_resolution_set_still_admitted() -> None:
    """Documented intentional inert resolutions still work (the delta is storage-only)."""

    # torch inert data TYPEs and reconstructors that legit metadata references. These
    # RESOLVE (the type object is an inert reference); only their CONSTRUCTION on a
    # REDUCE/NEWOBJ belt is denied (r63 -- see test_r63_construct_deny.py). Keeping
    # ``torch.Tensor`` / ``torch.FloatTensor`` resolvable here is exactly the
    # resolution-admitted half the construction-refused half must NOT mask.
    for module, name in (
        ("torch", "Size"),
        ("torch", "Tensor"),
        ("torch", "FloatTensor"),
        ("torch.nn.parameter", "Parameter"),
        ("torch._utils", "_rebuild_tensor_v2"),
    ):
        assert _resolve_or_none(module, name) is not None, f"{module}.{name} should resolve"
    # numpy reconstructors + a representative torchlens DATA type.
    assert _resolve_or_none("numpy", "ndarray") is not None
    assert _resolve_or_none("torchlens._io", "BlobRef") is not None


# --------------------------------------------------------------------------- #
# 6. Over-trigger pins: legitimate bundles still load AND run under both layers.
# --------------------------------------------------------------------------- #


class _StatefulLinear(nn.Module):
    """Runnable-eligible parameterized graph with a persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value)) * self.scale


class _ControlFlow(nn.Module):
    """Control-flow graph; a portable save embeds a predicate tensor + method refs."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.sum() > 0:
            return torch.relu(value) * 2
        return torch.sigmoid(value) - 1


@pytest.mark.smoke
def test_legit_bundles_round_trip_under_the_gate(tmp_path: Path) -> None:
    """Portable + all runnable levels still load AND run (no over-trigger)."""

    x = torch.randn(2, 4)
    keep_alive: list[nn.Module] = []

    portable = tmp_path / "portable"
    cf = _ControlFlow()
    keep_alive.append(cf)
    tl.trace(cf, x, layers_to_save="all").save(portable, level="portable")
    assert tl.load(portable) is not None

    for suffix, kwargs in (
        ("runnable", {}),
        ("runnable_weights", {"include_weights": True}),
        ("runnable_activations", {"include_activations": True}),
    ):
        model = _StatefulLinear()
        keep_alive.append(model)
        path = tmp_path / suffix
        tl.trace(model, x, save=tl.func("relu"), capture=_CAP).save(
            path, level="runnable", **kwargs
        )
        loaded = tl.load(path)
        assert loaded is not None, suffix
        result = loaded.run(inputs=x)
        assert tuple(result.output.shape) == (2, 4), suffix

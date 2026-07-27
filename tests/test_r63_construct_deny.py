"""Round-63 immunizer: bare CONSTRUCTION of torch tensor / numpy ndarray types is denied.

The r49 belt (``tests/test_r49_unpickler_weights_only_baseline.py``) closes torch STORAGE
construction, but a torch storage is not the only pickle-reachable attacker-sized
allocator. A hostile ``metadata.pkl`` can REDUCE/NEWOBJ/NEWOBJ_EX-construct

    * ``torch.Tensor(N)``            -> an N-element uninitialized-heap tensor,
    * ``torch.FloatTensor(N)``       -> ditto (a legacy ``tensortype`` class, NOT a
                                        ``torch.Tensor`` subclass, so the r49 storage
                                        check and a naive ``issubclass(_, torch.Tensor)``
                                        BOTH miss it),
    * ``numpy.ndarray((N,))``        -> an N-element uninitialized ndarray

at PLAIN ``tl.load()`` time (no trust, no ``.run()``) -- an attacker-sized allocation DoS
plus an uninitialized-heap read. The r49 belt refused only storage TYPES; these tensor /
ndarray types passed.

The fix (``torchlens/_io/_safe_unpickle.py``): a shared ``_is_alloc_constructor_type``
predicate (storage OR ``torch.Tensor`` subclass OR legacy ``tensortype`` class OR
``numpy.ndarray`` subclass) is applied on ALL THREE construction belts (``load_reduce`` /
``load_newobj`` / ``load_newobj_ex``), reusing the storage-deny refusal path (no new error
code). The type OBJECTS stay RESOLVABLE at ``find_class`` -- legit ``.tlspec`` metadata
references ``torch.Tensor`` / ``torch.FloatTensor`` / ``numpy.ndarray`` as inert dtype /
module_type values -- and the legit payload reconstructors (``_rebuild_tensor_v2`` /
numpy ``_frombuffer`` / ``_reconstruct``, all FUNCTIONS not alloc TYPES, plus the wrapped
``_safe_load_from_bytes``) are untouched. Zero-regression is PROVEN: a real
embedded-weights ``.tlspec`` performs many REDUCE/NEWOBJ ops and ZERO tensor/ndarray
constructions, so it round-trips (save -> load -> run) unchanged.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import _safe_unpickle
from torchlens._io._safe_unpickle import (
    _SAFE_EXPLICIT_GLOBALS,
    _SAFE_TORCHLENS_TYPES,
    SafeBundleUnpickler,
    _is_alloc_constructor_type,
    _is_torch_storage_type,
)
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# Hand-assembled PROTO2 construction gadgets (GLOBAL + belt opcode).
# --------------------------------------------------------------------------- #


def _int_arg(value: int) -> bytes:
    """A single LONG1-encoded int argument (a size big enough to be a real DoS)."""

    body = value.to_bytes((value.bit_length() + 8) // 8, "little", signed=True) if value else b""
    return pickle.LONG1 + bytes([len(body)]) + body


def _shape_tuple_arg(value: int) -> bytes:
    """A single ``(value,)`` shape tuple argument (numpy ndarray ctor form)."""

    return pickle.MARK + _int_arg(value) + pickle.TUPLE


def _global(module: str, name: str) -> bytes:
    return pickle.GLOBAL + (module + "\n" + name + "\n").encode()


def _reduce_gadget(module: str, name: str, arg: bytes) -> bytes:
    """``GLOBAL(module,name)`` then ``REDUCE`` over ``(arg,)`` -- bare construction."""

    return (
        pickle.PROTO
        + bytes([2])
        + _global(module, name)
        + pickle.MARK
        + arg
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


def _newobj_gadget(module: str, name: str, arg: bytes) -> bytes:
    """``GLOBAL(cls)`` + ``(arg,)`` + ``NEWOBJ`` -- ``cls.__new__(cls, arg)``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _global(module, name)
        + pickle.MARK
        + arg
        + pickle.TUPLE
        + pickle.NEWOBJ
        + pickle.STOP
    )


def _newobj_ex_gadget(module: str, name: str, arg: bytes) -> bytes:
    """``GLOBAL(cls)`` + ``(arg,)`` + ``{}`` + ``NEWOBJ_EX``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _global(module, name)
        + pickle.MARK
        + arg
        + pickle.TUPLE
        + pickle.EMPTY_DICT
        + pickle.NEWOBJ_EX
        + pickle.STOP
    )


_ALLOC_TARGETS = (
    ("torch", "Tensor", _int_arg(6_000_000_000)),
    ("torch", "FloatTensor", _int_arg(6_000_000_000)),
    ("numpy", "ndarray", _shape_tuple_arg(6_000_000_000)),
)


@pytest.mark.smoke
@pytest.mark.parametrize("module,name,arg", _ALLOC_TARGETS)
def test_reduce_construction_of_tensor_ndarray_refused(module: str, name: str, arg: bytes) -> None:
    """A PROTO2 GLOBAL+REDUCE of Tensor/FloatTensor/ndarray RAISES (never allocates)."""

    payload = _reduce_gadget(module, name, arg)
    with pytest.raises(pickle.UnpicklingError, match="allocation-constructor"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


@pytest.mark.smoke
@pytest.mark.parametrize("module,name,arg", _ALLOC_TARGETS)
def test_newobj_construction_of_tensor_ndarray_refused(module: str, name: str, arg: bytes) -> None:
    """A PROTO2 NEWOBJ of Tensor/FloatTensor/ndarray RAISES (never allocates)."""

    payload = _newobj_gadget(module, name, arg)
    with pytest.raises(pickle.UnpicklingError, match="allocation-constructor"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


@pytest.mark.smoke
@pytest.mark.parametrize("module,name,arg", _ALLOC_TARGETS)
def test_newobj_ex_construction_of_tensor_ndarray_refused(
    module: str, name: str, arg: bytes
) -> None:
    """A PROTO2 NEWOBJ_EX of Tensor/FloatTensor/ndarray RAISES (never allocates)."""

    payload = _newobj_ex_gadget(module, name, arg)
    with pytest.raises(pickle.UnpicklingError, match="allocation-constructor"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


@pytest.mark.smoke
def test_white_box_belts_refuse_tensor_and_ndarray() -> None:
    """The three belts refuse a Tensor/ndarray type placed directly on the stack."""

    for target in (torch.Tensor, torch.FloatTensor, np.ndarray):
        u = SafeBundleUnpickler(io.BytesIO(b""))
        u.stack = [target, (5,)]  # type: ignore[attr-defined]
        with pytest.raises(pickle.UnpicklingError, match="via REDUCE"):
            u.load_reduce()

        u = SafeBundleUnpickler(io.BytesIO(b""))
        u.stack = [target, (5,)]  # type: ignore[attr-defined]
        with pytest.raises(pickle.UnpicklingError, match="via NEWOBJ"):
            u.load_newobj()

        u = SafeBundleUnpickler(io.BytesIO(b""))
        u.stack = [target, (5,), {}]  # type: ignore[attr-defined]
        with pytest.raises(pickle.UnpicklingError, match="via NEWOBJ_EX"):
            u.load_newobj_ex()


# --------------------------------------------------------------------------- #
# Resolution (inert type reference) is STILL admitted -- construction != resolution.
# --------------------------------------------------------------------------- #


def _resolve_or_none(module: str, name: str) -> object:
    try:
        return SafeBundleUnpickler(io.BytesIO(b"")).find_class(module, name)
    except Exception:  # noqa: BLE001 - a denied/unresolvable name resolves to nothing
        return None


@pytest.mark.smoke
def test_tensor_ndarray_types_still_resolve_as_inert_references() -> None:
    """``find_class`` still hands back the tensor/ndarray/param TYPE objects (metadata needs them)."""

    assert _resolve_or_none("torch", "Tensor") is torch.Tensor
    assert _resolve_or_none("torch", "FloatTensor") is torch.FloatTensor
    assert _resolve_or_none("torch.nn.parameter", "Parameter") is torch.nn.Parameter
    assert _resolve_or_none("numpy", "ndarray") is np.ndarray


@pytest.mark.smoke
def test_legit_reconstructors_are_not_alloc_types_and_reduce_is_allowed() -> None:
    """The FUNCTION reconstructors resolve and are NOT flagged as alloc constructors."""

    rebuild = _resolve_or_none("torch._utils", "_rebuild_tensor_v2")
    assert callable(rebuild) and not isinstance(rebuild, type)
    assert not _is_alloc_constructor_type(rebuild)
    # numpy reconstructor helpers are functions, not alloc types.
    frombuffer = _resolve_or_none("numpy._core.numeric", "_frombuffer") or _resolve_or_none(
        "numpy.core.numeric", "_frombuffer"
    )
    assert frombuffer is not None and not _is_alloc_constructor_type(frombuffer)

    # A REDUCE whose func is the (function) reconstructor is NOT refused by the belt --
    # the alloc-guard fires on TYPES only. White-box: push func + args and run the belt;
    # the belt must never raise ITS alloc refusal (a downstream ctor error is fine).
    u = SafeBundleUnpickler(io.BytesIO(b""))
    u.stack = [rebuild, ()]  # type: ignore[attr-defined]
    try:
        u.load_reduce()
    except pickle.UnpicklingError as exc:
        assert "allocation-constructor" not in str(exc), "belt wrongly refused a function reduce"
    except Exception:  # noqa: BLE001 - base REDUCE ran the reconstructor: belt admitted, fine
        pass


# --------------------------------------------------------------------------- #
# Constructor-semantics drift guard: the admitted-CONSTRUCTABLE alloc set is EMPTY.
# --------------------------------------------------------------------------- #


def _reduce_belt_admits_construction(resolved: object) -> bool:
    """Return whether the REDUCE belt would let ``resolved`` be CONSTRUCTED (not refused)."""

    u = SafeBundleUnpickler(io.BytesIO(b""))
    u.stack = [resolved, ()]  # type: ignore[attr-defined]
    try:
        u.load_reduce()
    except pickle.UnpicklingError as exc:
        # Our alloc refusal is the only UnpicklingError the belt itself raises.
        return "allocation-constructor" not in str(exc)
    except Exception:  # noqa: BLE001 - base REDUCE ran (belt admitted) then ctor failed
        return True
    return True


@pytest.mark.smoke
def test_admitted_constructable_alloc_set_is_empty_over_baseline_and_allowlists() -> None:
    """No admitted (module,name) resolves to an alloc type the belt permits constructing.

    Enumerated over BOTH torch's ``weights_only`` baseline global set (which lists the
    real storage AND tensor classes by name) AND TorchLens's own explicit allowlists. For
    every resolved object that IS an allocation-constructor type (storage / tensor /
    ndarray), the REDUCE belt MUST refuse its construction -- so the admitted-CONSTRUCTABLE
    alloc set is EMPTY. This is the anti-drift invariant: it stays green when a type
    resolves inertly (``torch.Tensor`` / ``numpy.ndarray`` DO resolve) and goes RED only if
    a future change lets such a type be CONSTRUCTED on a belt.
    """

    from torch import _weights_only_unpickler as wou

    baseline = wou._get_allowed_globals()
    baseline_alloc = {k: v for k, v in baseline.items() if _is_alloc_constructor_type(v)}
    # Sanity: the baseline really does list constructable storage + tensor classes (else
    # the subset claim is vacuous).
    assert baseline_alloc, "torch baseline unexpectedly lists no constructable alloc type"

    admitted_constructable: list[str] = []

    def _check(module: str, name: str) -> None:
        resolved = _resolve_or_none(module, name)
        if resolved is None:
            return
        if _is_alloc_constructor_type(resolved) and _reduce_belt_admits_construction(resolved):
            admitted_constructable.append(f"{module}.{name}")

    for key in baseline_alloc:
        module, _, name = key.rpartition(".")
        _check(module, name)
    for module, name in set(_SAFE_EXPLICIT_GLOBALS) | set(_SAFE_TORCHLENS_TYPES):
        _check(module, name)

    assert admitted_constructable == [], (
        "SafeBundleUnpickler CONSTRUCTS an allocation type (storage/tensor/ndarray) from "
        "an admitted global -- the tensor/ndarray-construct regap is re-opened: "
        f"{admitted_constructable}"
    )


@pytest.mark.smoke
def test_baseline_lists_tensor_classes_that_resolve_but_do_not_construct() -> None:
    """Non-vacuity: the baseline lists a TENSOR class; it resolves inertly yet cannot construct."""

    from torch import _weights_only_unpickler as wou

    baseline = wou._get_allowed_globals()
    tensor_keys = [
        k
        for k, v in baseline.items()
        if _is_alloc_constructor_type(v) and not _is_torch_storage_type(v)
    ]
    assert tensor_keys, "torch baseline unexpectedly lists no constructable tensor class"
    for key in tensor_keys:
        module, _, name = key.rpartition(".")
        resolved = _resolve_or_none(module, name)
        # It RESOLVES (inert type reference)...
        assert resolved is not None, f"{key} should resolve as an inert type reference"
        # ...but the belt REFUSES constructing it.
        assert not _reduce_belt_admits_construction(resolved), f"{key} construction not refused"


# --------------------------------------------------------------------------- #
# Zero-regression: a real embedded-weights .tlspec still saves / loads / runs, and the
# legit load path performs ZERO alloc-type constructions.
# --------------------------------------------------------------------------- #


class _StatefulLinear(nn.Module):
    """Runnable-eligible parameterized graph with a persistent + non-persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("tmp", torch.zeros(4), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value)) * self.scale + self.tmp


@pytest.mark.smoke
def test_embedded_weights_tlspec_round_trips_with_zero_alloc_constructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Embedded-weights save/load/run is green AND flags ZERO alloc constructions on load."""

    x = torch.randn(2, 4)
    model = _StatefulLinear()
    path = tmp_path / "runnable_weights"
    # The non-persistent buffer triggers the required disclosure warning on save (declared
    # state that ships even without include_weights) -- expected, and it exercises the
    # embedded-weights + non-persistent-buffer path the r63 change must not regress.
    with pytest.warns(UserWarning, match="NON-persistent buffers"):
        tl.trace(model, x, save=tl.func("relu"), capture=_CAP).save(
            path, level="runnable", include_weights=True
        )

    # Instrument the SHARED predicate: record any object the belts flag as an alloc
    # constructor during the legit load. The zero-regression claim is that a legit
    # embedded-weights bundle performs MANY REDUCE/NEWOBJ ops but ZERO tensor/ndarray/
    # storage constructions, so nothing is ever flagged.
    flagged: list[str] = []
    real_predicate = _safe_unpickle._is_alloc_constructor_type

    def _spy(obj: object) -> bool:
        result = real_predicate(obj)
        if result:
            flagged.append(
                f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', obj)!r}"
            )
        return result

    monkeypatch.setattr(_safe_unpickle, "_is_alloc_constructor_type", _spy)

    loaded = tl.load(path)
    assert loaded is not None
    assert flagged == [], f"legit embedded-weights load flagged an alloc construction: {flagged}"

    result = loaded.run(inputs=x)
    assert tuple(result.output.shape) == (2, 4)

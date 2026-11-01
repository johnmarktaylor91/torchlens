"""The C-backed metadata unpickler decides exactly as the per-opcode interpreter.

``SafeBundleUnpickler`` pre-scans the opcode stream, then loads with the C
``pickle.Unpickler`` behind call-time guards; ``_PurePythonBundleUnpickler`` is the
same policy enforced per opcode on the pure-Python machine. This file is the
equivalence proof: a corpus of honest streams, hostile streams from every refusal
class and generated variants of both is run through both machines and must give the
same accept / refuse decision, the same exception type and the same refusal class,
plus the mechanism pins (fast path taken for honest streams, guards never leak into the
loaded object graph, the two deliberate tightenings).
"""

from __future__ import annotations

import collections
import io
import pickle
import pickletools
import random
import struct
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._canonical_pickle import dump_canonical_metadata
from torchlens._io._pickle_prescan import (
    GuardedCallable,
    OpcodeCeilingExceeded,
    StreamNeedsInterpreter,
    scan_opcodes,
)
from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
    _PurePythonBundleUnpickler,
)
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.options import CaptureOptions

# --------------------------------------------------------------------------- #
# Stream builders (raw opcodes, version-robust)
# --------------------------------------------------------------------------- #


def _global(module: str, name: str) -> bytes:
    return pickle.GLOBAL + module.encode() + b"\n" + name.encode() + b"\n"


def _int(value: int) -> bytes:
    return pickle.BININT + struct.pack("<i", value)


def _bytes(data: bytes) -> bytes:
    return pickle.BINBYTES + struct.pack("<I", len(data)) + data


def _reduce(module: str, name: str, *args: bytes) -> bytes:
    return (
        pickle.PROTO
        + b"\x02"
        + _global(module, name)
        + pickle.MARK
        + b"".join(args)
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


def _newobj(module: str, name: str, *args: bytes) -> bytes:
    return (
        pickle.PROTO
        + b"\x02"
        + _global(module, name)
        + pickle.MARK
        + b"".join(args)
        + pickle.TUPLE
        + pickle.NEWOBJ
        + pickle.STOP
    )


def _torch_saved(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def _embedded_tensor_then_build() -> bytes:
    """``_load_from_bytes -> Tensor -> BUILD``: the storage-rebind vector."""

    blob = _torch_saved(torch.zeros(3))
    return (
        pickle.PROTO
        + b"\x02"
        + _global("torch.storage", "_load_from_bytes")
        + pickle.MARK
        + _bytes(blob)
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.EMPTY_DICT
        + pickle.BUILD
        + pickle.STOP
    )


def _inline_tensor_then_build() -> bytes:
    """The writer's own tensor reduction followed by BUILD: the storage-rebind vector today."""

    buffer = io.BytesIO()
    dump_canonical_metadata(torch.zeros(3), buffer)
    data = buffer.getvalue()
    assert data.endswith(pickle.STOP)
    body = bytearray(data[:-1])
    # STOP sat inside the last frame: shrink that frame's declaration by the byte removed.
    frames = [(pos, arg) for op, arg, pos in pickletools.genops(data) if op.name == "FRAME"]
    if frames and frames[-1][0] + 9 + frames[-1][1] == len(data):
        pos, arg = frames[-1]
        body[pos + 1 : pos + 9] = struct.pack("<Q", arg - 1)
    return bytes(body) + pickle.EMPTY_DICT + pickle.BUILD + pickle.STOP


def _getattr_then_call(name: str, *args: bytes) -> bytes:
    """``getattr(_VariableFunctionsClass, name)(*args)``: the mediated-factory vector."""

    return (
        pickle.PROTO
        + b"\x02"
        + _global("builtins", "getattr")
        + pickle.MARK
        + _global("torch._C", "_VariableFunctionsClass")
        + pickle.SHORT_BINUNICODE
        + bytes([len(name)])
        + name.encode()
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.MARK
        + b"".join(args)
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


def _getattr_value_and_call() -> bytes:
    """A ``getattr`` result stored in the root list AND called: the interpreter must run it."""

    return (
        pickle.PROTO
        + b"\x02"
        + pickle.EMPTY_LIST
        + _global("builtins", "getattr")
        + pickle.MARK
        + _global("torch._C", "TensorBase")
        + pickle.SHORT_BINUNICODE
        + b"\x03sum"
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.BINPUT
        + b"\x01"
        + pickle.APPEND
        + pickle.BINGET
        + b"\x01"
        + pickle.MARK
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.APPEND
        + pickle.STOP
    )


def _honest_values() -> dict[str, Any]:
    """Values the TorchLens writer legitimately puts in metadata."""

    return {
        "scalars": {"a": 1, "b": 2.5, "c": "x", "d": None, "e": True},
        "bytes_payload": {"f": b"\x00\x01"},
        "containers": [1, (2, 3), {4: [5, 6]}, {7, 8}, frozenset({9})],
        "ordered": collections.OrderedDict([("k", 1), ("j", [2, 3])]),
        "counter": collections.Counter("abracadabra"),
        "torch_values": {"dtype": torch.float32, "size": torch.Size((2, 3)), "device": "cpu"},
        "numpy_scalar": np.float64(1.5),
        "module_identity": nn.Identity(),
        "module_type": nn.Linear,
        "tensor_method": getattr(torch._C.TensorBase, "sum"),  # noqa: B009 -- the pickled shape
        "nested": {"layers": [{"name": f"l{i}", "shape": (i, i + 1)} for i in range(20)]},
        "shared": (lambda s: [s, s, {"again": s}])(["memo", "shared"]),
    }


#: Values whose protocol-2 pickle routes through a stdlib REDUCE (``_codecs.encode`` for
#: bytes): the policy refuses that global on both machines, so no honest writer emits it.
_PROTOCOL_3_PLUS_LABELS = frozenset({"bytes_payload", "numpy_scalar"})


def _honest_streams() -> dict[str, bytes]:
    """Honest streams: stock pickles per protocol plus the metadata writer's own output."""

    streams: dict[str, bytes] = {}
    for label, value in _honest_values().items():
        protocols = (4, 5) if label in _PROTOCOL_3_PLUS_LABELS else (2, 4, 5)
        for protocol in protocols:
            data = pickle.dumps(value, protocol=protocol)
            if protocol == 2 and b"__builtin__\n" in data:
                # Protocol 2 names builtins by their Python 2 module, which the policy
                # does not admit; no TorchLens writer emits protocol 2.
                continue
            streams[f"{label}@p{protocol}"] = data
    # A tensor pickles through the writer's inline reduction; the stock torch reduction
    # (``torch.storage._load_from_bytes``) is refused by name on both machines.
    buffer = io.BytesIO()
    dump_canonical_metadata(
        {"tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3), "note": "x"}, buffer
    )
    streams["tensor@canonical"] = buffer.getvalue()
    return streams


def _hostile_streams() -> dict[str, bytes]:
    """One stream per refusal class the interpreter enforces."""

    blob = _torch_saved(torch.zeros(3))
    reconstruct = next(
        module
        for module in ("numpy._core.multiarray", "numpy.core.multiarray")
        if __import__(module, fromlist=["_reconstruct"])
    )
    return {
        # allowlist refusals at find_class
        "os_system": _reduce("os", "system", pickle.SHORT_BINUNICODE + b"\x02id"),
        "posix_system": _reduce("posix", "system", pickle.SHORT_BINUNICODE + b"\x02id"),
        "subprocess_popen": _reduce("subprocess", "Popen", pickle.EMPTY_LIST),
        "builtins_eval": _reduce("builtins", "eval", pickle.SHORT_BINUNICODE + b"\x031+1"),
        "torch_load": _reduce("torch.serialization", "load", pickle.SHORT_BINUNICODE + b"\x01x"),
        "storage_ctor": _reduce("torch", "UntypedStorage", _int(16)),
        "file_reader": _reduce("torch._C", "PyTorchFileReader", pickle.SHORT_BINUNICODE + b"\x01x"),
        "dotted_rebuild": _reduce("torch._utils", "_rebuild_tensor_v2.__globals__.get"),
        "lazy_import_ref": _newobj("torchlens.intervention.save", "LazyImportRef"),
        "pre3_module": _global("torchlens.data_classes.trace", "Trace") + pickle.STOP,
        # allocation belt: constructor types
        "tensor_reduce": _reduce("torch", "Tensor", _int(5)),
        "float_tensor_reduce": _reduce("torch", "FloatTensor", _int(5)),
        "ndarray_reduce": _reduce("numpy", "ndarray", _int(5)),
        "tensor_newobj": _newobj("torch", "Tensor", _int(5)),
        "ndarray_newobj": _newobj("numpy", "ndarray", _int(5)),
        "tensor_newobj_ex": (
            pickle.PROTO
            + b"\x04"
            + _global("torch", "Tensor")
            + pickle.MARK
            + _int(5)
            + pickle.TUPLE
            + pickle.EMPTY_DICT
            + pickle.NEWOBJ_EX
            + pickle.STOP
        ),
        "tensor_inst": b"(I5\nitorch\nFloatTensor\n.",
        "ndarray_obj": b"(cnumpy\nndarray\nI5\no.",
        # allocation belt: mediators
        "reconstruct_ndarray": _reduce(
            reconstruct,
            "_reconstruct",
            _global("numpy", "ndarray"),
            pickle.MARK + _int(1000) + pickle.TUPLE,
            pickle.SHORT_BINBYTES + b"\x01b",
        ),
        # the honest numpy pickling shape itself mediates ndarray.__new__; refused by design
        "ndarray_pickle_shape": pickle.dumps(
            np.arange(12, dtype=np.float64).reshape(3, 4), protocol=4
        ),
        "bytearray_sized": _reduce("builtins", "bytearray", _int(1000)),
        "bytes_sized": _reduce("builtins", "bytes", _int(1000)),
        "linear_sized": _reduce("torch.nn.modules.linear", "Linear", _int(16), _int(16)),
        "linear_newobj_sized": _newobj("torch.nn.modules.linear", "Linear", _int(16), _int(16)),
        "torch_empty_factory": _getattr_then_call("empty", _int(1000)),
        "torch_zeros_factory": _getattr_then_call("zeros", _int(1000)),
        "from_file_getattr": _getattr_then_call("from_file", pickle.SHORT_BINUNICODE + b"\x01x"),
        # BUILD on a tensor
        "embedded_tensor_build": _embedded_tensor_then_build(),
        "inline_tensor_build": _inline_tensor_then_build(),
        "rebuild_tensor_build": (
            pickle.PROTO
            + b"\x02"
            + _global("torch._utils", "_rebuild_tensor_v2")
            + pickle.MARK
            + _global("torch.storage", "_load_from_bytes")
            + pickle.MARK
            + _bytes(blob)
            + pickle.TUPLE
            + pickle.REDUCE
            + _int(0)
            + pickle.MARK
            + _int(3)
            + pickle.TUPLE
            + pickle.MARK
            + _int(1)
            + pickle.TUPLE
            + pickle.NEWFALSE
            + _reduce_body("collections", "OrderedDict")
            + pickle.TUPLE
            + pickle.REDUCE
            + pickle.EMPTY_DICT
            + pickle.BUILD
            + pickle.STOP
        ),
        # foreign callable executed on an untrusted load
        "deferred_called": _reduce("some_user_module", "recipe"),
        # persistent ids / extensions / buffers
        "persid": pickle.PROTO + b"\x02" + pickle.BINPERSID + pickle.STOP,
        "ext1": pickle.PROTO + b"\x02" + pickle.EXT1 + b"\x01" + pickle.STOP,
        "frame_overdeclared": pickle.PROTO
        + b"\x04"
        + pickle.FRAME
        + struct.pack("<Q", 2**40)
        + pickle.EMPTY_LIST
        + pickle.STOP,
        "memo_index_above_count": pickle.PROTO
        + b"\x02"
        + pickle.EMPTY_LIST
        + pickle.LONG_BINPUT
        + struct.pack("<I", 2**30)
        + pickle.STOP,
        "next_buffer": pickle.PROTO + b"\x05" + pickle.NEXT_BUFFER + pickle.STOP,
        # corruption facets
        "build_underflow": pickle.BUILD + pickle.STOP,
        "reduce_underflow": pickle.REDUCE + pickle.STOP,
        "binint_truncated": pickle.BININT + b"\x01\x02",
        "bad_utf8": pickle.SHORT_BINUNICODE + b"\x01\xff",
        "missing_memo": pickle.BINGET + b"\x00" + pickle.STOP,
        "empty": b"",
        "no_stop": pickle.EMPTY_LIST,
        "unknown_opcode": b"\xff" + pickle.STOP,
        "stack_global_non_str": pickle.PROTO
        + b"\x04"
        + _int(1)
        + _int(2)
        + pickle.STACK_GLOBAL
        + pickle.STOP,
    }


def _reduce_body(module: str, name: str) -> bytes:
    return _global(module, name) + pickle.MARK + pickle.TUPLE + pickle.REDUCE


def _variants(streams: dict[str, bytes], seed: int = 20261101) -> dict[str, bytes]:
    """Generated variants: truncations, byte flips and global substitutions."""

    rng = random.Random(seed)
    out: dict[str, bytes] = {}
    substitutions = (
        (b"collections\nOrderedDict", b"os\nsystem"),
        (b"torch._utils\n_rebuild_tensor_v2", b"torch\nTensor"),
        (b"builtins\ngetattr", b"builtins\neval"),
        (b"torch.storage\n_load_from_bytes", b"torch\nUntypedStorage"),
        (b"numpy\nndarray", b"torch\nFloatTensor"),
    )
    for label, data in streams.items():
        for cut in sorted({len(data) // 3, len(data) // 2, len(data) - 1}):
            if 0 < cut < len(data):
                out[f"{label}:cut{cut}"] = data[:cut]
        for flip in range(3):
            if len(data) > 2:
                pos = rng.randrange(1, len(data) - 1)
                mutated = bytearray(data)
                mutated[pos] ^= 1 << rng.randrange(8)
                out[f"{label}:flip{flip}"] = bytes(mutated)
        for old, new in substitutions:
            if old in data:
                target = new.split(b"\n")[1].decode()
                out[f"{label}:sub{target}"] = data.replace(old, new)
        out[f"{label}:trailing"] = data + b"junk"
    return out


# --------------------------------------------------------------------------- #
# The differential oracle
# --------------------------------------------------------------------------- #

_LoaderFactory = Callable[[bytes], Any]


def _normalize(value: Any, depth: int = 0) -> Any:
    """A comparable rendering of a loaded object graph (no identities, no addresses)."""

    if depth > 12:
        return "<deep>"
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape), value.detach().flatten().tolist())
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.flatten().tolist())
    if isinstance(value, np.generic):
        return ("npscalar", str(value.dtype), value.item())
    if isinstance(value, (str, bytes, bytearray, int, float, complex, bool, type(None))):
        return value
    if isinstance(value, dict):
        return (
            "dict",
            [(_normalize(k, depth + 1), _normalize(v, depth + 1)) for k, v in value.items()],
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_normalize(v, depth + 1) for v in value])
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, sorted(repr(_normalize(v, depth + 1)) for v in value))
    if isinstance(value, (torch.dtype, torch.Size, torch.device, slice)):
        return repr(value)
    if isinstance(value, type) or callable(value) and not hasattr(value, "__dict__"):
        return ("callable", type(value).__qualname__, getattr(value, "__qualname__", ""))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            [(f.name, _normalize(getattr(value, f.name), depth + 1)) for f in fields(value)],
        )
    state = getattr(value, "__dict__", None)
    if state is None and hasattr(type(value), "__slots__"):
        state = {
            slot: getattr(value, slot) for slot in type(value).__slots__ if hasattr(value, slot)
        }
    if isinstance(state, dict):
        return (type(value).__qualname__, _normalize(state, depth + 1))
    return (type(value).__qualname__,)


def _refusal_class(exc: BaseException) -> str:
    """Compare refusals by class: belt and allowlist texts are shared, corruption text is not."""

    text = str(exc)
    for prefix in (
        "Blocked mediated allocation via",
        "Blocked allocation-constructor",
        "Blocked BUILD applied",
    ):
        if text.startswith(prefix):
            return prefix
    if text.startswith("Blocked") or text.startswith("Refusing to execute foreign"):
        return text
    if text.startswith("corrupt or malformed pickle stream") or text.startswith(
        "A load persistent id"
    ):
        return "corrupt"
    return "corrupt"


def _outcome(factory: _LoaderFactory, data: bytes) -> tuple[Any, ...]:
    try:
        value = factory(data)
    except pickle.UnpicklingError as exc:
        return ("refused", "UnpicklingError", _refusal_class(exc))
    except Exception as exc:  # noqa: BLE001 -- an application failure keeps its identity
        return ("raised", type(exc).__name__)
    return ("ok", _normalize(value))


def _new_path(data: bytes) -> Any:
    return SafeBundleUnpickler(io.BytesIO(data)).load()


def _old_path(data: bytes) -> Any:
    """The per-opcode machine behind the reader pass both machines share.

    The reader pass refuses what neither machine should see (a memo index above the
    opcode count, an extension code, an over-declared FRAME); a stream it cannot walk is
    the interpreter's to judge, in opcode order.
    """

    try:
        scan_opcodes(data, None)
    except StreamNeedsInterpreter:
        pass
    return _PurePythonBundleUnpickler(io.BytesIO(data)).load()


_HONEST = _honest_streams()
_HOSTILE = _hostile_streams()
_CORPUS = {**_HONEST, **_HOSTILE, **_variants({**_HONEST, **_HOSTILE})}


@pytest.mark.parametrize("label", sorted(_HONEST) + sorted(_HOSTILE))
def test_corpus_decisions_match_the_interpreter(label: str) -> None:
    """Every honest and hostile stream gets the same decision on both machines."""

    data = _CORPUS[label]
    assert _outcome(_new_path, data) == _outcome(_old_path, data)


@pytest.mark.smoke
def test_corpus_decisions_match_the_interpreter_in_one_pass() -> None:
    """The whole corpus, as one smoke item: every mismatch in one report."""

    mismatches = {
        label: (_outcome(_new_path, data), _outcome(_old_path, data))
        for label, data in {**_HONEST, **_HOSTILE}.items()
        if _outcome(_new_path, data) != _outcome(_old_path, data)
    }
    assert not mismatches, f"{len(mismatches)} streams differ: {mismatches}"


def test_generated_variants_match_the_interpreter() -> None:
    """Every generated variant gets the same decision on both machines (one report)."""

    variants = _variants({**_HONEST, **_HOSTILE})
    mismatches = {
        label: (_outcome(_new_path, data), _outcome(_old_path, data))
        for label, data in variants.items()
        if _outcome(_new_path, data) != _outcome(_old_path, data)
    }
    assert not mismatches, f"{len(mismatches)} of {len(variants)} variants differ: {mismatches}"


@pytest.mark.smoke
def test_hostile_corpus_is_refused_by_both_machines() -> None:
    """The hostile corpus is not vacuous: every entry refuses (or raises) on both paths."""

    for label, data in _HOSTILE.items():
        for factory in (_new_path, _old_path):
            outcome = _outcome(factory, data)
            assert outcome[0] in ("refused", "raised"), (label, factory.__name__, outcome)


def test_honest_corpus_takes_the_fast_path() -> None:
    """Every honest stream loads on the C machine; no honest shape needs the interpreter."""

    for label, data in _HONEST.items():
        unpickler = SafeBundleUnpickler(io.BytesIO(data))
        unpickler.load()
        assert unpickler.load_path == "c", (label, unpickler.fallback_reason)


# --------------------------------------------------------------------------- #
# Mechanism pins
# --------------------------------------------------------------------------- #


def test_guards_never_reach_the_loaded_object_graph() -> None:
    """A callable stored as a VALUE comes back as itself, not wrapped in a guard."""

    value = {"method": getattr(torch._C.TensorBase, "sum"), "ctor": collections.OrderedDict}  # noqa: B009
    loaded = _new_path(pickle.dumps(value, protocol=4))
    assert loaded["method"] is value["method"]
    assert loaded["ctor"] is collections.OrderedDict
    assert not any(isinstance(item, GuardedCallable) for item in loaded.values())


def test_build_on_a_reduce_result_refuses_before_setstate_runs() -> None:
    """The guard refuses the BUILD target at the producing REDUCE; ``__setstate__`` never runs."""

    calls = 0
    original = torch.Tensor.__setstate__

    def _spy(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    torch.Tensor.__setstate__ = _spy  # type: ignore[method-assign, assignment]
    try:
        with pytest.raises(pickle.UnpicklingError, match="BUILD applied to a torch"):
            _new_path(_inline_tensor_then_build())
        # The stock torch reduction is refused by name before any tensor exists.
        with pytest.raises(pickle.UnpicklingError, match="_load_from_bytes"):
            _new_path(_embedded_tensor_then_build())
    finally:
        torch.Tensor.__setstate__ = original  # type: ignore[method-assign]
    assert calls == 0


def test_value_and_call_shape_runs_the_interpreter_with_the_same_decision() -> None:
    """A REDUCE result used both as a value and as a callable takes the interpreter."""

    data = _getattr_value_and_call()
    unpickler = SafeBundleUnpickler(io.BytesIO(data))
    new = _outcome(lambda _d: unpickler.load(), data)
    assert unpickler.load_path == "pure_python"
    assert new == _outcome(_old_path, data)


def test_opcode_ceiling_fires_in_the_prescan() -> None:
    """``max_opcodes`` refuses before any object is built, on both paths."""

    data = pickle.dumps([0] * 1000)
    with pytest.raises(OpcodeCeilingExceeded):
        SafeBundleUnpickler(io.BytesIO(data), max_opcodes=100).load()
    with pytest.raises(OpcodeCeilingExceeded):
        SafeBundleUnpickler(io.BytesIO(_getattr_value_and_call()), max_opcodes=3).load()
    counted = SafeBundleUnpickler(io.BytesIO(data), max_opcodes=10_000)
    assert counted.load() == [0] * 1000
    assert counted.prescan_opcodes == sum(1 for _ in pickletools.genops(data))
    # below the byte threshold the ceiling is unreachable and is not applied
    skipped = SafeBundleUnpickler(io.BytesIO(data), max_opcodes=100, count_opcodes_above=len(data))
    assert skipped.load() == [0] * 1000


def test_absurd_memo_index_is_refused_before_either_machine_runs() -> None:
    """A memo PUT index above the opcode count is refused (the C memo-table amplifier)."""

    huge = (
        pickle.PROTO + b"\x02" + pickle.EMPTY_LIST + pickle.LONG_BINPUT + struct.pack("<I", 2**30)
    )
    with pytest.raises(pickle.UnpicklingError, match="memo PUT index"):
        _new_path(huge + pickle.STOP)
    # the flipped-SETITEM shape the generated variants produce
    flipped = pickle.PROTO + b"\x02" + pickle.EMPTY_DICT + b"r" + b"\xff\xff\xff\x7f" + pickle.STOP
    with pytest.raises(pickle.UnpicklingError, match="memo PUT index"):
        _new_path(flipped)
    sane = pickle.PROTO + b"\x02" + pickle.EMPTY_LIST + pickle.BINPUT + b"\x01" + pickle.STOP
    assert _new_path(sane) == []


def test_application_failures_keep_their_identity() -> None:
    """An exception from admitted application code propagates raw, as before."""

    # ``Counter(1)`` fails inside ``collections`` (Python frames, so an application
    # failure on both machines); a C-only failure such as ``OrderedDict(1, 2, 3)`` has no
    # application frame and both machines report it as stream corruption.
    data = _reduce("collections", "Counter", _int(1))
    with pytest.raises(TypeError):
        _new_path(data)
    with pytest.raises(TypeError):
        _old_path(data)
    c_only = _reduce("collections", "OrderedDict", _int(1), _int(2), _int(3))
    assert _outcome(_new_path, c_only) == _outcome(_old_path, c_only)
    assert _outcome(_new_path, c_only)[:2] == ("refused", "UnpicklingError")


def test_a_refusal_fires_in_stream_order_after_an_earlier_failure() -> None:
    """A blocked global after a failing REDUCE: both machines report the REDUCE failure."""

    data = (
        pickle.PROTO
        + b"\x02"
        + pickle.EMPTY_LIST
        + _reduce_body("collections", "Counter").replace(
            pickle.MARK + pickle.TUPLE, pickle.MARK + _int(1) + pickle.TUPLE
        )
        + pickle.APPEND
        + _global("os", "system")
        + pickle.APPEND
        + pickle.STOP
    )
    assert _outcome(_new_path, data) == _outcome(_old_path, data) == ("raised", "TypeError")
    refused_first = (
        pickle.PROTO
        + b"\x02"
        + pickle.EMPTY_LIST
        + _global("os", "system")
        + pickle.APPEND
        + _reduce_body("collections", "Counter")
        + pickle.APPEND
        + pickle.STOP
    )
    assert _outcome(_new_path, refused_first) == _outcome(_old_path, refused_first)
    assert _outcome(_new_path, refused_first)[:2] == ("refused", "UnpicklingError")


class _TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 6)
        self.second = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


def test_saved_bundle_loads_on_the_fast_path_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine bundle, with a metadata tensor, loads through the C machine unchanged."""

    seen: list[tuple[str | None, int]] = []
    original = _RenameAwareUnpickler.load

    def _spy(self: _RenameAwareUnpickler) -> Any:
        try:
            return original(self)
        finally:
            seen.append((self.load_path, self.prescan_opcodes))

    monkeypatch.setattr(_RenameAwareUnpickler, "load", _spy)
    trace = tl.trace(_TwoLayer(), torch.ones(2, 4), capture=CaptureOptions(cache=False))
    trace.annotations["probe"] = torch.arange(4, dtype=torch.float32)
    path = tmp_path / "bundle.tlspec"
    trace.save(path)
    saved_metadata = (path / "metadata.pkl").read_bytes()
    assert seen and all(load_path == "c" for load_path, _ in seen), seen  # the save preflight
    seen.clear()
    restored = tl.load(path)
    assert seen and seen[0][0] == "c", seen
    assert torch.equal(restored.annotations["probe"], torch.arange(4, dtype=torch.float32))
    assert [layer.label for layer in restored.layers] == [layer.label for layer in trace.layers]
    # The interpreter agrees with the fast path on the real metadata bytes.
    assert _outcome(_new_path, saved_metadata)[0] == "ok"
    assert _outcome(_new_path, saved_metadata) == _outcome(_old_path, saved_metadata)

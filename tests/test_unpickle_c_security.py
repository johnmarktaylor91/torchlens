"""Adversarial security review of the C-backed metadata unpickler's load path.

``SafeBundleUnpickler`` reads an untrusted ``metadata.pkl`` once, walks its opcodes with
a tight reader, plans the load with a dry run of the C machine over recording probes and
then loads it with the C ``pickle.Unpickler`` behind call-time guards; a stream the plan
cannot classify runs through the per-opcode pure-Python machine instead. This file is the
attack record for that path: every stream shape tried against the three claims the design
makes -- the reader pass admits no allocation amplifier, the plan cannot be deviated from
by the real run, and the two machines decide alike -- whether or not it found a hole.

Streams that would allocate without bound are loaded in a SUBPROCESS under an address-space
limit, so a regression fails this file instead of the test runner's host.

No test here carries the ``smoke`` mark: several of them spawn a child interpreter, and the
smoke tier is already at its size ceiling. This file is a targeted suite, run by name.
"""

from __future__ import annotations

import io
import pickle
import struct
import subprocess
import sys
import textwrap
from typing import Any

import pytest

from torchlens._io._pickle_prescan import prescan_stream
from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
    _build_refusal,
    _build_type_refusal,
    _construction_refusal,
    _PurePythonBundleUnpickler,
)

# --------------------------------------------------------------------------- #
# Stream builders
# --------------------------------------------------------------------------- #

_P2 = pickle.PROTO + b"\x02"


def _global(module: str, name: str) -> bytes:
    return pickle.GLOBAL + module.encode() + b"\n" + name.encode() + b"\n"


def _unicode(text: str) -> bytes:
    return pickle.SHORT_BINUNICODE + bytes([len(text)]) + text.encode()


def _int(value: int) -> bytes:
    return pickle.BININT + struct.pack("<i", value)


def _frame(size: int) -> bytes:
    return pickle.FRAME + struct.pack("<Q", size)


_ORDERED_DICT = _global("collections", "OrderedDict")
_CALL = pickle.MARK + pickle.TUPLE + pickle.REDUCE


def _new_path(data: bytes) -> Any:
    return SafeBundleUnpickler(io.BytesIO(data)).load()


def _old_path(data: bytes) -> Any:
    return _PurePythonBundleUnpickler(io.BytesIO(data)).load()


def _decision(loader: Any, data: bytes) -> tuple[str, ...]:
    """The comparable verdict of one machine on one stream."""

    try:
        value = loader(data)
    except pickle.UnpicklingError:
        return ("refused",)
    except Exception as exc:  # noqa: BLE001 -- an application failure keeps its identity
        return ("raised", type(exc).__name__)
    return ("ok", repr(value))


def _both_refuse(data: bytes, match: str | None = None) -> None:
    """Assert both machines refuse ``data`` with ``UnpicklingError``."""

    with pytest.raises(pickle.UnpicklingError, match=match):
        _new_path(data)
    with pytest.raises(pickle.UnpicklingError, match=match):
        _old_path(data)


def _agree(data: bytes) -> tuple[str, ...]:
    """Assert the two machines decide alike on ``data`` and return the decision."""

    new = _decision(_new_path, data)
    old = _decision(_old_path, data)
    assert new == old, f"machines disagree: C={new!r} interpreter={old!r}"
    return new


def _load_in_subprocess(data: bytes, address_space_bytes: int = 4 * 1024**3) -> str:
    """Load ``data`` in a child capped by ``RLIMIT_AS``; return its one-word verdict.

    An allocation amplifier must not be able to take the test runner's host down, so the
    streams that would allocate without bound are only ever loaded here. The limit is
    applied AFTER the imports, so it bounds the load and not torch's own address space.
    """

    program = textwrap.dedent(
        """
        import io, pickle, resource, sys
        from torchlens._io._safe_unpickle import SafeBundleUnpickler
        data = sys.stdin.buffer.read()
        limit = {limit}
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        try:
            SafeBundleUnpickler(io.BytesIO(data)).load()
        except pickle.UnpicklingError:
            print("refused")
        except MemoryError:
            print("memoryerror")
        except Exception as exc:
            print("raised:" + type(exc).__name__)
        else:
            print("loaded")
        """
    ).format(limit=address_space_bytes)
    completed = subprocess.run(  # noqa: S603 -- fixed argv, this interpreter
        [sys.executable, "-I", "-c", program],
        input=data,
        capture_output=True,
        timeout=180,
    )
    if completed.returncode != 0:
        return f"exit:{completed.returncode}"
    return completed.stdout.decode().strip()


# --------------------------------------------------------------------------- #
# 1. The reader pass: memo indices, frames and declared lengths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "put_opcode",
    [
        pickle.BINPUT + b"\xff",
        pickle.LONG_BINPUT + struct.pack("<I", 2**31 - 1),
        pickle.LONG_BINPUT + struct.pack("<I", 2**32 - 1),
        pickle.PUT + b"4294967295\n",
        pickle.PUT + b" 4294967295\n",
        pickle.PUT + b"000000009999\n",
    ],
)
def test_memo_put_index_above_the_opcode_count_is_refused(put_opcode: bytes) -> None:
    """Every memo-writing spelling is held to the opcode count, text forms included.

    The C machine sizes its memo table from the PUT index, so a five-byte opcode could
    demand billions of slots; the index is bounded by the opcodes seen so far, which an
    honest (sequentially memoizing) pickler never exceeds.
    """

    data = _P2 + pickle.NONE + put_opcode + pickle.STOP
    with pytest.raises(pickle.UnpicklingError, match="memo PUT index above the opcode count"):
        _new_path(data)


def test_the_memo_index_ceiling_admits_an_honest_sequential_stream() -> None:
    """The guard is a ceiling on dishonesty, not on size: real streams memoize densely."""

    honest = pickle.dumps([{"k": index} for index in range(200)], protocol=4)
    assert _new_path(honest) == [{"k": index} for index in range(200)]
    assert pickle.BINPUT[0] in honest or pickle.MEMOIZE[0] in honest


def test_an_absurd_memo_index_never_allocates_the_table() -> None:
    """The amplifier is refused inside a 2 GiB child, not merely refused eventually."""

    data = _P2 + pickle.NONE + pickle.LONG_BINPUT + struct.pack("<I", 2**32 - 1) + pickle.STOP
    assert _load_in_subprocess(data) == "refused"


@pytest.mark.parametrize(
    "length_opcode",
    [
        pickle.BINBYTES8 + struct.pack("<Q", 2**42),
        pickle.BYTEARRAY8 + struct.pack("<Q", 2**42),
        pickle.BINUNICODE8 + struct.pack("<Q", 2**42),
        pickle.BINBYTES + struct.pack("<I", 2**31),
        pickle.BINSTRING + struct.pack("<I", 2**31 + 5),
        pickle.LONG4 + struct.pack("<i", 2**30),
    ],
)
def test_a_declared_length_above_the_stream_never_allocates(length_opcode: bytes) -> None:
    """A length-prefixed opcode may only promise what the remaining bytes can pay for."""

    data = _P2 + length_opcode + b"ab" + pickle.STOP
    assert _load_in_subprocess(data) == "refused"
    _both_refuse(data)


def test_a_frame_longer_than_the_stream_is_refused_by_the_reader() -> None:
    """A FRAME may not declare more bytes than the stream holds.

    The reader decides it before either machine runs, so the frame length can never
    become a read size.
    """

    body = pickle.NONE + pickle.STOP
    data = _P2 + _frame(2**40) + body
    with pytest.raises(pickle.UnpicklingError, match="FRAME declares more bytes than remain"):
        _new_path(data)
    assert _load_in_subprocess(data) == "refused"


def test_an_exact_frame_still_loads() -> None:
    """The frame check is a bound on the declaration, not a refusal of framing."""

    body = pickle.NONE + pickle.STOP
    assert _new_path(_P2 + _frame(len(body)) + body) is None
    assert _new_path(pickle.dumps({"a": [1, 2, 3]}, protocol=4)) == {"a": [1, 2, 3]}


def test_repeated_empty_container_growth_is_bounded_by_the_stream() -> None:
    """``EMPTY_LIST`` plus ``APPENDS`` grows only with the items the stream spells out."""

    data = _P2 + pickle.EMPTY_LIST + pickle.MARK + pickle.NONE * 50 + pickle.APPENDS + pickle.STOP
    assert _agree(data) == ("ok", repr([None] * 50))


# --------------------------------------------------------------------------- #
# 2. The dry run: no admitted code runs, and the plan binds the real run
# --------------------------------------------------------------------------- #


def test_extension_codes_never_resolve_through_the_process_wide_registry() -> None:
    """EXT1 / EXT2 / EXT4 are refused by both machines before anything is cached.

    Both pickle machines memoize what an extension code resolved to in the shared
    ``copyreg._extension_cache``. The dry run resolves globals to RECORDING PROBES, so an
    admitted extension code would publish a probe class into that process-wide cache,
    where every later unpickler -- including unrelated ``pickle.load`` callers in the same
    process -- would read it instead of the registered class. No bundle writer emits an
    extension code, so both machines refuse one and neither writes the cache.
    """

    program = textwrap.dedent(
        """
        import copyreg, io, pickle, sys
        copyreg.add_extension("collections", "OrderedDict", 177)
        from torchlens._io._safe_unpickle import (
            SafeBundleUnpickler,
            _PurePythonBundleUnpickler,
        )
        data = (
            pickle.PROTO + b"\\x02" + pickle.EXT1 + bytes([177])
            + pickle.MARK + pickle.TUPLE + pickle.REDUCE + pickle.STOP
        )
        verdicts = []
        for machine in (SafeBundleUnpickler, _PurePythonBundleUnpickler):
            try:
                machine(io.BytesIO(data)).load()
            except pickle.UnpicklingError:
                verdicts.append("refused")
            else:
                verdicts.append("loaded")
        print(" ".join(verdicts), "cache=" + repr(dict(copyreg._extension_cache)))
        """
    )
    completed = subprocess.run(  # noqa: S603 -- fixed argv, this interpreter
        [sys.executable, "-I", "-c", program], capture_output=True, timeout=180
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stdout.decode().strip() == "refused refused cache={}"


def test_a_reduce_result_that_is_built_and_called_takes_the_interpreter() -> None:
    """A guard may never stand in for its own result at a BUILD.

    When one REDUCE result is both a BUILD target and a later REDUCE callable, the C
    machine would apply the BUILD to the ``GuardedCallable`` the guard returned, not to
    the object the plan checked. The pre-scan refuses to plan that shape, so the stream
    runs through the interpreter and both machines decide alike.
    """

    data = (
        _P2
        + _ORDERED_DICT
        + _CALL
        + pickle.BINPUT
        + b"\x01"
        + pickle.EMPTY_DICT
        + pickle.BUILD
        + pickle.POP
        + pickle.BINGET
        + b"\x01"
        + _CALL
        + pickle.STOP
    )
    unpickler = SafeBundleUnpickler(io.BytesIO(data))
    assert _decision(lambda _d: unpickler.load(), data) == _decision(_old_path, data)
    assert unpickler.load_path == "pure_python"


def test_a_build_on_a_bare_global_takes_the_interpreter() -> None:
    """A BUILD landing on a global is a shape the dry run will not plan.

    The dry run holds a recording probe where the real run holds the resolved object, so
    the BUILD's outcome is not knowable statically; the interpreter runs it per opcode.
    """

    data = _P2 + _global("torch", "float32") + pickle.EMPTY_DICT + pickle.BUILD + pickle.STOP
    unpickler = SafeBundleUnpickler(io.BytesIO(data))
    assert _decision(lambda _d: unpickler.load(), data) == _decision(_old_path, data)
    assert unpickler.load_path == "pure_python"


def test_more_reduces_than_planned_are_refused() -> None:
    """The guard takes planned sites in order and runs out rather than improvising."""

    plan = prescan_stream(
        _P2 + _ORDERED_DICT + _CALL + pickle.STOP,
        resolve=SafeBundleUnpickler(io.BytesIO(b""))._resolve_global,
        construction_refusal=_construction_refusal,
        build_refusal=_build_refusal,
        build_type_refusal=_build_type_refusal,
    )
    plan.take_reduce_site()
    with pytest.raises(pickle.UnpicklingError, match="more often than the pre-scan planned"):
        plan.take_reduce_site()


def test_no_guard_survives_into_a_second_load_in_the_same_process() -> None:
    """Two loads of the same bytes in one process give the same value and the same path."""

    data = pickle.dumps({"a": [1, 2, 3], "b": ("x", None)}, protocol=4)
    first = SafeBundleUnpickler(io.BytesIO(data))
    second = SafeBundleUnpickler(io.BytesIO(data))
    assert first.load() == second.load() == {"a": [1, 2, 3], "b": ("x", None)}
    assert first.load_path == second.load_path == "c"


@pytest.mark.parametrize(
    "label,stream",
    [
        (
            "callable_reached_through_a_memo_get",
            _P2
            + _ORDERED_DICT
            + pickle.BINPUT
            + b"\x01"
            + pickle.POP
            + pickle.BINGET
            + b"\x01"
            + _CALL
            + pickle.STOP,
        ),
        (
            "global_stored_and_also_called",
            _P2
            + pickle.EMPTY_LIST
            + _ORDERED_DICT
            + pickle.BINPUT
            + b"\x01"
            + pickle.APPEND
            + pickle.BINGET
            + b"\x01"
            + _CALL
            + pickle.APPEND
            + pickle.STOP,
        ),
        (
            "construction_whose_argument_is_built_at_run_time",
            _P2
            + _ORDERED_DICT
            + _ORDERED_DICT
            + _CALL
            + pickle.TUPLE1
            + pickle.NEWOBJ
            + pickle.STOP,
        ),
        (
            "stack_global_spelling_of_a_call",
            pickle.PROTO
            + b"\x04"
            + _unicode("collections")
            + _unicode("OrderedDict")
            + pickle.STACK_GLOBAL
            + _CALL
            + pickle.STOP,
        ),
        (
            "legacy_inst_construction",
            pickle.MARK + pickle.INST + b"collections\nOrderedDict\n" + pickle.STOP,
        ),
        (
            "legacy_obj_construction",
            pickle.MARK + _ORDERED_DICT + pickle.OBJ + pickle.STOP,
        ),
        (
            "self_referential_argument_tree",
            _P2
            + _global("builtins", "dict")
            + pickle.EMPTY_DICT
            + pickle.BINPUT
            + b"\x01"
            + _unicode("k")
            + pickle.BINGET
            + b"\x01"
            + pickle.SETITEM
            + pickle.TUPLE1
            + pickle.REDUCE
            + pickle.STOP,
        ),
    ],
)
def test_indirect_callable_shapes_decide_alike(label: str, stream: bytes) -> None:
    """A callable reached indirectly still reaches one decision on both machines."""

    assert _agree(stream)[0] in ("ok", "refused", "raised")


# --------------------------------------------------------------------------- #
# 3. The allocation belt still refuses what it refused, through both machines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,stream",
    [
        (
            "sized_bytearray_via_reduce",
            _P2
            + _global("builtins", "bytearray")
            + pickle.MARK
            + _int(2**30)
            + pickle.TUPLE
            + pickle.REDUCE
            + pickle.STOP,
        ),
        (
            "sized_bytearray_via_newobj",
            _P2
            + _global("builtins", "bytearray")
            + pickle.MARK
            + _int(2**30)
            + pickle.TUPLE
            + pickle.NEWOBJ
            + pickle.STOP,
        ),
        (
            "sized_bytearray_via_inst",
            pickle.MARK + _int(2**30) + pickle.INST + b"builtins\nbytearray\n" + pickle.STOP,
        ),
        (
            "sized_bytearray_via_obj",
            pickle.MARK + _global("builtins", "bytearray") + _int(2**30) + pickle.OBJ + pickle.STOP,
        ),
        (
            "tensor_construction",
            _P2 + _global("torch", "Tensor") + pickle.EMPTY_TUPLE + pickle.NEWOBJ + pickle.STOP,
        ),
        (
            "parameter_construction",
            _P2
            + _global("torch.nn.parameter", "Parameter")
            + pickle.EMPTY_TUPLE
            + pickle.NEWOBJ
            + pickle.STOP,
        ),
    ],
)
def test_the_allocation_belt_refuses_through_both_machines(label: str, stream: bytes) -> None:
    """Each construction opcode reaches the belt on the fast path and the interpreter."""

    _both_refuse(stream, match="Blocked")
    assert _load_in_subprocess(stream) == "refused"


@pytest.mark.parametrize(
    "name",
    ["system", "popen", "eval", "exec", "__import__", "compile", "open", "getattr"],
)
def test_code_execution_globals_stay_denied_on_the_fast_path(name: str) -> None:
    """The pre-scan resolves through the same policy, so the old denials still fire."""

    module = "os" if name in ("system", "popen") else "builtins"
    stream = _P2 + _global(module, name) + pickle.MARK + pickle.TUPLE + pickle.REDUCE + pickle.STOP
    with pytest.raises(pickle.UnpicklingError):
        _new_path(stream)
    with pytest.raises(pickle.UnpicklingError):
        _old_path(stream)


def test_a_global_the_prescan_never_saw_is_refused_by_the_c_phase() -> None:
    """``find_class`` serves pre-resolved pairs only; an unseen pair is a changed stream."""

    unpickler = SafeBundleUnpickler(io.BytesIO(_P2 + _ORDERED_DICT + _CALL + pickle.STOP))
    unpickler.load()
    plan = prescan_stream(
        _P2 + _ORDERED_DICT + _CALL + pickle.STOP,
        resolve=unpickler._resolve_global,
        construction_refusal=_construction_refusal,
        build_refusal=_build_refusal,
        build_type_refusal=_build_type_refusal,
    )
    unpickler._plan = plan
    try:
        with pytest.raises(pickle.UnpicklingError, match="the pre-scan did not see"):
            unpickler.find_class("collections", "Counter")
    finally:
        unpickler._plan = None


# --------------------------------------------------------------------------- #
# 4. Inherited exposure: a BUILD state the target's C code cannot survive
# --------------------------------------------------------------------------- #


def _dtype_build_with_state(state: bytes) -> bytes:
    """``numpy.dtype("f8", False, True)`` built by REDUCE, then BUILD with ``state``."""

    return (
        _P2
        + _global("numpy", "dtype")
        + pickle.MARK
        + _unicode("f8")
        + pickle.NEWFALSE
        + pickle.NEWTRUE
        + pickle.TUPLE
        + pickle.REDUCE
        + state
        + pickle.BUILD
        + pickle.STOP
    )


def _load_machine_in_child(data: bytes, machine: str) -> int:
    """Load ``data`` with ``machine`` in a child; return the child's return code."""

    program = textwrap.dedent(
        """
        import io, pickle, sys
        from torchlens._io._safe_unpickle import (
            SafeBundleUnpickler,
            _PurePythonBundleUnpickler,
        )
        machine = {machine}
        try:
            machine(io.BytesIO(sys.stdin.buffer.read())).load()
        except Exception:
            pass
        """
    ).format(machine=machine)
    return subprocess.run(  # noqa: S603 -- fixed argv, this interpreter
        [sys.executable, "-I", "-c", program], input=data, capture_output=True, timeout=180
    ).returncode


_SEVEN_ELEMENT_DTYPE_STATE = (
    pickle.MARK
    + _int(3)
    + _unicode("<")
    + pickle.NONE
    + _int(-1)
    + _int(-1)
    + _int(0)
    + pickle.TUPLE
)


def test_an_honest_numpy_dtype_still_round_trips_through_both_machines() -> None:
    """The dtype reconstruction legitimate metadata uses is unaffected by the gap below."""

    import numpy as np

    data = pickle.dumps(np.dtype("f8"), protocol=4)
    assert _new_path(data) == np.dtype("f8")
    assert _old_path(data) == np.dtype("f8")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Inherited from the pure-Python loader, not introduced by the C path: a BUILD whose "
        "state is a seven-element tuple reaches numpy's own dtype __setstate__, whose C code "
        "indexes the tuple without checking its arity and crashes the process. Both machines "
        "crash identically, so it is not a divergence. Closing it inside the loader needs the "
        "BUILD state at the guard, which runs one opcode earlier at the REDUCE and cannot see "
        "it; refusing every BUILD on a numpy dtype would refuse honest metadata, and routing "
        "them to the interpreter would put every ndarray-bearing bundle on the slow path. The "
        "load site's artifact-size and opcode ceilings still bound what reaches it."
    ),
)
def test_a_malformed_dtype_build_state_does_not_crash_the_process() -> None:
    """A hostile BUILD state must be refused, never executed into a segmentation fault."""

    data = _dtype_build_with_state(_SEVEN_ELEMENT_DTYPE_STATE)
    assert _load_machine_in_child(data, "SafeBundleUnpickler") == 0
    assert _load_machine_in_child(data, "_PurePythonBundleUnpickler") == 0


def test_the_malformed_dtype_build_state_treats_both_machines_alike() -> None:
    """Whatever the gap above does, it does to both machines; neither is the weaker path."""

    data = _dtype_build_with_state(_SEVEN_ELEMENT_DTYPE_STATE)
    assert _load_machine_in_child(data, "SafeBundleUnpickler") == _load_machine_in_child(
        data, "_PurePythonBundleUnpickler"
    )

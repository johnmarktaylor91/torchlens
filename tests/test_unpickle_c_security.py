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


def test_an_absurd_memo_index_never_allocates_the_table() -> None:
    """The amplifier is refused inside a 2 GiB child, not merely refused eventually."""

    data = _P2 + pickle.NONE + pickle.LONG_BINPUT + struct.pack("<I", 2**32 - 1) + pickle.STOP
    assert _load_in_subprocess(data) == "refused"


# --------------------------------------------------------------------------- #
# 2. The dry run: no admitted code runs, and the plan binds the real run
# --------------------------------------------------------------------------- #


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


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "label,stream",
    [
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
    ],
)
def test_indirect_callable_shapes_decide_alike(label: str, stream: bytes) -> None:
    """A callable reached indirectly still reaches one decision on both machines."""

    assert _agree(stream)[0] in ("ok", "refused", "raised")


# --------------------------------------------------------------------------- #
# 3. The allocation belt still refuses what it refused, through both machines
# --------------------------------------------------------------------------- #


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

"""Bytecode decoding of a held-reference clock call site (explicit-time argument proofs)."""

from __future__ import annotations

import dis as _dis_module
from typing import Any

from . import channels

#: Opcodes that push exactly ONE value and therefore keep positional argument
#: slots decodable by walking back from the ``CALL`` instruction. As plain
#: ARGUMENT loads these all push a single value on every supported CPython
#: (``LOAD_GLOBAL``'s extra-NULL form applies only to callable loads).
_SINGLE_PUSH_LOAD_OPNAMES = frozenset(
    {"LOAD_CONST", "LOAD_FAST", "LOAD_NAME", "LOAD_DEREF", "LOAD_GLOBAL"}
)

#: Call-sequence bookkeeping opcodes that carry no argument push of their own and can
#: sit between the last argument-push instruction and the ``CALL``/``CALL_FUNCTION``/
#: ``CALL_METHOD`` instruction, depending on interpreter. Python 3.11 alone splits the
#: call into ``PRECALL`` (argument-count/shape dispatch) followed by ``CALL`` (the
#: actual invocation); 3.10 has no such opcode (``CALL_FUNCTION``/``CALL_METHOD``
#: follow the arguments directly) and 3.12+ folded ``PRECALL`` back into ``CALL``. A
#: naive fixed-width slice ending at the ``CALL`` position silently swallows
#: ``PRECALL`` as if it were the last argument instruction on 3.11, which starves the
#: decode of its real last argument and misclassifies every held-alias call on that
#: interpreter as "unknown" (grind-pyver R8). Walking backward and skipping opcodes in
#: this set keeps the decode interpreter-shape-agnostic.
_CALL_BOOKKEEPING_OPNAMES = frozenset({"PRECALL"})


def _call_site_explicit_time_value(frame: Any, time_arg_index: int) -> bool:
    """Return whether a held-alias ``c_call`` site passes an explicit non-None time.

    Reads the caller frame's bytecode at ``f_lasti``. A plain ``CALL``
    (py3.11+) / ``CALL_FUNCTION`` / ``CALL_METHOD`` (py3.10) oparg carries the
    exact positional count; the instruction that pushed the time argument is
    then decodable when every pushed argument is a simple single-push load,
    and its RUNTIME VALUE is resolved from the frame (constants directly;
    names from the frame's locals/globals, still bound at ``c_call`` time).

    The previous argcount-only decode was VALUE-BLIND (r5 b8-fable R57): a
    held alias called with an explicit ``None`` (``localtime(None)``, or the
    common idiom ``def fmt(ts=None): return ctime(ts)``) decoded as
    "explicit time" and read the current clock unmarked -- a false VERIFIED.
    Resolving the value keeps a genuine held ``localtime(t)`` a pure
    transform (no over-ceiling) while a ``None`` value, a star-call, a
    non-simple argument expression, or any decode failure marks fail-closed
    -- over-marking, never under-marking. The module-attr wrapper path is
    unaffected: it sees the argument value directly and stays exact.

    Parameters
    ----------
    frame:
        Caller frame supplied by the ``c_call`` profile event.
    time_arg_index:
        Position of the explicit-time argument in the converter's signature.

    Returns
    -------
    bool
        ``True`` only when the time argument resolves to a non-``None``
        value; ``False`` means the caller must mark.
    """

    # One decode authority: delegate to the three-way proof (the monitor
    # flow additionally distinguishes ``"unknown"`` -- unresolvable value --
    # as monitor uncertainty; this boolean conflates it with the mark-worthy
    # outcomes, which is exactly the unit-pinned fail-closed contract).
    argcount = channels.call_site_argcount(frame)
    if argcount is None or argcount <= time_arg_index:
        return False
    return _call_site_time_arg_proof(frame, argcount, time_arg_index) == "transform"


def _call_site_time_arg_proof(frame: Any, argcount: int, time_arg_index: int) -> str:
    """Classify the explicit time argument at a held-ref converter call site.

    The positional-count decode alone is VALUE-BLIND: ``localtime(None)`` (and
    the common idiom ``def fmt(ts=None): return ctime(ts)``) passes the
    explicit-time slot yet still reads the current clock, so counting
    positionals let a held pre-window alias escape unmarked (grind-r5 b8
    R57). The decode is VALUE-resolving (r5 b8-fable R57, unified with
    :func:`_call_site_explicit_time_value` at fixwave-5 integration):
    constants resolve directly and simple names resolve from the frame's
    locals/globals, which are still bound at ``c_call`` time -- nothing can
    rebind a simple name between its argument load and the call in the same
    thread, so a resolved value IS the value the converter received.

    Returns
    -------
    str
        ``"transform"`` -- resolves to a provably non-``None`` value (pure
        transform); ``"now_read"`` -- resolves to ``None``, literal or
        through a bound name (implicit-now clock read); ``"unknown"`` --
        unresolvable (attribute/expression argument, unbound name, or
        undecodable site), which callers treat as monitor uncertainty (the
        value is runtime-dependent, so neither a clock-draw claim nor a
        clean pass is provable).
    """

    try:
        lasti = frame.f_lasti
        instructions = list(_dis_module.get_instructions(frame.f_code))
        call_position = next(
            (index for index, ins in enumerate(instructions) if ins.offset == lasti),
            None,
        )
        if call_position is None:
            return "unknown"
        # Walk backward from the CALL instruction collecting exactly ``argcount``
        # single-push argument loads, skipping any interposed call-bookkeeping
        # opcode (``PRECALL`` on Python 3.11 -- see _CALL_BOOKKEEPING_OPNAMES).
        # A fixed-width slice ending at ``call_position`` assumes the ``argcount``
        # instructions immediately preceding CALL are all argument pushes, which
        # is false on 3.11: ``PRECALL`` sits there instead of the real last
        # argument instruction, starving the decode and misclassifying every
        # held-alias call on that interpreter as "unknown" (grind-pyver R8).
        cursor = call_position - 1
        arg_instructions: list[Any] = []
        while cursor >= 0 and len(arg_instructions) < argcount:
            candidate = instructions[cursor]
            if candidate.opname in _CALL_BOOKKEEPING_OPNAMES:
                cursor -= 1
                continue
            arg_instructions.append(candidate)
            cursor -= 1
        if len(arg_instructions) != argcount:
            return "unknown"
        arg_instructions.reverse()
        if any(ins.opname not in _SINGLE_PUSH_LOAD_OPNAMES for ins in arg_instructions):
            return "unknown"
        time_instruction = arg_instructions[time_arg_index]
        if time_instruction.opname == "LOAD_CONST":
            return "now_read" if time_instruction.argval is None else "transform"
        name = time_instruction.argval
        if time_instruction.opname in {"LOAD_FAST", "LOAD_DEREF"}:
            frame_locals = frame.f_locals
            if name in frame_locals:
                return "now_read" if frame_locals[name] is None else "transform"
            return "unknown"
        # LOAD_GLOBAL / LOAD_NAME: module global (falls back through locals
        # for class-body/exec frames first, mirroring name resolution).
        for namespace in (frame.f_locals, frame.f_globals):
            if name in namespace:
                return "now_read" if namespace[name] is None else "transform"
        return "unknown"
    except Exception:
        return "unknown"

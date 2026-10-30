"""Call-site keys for layer grouping (step 7, ``detector="callsite_v1"``).

The grouping rule (``docs/guides/grouping.md``): an operation is a layer, and
running the same place in a module's code again is another pass of it. The
place is the chain of Python frames inside the innermost module call, from that
module's ``forward`` down to the frame that called the torch function, each frame
reduced to ``(code unit, instruction)``. Argument values, tensor shapes, shared
weights and graph shape are never evidence.

This module turns the raw frames the capture walk already visits
(``introspection._get_code_context``) into that chain. Everything is cached per
code object or per raw stack, so the per-op cost is one dict lookup on a hit.
"""

from __future__ import annotations

import dis
import sys
import weakref
from types import CodeType
from typing import Any

#: The one place the module-call choice lives (decided True, 2026-10-29: calling
#: one module object again is another module call, ``m:1``, ``m:2``, ``m:3``).
#: True: a module instance is a scope, so each module call, from any line or
#: through any parent, runs every op inside it once more, and the op's layer gets
#: one pass per module call. False: the caller's frames stay in the chain, so
#: ``self.m(x)`` on two lines gives two layers while module calls still read
#: ``m:1`` and ``m:2``.
MODULE_CALL_IS_PASS = True

#: Marks a call-site equivalence class (``<type>@cs<N>``).
CALL_SITE_CLASS_MARKER = "@cs"

_HAS_CO_POSITIONS = sys.version_info >= (3, 11)

# Code objects of TorchLens's module forward wrappers (``decorated_forward``),
# registered by the torch backend at import. A frame running one of them is the
# boundary between a module call and the code that called the module.
_BOUNDARY_CODES: set[CodeType] = set()

# Per-code memos, held weakly so code built at run time (``exec``, lambdas in
# reloaded modules) is not pinned by the cache.
_INSTRUCTION_MAPS: weakref.WeakKeyDictionary[CodeType, Any] = weakref.WeakKeyDictionary()
_CODE_TOKENS: weakref.WeakKeyDictionary[CodeType, tuple[str, str, int, bytes]] = (
    weakref.WeakKeyDictionary()
)


def register_module_boundary_code(code: CodeType) -> None:
    """Mark ``code`` as a module forward wrapper (a call-site chain boundary)."""

    _BOUNDARY_CODES.add(code)


def is_module_boundary_code(code: CodeType) -> bool:
    """Whether a frame running ``code`` is a TorchLens module forward wrapper."""

    return code in _BOUNDARY_CODES


def _code_token(code: CodeType) -> tuple[str, str, int, bytes]:
    """Content identity of one code unit, so code recompiled per call is one unit."""

    token = _CODE_TOKENS.get(code)
    if token is None:
        qualname = getattr(code, "co_qualname", code.co_name)
        token = (code.co_filename, qualname, code.co_firstlineno, code.co_code)
        _CODE_TOKENS[code] = token
    return token


def _positions_map(code: CodeType) -> list[Any]:
    """Source span per code unit on Python 3.11+ (``co_positions``)."""

    return list(code.co_positions())  # type: ignore[attr-defined,unused-ignore]


def _while_copy_map(code: CodeType) -> dict[int, int]:
    """Fold compiler copies of a ``while`` condition onto their entry copy (3.10).

    CPython emits a ``while`` condition twice: at loop entry and at the back
    edge. A later run of same-line instructions maps onto an earlier same-line
    run when the later run ends in a backward jump and the two runs have the same
    instructions with jump targets ignored. Everything else keeps its offset.
    """

    runs: list[tuple[int | None, list[Any]]] = []
    line: int | None = None
    for instruction in dis.get_instructions(code):
        if instruction.starts_line is not None:
            line = instruction.starts_line
        if runs and runs[-1][0] == line and instruction.starts_line is None:
            runs[-1][1].append(instruction)
        else:
            runs.append((line, [instruction]))

    def _signature(run: list[Any]) -> tuple[Any, ...]:
        """Instructions of one run with jump targets erased."""

        return tuple(
            "jump"
            if ins.opcode in dis.hasjabs or ins.opcode in dis.hasjrel
            else (ins.opname, ins.argrepr)
            for ins in run
        )

    folded: dict[int, int] = {}
    first_by_signature: dict[tuple[Any, ...], list[Any]] = {}
    for run_line, run in runs:
        key = (run_line, _signature(run))
        earlier = first_by_signature.get(key)
        last = run[-1]
        backward = (
            isinstance(last.argval, int)
            and last.argval < last.offset
            and (last.opcode in dis.hasjabs or last.opcode in dis.hasjrel)
        )
        if earlier is not None and backward:
            for copy, original in zip(run, earlier):
                folded[copy.offset] = original.offset
        elif earlier is None:
            first_by_signature[key] = run
    return folded


def instruction_id(code: CodeType, lasti: int) -> Any:
    """The source-level identity of the instruction at ``lasti`` in ``code``.

    Python 3.11+: its source span ``(line, end_line, col, end_col)``, which
    compiler copies of one expression share; the offset when the span is
    missing. Python 3.10: the offset, with ``while``-condition copies folded.
    """

    if _HAS_CO_POSITIONS:
        positions = _INSTRUCTION_MAPS.get(code)
        if positions is None:
            positions = _positions_map(code)
            _INSTRUCTION_MAPS[code] = positions
        index = lasti // 2
        if 0 <= index < len(positions):
            span = positions[index]
            if span[0] is not None:
                return span
        return ("offset", lasti)
    folded = _INSTRUCTION_MAPS.get(code)
    if folded is None:
        folded = _while_copy_map(code)
        _INSTRUCTION_MAPS[code] = folded
    return folded.get(lasti, lasti)


def collapse_recursion(tokens: list[tuple[Any, Any]]) -> tuple[tuple[Any, Any], ...]:
    """Loop-erase recursion: a code unit that reappears replaces its earlier entry.

    Scanning outermost first, when a code unit is already in the chain its
    earlier token and everything after it are dropped before the new token is
    appended, so every recursion depth maps to one chain while different
    external entry points stay distinct.
    """

    out: list[tuple[Any, Any]] = []
    position: dict[Any, int] = {}
    for code_token, instruction in tokens:
        earlier = position.get(code_token)
        if earlier is not None:
            for dropped, _ in out[earlier:]:
                position.pop(dropped, None)
            del out[earlier:]
        position[code_token] = len(out)
        out.append((code_token, instruction))
    return tuple(out)


def module_scope_entry(module: Any) -> tuple[tuple[str, int], int]:
    """A raw chain entry for a module whose hook an op runs in.

    Hooks run in ``_call_impl`` before the module's forward wrapper, so the
    module is not the op's scope; its identity in the chain keeps the hooks of
    distinct modules (two weight-normed convolutions in one ``Sequential``)
    apart while one module called again still repeats its hook's place.
    """

    return (("module", id(module)), 0)


def canonical_chain(raw: tuple[tuple[Any, int], ...]) -> tuple[tuple[Any, Any], ...]:
    """Turn raw ``(code, f_lasti)`` frames, innermost first, into a call-site chain.

    Module scope entries (:func:`module_scope_entry`) pass through unchanged.
    """

    tokens = [
        (code, lasti)
        if isinstance(code, tuple)
        else (_code_token(code), instruction_id(code, lasti))
        for code, lasti in reversed(raw)
    ]
    return collapse_recursion(tokens)

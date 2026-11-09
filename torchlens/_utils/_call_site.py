"""Call-site keys for layer grouping (step 7, ``detector="callsite_v2"``).

The grouping rule (``docs/guides/grouping.md``): an operation is a layer, and
running the same place in the code again is another pass of it. The place is
the chain of Python frames inside the innermost STATEFUL module call (a module
that registers a parameter or buffer, itself or below it; the root model when
none), from that module's ``forward`` down to the frame that called the torch
function, each frame reduced to ``(code unit, instruction)``. A stateless module
crossed on the way is read as the function it is: its frames stay in the chain
behind one identity token, so one ``nn.ReLU`` called from two lines is two
layers while two ``nn.ReLU`` objects called from one loop line stay apart.
Argument values, tensor shapes, shared weights and graph shape are never
evidence.

This module turns the raw frames the capture walk already visits
(``introspection._get_code_context``) into that chain. Everything is cached per
code object or per raw stack, so the per-op cost is one dict lookup on a hit.
"""

from __future__ import annotations

import dis
import sys
import weakref
from collections.abc import Callable, Iterable
from types import CodeType, FrameType
from typing import Any

#: Marks a call-site equivalence class (``<type>@cs<N>``).
CALL_SITE_CLASS_MARKER = "@cs"


def call_site_scope(addresses: Iterable[str], stateful: frozenset[str] | None) -> str:
    """The grouping scope of an op: the innermost stateful module on its module stack.

    ``addresses`` is the op's module stack, outermost first. A module call is a
    pass of the ops inside it only for a stateful module (the 2026-10-29 ruling,
    narrowed to modules that own state on 2026-11-08); ``""`` (the root) when no
    stateful module encloses the op. ``stateful`` is ``None`` before the session
    decided it, and then every module is a scope.
    """

    stack = list(addresses)
    if stateful is None:
        return stack[-1] if stack else ""
    for address in reversed(stack):
        if address in stateful:
            return address
    return ""


def stop_or_cross(frame: FrameType, position: int, crossed: list[tuple[int, Any]]) -> int:
    """Stop the call-site chain at a module forward wrapper frame, or cross it.

    ``position`` is the number of survivors (user frames, innermost first) the
    capture walk kept below this wrapper. A stateless module's wrapper is
    crossed: ``(position, module)`` is appended to ``crossed`` (its identity
    becomes a token at that place in the chain) and ``-1`` is returned, so the
    walk goes on to the stateful module or root that called it. A stateful
    module's wrapper ends the chain: ``position`` is returned. The wrapper
    (``model_prep.module_forward_decorator``) holds its ``module`` (closure)
    and the ``trace`` it records into (local), read the way the walk reads
    ``_call_impl``'s ``self``; the stateful set is that trace's
    (``ModuleCaptureWorkspace.stateful_module_ids``). A wrapper that records
    nothing, or a trace that decided no set, ends the chain as before.
    """

    names = frame.f_locals
    workspace = getattr(names.get("trace"), "_module_capture_ws", None)
    stateful_ids = getattr(workspace, "stateful_module_ids", None)
    module = names.get("module")
    if stateful_ids is not None and module is not None and id(module) not in stateful_ids:
        crossed.append((position, module))
        return -1
    return position


def chain_boundary(inner_boundary_pos: int, outer_boundary_pos: int, last_forward_pos: int) -> int:
    """How many survivors (innermost first) belong to the op's call-site chain.

    The chain stops at the innermost stateful module's forward wrapper
    (:func:`stop_or_cross`). With no stateful wrapper it runs up to the root
    model's forward (the outermost wrapper or ``forward`` survivor).
    """

    if inner_boundary_pos >= 0:
        return inner_boundary_pos
    return max(outer_boundary_pos, last_forward_pos + 1)


def call_site_class(
    index: dict[Any, int], key: tuple[Any, ...], layer_type: str, slot: int | None
) -> str:
    """Return the call-site class ``<type>@cs<N>`` of one op output.

    ``key`` is ``(scope address, call-site chain, func_name, output slot)``
    (:func:`call_site_scope`); ``N`` is its first-appearance number in ``index``, the per-capture
    table the caller owns, so equal keys get equal classes and the string is
    deterministic for the same model code. Exhaustive capture and predicate
    recording mint through this one function, so a cooked recording groups by
    the same classes as ``tl.trace``.
    """

    number = index.get(key)
    if number is None:
        number = len(index) + 1
        index[key] = number
    site_class = f"{layer_type}{CALL_SITE_CLASS_MARKER}{number}"
    if slot is not None:
        site_class += f"_outindex{slot}"
    return site_class


_HAS_CO_POSITIONS = sys.version_info >= (3, 11)

# Code objects of TorchLens's module forward wrappers (``decorated_forward``),
# registered by the torch backend when it decorates submodule forwards. A frame running one of them is the
# boundary between a module call and the code that called the module.
_BOUNDARY_CODES: set[CodeType] = set()

# Per-code memos, held weakly so code built at run time (``exec``, lambdas in
# reloaded modules) is not pinned by the cache.
_INSTRUCTION_MAPS: weakref.WeakKeyDictionary[CodeType, Any] = weakref.WeakKeyDictionary()
_CODE_TOKENS: weakref.WeakKeyDictionary[CodeType, tuple[str, str, int, bytes]] = (
    weakref.WeakKeyDictionary()
)


def register_module_boundary_wrapper(factory: Callable[..., Any], inner_name: str) -> None:
    """Mark the wrapper ``factory`` builds (its nested ``inner_name``) as a boundary.

    A frame running that code is the boundary between a module call and the
    code that called the module. Idempotent: the code object is a constant of
    ``factory``.
    """

    for const in factory.__code__.co_consts:
        if isinstance(const, CodeType) and const.co_name == inner_name:
            _BOUNDARY_CODES.add(const)


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
            for copy, original in zip(run, earlier, strict=True):
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
    """A raw chain entry for a module crossed inside the op's scope.

    Two kinds of module get one: a module whose hook the op runs in (hooks run
    in ``_call_impl`` before the module's forward wrapper, so the module is not
    the op's scope), and a stateless module the op runs inside. Its identity in
    the chain keeps distinct modules apart (two weight-normed convolutions in
    one ``Sequential``, two ``nn.ReLU`` objects called from ``Sequential``'s
    loop line) while one module called again from one place repeats its place.
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


_Survivor = tuple[FrameType, CodeType, int, int]


def select_context_frames(
    survivors: list[_Survivor], last_forward_pos: int, anchor_frame: FrameType | None
) -> tuple[list[_Survivor], _Survivor | None]:
    """Select an op's context frames from the capture walk's survivors, plus its call site.

    ``survivors`` are the user frames ``introspection._get_code_context``
    kept, innermost first. The context is outermost first, like a traceback:
    the call site (the first survivor above the outermost ``forward`` frame,
    or the verified anchor frame), then the outermost ``forward`` frame down
    to the innermost frame, which ran the op and is always LAST. With no
    ``forward`` frame the context is empty and there is no call site.
    """

    if last_forward_pos < 0:
        return [], None
    call_site: _Survivor | None = None
    if last_forward_pos + 1 < len(survivors):
        call_site = survivors[last_forward_pos + 1]
    elif anchor_frame is not None:
        call_site = (anchor_frame, anchor_frame.f_code, anchor_frame.f_lineno, anchor_frame.f_lasti)
    inner = survivors[last_forward_pos::-1]
    return ([call_site, *inner] if call_site is not None else inner), call_site

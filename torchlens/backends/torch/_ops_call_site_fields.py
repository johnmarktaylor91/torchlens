"""Call-site family of the shared op fields: the code context and the call-site chain.

One of the per-family modules split out of ``_ops_shared_fields`` along the
families of ``torchlens/_trace_core/field_table.py`` (here ``site``), so a lane
changing how the call site is observed edits this module alone.

Per op the capture walks the frames below the innermost module forward wrapper
reading only each code object and instruction offset (TorchLens's own frames
are skipped), and does one lookup on that chain plus an interned id for
everything above the wrapper. The id is computed once per module call, on the
first op inside it, from the frames between that wrapper and the next one out
plus that wrapper's id, so the per-op cost does not grow with stack depth. The
module call's record is read from the top of the module stack while the wrapper
is between pushing and popping it, without a snapshot of the frame's locals.
A first-time site is built by ``_call_site_parts`` from the module call's upper
part, computed once per module call, plus the op's own frames; the first op of
a capture, and any op outside a module call, runs
:func:`~torchlens._utils.introspection._get_code_context` as before. The result
is reused for every later op whose key is equal. Codes are held alive by the
table, which lives for one capture only.

The key determines everything that walk reads: every frame's code object and
offset up to the call-site anchor (or the top of the stack), the module of
every ``_call_impl`` frame and every stateless module wrapper crossed before
the innermost stateful module call (the chain's module tokens), and which
wrappers are stateful. So a hit returns exactly what the walk would return.
"""

import sys
from types import CodeType, FrameType
from typing import TYPE_CHECKING, Any

from ..._utils._call_site import _BOUNDARY_CODES, _BOUNDARY_NAMES, _BOUNDARY_PUSHED
from ..._utils.introspection import _CODE_CONTEXT_ANCHOR_KEY, _get_code_context
from ._call_site_parts import (
    build_from_parts,
    is_anchor as _is_anchor,
    is_internal as _is_internal,
    module_record,
    stateful as _stateful,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_predicate_call_site_chain",
    "_set_call_site_fields",
)

#: End markers of a site key: the walk stopped at the call-site anchor, or ran
#: off the top of the stack (before the anchor exists).
_AT_ANCHOR = "anchor"
_AT_TOP = "top"
#: Tags of an interned remainder: the innermost module call is stateful (the
#: chain ends there) or stateless (crossed; its module is a chain token).
_STATEFUL = "stateful"
_CROSSED = "crossed"


class _CallSiteMemo:
    """Per-capture site table, held on the raw graph workspace (dropped after capture)."""

    __slots__ = (
        "anchor",
        "cache",
        "entries",
        "interns",
        "last_ids",
        "last_record",
        "locations",
        "module_ws",
        "pins",
        "sites",
        "uppers",
    )

    #: Whether a miss is built from the per-module-call upper part plus the op's
    #: lower part (``_call_site_parts``); ``False`` takes the full walk instead.
    build_misses = True

    def __init__(self, cache: dict[Any, Any], module_ws: Any = None) -> None:
        """Start an empty table bound to one code-context cache and module workspace."""

        #: The ``_code_context_cache`` the values came from; a new one resets the table.
        self.cache = cache
        #: The trace's module capture workspace: its module stack and stateful set.
        self.module_ws = module_ws
        #: Site key -> ``(code_context list, call-site chain)``; the list is never handed out.
        self.sites: dict[Any, tuple[list[Any], Any]] = {}
        #: ``id(ModuleStackFrame)`` -> ``(frame record, ids remainder, plain remainder)``
        #: for each module call met; the record is held so its id is not reused.
        self.entries: dict[int, tuple[Any, int, int]] = {}
        #: Remainder content -> small int.
        self.interns: dict[Any, int] = {}
        #: ``id(code)`` -> code for every code a key names by id, so no id is
        #: reused while the table lives (the capture).
        self.pins: dict[int, CodeType] = {}
        #: ``id(ModuleStackFrame)`` -> upper part above that module call (miss path).
        self.uppers: dict[int, Any] = {}
        #: ``(id(code), f_lasti, f_lineno)`` -> ``FuncCallLocation`` (miss path).
        self.locations: dict[tuple[int, int, int], Any] = {}
        #: The module call the previous op's key ended in, and its ids remainder.
        self.last_record: Any = None
        self.last_ids = 0
        #: ``(id, code, f_lasti, f_lineno)`` of the anchor frame, or ``None``.
        self.anchor: tuple[int, CodeType, int, int] | None = cache.get(_CODE_CONTEXT_ANCHOR_KEY)


def _intern(memo: _CallSiteMemo, content: Any) -> int:
    """Return the small int naming ``content`` in this capture."""

    interns = memo.interns
    number = interns.get(content)
    if number is None:
        number = len(interns)
        interns[content] = number
    return number


def _module_entry(
    memo: _CallSiteMemo, wrapper: FrameType, depth: int
) -> tuple[Any, int, int] | None:
    """Interned remainders above one module forward wrapper frame, once per module call.

    Returns ``(record, ids, plain)``: ``plain`` names the frames above the wrapper
    (code objects and offsets) up to the anchor or the top; ``ids`` names what the
    walk reads from the wrapper outwards while no stateful module call has been
    passed yet (the wrapper's own statefulness, its module when stateless, and the
    module of each ``_call_impl`` frame above it). ``None`` when the wrapper holds
    no module stack frame (a pass-through or predicate-mode call): the caller then
    takes the full walk. Codes are named by id and pinned in ``memo.pins``;
    TorchLens's own frames are left out (the walk never keeps or reads them).
    """

    record, next_depth = module_record(memo, wrapper, depth)
    if record is None:
        return None
    entry = memo.entries.get(id(record))
    if entry is not None and entry[0] is record:
        return entry
    stateful = _stateful(memo, record)
    walked = _remainder_walk(memo, wrapper, stateful, next_depth)
    if walked is None:
        return None
    plain, ids, tail = walked
    pins = memo.pins
    if isinstance(tail, tuple):
        plain_number = _intern(memo, (plain, tail[2]))
        ids_tail: Any = tail[1]
    else:
        plain_number = _intern(memo, (plain, tail))
        ids_tail = tail
    if stateful:
        ids_number = _intern(memo, (_STATEFUL, plain_number))
    else:
        wrapper_id = id(wrapper.f_code)
        pins[wrapper_id] = wrapper.f_code
        ids_number = _intern(memo, (_CROSSED, wrapper_id, record.module_id, ids, ids_tail))
    entry = (record, ids_number, plain_number)
    memo.entries[id(record)] = entry
    return entry


def _remainder_walk(
    memo: _CallSiteMemo, wrapper: FrameType, stateful: bool, next_depth: int
) -> tuple[Any, Any, Any] | None:
    """Walk the frames between ``wrapper`` and the next wrapper out (or the anchor or top).

    Returns ``(plain, ids, tail)``: the nested frame tuples for both remainders
    and the parent's entry, ``_AT_ANCHOR`` or ``_AT_TOP``. ``None`` when a wrapper
    further out holds no module stack frame.
    """

    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    pins = memo.pins
    plain: Any = None
    ids: Any = None
    frame = wrapper.f_back
    while frame is not None:
        code = frame.f_code
        name = code.co_name
        if name in _BOUNDARY_NAMES and code in _BOUNDARY_CODES:
            parent = _module_entry(memo, frame, next_depth)
            return None if parent is None else (plain, ids, parent)
        if code is anchor_code and _is_anchor(anchor, frame):  # type: ignore[arg-type]
            return plain, ids, _AT_ANCHOR
        if name == "_call_impl" or not _is_internal(code):
            code_id = id(code)
            pins[code_id] = code
            lasti = frame.f_lasti
            plain = (plain, code_id, lasti)
            if not stateful:
                if name == "_call_impl":
                    ids = (ids, code_id, lasti, id(frame.f_locals.get("self")))
                else:
                    ids = (ids, code_id, lasti)
        frame = frame.f_back
    return plain, ids, _AT_TOP


def _site_key(memo: _CallSiteMemo, frame: FrameType | None) -> Any:
    """The site key of the op being recorded, walking up from ``frame``.

    Walks to the innermost module wrapper (or the anchor, or the top) naming each
    kept-kind frame by its code's id and offset, and each ``_call_impl`` frame
    also by its module; TorchLens's own frames are skipped, since the walk never
    keeps or reads them. ``None`` when the wrapper holds no module stack frame
    (the full walk applies). Consecutive ops in one module call reuse its
    remainder without touching the wrapper frame's locals.
    """

    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    pins = memo.pins
    key: Any = None
    while frame is not None:
        code = frame.f_code
        name = code.co_name
        if name in _BOUNDARY_NAMES and code in _BOUNDARY_CODES:
            lasti = frame.f_lasti
            for boundary, after, before in _BOUNDARY_PUSHED[name]:
                if code is boundary and after < lasti < before:
                    stack = memo.module_ws.exhaustive_module_stack
                    if stack and stack[-1] is memo.last_record:
                        return (key, memo.last_ids)
                    break
            entry = _module_entry(memo, frame, 1)
            if entry is None:
                return None
            memo.last_record = entry[0]
            memo.last_ids = entry[1]
            return (key, entry[1])
        if code is anchor_code and _is_anchor(anchor, frame):  # type: ignore[arg-type]
            return (key, _AT_ANCHOR)
        if name == "_call_impl":
            code_id = id(code)
            pins[code_id] = code
            key = (key, code_id, frame.f_lasti, id(frame.f_locals.get("self")))
        elif not _is_internal(code):
            code_id = id(code)
            pins[code_id] = code
            key = (key, code_id, frame.f_lasti)
        frame = frame.f_back
    return (key, _AT_TOP)


def _full_walk(self: "Trace", cache: dict[Any, Any]) -> tuple[list[Any], Any]:
    """The full code-context walk for the op being recorded (two frames up)."""

    call_site_out: list[Any] = []
    code_context = _get_code_context(
        self.num_context_lines,
        source_loading_enabled=self._options_save_source_context,
        disable_col_offset=False,
        context_cache=cache,
        call_site_out=call_site_out,
    )
    return code_context, call_site_out[0]


def _set_call_site_fields(self: "Trace", fields_dict: dict[str, Any], func_call_id: int) -> None:
    """Record the call's ``code_context`` and stage its call-site chain for the row commit.

    One site-key walk and one table lookup per op. A miss is built from the
    module call's upper part and the op's own frames (``_call_site_parts``), or
    takes the full walk where that does not apply. The first op of a capture
    takes the full walk before its key, so the call-site anchor that walk
    establishes ends every key, the first included.
    """

    workspace = self._raw_graph_ws
    memo = workspace._call_site_memo
    cache = self.__dict__.get("_code_context_cache")
    if memo is None or memo.cache is not cache:
        if cache is None:
            cache = {}
            self._code_context_cache = cache
        memo = _CallSiteMemo(cache, self._module_capture_ws)
        workspace._call_site_memo = memo
        code_context, chain = _full_walk(self, cache)
        if memo.anchor is None:
            memo.anchor = cache.get(_CODE_CONTEXT_ANCHOR_KEY)
        key = _site_key(memo, sys._getframe(1))
    else:
        frame = sys._getframe(1)
        key = _site_key(memo, frame)
        try:
            context, chain = memo.sites[key]
        except KeyError:
            pass
        else:
            fields_dict["code_context"] = [*context]
            workspace._pending_call_site = (func_call_id, chain)
            return
        built = None
        if key is not None and memo.build_misses:
            built = build_from_parts(memo, frame, self)
        if built is None:
            code_context, chain = _full_walk(self, cache)
            if memo.anchor is None:  # a full walk may set the anchor
                memo.anchor = cache.get(_CODE_CONTEXT_ANCHOR_KEY)
                key = _site_key(memo, frame)
        else:
            code_context, chain = built
    if key is not None:
        memo.sites[key] = (code_context, chain)
    fields_dict["code_context"] = [*code_context]
    workspace._pending_call_site = (func_call_id, chain)


def _predicate_call_site_chain(trace: "Trace") -> tuple[Any, ...]:
    """Return the call-site chain of the torch call being recorded.

    The same frame walk exhaustive capture runs for its code context
    (:func:`_set_call_site_fields`), asked only for the chain: no source lines, no
    column offsets, and the per-capture anchor in ``trace._code_context_cache``
    stops the walk at the capture call. Empty when no Python frame lies
    inside the innermost stateful module call. The key and class string it
    feeds are the ones exhaustive capture mints (``_ops_activations``), so the
    cooked Trace groups by call site (``callsite_v2``) with the same result.
    """

    context_cache = trace._code_context_cache
    call_site_out: list[Any] = []
    _get_code_context(
        0,
        source_loading_enabled=False,
        disable_col_offset=True,
        context_cache=context_cache,
        call_site_out=call_site_out,
    )
    return call_site_out[0] if call_site_out else ()

"""Call-site family of the shared op fields: the code context and the call-site chain.

One of the per-family modules split out of ``_ops_shared_fields`` along the
families of ``torchlens/_trace_core/field_table.py`` (here ``site``), so a lane
changing how the call site is observed edits this module alone.

Per op the capture walks the frames below the innermost module forward wrapper
reading only each code object and instruction offset, and does one lookup on
that chain plus an interned id for everything above the wrapper. The id is
computed once per module call, on the first op inside it, from the frames
between that wrapper and the next one out plus that wrapper's id, so the
per-op cost does not grow with stack depth. A miss runs
:func:`~torchlens._utils.introspection._get_code_context` as before, which
builds the line numbers, ``FuncCallLocation`` records and source text and
caches them per call site; the result is reused for every later op whose key
is equal.

The key determines everything that walk reads: every frame's code object and
offset up to the call-site anchor (or the top of the stack), the module of
every ``_call_impl`` frame and every stateless module wrapper crossed before
the innermost stateful module call (the chain's module tokens), and which
wrappers are stateful. So a hit returns exactly what the walk would return.
"""

import sys
from types import CodeType, FrameType
from typing import TYPE_CHECKING, Any

from ..._utils import _call_site
from ..._utils._call_site import _BOUNDARY_CODES
from ..._utils.introspection import _CODE_CONTEXT_ANCHOR_KEY, _get_code_context

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

    __slots__ = ("anchor", "cache", "entries", "interns", "sites")

    def __init__(self, cache: dict[Any, Any]) -> None:
        """Start an empty table bound to one code-context cache."""

        #: The ``_code_context_cache`` the values came from; a new one resets the table.
        self.cache = cache
        #: Site key -> ``(code_context tuple, call-site chain)``.
        self.sites: dict[Any, tuple[tuple[Any, ...], Any]] = {}
        #: ``id(ModuleStackFrame)`` -> ``(frame record, ids remainder, plain remainder)``
        #: for each module call met; the record is held so its id is not reused.
        self.entries: dict[int, tuple[Any, int, int]] = {}
        #: Remainder content -> small int.
        self.interns: dict[Any, int] = {}
        #: ``(id, code, f_lasti, f_lineno)`` of the anchor frame, or ``None``.
        self.anchor: tuple[int, CodeType, int, int] | None = cache.get(_CODE_CONTEXT_ANCHOR_KEY)


def _is_anchor(anchor: tuple[int, CodeType, int, int], frame: FrameType) -> bool:
    """Whether ``frame`` is the verified anchor frame (the check ``_get_code_context`` makes)."""

    return (
        id(frame) == anchor[0]
        and frame.f_code is anchor[1]
        and frame.f_lasti == anchor[2]
        and frame.f_lineno == anchor[3]
    )


def _intern(memo: _CallSiteMemo, content: Any) -> int:
    """Return the small int naming ``content`` in this capture."""

    interns = memo.interns
    number = interns.get(content)
    if number is None:
        number = len(interns)
        interns[content] = number
    return number


def _module_entry(memo: _CallSiteMemo, wrapper: FrameType) -> tuple[Any, int, int] | None:
    """Interned remainders above one module forward wrapper frame, once per module call.

    Returns ``(record, ids, plain)``: ``plain`` names the frames above the wrapper
    (code objects and offsets) up to the anchor or the top; ``ids`` names what the
    walk reads from the wrapper outwards while no stateful module call has been
    passed yet (the wrapper's own statefulness, its module when stateless, and the
    module of each ``_call_impl`` frame above it). ``None`` when the wrapper holds
    no module stack frame (a pass-through or predicate-mode call): the caller then
    takes the full walk.
    """

    # The wrapper's ``ModuleStackFrame`` (``model_prep.decorated_forward``); unbound
    # until the module call is pushed, and never bound on a pass-through call.
    record = wrapper.f_locals.get("frame")
    if record is None:
        return None
    entry = memo.entries.get(id(record))
    if entry is not None and entry[0] is record:
        return entry
    stateful = _call_site.stop_or_cross(wrapper, 0, []) >= 0
    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    plain: Any = None
    ids: Any = None
    tail: Any = _AT_TOP
    frame = wrapper.f_back
    while frame is not None:
        code = frame.f_code
        if code in _BOUNDARY_CODES:
            parent = _module_entry(memo, frame)
            if parent is None:
                return None
            tail = parent
            break
        if code is anchor_code and _is_anchor(anchor, frame):  # type: ignore[arg-type]
            tail = _AT_ANCHOR
            break
        lasti = frame.f_lasti
        plain = (plain, code, lasti)
        if not stateful:
            if code.co_name == "_call_impl":
                ids = (ids, code, lasti, id(frame.f_locals.get("self")))
            else:
                ids = (ids, code, lasti)
        frame = frame.f_back
    if isinstance(tail, tuple):
        plain_number = _intern(memo, (plain, tail[2]))
        ids_tail: Any = tail[1]
    else:
        plain_number = _intern(memo, (plain, tail))
        ids_tail = tail
    if stateful:
        ids_number = _intern(memo, (_STATEFUL, plain_number))
    else:
        module_id = id(wrapper.f_locals.get("module"))
        ids_number = _intern(memo, (_CROSSED, wrapper.f_code, module_id, ids, ids_tail))
    entry = (record, ids_number, plain_number)
    memo.entries[id(record)] = entry
    return entry


def _set_call_site_fields(self: "Trace", fields_dict: dict[str, Any], func_call_id: int) -> None:
    """Record the call's ``code_context`` and stage its call-site chain for the row commit.

    Hot path: walk to the innermost module wrapper (or the anchor) building the
    site key, one table lookup. Complexity is the inlined walk, kept in one
    function so a hit costs no extra Python call.
    """

    workspace = self._raw_graph_ws
    memo = workspace._call_site_memo
    cache = self.__dict__.get("_code_context_cache")
    if memo is None or memo.cache is not cache:
        if cache is None:
            cache = {}
            self._code_context_cache = cache
        memo = _CallSiteMemo(cache)
        workspace._call_site_memo = memo
    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    key: Any = None
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if code in _BOUNDARY_CODES:
            entry = _module_entry(memo, frame)
            key = None if entry is None else (key, entry[1])
            break
        if code is anchor_code and _is_anchor(anchor, frame):  # type: ignore[arg-type]
            key = (key, _AT_ANCHOR)
            break
        if code.co_name == "_call_impl":
            key = (key, code, frame.f_lasti, id(frame.f_locals.get("self")))
        else:
            key = (key, code, frame.f_lasti)
        frame = frame.f_back
    else:
        key = (key, _AT_TOP)
    try:
        context, chain = memo.sites[key]
    except KeyError:
        code_context, chain = _walk_call_site(self, cache)
        if key is not None:
            memo.sites[key] = (tuple(code_context), chain)
        if memo.anchor is None:
            memo.anchor = cache.get(_CODE_CONTEXT_ANCHOR_KEY)
        fields_dict["code_context"] = code_context
    else:
        fields_dict["code_context"] = [*context]
    workspace._pending_call_site = (func_call_id, chain)


def _walk_call_site(self: "Trace", cache: dict[Any, Any]) -> tuple[list[Any], Any]:
    """The full frame walk: the op's ``code_context`` list and its call-site chain."""

    call_site_out: list[Any] = []
    code_context = _get_code_context(
        self.num_context_lines,
        source_loading_enabled=self._options_save_source_context,
        disable_col_offset=False,
        context_cache=cache,
        call_site_out=call_site_out,
    )
    return code_context, call_site_out[0]


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

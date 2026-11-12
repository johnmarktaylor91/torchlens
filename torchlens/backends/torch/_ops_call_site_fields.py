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

The key determines everything that walk reads except module identity: every
frame's code object, file and offset up to the call-site anchor (or the top of
the stack), and which wrappers are stateful. The modules the walk reads (that
of every ``_call_impl`` frame and every stateless module wrapper crossed before
the innermost stateful module call: the chain's module tokens) are not in the
key; it holds which of them are equal (first-appearance slots), and the
modules themselves come from the module calls on the module stack, once per
module call. A hit with the cached op's modules returns its value; a hit with
other modules joins the cached chain with them (``_call_site_join``). So a hit
returns exactly what the walk would return, and a module model whose ops run
in distinct module instances from one code path hits the table.

The key walk skips runs of TorchLens frames it has met before by code identity
(``_call_site_runs``), and names codes by ``id``, with every code a stored key
names held by the table; both leave the key what the walk made before.
"""

import sys
from types import CodeType, FrameType
from typing import TYPE_CHECKING, Any

from ..._utils._call_site import _BOUNDARY_CODES, _BOUNDARY_NAMES, _BOUNDARY_PUSHED
from ..._utils._package_frames import _VERDICTS
from ..._utils.introspection import _CODE_CONTEXT_ANCHOR_KEY, _get_code_context
from ._call_site_join import join, make_template, slots
from ._call_site_parts import (
    build_from_parts,
    is_internal as _is_internal,
    module_record,
    stateful as _stateful,
)
from ._call_site_runs import RUN_BUDGET, pin_codes, skip_run

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
#: The slot pattern of a key naming one module (``slots((m,))[1]``).
_ONE_SLOT = (0,)
#: Tags of an interned remainder: the innermost module call is stateful (the
#: chain ends there) or stateless (crossed; its module is a chain token).
_STATEFUL = "stateful"
_CROSSED = "crossed"


class _CallSiteMemo:
    """Per-capture site table, held on the raw graph workspace (dropped after capture)."""

    __slots__ = (
        "anchor",
        "cache",
        "chain_key",
        "entries",
        "entry_contexts",
        "interns",
        "last_ids",
        "last_mods",
        "last_record",
        "locations",
        "module_ws",
        "mods",
        "pinned",
        "run_boundaries",
        "run_budget",
        "runs",
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
        #: Site key -> ``(code_context list, call-site chain, its op's modules, template)``;
        #: the list is never handed out. The template (``_call_site_join``) is ``None``
        #: when the chain cannot be joined with other modules.
        self.sites: dict[Any, tuple[list[Any], Any, tuple[int, ...], Any]] = {}
        #: ``id(ModuleStackFrame)`` -> ``(frame record, ids remainder, plain remainder,
        #: modules)`` for each module call met; the record is held so its id is not
        #: reused. ``modules`` are the distinct module ids the remainders name by slot.
        self.entries: dict[int, tuple[Any, int, int, tuple[int, ...]]] = {}
        #: Remainder content -> small int.
        self.interns: dict[Any, int] = {}
        #: Plain remainder number -> module entry's code context (``_module_entry_site``).
        self.entry_contexts: dict[int, tuple[Any, ...]] = {}
        #: ``id(ModuleStackFrame)`` -> upper part above that module call (miss path).
        self.uppers: dict[int, Any] = {}
        #: ``(code, co_filename, f_lasti, f_lineno)`` -> ``FuncCallLocation``: the
        #: full walk and the miss path build each location once per capture
        #: (``introspection.build_locations``).
        self.locations: dict[tuple[CodeType, str, int, int], Any] = {}
        #: The module call the previous op's key ended in, and its ids remainder.
        self.last_record: Any = None
        self.last_ids = 0
        self.last_mods: tuple[int, ...] = ()
        #: The distinct module ids of the op whose key ``_site_key`` returned last.
        self.mods: tuple[int, ...] = ()
        #: The raw-frames cache key of the chain ``build_from_parts`` built last.
        self.chain_key: Any = None
        #: ``(id, code, f_lasti, f_lineno)`` of the anchor frame, or ``None``.
        self.anchor: tuple[int, CodeType, int, int] | None = cache.get(_CODE_CONTEXT_ANCHOR_KEY)
        #: ``id(first code)`` -> ``[(first code, later codes)]``: runs of frames the key
        #: walk skips (``_call_site_runs``), at most ``run_budget`` more of them.
        self.runs: dict[int, list[tuple[CodeType, tuple[CodeType, ...]]]] = {}
        self.run_budget = RUN_BUDGET
        self.run_boundaries = len(_BOUNDARY_CODES)
        #: ``id(code)`` -> code for every code a stored site key names by id.
        self.pinned: dict[int, CodeType] = {}


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
) -> tuple[Any, int, int, tuple[int, ...]] | None:
    """Interned remainders above one module forward wrapper frame, once per module call.

    Returns ``(record, ids, plain, modules)``: ``plain`` names the frames above the
    wrapper (code objects and offsets) up to the anchor or the top; ``ids`` names
    what the walk reads from the wrapper outwards while no stateful module call has
    been passed yet (the wrapper's own statefulness and, when it is stateless, the
    slots of its module and of each ``_call_impl`` frame's module above it).
    ``modules`` are those modules' distinct ids, in slot order. ``None`` when the wrapper holds
    no module stack frame (a pass-through or predicate-mode call): the caller then
    takes the full walk. Codes are named as in :func:`_site_key`; TorchLens's
    own frames are left out (the walk never keeps or reads them).
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
    plain, ids, tail, own = walked
    above: tuple[int, ...] = ()
    if isinstance(tail, tuple):
        plain_number = _intern(memo, (plain, tail[2]))
        ids_tail: Any = tail[1]
        above = tail[3]
    else:
        plain_number = _intern(memo, (plain, tail))
        ids_tail = tail
    modules: tuple[int, ...] = ()
    if stateful:
        ids_number = _intern(memo, (_STATEFUL, plain_number))
    else:
        modules, pattern = slots((record.module_id, *own, *above))
        ids_number = _intern(memo, (_CROSSED, wrapper.f_code, ids, ids_tail, pattern))
    entry = (record, ids_number, plain_number, modules)
    memo.entries[id(record)] = entry
    return entry


def _remainder_walk(
    memo: _CallSiteMemo, wrapper: FrameType, stateful: bool, next_depth: int
) -> tuple[Any, Any, Any, list[int]] | None:
    """Walk the frames between ``wrapper`` and the next wrapper out (or the anchor or top).

    Returns ``(plain, ids, tail, modules)``: the nested frame tuples for both
    remainders, the parent's entry, ``_AT_ANCHOR`` or ``_AT_TOP``, and the module
    of each ``_call_impl`` frame ``ids`` names, innermost first. ``None`` when a
    wrapper further out holds no module stack frame.
    """

    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    plain: Any = None
    ids: Any = None
    modules: list[int] = []
    frame = wrapper.f_back
    while frame is not None:
        code = frame.f_code
        name = code.co_name
        if name in _BOUNDARY_NAMES and code in _BOUNDARY_CODES:
            parent = _module_entry(memo, frame, next_depth)
            return None if parent is None else (plain, ids, parent, modules)
        if (
            code is anchor_code
            and frame.f_lasti == anchor[2]  # type: ignore[index]
            and frame.f_lineno == anchor[3]  # type: ignore[index]
            and id(frame) == anchor[0]  # type: ignore[index]
        ):  # ``_is_anchor``, inlined
            return plain, ids, _AT_ANCHOR, modules
        if name == "_call_impl":
            internal = False
        else:
            try:
                internal = _VERDICTS[code.co_filename]
            except KeyError:
                internal = _is_internal(code)
        if not internal:
            filename = code.co_filename
            lasti = frame.f_lasti
            plain = (plain, code, filename, lasti)
            if not stateful:
                ids = (ids, code, filename, lasti)
                if name == "_call_impl":
                    modules.append(id(frame.f_locals["self"]))
        frame = frame.f_back
    return plain, ids, _AT_TOP, modules


def _site_key(memo: _CallSiteMemo, frame: FrameType | None) -> Any:
    """The op's site key from ``frame`` up (code ids, files, offsets, module slots); ``None``: full walk."""

    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    key: Any = None
    lower: tuple[int, ...] | None = None  # ``_call_impl`` modules below the wrapper
    tail: Any = _AT_TOP
    upper: tuple[int, ...] = ()
    while frame is not None:
        code = frame.f_code
        name = code.co_name
        if name in _BOUNDARY_NAMES and code in _BOUNDARY_CODES:
            lasti = frame.f_lasti
            record = None  # the module call's record, while the wrapper holds it pushed
            for boundary, after, before in _BOUNDARY_PUSHED[name]:
                if code is boundary and after < lasti < before:
                    stack = memo.module_ws.exhaustive_module_stack
                    record = stack[-1] if stack else None
                    break
            if record is not None and record is memo.last_record:
                tail, upper = memo.last_ids, memo.last_mods
                break
            entry = None if record is None else memo.entries.get(id(record))
            if entry is None or entry[0] is not record:  # a module call not met before
                entry = _module_entry(memo, frame, 1)
                if entry is None:
                    return None
            memo.last_record, memo.last_ids, _plain, memo.last_mods = entry
            tail, upper = entry[1], entry[3]
            break
        if (
            code is anchor_code
            and frame.f_lasti == anchor[2]  # type: ignore[index]
            and frame.f_lineno == anchor[3]  # type: ignore[index]
            and id(frame) == anchor[0]  # type: ignore[index]
        ):  # ``_is_anchor``, inlined: one call fewer per op
            tail = _AT_ANCHOR
            break
        if name == "_call_impl":
            key = (key, id(code), code.co_filename, frame.f_lasti)
            lower = (*(lower or ()), id(frame.f_locals["self"]))
        else:
            filename = code.co_filename
            try:
                internal = _VERDICTS[filename]
            except KeyError:
                internal = _is_internal(code)
            if internal:
                frame = skip_run(memo, frame, code)
                continue
            key = (key, id(code), filename, frame.f_lasti)
        frame = frame.f_back
    if lower is None:
        memo.mods = upper
        return (key, tail)
    if upper or lower[1:]:
        return _with_lower(memo, (key, tail), lower, upper)
    memo.mods = lower  # one module, no call: ``_with_lower`` gives the same key
    return (key, tail, _ONE_SLOT)


def _with_lower(
    memo: _CallSiteMemo, key: tuple[Any, Any], lower: tuple[int, ...], upper: tuple[int, ...]
) -> tuple[Any, ...]:
    """``key`` with the slots of the modules below the wrapper and above it (a hook's module)."""

    memo.mods, pattern = slots((*lower, *upper))
    return (*key, pattern)


def _full_walk(self: "Trace", memo: _CallSiteMemo) -> tuple[list[Any], Any]:
    """The full code-context walk for the op being recorded (two frames up).

    Its locations come from and go to ``memo.locations``, shared with the miss
    path, so a first-time site under a deep stack builds only its new frames.
    """

    call_site_out: list[Any] = [memo.locations]  # the walk appends the chain
    code_context = _get_code_context(
        self.num_context_lines,
        source_loading_enabled=self._options_save_source_context,
        disable_col_offset=False,
        context_cache=memo.cache,
        call_site_out=call_site_out,
    )
    return code_context, call_site_out[-1]


def _set_call_site_fields(self: "Trace", fields_dict: dict[str, Any], func_call_id: int) -> None:
    """Record the call's ``code_context`` and stage its call-site chain for the row commit.

    One site-key walk and one table lookup per op. A hit whose modules differ
    from the cached op's joins the cached chain with this op's modules. A miss
    is built from the module call's upper part and the op's own frames
    (``_call_site_parts``), or takes the full walk where that does not apply.
    The first op of a capture takes the full walk before its key, so the
    call-site anchor that walk establishes ends every key, the first included.
    """

    workspace = self._raw_graph_ws
    memo = workspace._call_site_memo
    cache = self.__dict__.get("_code_context_cache")
    template = None
    if memo is None or memo.cache is not cache:
        if cache is None:
            cache = {}
            self._code_context_cache = cache
        memo = _CallSiteMemo(cache, self._module_capture_ws)
        workspace._call_site_memo = memo
        code_context, chain = _full_walk(self, memo)
        if memo.anchor is None:
            memo.anchor = cache.get(_CODE_CONTEXT_ANCHOR_KEY)
        frame = sys._getframe(1)
        key = _site_key(memo, frame)
    else:
        frame = sys._getframe(1)
        key = _site_key(memo, frame)
        try:
            context, chain, modules, joinable = memo.sites[key]
        except KeyError:
            pass
        else:
            if modules == memo.mods or joinable is not None:
                if modules != memo.mods:
                    chain = join(chain, joinable, memo.mods, cache)
                fields_dict["code_context"] = [*context]
                workspace._pending_call_site = (func_call_id, chain)
                return
        built = None
        if key is not None and memo.build_misses:
            built = build_from_parts(memo, frame, self)
        if built is None:
            code_context, chain = _full_walk(self, memo)
            if memo.anchor is None:  # a full walk may set the anchor
                memo.anchor = cache.get(_CODE_CONTEXT_ANCHOR_KEY)
                key = _site_key(memo, frame)
        else:
            code_context, chain = built
            if memo.mods:
                template = make_template(chain, memo.mods, memo.chain_key)
    if key is not None:
        pin_codes(memo, frame)
        memo.sites[key] = (code_context, chain, memo.mods, template)
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

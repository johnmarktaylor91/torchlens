"""Build a first-time call site from the module call's upper part plus the op's own frames.

The call-site table (``_ops_call_site_fields``) serves a repeated site from the
table. A first-time site used to take the full frame walk
(:func:`torchlens._utils.introspection._get_code_context`), whose cost grows with
every frame between the op and the capture call. Here the walk is split at the
innermost module forward wrapper:

- the upper part, everything from that wrapper up to the call-site anchor (or
  the top), is built once per module call and reused by every op inside it. It
  is composed from the frames between this wrapper and the next one out plus
  that wrapper's own upper part, so each module call walks only its own frames;
- the lower part, the op's frames below the wrapper, is walked per first-time
  site.

The upper part keeps what the full walk would have read above the wrapper,
with positions counted from the wrapper: the kept frames as ``(code, f_lasti)``
for the call-site chain, the outermost ``forward`` frame, where a stateful
module call ends the chain, the outermost wrapper, the module tokens of
crossed stateless modules and hooks, and the ``FuncCallLocation`` records of the
kept frames the code context takes from above the wrapper. The op's result is
then the same list and chain the full walk returns. The chain is built outermost
first, so the upper part's share is kept as a state the op's own frames resume
from. Locations are built once per ``(code, f_lasti, f_lineno)`` in a capture,
shared with the full walk (``introspection.build_locations``). No frame is held
past the call that walks it.
"""

from types import CodeType, FrameType
from typing import Any, NamedTuple

from ..._utils._call_site import (
    _BOUNDARY_CODES,
    _BOUNDARY_NAMES,
    _BOUNDARY_PUSHED,
    _code_token,
    chain_boundary,
    instruction_id,
    module_id_scope_entry,
    module_scope_entry,
)
from ..._utils._package_frames import _VERDICTS, is_torchlens_code
from ..._utils.introspection import build_locations

__all__ = ("build_from_parts", "is_anchor", "is_internal", "module_record", "stateful")

_Kept = tuple[FrameType, CodeType, int, int]


class _Upper(NamedTuple):
    """What the full walk reads above one module forward wrapper, positions from the wrapper."""

    #: The module call's ``ModuleStackFrame``; ``None`` above the outermost wrapper.
    record: Any
    #: Kept frames above the wrapper as ``(code, f_lasti)``, innermost first.
    raw: list[tuple[CodeType, int]]
    #: Position of the outermost ``forward`` frame among them, or -1.
    forward: int
    #: Position where a stateful module call ends the chain, or -1.
    inner: int
    #: Position of the outermost wrapper (0 is this one); ``None`` above it.
    outer: int | None
    #: ``(position, chain entry)`` of each module token met while no stateful
    #: module call has ended the chain.
    scopes: list[tuple[int, Any]]
    #: The code context's locations from above the wrapper (call site first),
    #: when the outermost ``forward`` frame lies above it; else ``None``.
    prefix: list[Any] | None
    #: Location of the first kept frame above the wrapper, else of the anchor
    #: frame; ``None`` when there is neither.
    first: Any
    #: Chain state after this part's first ``n`` kept frames, per ``n`` (``_upper_state``).
    states: dict[int, tuple[list[Any], dict[Any, int]]]


def is_anchor(anchor: tuple[int, CodeType, int, int], frame: FrameType) -> bool:
    """Whether ``frame`` is the verified anchor frame (the check ``_get_code_context`` makes)."""

    return (
        id(frame) == anchor[0]
        and frame.f_code is anchor[1]
        and frame.f_lasti == anchor[2]
        and frame.f_lineno == anchor[3]
    )


def is_internal(code: CodeType) -> bool:
    """Whether ``code`` is TorchLens's own (the walk never keeps such a frame)."""

    try:
        return _VERDICTS[code.co_filename]
    except KeyError:
        return is_torchlens_code(code)


def module_record(memo: Any, wrapper: FrameType, depth: int) -> tuple[Any, int]:
    """The ``ModuleStackFrame`` of the module call ``wrapper`` runs, and the next depth.

    ``depth`` counts module wrapper frames from the op outwards (1 for the
    innermost). While every wrapper so far is suspended between pushing its
    record and popping it (``_call_site._BOUNDARY_PUSHED``), the record is the
    ``depth``-th from the top of the module stack, read without the frame's
    locals. Otherwise (and from then on, depth 0) it is the wrapper's ``frame``
    local, unbound (``None``) on a pass-through or predicate-mode call.
    """

    if depth:
        lasti = wrapper.f_lasti
        code = wrapper.f_code
        for boundary, after, before in _BOUNDARY_PUSHED.get(code.co_name, ()):
            if code is boundary and after < lasti < before:
                stack = memo.module_ws.exhaustive_module_stack
                if len(stack) >= depth:
                    return stack[-depth], depth + 1
    return wrapper.f_locals.get("frame"), 0


def stateful(memo: Any, record: Any) -> bool:
    """Whether module call ``record`` ends the call-site chain (``_call_site.stop_or_cross``)."""

    stateful_ids = getattr(memo.module_ws, "stateful_module_ids", None)
    return stateful_ids is None or record.module_id in stateful_ids


def _locations(memo: Any, kept: list[_Kept], options: tuple[Any, ...]) -> list[Any]:
    """The locations of ``kept`` frames, each built once per capture (``build_locations``)."""

    return build_locations(kept, *options)


def _combine(
    memo: Any,
    record: Any,
    walked: tuple[int, list[tuple[int, Any]], list[_Kept], int],
    parent: _Upper,
    options: tuple[Any, ...],
) -> _Upper:
    """This wrapper's upper part: its own frames ``walked`` followed by ``parent``'s."""

    inner, scopes, between, forward = walked
    count = len(between)
    raw = [(code, lasti) for _frame, code, _lineno, lasti in between] + parent.raw
    prefix: list[Any] | None = None
    if parent.forward >= 0:
        forward = count + parent.forward
        prefix = [*parent.prefix, *_locations(memo, between[::-1], options)]  # type: ignore[misc]
    elif forward >= 0:
        above = forward + 1
        call_site = (
            _locations(memo, [between[above]], options)[0] if above < count else parent.first
        )
        prefix = _locations(memo, between[forward::-1], options)
        if call_site is not None:
            prefix = [call_site, *prefix]
    first = _locations(memo, between[:1], options)[0] if between else parent.first
    if inner < 0:
        inner = count + parent.inner if parent.inner >= 0 else -1
        scopes = scopes + [(count + position, entry) for position, entry in parent.scopes]
    outer = 0 if parent.outer is None else count + parent.outer
    return _Upper(record, raw, forward, inner, outer, scopes, prefix, first, {})


def _upper(memo: Any, wrapper: FrameType, depth: int, options: tuple[Any, ...]) -> _Upper | None:
    """The upper part above module wrapper frame ``wrapper``, once per module call.

    ``None`` when this wrapper or one further out holds no module stack frame
    (the caller then takes the full walk).
    """

    record, next_depth = module_record(memo, wrapper, depth)
    if record is None:
        return None
    upper = memo.uppers.get(id(record))
    if upper is not None and upper.record is record:
        return upper
    inner = 0 if stateful(memo, record) else -1
    scopes = [] if inner == 0 else [(0, module_id_scope_entry(record.module_id))]
    between: list[_Kept] = []
    forward = -1
    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    parent: _Upper | None = None
    anchor_first = None
    frame = wrapper.f_back
    while frame is not None:
        code = frame.f_code
        name = code.co_name
        if name in _BOUNDARY_NAMES and code in _BOUNDARY_CODES:
            parent = _upper(memo, frame, next_depth, options)
            if parent is None:
                return None
            break
        if code is anchor_code and is_anchor(anchor, frame):
            kept = (frame, code, frame.f_lineno, frame.f_lasti)
            anchor_first = _locations(memo, [kept], options)[0]
            break
        if name == "_call_impl":
            if inner < 0:
                scopes.append((len(between), module_scope_entry(frame.f_locals.get("self"))))
        elif "_call_impl" not in name and not is_internal(code):
            if name == "forward":
                forward = len(between)
            between.append((frame, code, frame.f_lineno, frame.f_lasti))
        frame = frame.f_back
    if parent is None:
        parent = _Upper(None, [], -1, -1, None, [], None, anchor_first, {})
    upper = _combine(memo, record, (inner, scopes, between, forward), parent, options)
    memo.uppers[id(record)] = upper
    return upper


def _interleave(raw_kept: list[Any], scopes: list[tuple[int, Any]], boundary: int) -> list[Any]:
    """The first ``boundary`` kept frames with the module tokens met among them, innermost first."""

    raw: list[Any] = []
    pending = iter(scopes)
    scope = next(pending, None)
    for position, entry in enumerate(raw_kept[:boundary]):
        while scope is not None and scope[0] <= position:
            raw.append(scope[1])
            scope = next(pending, None)
        raw.append(entry)
    while scope is not None and scope[0] <= boundary:
        raw.append(scope[1])
        scope = next(pending, None)
    return raw


def _collapse(out: list[Any], position: dict[Any, int], raw: list[Any]) -> None:
    """Continue ``_call_site.canonical_chain`` over ``raw`` (innermost first) from a saved state."""

    for code, lasti in reversed(raw):
        if isinstance(code, tuple):
            code_token, instruction = code, lasti
        else:
            code_token, instruction = _code_token(code), instruction_id(code, lasti)
        earlier = position.get(code_token)
        if earlier is not None:
            for dropped, _ in out[earlier:]:
                position.pop(dropped, None)
            del out[earlier:]
        position[code_token] = len(out)
        out.append((code_token, instruction))


def _upper_state(upper: _Upper, count: int) -> tuple[list[Any], dict[Any, int]]:
    """The chain state after the upper part's first ``count`` kept frames, once per module call."""

    state = upper.states.get(count)
    if state is None:
        state = ([], {})
        _collapse(*state, _interleave(upper.raw, upper.scopes, count))
        upper.states[count] = state
    return state


def _chain(
    lower: list[tuple[CodeType, int]], scopes: list[tuple[int, Any]], boundary: int, upper: _Upper
) -> tuple[Any, ...]:
    """The op's call-site chain (``introspection._call_site_chain``), per op only its own frames.

    The chain runs outermost first, so the upper part's share is a saved state
    the op's own frames continue from.
    """

    count = len(lower)
    if boundary < count:
        out: list[Any] = []
        position: dict[Any, int] = {}
    else:
        saved_out, saved_position = _upper_state(upper, boundary - count)
        out, position = list(saved_out), dict(saved_position)
    _collapse(out, position, _interleave(lower, scopes, min(boundary, count)))
    return tuple(out)


def _context(
    memo: Any, lower: list[_Kept], forward: int, upper: _Upper, options: tuple[Any, ...]
) -> list[Any]:
    """The op's code context: outermost first, the frame that ran the op last."""

    if upper.forward >= 0:
        return [*upper.prefix, *_locations(memo, lower[::-1], options)]  # type: ignore[misc]
    if forward < 0:
        return []
    above = forward + 1
    call_site = _locations(memo, [lower[above]], options)[0] if above < len(lower) else upper.first
    kept = _locations(memo, lower[forward::-1], options)
    return kept if call_site is None else [call_site, *kept]


def build_from_parts(
    memo: Any, frame: FrameType | None, trace: Any
) -> tuple[list[Any], Any] | None:
    """The op's ``(code_context, call-site chain)``, walking only its own frames.

    ``frame`` is the frame below which nothing is the user's (the caller of the
    call-site step). ``None`` when the op runs outside any module call, or a
    wrapper on the way holds no module stack frame: the full walk applies.
    """

    anchor = memo.anchor
    anchor_code = anchor[1] if anchor is not None else None
    lower: list[_Kept] = []
    scopes: list[tuple[int, Any]] = []
    forward = -1
    while frame is not None:
        code = frame.f_code
        name = code.co_name
        if name in _BOUNDARY_NAMES and code in _BOUNDARY_CODES:
            break
        if code is anchor_code and is_anchor(anchor, frame):
            return None
        if name == "_call_impl":
            scopes.append((len(lower), module_scope_entry(frame.f_locals.get("self"))))
        elif "_call_impl" not in name:
            try:
                internal = _VERDICTS[code.co_filename]
            except KeyError:
                internal = is_torchlens_code(code)
            if not internal:
                if name == "forward":
                    forward = len(lower)
                lower.append((frame, code, frame.f_lineno, frame.f_lasti))
        frame = frame.f_back
    if frame is None:
        return None
    options = (trace.num_context_lines, trace._options_save_source_context, False, memo.cache)
    upper = _upper(memo, frame, 1, options)
    if upper is None:
        return None
    count = len(lower)
    last_forward = count + upper.forward if upper.forward >= 0 else forward
    inner = count + upper.inner if upper.inner >= 0 else -1
    boundary = chain_boundary(inner, count + upper.outer, last_forward)  # type: ignore[operator]
    raw_kept = [(code, lasti) for _frame, code, _lineno, lasti in lower]
    chain = _chain(raw_kept, scopes, boundary, upper) if boundary > 0 else ()
    return _context(memo, lower, forward, upper, options), chain

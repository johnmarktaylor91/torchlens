"""The one read-only narration observer slot (snoop D1; lane F28).

``EchoSession`` is installed at capture entry as the runtime-only attribute
``trace.__dict__["_echo_session"]`` (never a declared Trace field, never
persisted) and invoked EXACTLY ONCE per capture event from the thin verified
seams on both tiers: the record op path, the trace op path (with the
build-context-when-echo fix), input/buffer sources, and module enter/exit.
The hot-path contract, non-negotiable per the panel memo:

- exactly once per event (the save-predicate double-evaluation receipt);
- receives the frozen metadata record; touches the live tensor ONLY under an
  explicitly armed stats policy; never clones into retention, never saves,
  never replaces;
- mints no intervention state, probes no content, alters no enrichment;
- runs after hooks/interventions produced the effective output and after the
  event commits;
- observer/sink failures are instrumentation failures: warn once, disable
  the narrator, continue capture; if a model exception is active the
  secondary failure becomes a note -- THE ORIGINAL EXCEPTION ALWAYS WINS.

The slot's callable protocol is fixed in wave 1 (frozen record in, string
out, exactly once, never the tensor unless stats is armed) even though wave
1 ships only the built-in narrator: later mounts (``tl.tap`` migration, user
formatters) are a change of mount, not a redesign.
"""

from __future__ import annotations

import contextlib
import warnings
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .._state import pause_logging
from ..errors._base import TorchLensWarning
from ._errors import EchoStatsError
from ._event import NarrationEvent, NarrationStats
from ._format import compact_device, compact_dtype, format_shape, render_line
from ._sink import EchoSink
from ._stats import exact_stats, reuse_stats, sampled_stats

if TYPE_CHECKING:
    import torch

    from ..ir.predicate import RecordContext
    from ..options import EchoOptions

__tl_layer__ = "L5"

#: Note-attached tail is bounded so a PEP-678 note never balloons a traceback.
_NOTE_TAIL_CHAR_BUDGET = 2000


@dataclass
class _ModuleFrame:
    """Session-side mirror of one active module call (held-ancestor state)."""

    address: str
    module_type: str
    depth: int
    printed: bool = False
    op_count: int = 0


@dataclass
class _AttemptedCall:
    """One in-flight wrapped call recorded before execution (crash marker)."""

    func_name: str
    inputs: str
    device_hint: str | None = None


def _describe_attempt_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, str | None]:
    """Describe a wrapped call's inputs for the attempted-op marker.

    Parameters
    ----------
    args:
        Positional arguments of the wrapped call.
    kwargs:
        Keyword arguments of the wrapped call.

    Returns
    -------
    tuple[str, str | None]
        ``(descriptor, device_hint)``: shallow tensor descriptors plus short
        non-tensor kwarg reprs (bounded), and the first tensor device seen.
    """

    import torch

    parts: list[str] = []
    device_hint: str | None = None
    for value in args:
        if isinstance(value, torch.Tensor):
            token = " ".join(
                token
                for token in (
                    format_shape(tuple(value.shape)),
                    compact_dtype(value.dtype),
                    str(value.device),
                )
                if token
            )
            parts.append(token)
            if device_hint is None:
                device_hint = value.device.type
    shown_kwargs = 0
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}={format_shape(tuple(value.shape))}")
            continue
        rendered = repr(value)
        if len(rendered) <= 40 and shown_kwargs < 4:
            parts.append(f"{key}={rendered}")
            shown_kwargs += 1
    return ", ".join(parts), device_hint


class EchoSession:
    """Per-capture narration state machine behind ``echo=`` (both tiers).

    Parameters
    ----------
    options:
        Normalized ``EchoOptions``.
    tier:
        ``"trace"`` or ``"record"`` -- disclosure only; the rendered lines
        are field-identical across tiers by construction (tier parity gate).
    """

    def __init__(self, options: EchoOptions, tier: str) -> None:
        """Resolve the sink, the scope, and the per-pass counters."""

        from ..intervention.selectors import BaseSelector
        from ..ir.selector_eval import split_followed_by_conjunction

        self.options = options
        self.tier = tier
        self.disabled = False
        self._sink = EchoSink(options.sink)
        select = options.select
        self._modules_only = select == "modules"
        self._select_all = select is True or self._modules_only
        self._selector = None if self._select_all else select
        self._selector_is_base = isinstance(self._selector, BaseSelector)
        self._followed_split = (
            split_followed_by_conjunction(self._selector) if self._selector_is_base else None
        )
        self._stats_mode = options.stats
        self._tail_maxlen = max(int(options.tail_on_error), 0)
        self._tail: deque[str] = deque(maxlen=self._tail_maxlen or 1)
        self._module_stack: list[_ModuleFrame] = []
        self._attempted: list[_AttemptedCall] = []
        self._emitted_late: set[int] = set()
        self.pass_index = 1
        self.ordinal = 0
        #: Ordinal shown on lines: the session's own op/source event counter.
        #: Tier-honest AND tier-uniform (the pinned parity gate): both tiers
        #: count the same narratable events even though their internal
        #: capture event indexes differ.
        self._ordinal_by_event: dict[int, int] = {}
        self.events_seen = 0
        self.ops_seen = 0
        self.lines_emitted = 0
        self.lines_matched = 0
        self.suppressed_count = 0
        self._suppression_announced = False
        self._saw_cuda = False
        self._last_completed: tuple[int, str] | None = None
        self._last_narrated: tuple[int, str] | None = None
        self._finished = False

    # ------------------------------------------------------------------ #
    # lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def bind_trace(self, trace: Any) -> None:
        """Install this session on a (per-pass) capture trace."""

        trace.__dict__["_echo_session"] = self

    def reset_pass(self, pass_index: int) -> None:
        """Reset per-forward state for a repeated ``Recorder`` rollout.

        Ordinals, suppression counters, attempted-call state, and the tail
        deque reset each pass; the configured narrator (scope, sink, stats
        mode) is kept.
        """

        self.pass_index = pass_index
        self.ordinal = 0
        self._ordinal_by_event.clear()
        self.suppressed_count = 0
        self._suppression_announced = False
        self._tail.clear()
        self._attempted.clear()
        self._module_stack.clear()
        self._emitted_late.clear()
        self._last_completed = None
        self._last_narrated = None

    # ------------------------------------------------------------------ #
    # scope                                                               #
    # ------------------------------------------------------------------ #

    def _matches(self, ctx: RecordContext) -> bool:
        """Evaluate the narration scope for one op/source context."""

        if self._modules_only:
            return False
        if self._select_all:
            return True
        selector = self._selector
        if selector is None:
            return False
        if self._followed_split is not None:
            # Buffered followed_by: the successor's fire releases recent
            # candidates (handled in emit_op); the candidate alone stays held.
            return False
        try:
            return bool(selector(ctx))
        except Exception as exc:  # noqa: BLE001 - instrumentation failure disables echo, never crashes capture
            self._disable(exc)
            return False

    # ------------------------------------------------------------------ #
    # emission                                                            #
    # ------------------------------------------------------------------ #

    def _deliver(self, event: NarrationEvent) -> None:
        """Render one selected event, feed the tail, and write the sink."""

        line = render_line(event)
        if self._tail_maxlen:
            self._tail.append(line)
        if event.kind in ("op", "input", "buffer"):
            self._last_narrated = (event.ordinal, event.label)
        if self.options.on_error_only:
            return
        max_lines = self.options.max_lines
        if max_lines is not None and self.lines_emitted >= max_lines:
            self.suppressed_count += 1
            if not self._suppression_announced:
                self._suppression_announced = True
                self._sink.write_line(
                    f"-- echo: max_lines={max_lines} reached; further lines suppressed "
                    "(still counted; the crash tail stays uncapped) --"
                )
            return
        self._sink.write_line(line)
        self.lines_emitted += 1

    def _flush_held_ancestors(self) -> None:
        """Print held ancestor enter lines before the first descendant match."""

        for frame in self._module_stack:
            if not frame.printed:
                frame.printed = True
                self._deliver(
                    NarrationEvent(
                        kind="module_enter",
                        ordinal=self.ordinal,
                        label=frame.address,
                        address=frame.address,
                        module_type=frame.module_type,
                        module_depth=frame.depth,
                        tier=self.tier,
                    )
                )

    def _stats_for(
        self, ctx: RecordContext, tensor: torch.Tensor | None, trace: Any
    ) -> NarrationStats | None:
        """Compute the armed stats rung for one narrated op, if any."""

        mode = self._stats_mode
        if mode == "off":
            return None
        numel = 1
        for dim in ctx.shape or ():
            numel *= dim
        if mode == "reuse":
            return reuse_stats(trace, ctx.raw_label or ctx.label, numel)
        if tensor is None:
            return None
        if mode == "sampled":
            return sampled_stats(tensor, identity=ctx.raw_label or ctx.label)
        if mode == "exact":
            return exact_stats(tensor, identity=ctx.raw_label or ctx.label)
        return None

    def _event_from_ctx(
        self,
        ctx: RecordContext,
        *,
        kind: str,
        ordinal: int,
        intervened: bool = False,
        stats: NarrationStats | None = None,
    ) -> NarrationEvent:
        """Build the frozen narration record for one capture context.

        Late (out-of-order lookback) rows stamp ``late=`` afterwards via
        ``dataclasses.replace`` -- the one rare field stays off this hot
        signature.
        """

        device = compact_device(ctx.tensor_device)
        if device is not None and device.startswith("cuda"):
            self._saw_cuda = True
        return NarrationEvent(
            kind=kind,
            ordinal=ordinal,
            label=ctx.label,
            pass_index=ctx.pass_index if ctx.pass_index else self.pass_index,
            step_index=ctx.step_index,
            layer_type=ctx.layer_type,
            func_name=ctx.func_name,
            address=ctx.address if kind == "op" else (ctx.input_output_address or ctx.address),
            module_type=ctx.module_type,
            module_depth=len(self._module_stack),
            shape=ctx.shape,
            dtype=compact_dtype(ctx.dtype),
            device=device,
            output_index=ctx.output_index,
            intervened=intervened,
            late=None,
            stats=stats,
            tier=self.tier,
        )

    def emit_op(
        self,
        ctx: RecordContext,
        *,
        tensor: torch.Tensor | None = None,
        trace: Any = None,
        intervened: bool = False,
    ) -> None:
        """Narrate one committed op event (exactly once per event).

        Parameters
        ----------
        ctx:
            Frozen capture context for the committed event.
        tensor:
            Effective (post-intervention) live output. Read ONLY when a
            value-reading stats rung is armed.
        trace:
            Active capture trace (reuse-rung store reads).
        intervened:
            Whether a hook/intervention replaced this op's output.
        """

        if self.disabled or self._finished:
            return
        with pause_logging():
            try:
                self.events_seen += 1
                self.ops_seen += 1
                self.ordinal += 1
                self._ordinal_by_event[ctx.event_index] = self.ordinal
                if len(self._ordinal_by_event) > 4096:
                    floor = self.ordinal - 2048
                    self._ordinal_by_event = {
                        key: value for key, value in self._ordinal_by_event.items() if value > floor
                    }
                self._last_completed = (self.ordinal, ctx.label)
                for frame in self._module_stack:
                    frame.op_count += 1
                if self._modules_only:
                    return
                if self._followed_split is not None:
                    self._emit_followed_by(ctx, tensor=tensor, trace=trace, intervened=intervened)
                    return
                if not self._select_all and not self._matches(ctx):
                    return
                self.lines_matched += 1
                self._flush_held_ancestors()
                stats = self._stats_for(ctx, tensor, trace)
                self._deliver(
                    self._event_from_ctx(
                        ctx, kind="op", ordinal=self.ordinal, intervened=intervened, stats=stats
                    )
                )
            except EchoStatsError:
                raise
            except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
                self._disable(exc)

    def _emit_followed_by(
        self,
        ctx: RecordContext,
        *,
        tensor: torch.Tensor | None,
        trace: Any,
        intervened: bool,
    ) -> None:
        """Release buffered earlier matches when the future condition fires.

        Bounded ``followed_by``: when the successor fires, candidates in the
        bounded lookback window are narrated marked ``late=N`` with their
        ORIGINAL event ordinal retained -- "live" is never overstated.
        """

        followed, candidate = self._followed_split  # type: ignore[misc]
        inner = followed.inner
        if not callable(inner) or not bool(inner(ctx)):
            return
        for recent in ctx.recent_ops:
            if recent.event_index in self._emitted_late:
                continue
            try:
                matched = bool(candidate(recent))
            except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
                self._disable(exc)
                return
            if not matched:
                continue
            recent_ordinal = self._ordinal_by_event.get(recent.event_index)
            if recent_ordinal is None:
                # Outside the retained window: an unknowable lateness would
                # overstate "live"; the release is skipped, never guessed.
                continue
            self._emitted_late.add(recent.event_index)
            self.lines_matched += 1
            self._flush_held_ancestors()
            late_event = self._event_from_ctx(recent, kind="op", ordinal=recent_ordinal)
            self._deliver(replace(late_event, late=self.ordinal - recent_ordinal))
        del tensor, trace, intervened

    def emit_source(self, ctx: RecordContext) -> None:
        """Narrate one input/buffer source event."""

        if self.disabled or self._finished:
            return
        with pause_logging():
            try:
                self.events_seen += 1
                self.ordinal += 1
                self._ordinal_by_event[ctx.event_index] = self.ordinal
                if self._modules_only or self._followed_split is not None:
                    return
                if not self._select_all and not self._matches(ctx):
                    return
                self.lines_matched += 1
                self._flush_held_ancestors()
                self._deliver(self._event_from_ctx(ctx, kind=str(ctx.kind), ordinal=self.ordinal))
            except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
                self._disable(exc)

    def emit_module_enter(self, address: str, module_type: str) -> None:
        """Track one module entry; print it now or hold it for a match."""

        if self.disabled or self._finished:
            return
        with pause_logging():
            try:
                self.events_seen += 1
                frame = _ModuleFrame(
                    address=address, module_type=module_type, depth=len(self._module_stack)
                )
                self._module_stack.append(frame)
                if self._select_all:
                    # Unscoped narration prints structure lines immediately
                    # (default ON with echo=True; the review's op-only default is a
                    # recorded dissent, snoop s10 D3).
                    frame.printed = True
                    self._deliver(
                        NarrationEvent(
                            kind="module_enter",
                            ordinal=self.ordinal,
                            label=address,
                            address=address,
                            module_type=module_type,
                            module_depth=frame.depth,
                            tier=self.tier,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
                self._disable(exc)

    def emit_module_exit(self, address: str) -> None:
        """Close one module frame; print the exit line if the enter printed."""

        if self.disabled or self._finished:
            return
        with pause_logging():
            try:
                self.events_seen += 1
                if not self._module_stack:
                    return
                frame = self._module_stack.pop()
                if frame.printed:
                    self._deliver(
                        NarrationEvent(
                            kind="module_exit",
                            ordinal=self.ordinal,
                            label=address,
                            address=frame.address,
                            module_type=frame.module_type,
                            module_depth=frame.depth,
                            text=f"{frame.op_count} ops",
                            tier=self.tier,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
                self._disable(exc)

    def abandon_module_frame(self) -> None:
        """Pop one module frame WITHOUT an exit line (module raised).

        A module whose forward raised never completed; printing an exit line
        would claim completion. The printed enter line without its exit is
        the honest crash shape; this keeps the depth mirror consistent when
        user code catches the exception and the forward continues.
        """

        if self._module_stack:
            self._module_stack.pop()

    def note_line(self, text: str) -> None:
        """Append one disclosure note to the live stream and the crash tail."""

        if self.disabled or self._finished:
            return
        with pause_logging():
            try:
                self._deliver(
                    NarrationEvent(
                        kind="note", ordinal=self.ordinal, label="", text=text, tier=self.tier
                    )
                )
            except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
                self._disable(exc)

    # ------------------------------------------------------------------ #
    # attempted-op marker                                                 #
    # ------------------------------------------------------------------ #

    def attempted_push(self, func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        """Record one in-flight wrapped call before it executes.

        The op that raises never produces an output to log, so this marker is
        the only honest source for the crash tail's last line. It is a
        recorded fact, always QUALIFIED as "attempted", never "culprit"
        (nested calls and async CUDA errors make the stronger claim unsound).
        The wrapper pops on normal return only: an exception leaves the
        in-flight chain visible to the tail, and the truncating pop
        self-heals entries stranded by user-caught exceptions. Cost when echo
        is off: one dict read at the wrapper, before the per-op clock starts.

        Returns
        -------
        int
            Stack depth token; pass it to :meth:`attempted_pop` on return.
        """

        if self.disabled:
            return len(self._attempted)
        try:
            with pause_logging():
                depth = len(self._attempted)
                inputs, device_hint = _describe_attempt_inputs(args, kwargs)
                self._attempted.append(
                    _AttemptedCall(func_name=func_name, inputs=inputs, device_hint=device_hint)
                )
                return depth
        except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
            self._disable(exc)
            return len(self._attempted)

    def attempted_pop(self, depth: int) -> None:
        """Clear the in-flight record on normal return.

        Truncating at ``depth`` also self-heals entries stranded by user
        code that caught an op exception and continued the forward.
        """

        if depth < len(self._attempted):
            del self._attempted[depth:]

    # ------------------------------------------------------------------ #
    # completion / failure                                                #
    # ------------------------------------------------------------------ #

    def _zero_match_expected(self) -> bool:
        """Whether a zero-match disclosure applies (selector scopes only)."""

        return not self._select_all and not self.options.on_error_only

    def finish(self, status: str = "complete") -> None:
        """Emit the completion footer and the zero-match disclosure.

        Parameters
        ----------
        status:
            ``"complete"`` or ``"halted"`` -- halted is NOT failed and the
            footer says so via the typed outcome vocabulary.
        """

        if self.disabled or self._finished:
            return
        self._finished = True
        try:
            if self._zero_match_expected() and self.lines_matched == 0 and self.ops_seen:
                warnings.warn(
                    TorchLensWarning(
                        f"echo narrated zero events: selector {self._selector!r} matched "
                        f"nothing across {self.ops_seen} op events in a {status} capture. "
                        "Remedy: check the selector against trace labels, or use echo=True",
                        code="echo_zero_match",
                    ),
                    stacklevel=2,
                )
            if self.options.on_error_only:
                return
            suffix = " (capture halted, not failed)" if status == "halted" else ""
            footer = (
                f"-- echo: {self.lines_emitted} lines narrated, {self.events_seen} events seen, "
                f"stats={self._stats_mode}{suffix} --"
            )
            if self.suppressed_count:
                footer = footer[:-3] + f", {self.suppressed_count} lines suppressed --"
            self._sink.write_line(footer)
            self._sink.flush()
            self._sink.close()
        except Exception as exc:  # noqa: BLE001 - observer failure disables echo, never the capture
            self._disable(exc)

    def render_failure_tail(self, exc: BaseException) -> list[str]:
        """Render the crash tail block, oldest line first (snoop D5)."""

        lines: list[str] = [f"!! forward failed: {type(exc).__name__}: {exc}"]
        if self._last_completed is not None:
            lines.append(
                f"!! last completed event={self._last_completed[0]} op={self._last_completed[1]}"
            )
        if self._last_narrated is not None:
            lines.append(
                f"!! last narrated event={self._last_narrated[0]} op={self._last_narrated[1]}"
            )
        if self._zero_match_expected() and self.lines_matched == 0:
            lines.append(
                "-- echo: zero matches before failure (the crash may precede the scope) --"
            )
        if self._tail:
            lines.append(f"-- last {len(self._tail)} selected events --")
            lines.extend(self._tail)
        if self._attempted:
            attempt = self._attempted[-1]
            lines.append(
                f"!! attempted call={attempt.func_name}  inputs: {attempt.inputs}  "
                "(not proven culprit)"
            )
            if attempt.device_hint == "cuda" or self._saw_cuda:
                # The CUDA caveat prints in the OUTPUT, not just the docs: an
                # async kernel error surfaces at a later sync point, so the
                # marker can name an innocent op.
                lines.append(
                    "!! note: CUDA kernel errors surface asynchronously; rerun with "
                    "CUDA_LAUNCH_BLOCKING=1 for exact attribution"
                )
        lines.append("-- end echo tail; original exception re-raised --")
        return lines

    def on_forward_failure(self, exc: BaseException) -> str | None:
        """Flush the crash tail to the sink and return the bounded note text.

        The tail is REPRINTED even if the lines already appeared live --
        locating the frontier in a long scrollback IS the feature. The
        original exception always wins: any failure in here degrades to a
        warning-note, never a raise.

        Returns
        -------
        str | None
            Bounded tail text for a PEP-678 note, or ``None``.
        """

        if self.disabled or self._finished:
            return None
        self._finished = True
        try:
            with pause_logging():
                lines = self.render_failure_tail(exc)
                for line in lines:
                    self._sink.write_line(line)
                self._sink.flush()
                self._sink.close()
                note = "\n".join(["[torchlens echo tail]", *lines])
                if len(note) > _NOTE_TAIL_CHAR_BUDGET:
                    note = note[:_NOTE_TAIL_CHAR_BUDGET] + "\n... (echo tail truncated)"
                return note
        except Exception as secondary:  # noqa: BLE001 - the original capture exception must win
            self._note_secondary_failure(exc, secondary)
            return None

    def on_interrupt(self) -> None:
        """Best-effort flush for KeyboardInterrupt/SystemExit (stay interrupts)."""

        with contextlib.suppress(Exception):
            self._sink.flush()

    # ------------------------------------------------------------------ #
    # instrumentation-failure policy                                      #
    # ------------------------------------------------------------------ #

    def _disable(self, exc: Exception) -> None:
        """Disable this narrator after an instrumentation failure (warn once)."""

        if self.disabled:
            return
        self.disabled = True
        warnings.warn(
            TorchLensWarning(
                f"echo narration disabled after an observer/sink failure: "
                f"{type(exc).__name__}: {exc}. Capture continues unaffected. "
                "Remedy: fix the sink or selector and re-run",
                code="echo_sink_disabled",
            ),
            stacklevel=3,
        )

    @staticmethod
    def _note_secondary_failure(primary: BaseException, secondary: Exception) -> None:
        """Attach a tail-rendering failure as a note; the original wins."""

        note = (
            "torchlens echo tail rendering also failed while handling this error: "
            f"{type(secondary).__name__}: {secondary}"
        )
        add_note = getattr(primary, "add_note", None)
        if add_note is not None:
            add_note(note)


def active_echo_session(trace: Any) -> EchoSession | None:
    """Return the armed echo session on a capture trace, if any.

    The ONE hot-path accessor the capture seams use: a plain dict read, no
    imports, near-zero cost when echo is off.
    """

    session = trace.__dict__.get("_echo_session") if hasattr(trace, "__dict__") else None
    return session if isinstance(session, EchoSession) else None


__all__ = ["EchoSession", "active_echo_session"]

"""Partial capture helpers for failed TorchLens forward ops."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ..capture.outcome import safe_exception_str
from ..data_classes._nonfinite import first_nonfinite_layer
from ..data_classes.field_policy import FieldPolicy
from ..errors import TorchLensError

if TYPE_CHECKING:
    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace
    from torchlens.debug._audit import TraceAudit


class PartialCaptureLookupError(TorchLensError, ValueError):
    """The exception carries no TorchLens partial capture state (R10).

    Subclasses ``ValueError`` so callers of the historical
    ``from_failed_capture`` contract keep working unchanged.
    """


_FAILED_CAPTURE_REGISTRY_LIMIT = 128
_FAILED_CAPTURE_REGISTRY: OrderedDict[
    int, tuple[weakref.ref[BaseException] | _NonWeakrefIdentityStub, Trace]
] = OrderedDict()
"""Fallback recovery table for exceptions that reject ``partial_log`` assignment.

Keyed by ``id(exception)``; the exception is retained WEAKLY and the stored
value is the partial ``Trace`` alone, never a ``PartialTrace``. The entry-count
cap was not by itself a real bound: an entry that reached the exception
strongly kept its ``__traceback__`` alive, and that pins every frame local --
the model, the inputs, the partial outputs -- so retained large-model failures
could pin GBs with no byte bound and no time eviction. Holding only the trace
breaks that chain (a partial trace records string-only error metadata and never
references the exception object).

Weak retention is sound because the only route to an entry is
``from_failed_capture(exc)``, whose caller must be HOLDING that exception; once
it dies the entry is unreachable garbage and the weakref callback drops it.
Keying on ``id()`` stays safe across id reuse for the reason it already was:
lookup re-checks referent identity, and a dead referent can never satisfy it.
Exception types that do not support weak references NEVER retain the exception
strongly (R37 REOPENED b2:C5: the old strong fallback let up to 128 failed
captures pin their whole exception graphs -- traceback, frame locals, model,
inputs -- until unrelated failures evicted them). They store an identity STUB
holding only the exception TYPE: lookup verifies id + exact type instead of
object identity, a deliberately weaker check for a diagnostic-only channel,
disclosed here rather than paid for in gigabytes.
Stub entries still strongly retain the partial TRACE (that is what makes a
later ``from_failed_capture`` recoverable at all): nothing weakref-able in a
non-weakrefable exception's retention graph exists to witness its death
(frames and tracebacks refuse weak references), so the trace pin is bounded
by the registry CAP rather than by liveness -- a bounded, disclosed cost,
categorically smaller than the unbounded exception-graph pin the stub
eliminates. :func:`_sweep_unreachable_strong_entries` remains as a belt for
the legacy strong entry shape only (a refcount sweep must never fire on
stubs, whose sole strong reference legitimately IS the registry).
"""

_FAILED_CAPTURE_RESULTS: weakref.WeakValueDictionary[int, PartialTrace] = (
    weakref.WeakValueDictionary()
)
"""Identity memo so repeated lookups of one exception return the same wrapper.

Weak-VALUED, so it adds no retention of its own: the wrapper it hands back does
reference the exception strongly, but only for as long as the CALLER keeps that
wrapper. Entries are re-derived on demand and re-checked against exception
identity, so a reused ``id()`` can never serve a foreign wrapper.
"""


@dataclass(frozen=True)
class PartialTrace:
    """Thin wrapper around raw capture state from a failed forward pass.

    Parameters
    ----------
    trace:
        Partially populated ``Trace`` whose raw layer entries were captured
        before the exception.
    original_exception:
        Exception raised by the failed capture.
    """

    trace: Trace
    original_exception: BaseException

    # R11: declared so the runtime field-declaration gates can sweep failed
    # partial products instead of being structurally blind to them. Failed
    # partials never persist (`to_trace`/`tl.save` refuse), so both fields
    # are session-time DROP.
    FIELD_POLICY: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "trace": FieldPolicy.DROP,
            "original_exception": FieldPolicy.DROP,
        }
    )

    # b1-opus-R06-1: the sanctioned outcome-delegation hop. ``outcome_for``
    # (and through it every capability gate) reads the inner trace's settled
    # stamp instead of this wrapper's empty ``__dict__`` -- one answer,
    # whether callers ask the wrapper or the trace.
    _OUTCOME_DELEGATE_FIELD: ClassVar[str] = "trace"

    @property
    def outcome(self) -> Any | None:
        """Return the ONE settled capture outcome (the inner trace's stamp)."""

        from ..capture.outcome import outcome_for

        return outcome_for(self)

    @classmethod
    def from_trace(cls, trace: Trace, exception: BaseException) -> PartialTrace:
        """Build a partial log wrapper from a failed capture's internal state.

        Parameters
        ----------
        trace:
            ``Trace`` instance active when capture failed.
        exception:
            Original exception raised during capture.

        Returns
        -------
        PartialTrace
            Wrapper exposing minimal inspection and graph rendering helpers.
        """

        _materialize_failed_capture_events(trace)
        return cls(trace=trace, original_exception=exception)

    @property
    def raw_layers(self) -> tuple[Op, ...]:
        """Return raw layer entries captured before failure.

        Returns
        -------
        tuple[Op, ...]
            Raw layer pass logs in capture order.
        """

        raw_graph_ws = self.trace.__dict__.get("_raw_graph_ws")
        return tuple(getattr(raw_graph_ws, "raw_layer_dict", {}).values())

    def first_nonfinite(self) -> str:
        """Return a text summary of the first raw non-finite tensor.

        Returns
        -------
        str
            Human-readable summary with layer, operation, shape, dtype, and parents.
        """

        layer = first_nonfinite_layer(self, kind="raw")
        if layer is not None:
            parents = ", ".join(getattr(layer, "parents", None) or []) or "none"
            return (
                "First non-finite captured tensor is in "
                f"layer {getattr(layer, '_label_raw', 'unknown')} "
                f"(op {getattr(layer, 'func_name', 'unknown')}), "
                f"shape={getattr(layer, 'shape', None)}, "
                f"dtype={getattr(layer, 'dtype', None)}, parents={parents}."
            )
        fields = getattr(self.original_exception, "fields", {})
        if "layer" in fields:
            return (
                "First non-finite captured tensor is in "
                f"layer {fields.get('layer')} (op {fields.get('op')}), "
                f"shape={fields.get('shape')}, dtype={fields.get('dtype')}, "
                f"parents={fields.get('parents', [])}."
            )
        return "No non-finite tensor values found in partial capture."

    def audit(self) -> TraceAudit:
        """Return an evidence-backed health report for this partial capture.

        Returns
        -------
        TraceAudit
            Structured failure and saved-payload findings. Full-trace checks are
            explicitly skipped because postprocessing did not complete.
        """

        from torchlens.debug._audit import audit_trace

        return audit_trace(self)

    def draw(self, vis_outpath: str | None = None, **_: Any) -> str:
        """Render the failed capture as minimal Graphviz DOT source.

        Parameters
        ----------
        vis_outpath:
            Accepted for API symmetry; no file is written by this minimal renderer.
        **_:
            Ignored rendering keyword arguments accepted for compatibility.

        Returns
        -------
        str
            DOT source for the raw operations captured before failure.
        """

        lines = [
            "digraph torchlens_partial {",
            '  graph [label="TorchLens partial capture", labelloc=t];',
            '  node [shape=box, style="rounded"];',
        ]
        for layer in self.raw_layers:
            raw_label = _raw_label(layer)
            shape = getattr(layer, "shape", None)
            dtype = getattr(layer, "dtype", None)
            func_name = getattr(layer, "func_name", "unknown")
            label = f"{raw_label}\\nop={func_name}\\nshape={shape}\\ndtype={dtype}"
            lines.append(f'  "{_dot_escape(raw_label)}" [label="{_dot_escape(label)}"];')
            for parent in getattr(layer, "parents", []) or []:
                lines.append(f'  "{_dot_escape(str(parent))}" -> "{_dot_escape(raw_label)}";')
        failure_label = _failure_label(self.original_exception)
        lines.append(f'  "__failure__" [shape=note, label="{_dot_escape(failure_label)}"];')
        if self.raw_layers:
            lines.append(f'  "{_dot_escape(_raw_label(self.raw_layers[-1]))}" -> "__failure__";')
        lines.append("}")
        return "\n".join(lines)

    def show(self, method: Literal["graph", "repr", "html"] = "graph", **kwargs: Any) -> str:
        """Display this partial log using a small set of inspection modes.

        Parameters
        ----------
        method:
            ``"graph"`` returns DOT, ``"repr"`` returns ``repr(self)``, and
            ``"html"`` returns a compact HTML fragment.
        **kwargs:
            Forwarded to ``draw`` for graph mode.

        Returns
        -------
        str
            Rendered graph, representation, or HTML fragment.
        """

        if method == "graph":
            return self.draw(**kwargs)
        if method == "repr":
            return repr(self)
        if method == "html":
            return self._repr_html_()
        raise ValueError("method must be 'graph', 'repr', or 'html'.")

    def _repr_html_(self) -> str:
        """Return a compact notebook HTML representation.

        Returns
        -------
        str
            HTML fragment summarizing the failed capture, generated through
            the CardTree presentation IR (treescope memo B1) with the
            failure-first contract: the partial/failure banner is the badge
            in the always-visible header, never folded.
        """

        from ..notebook.cardtree import Card, CardText, safe_card_html

        def build() -> Card:
            """Assemble the failure-first PartialTrace Card."""

            return Card(
                title="PartialTrace",
                badge="FAILED CAPTURE",
                kind="partial",
                children=(
                    CardText(f"raw_layers={len(self.raw_layers)}"),
                    CardText(self.first_nonfinite()),
                    CardText(
                        f"error: {safe_exception_str(self.original_exception)}",
                        role="notice",
                    ),
                ),
            )

        return safe_card_html(build)

    def __repr__(self) -> str:
        """Return a concise partial capture representation.

        Returns
        -------
        str
            Debug representation with raw layer count and original error type.
        """

        return (
            f"PartialTrace(raw_layers={len(self.raw_layers)}, "
            f"error={type(self.original_exception).__name__})"
        )


def from_failed_capture(exception: BaseException) -> PartialTrace:
    """Return the partial log attached to a failed TorchLens trace exception.

    Failed ``tl.trace(...)`` captures attach an honest partial as
    ``exception.partial_log`` before re-raising. This helper retrieves that
    wrapper. Fastlog ``tl.record(...)`` failed partials are separate
    ``Recording`` objects attached as ``exception.partial_recording`` or
    returned by ``on_forward_error="return_partial"``.

    Parameters
    ----------
    exception:
        Exception raised by ``torchlens.trace``.

    Returns
    -------
    PartialTrace
        Partial capture wrapper attached as ``exception.partial_log``.

    Raises
    ------
    PartialCaptureLookupError
        If the exception does not carry TorchLens partial capture state
        (a ``ValueError`` subclass, preserving the historical contract).
    """

    partial_log = getattr(exception, "partial_log", None)
    if isinstance(partial_log, PartialTrace):
        return partial_log
    _sweep_unreachable_strong_entries()
    exception_id = id(exception)
    registry_entry = _FAILED_CAPTURE_REGISTRY.get(exception_id)
    if registry_entry is not None and _held_matches(registry_entry[0], exception):
        _FAILED_CAPTURE_REGISTRY.move_to_end(exception_id)
        memoized = _FAILED_CAPTURE_RESULTS.get(exception_id)
        if memoized is not None and memoized.original_exception is exception:
            return memoized
        recovered = PartialTrace(trace=registry_entry[1], original_exception=exception)
        _FAILED_CAPTURE_RESULTS[exception_id] = recovered
        return recovered
    raise PartialCaptureLookupError("exception does not contain a TorchLens partial capture")


class _NonWeakrefIdentityStub:
    """Identity witness for a non-weakrefable registered exception.

    Holds only the exception TYPE (a long-lived class object), never the
    instance, so the registry cannot pin the exception graph. No liveness
    witness is constructible for the stub arm: the exception refuses weak
    references and so do its traceback and frames, so the entry (and the
    partial trace it strongly retains) is bounded by the registry cap
    rather than by liveness, disclosed on the registry docstring.
    """

    __slots__ = ("exc_type",)

    def __init__(self, exc_type: type[BaseException]) -> None:
        self.exc_type = exc_type


def _held_matches(
    held: weakref.ref[BaseException] | _NonWeakrefIdentityStub | BaseException,
    exception: BaseException,
) -> bool:
    """Return whether a registry slot identifies ``exception``.

    Weak entries verify OBJECT identity. Stub entries (non-weakrefable types)
    verify id-key + exact type -- weaker by construction, disclosed in the
    registry docstring.
    """

    if isinstance(held, weakref.ref):
        return held() is exception
    if isinstance(held, _NonWeakrefIdentityStub):
        return type(exception) is held.exc_type
    # Legacy strong entry shape (should not occur after R37); exact identity.
    return held is exception


def _sweep_unreachable_strong_entries() -> None:
    """Evict strong-fallback entries whose exception no caller can reach.

    Weak entries evict themselves through their weakref callback the moment
    the caller drops the exception. Strong entries (non-weakrefable exception
    types) have no callback, so without this sweep a dropped exception kept
    its whole traceback -- frame locals, model, inputs -- plus the partial
    trace pinned until 128 later failures evicted it. Swept at registration
    and lookup time by refcount: an exception whose only remaining reference
    is this registry's entry tuple can never be passed to
    ``from_failed_capture`` again, so its entry is unrecoverable garbage.
    """

    import sys

    for key, entry in list(_FAILED_CAPTURE_REGISTRY.items()):
        if isinstance(entry[0], weakref.ref):
            continue
        if isinstance(entry[0], _NonWeakrefIdentityStub):
            # Stub entries hold only the exception TYPE; their sole strong
            # reference legitimately IS the registry tuple, so a refcount
            # sweep would evict every stub immediately after registration.
            # No liveness witness is constructible for them (the exception,
            # its traceback, and its frames all refuse weak references);
            # they stay until the registry cap, disclosed on the stub class.
            continue
        # Sole-ownership baseline: the registry tuple's slot plus
        # getrefcount's own argument slot -> 2. Any caller-held reference
        # (including an in-flight ``except`` binding) raises it above that,
        # so miscounting can only KEEP an entry, never evict a live one.
        if sys.getrefcount(entry[0]) <= 2:
            del _FAILED_CAPTURE_REGISTRY[key]


def _register_failed_capture(exception: BaseException, partial_log: PartialTrace) -> None:
    """Retain partial recovery when an exception rejects attribute assignment.

    Parameters
    ----------
    exception:
        Original user exception that could not accept ``partial_log``.
    partial_log:
        Constructed partial capture associated with the exception by identity.

    Returns
    -------
    None
        Stores a bounded, weakly-held entry for :func:`from_failed_capture`.
    """

    _sweep_unreachable_strong_entries()
    exception_id = id(exception)
    held: weakref.ref[BaseException] | _NonWeakrefIdentityStub
    try:
        held = weakref.ref(exception, _drop_failed_capture_entry(exception_id))
    except TypeError:
        # Exception type does not support weak references: store an identity
        # stub (type only), NEVER the exception itself -- a strong entry pins
        # the traceback's frame locals (model, inputs) with no byte bound
        # (R37 REOPENED b2:C5). Lookup verifies id + exact type.
        held = _NonWeakrefIdentityStub(type(exception))
    # Store the TRACE, not the wrapper: a stored wrapper reaches the exception
    # strongly and would keep its own weak key alive forever (and with it the
    # traceback's frame locals).
    _FAILED_CAPTURE_REGISTRY[exception_id] = (held, partial_log.trace)
    _FAILED_CAPTURE_RESULTS[exception_id] = partial_log
    _FAILED_CAPTURE_REGISTRY.move_to_end(exception_id)
    while len(_FAILED_CAPTURE_REGISTRY) > _FAILED_CAPTURE_REGISTRY_LIMIT:
        _FAILED_CAPTURE_REGISTRY.popitem(last=False)


def _drop_failed_capture_entry(exception_id: int) -> Callable[[weakref.ref[Any]], None]:
    """Build the weakref callback that evicts one collected registry entry.

    Parameters
    ----------
    exception_id:
        ``id()`` of the registered exception, used as the registry key.

    Returns
    -------
    Callable[[weakref.ref[Any]], None]
        Callback that drops the entry if that exact weak reference still owns it.
    """

    def _drop(reference: weakref.ref[Any]) -> None:
        """Evict the entry this dead reference owned, if it is still current."""

        entry = _FAILED_CAPTURE_REGISTRY.get(exception_id)
        # Guard against id reuse: only evict when this very reference is the
        # one recorded, never a newer entry that happens to share the key.
        if entry is not None and entry[0] is reference:
            del _FAILED_CAPTURE_REGISTRY[exception_id]

    return _drop


def _materialize_failed_capture_events(trace: Trace) -> None:
    """Drain live capture records into raw layers for failed captures.

    Parameters
    ----------
    trace:
        Partially populated trace from a failed forward pass.

    Returns
    -------
    None
        Mutates the trace raw-layer lookup structures when live events are pending.
    """

    raw_graph_ws = trace.__dict__.get("_raw_graph_ws")
    if raw_graph_ws is not None and raw_graph_ws.raw_layer_dict:
        return
    events = getattr(trace, "capture_events", None)
    if events is None or not getattr(events, "op_events", None):
        return

    from torchlens.postprocess._materialize import materialize_from_events

    materialize_from_events(trace, events)


def _raw_label(layer: Op) -> str:
    """Return the raw label for a captured layer entry.

    Parameters
    ----------
    layer:
        Raw layer pass log.

    Returns
    -------
    str
        Raw tensor label, falling back to the raw layer label.
    """

    return str(getattr(layer, "_label_raw", getattr(layer, "_layer_label_raw", "unknown")))


def _dot_escape(value: str) -> str:
    """Escape a value for a Graphviz DOT string literal.

    Parameters
    ----------
    value:
        String value to escape.

    Returns
    -------
    str
        Escaped string.
    """

    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _failure_label(exception: BaseException) -> str:
    """Return a concise failure label for DOT output.

    Parameters
    ----------
    exception:
        Original capture exception.

    Returns
    -------
    str
        Failure label including the exception type and message.
    """

    return f"{type(exception).__name__}: {safe_exception_str(exception)}"


__all__ = ["PartialTrace", "from_failed_capture"]

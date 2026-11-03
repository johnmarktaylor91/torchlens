"""Partial capture helpers for failed TorchLens forward ops."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .._capture.outcome import safe_exception_str
from .._data_classes._nonfinite import first_nonfinite_layer
from .._data_classes.field_policy import FieldPolicy

if TYPE_CHECKING:
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace
    from torchlens.debug._audit import HealthReport
    from torchlens.report._explain import ExplainReport


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
    # partials never persist (`to_trace`/`Trace.save` refuse), so both fields
    # are session-time DROP.
    _FIELD_POLICY: ClassVar[Mapping[str, Any]] = MappingProxyType(
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

        from .._capture.outcome import outcome_for

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
        """Return the layer entries captured before failure.

        An aborted-nonfinite capture whose prefix finalization ran (observe
        item 1) has already traded its raw workspace for FINALIZED records;
        those records ARE the captured prefix, served in capture order with
        their public labels populated.

        Returns
        -------
        tuple[Op, ...]
            Layer pass logs in capture order.
        """

        raw_graph_ws = self.trace.__dict__.get("_raw_graph_ws")
        if raw_graph_ws is not None:
            return tuple(getattr(raw_graph_ws, "raw_layer_dict", {}).values())
        if self.trace.__dict__.get("_nonfinite_prefix_finalized", False):
            try:
                return tuple(self.trace._layer_list)
            except Exception:  # noqa: BLE001 - a partial never raises from inspection.
                return ()
        return ()

    def narrate(self, last: int | None = None, *, select: Any = None) -> str:
        """Render the recorded frontier in the live-narration grammar.

        The headline crash workflow (snoop D5, tail 1): works when ``narrate=``
        was never enabled, because the partial the library already attaches
        to the exception carries everything a line needs. The op that raised
        is ABSENT from the record (it never produced an output); the footer
        says so instead of presenting the last recorded op as the culprit.

        Parameters
        ----------
        last:
            Keep only the last ``last`` rows.
        select:
            Optional filter: a substring, or a callable over
            :class:`torchlens.types.NarrationEvent` rows.

        Returns
        -------
        str
            Rendered narration block (no trailing newline).
        """

        from torchlens.debug._snoop._narrate import narrate_partial

        return narrate_partial(self, last=last, select=select)

    def first_nonfinite(self) -> str:
        """Return a text summary of the first raw non-finite tensor.

        Returns
        -------
        str
            Human-readable summary with layer, operation, shape, dtype, and parents.
        """

        layer = first_nonfinite_layer(self, kind="raw")
        if layer is not None:
            parents = ", ".join(str(parent) for parent in getattr(layer, "parents", None) or [])
            return (
                "First non-finite captured tensor is in "
                f"layer {_display_label(layer)} "
                f"(op {getattr(layer, 'func_name', 'unknown')}), "
                f"shape={getattr(layer, 'shape', None)}, "
                f"dtype={getattr(layer, 'dtype', None)}, parents={parents or 'none'}."
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

    def audit(self) -> HealthReport:
        """Return an evidence-backed health report for this partial capture.

        Returns
        -------
        HealthReport
            Structured failure and saved-payload findings. Full-trace checks are
            explicitly skipped because postprocessing did not complete.
        """

        from torchlens.debug._audit import audit_trace

        return audit_trace(self)

    def draw(self, path: str | None = None, **_: Any) -> str:
        """Render the failed capture as minimal Graphviz DOT source.

        Parameters
        ----------
        path:
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
            node_label = _display_label(layer)
            shape = getattr(layer, "shape", None)
            dtype = getattr(layer, "dtype", None)
            func_name = getattr(layer, "func_name", "unknown")
            label = f"{node_label}\\nop={func_name}\\nshape={shape}\\ndtype={dtype}"
            lines.append(f'  "{_dot_escape(node_label)}" [label="{_dot_escape(label)}"];')
            for parent in getattr(layer, "parents", []) or []:
                lines.append(f'  "{_dot_escape(str(parent))}" -> "{_dot_escape(node_label)}";')
        failure_label = _failure_label(self.original_exception)
        lines.append(f'  "__failure__" [shape=note, label="{_dot_escape(failure_label)}"];')
        if self.raw_layers:
            lines.append(
                f'  "{_dot_escape(_display_label(self.raw_layers[-1]))}" -> "__failure__";'
            )
        lines.append("}")
        return "\n".join(lines)

    def show(self, kind: Literal["graph", "repr", "html"] | None = None, **kwargs: Any) -> str:
        """Display this partial log using a small set of inspection modes.

        Parameters
        ----------
        kind:
            ``"graph"`` returns DOT, ``"repr"`` returns ``repr(self)``, and
            ``"html"`` returns a compact HTML fragment.
        **kwargs:
            Forwarded to ``draw`` for graph mode.

        Returns
        -------
        str
            Rendered graph, representation, or HTML fragment.
        """

        kind = kind or "graph"
        if kind == "graph":
            return self.draw(**kwargs)
        if kind == "repr":
            return repr(self)
        if kind == "html":
            return self._repr_html_()
        raise ValueError("kind must be 'graph', 'repr', or 'html'.")

    def _repr_html_(self) -> str:
        """Return a compact notebook HTML representation.

        Returns
        -------
        str
            HTML fragment summarizing the failed capture, generated through
            the CardTree presentation IR (treescope memo B1/B2) with the
            failure-first contract: the partial/failure banner is the badge
            in the always-visible header, never folded.
        """

        from ..report._notebook.cards import partial_repr_html

        return partial_repr_html(self)

    def __treescope_repr__(self, path: Any, subtree_renderer: Any) -> Any:
        """Thin lazy treescope hook (treescope memo 3.2; lane F16).

        ``PartialTrace`` is the second dataclass treescope reflects into a
        megabyte dump; see ``Trace.__treescope_repr__`` for the contract
        (``NotImplemented`` fall-through on the disabled scope or fault).
        """
        try:
            from ..bridge.treescope import treescope_repr

            return treescope_repr(self, path, subtree_renderer)
        except Exception:  # noqa: BLE001 - degrade to treescope's default
            return NotImplemented

    def __repr__(self) -> str:
        """Badge-first one-line representation (F10; lovely matrix).

        ``partial`` leads -- state precedes detail when state invalidates
        detail (voice rule 7).

        Returns
        -------
        str
            One line: badge, captured-prefix size, failing exception type.
        """

        return (
            f"PartialTrace [partial] captured_prefix={len(self.raw_layers)} ops, "
            f"error={type(self.original_exception).__name__}"
        )

    def __str__(self) -> str:
        """Bounded failure-first card: badge, prefix, last op, error, exits."""

        lines = [self.__repr__()]
        raw = self.raw_layers
        if raw:
            first = getattr(raw[0], "_label_raw", None) or getattr(raw[0], "layer_label", "?")
            last = getattr(raw[-1], "_label_raw", None) or getattr(raw[-1], "layer_label", "?")
            lines.append(f"  captured {first} .. {last} (last success)")
        from .._utils.fail_open import fail_open

        nonfinite = fail_open(lambda: self.first_nonfinite(), lambda _error: None)
        if nonfinite and not nonfinite.startswith("No non-finite"):
            lines.append(f"  {nonfinite.splitlines()[0]}")
        error_text = str(self.original_exception).splitlines()
        if error_text:
            lines.append(f"  error: {error_text[0][:100]}")
        lines.append("  More: .raw_layers  .first_nonfinite()  .audit()  .draw()")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # F09 (sumfam D25/item 15): the five report-family members that used
    # to die with bare AttributeError refuse TYPED, pointing at the two
    # surfaces that DO answer on a partial capture. The interim contract:
    # degraded report or typed refusal, never AttributeError. The
    # capture-side label/identity finalization that would enable real
    # degraded tables is a capture-fence work item, not this wrapper's.

    def _refuse_partial_member(self, member: str) -> Any:
        """Raise the typed partial-capture refusal for one family member."""

        from .._errors import InvalidArgumentError

        raise InvalidArgumentError(
            f"{member} is unavailable on a PartialTrace: the failed capture never reached "
            "label/identity finalization, so a table here would be a hollow valid-schema "
            "zero report.",
            code="partial_trace_member_unavailable",
            remedy=(
                "Use partial.explain() for the failure diagnosis (what ran, the "
                "failing op, the boundary) or partial.audit() for the degraded findings; "
                "partial.raw_layers holds the committed per-op records."
            ),
        )

    def summary(self, **_kwargs: Any) -> Any:
        """Typed refusal (D25): orientation needs finalized identities."""

        return self._refuse_partial_member("summary()")

    def profile(self, *_args: Any, **_kwargs: Any) -> Any:
        """Typed refusal (D25): resource ranking needs finalized identities."""

        return self._refuse_partial_member("profile()")

    def to_pandas(self, **_kwargs: Any) -> Any:
        """Typed refusal (D25): the per-op ledger needs finalized identities."""

        return self._refuse_partial_member("to_pandas()")

    def to_agent_json(self, **_kwargs: Any) -> Any:
        """Typed refusal (D25): the machine map needs finalized identities."""

        return self._refuse_partial_member("to_agent_json()")

    def output_table(self, **_kwargs: Any) -> Any:
        """Typed refusal (D25): no outputs exist on a failed forward."""

        return self._refuse_partial_member("output_table()")

    def explain(
        self,
        *,
        audience: Literal["researcher", "practitioner", "auto"] = "auto",
        max_tokens: int | None = None,
    ) -> ExplainReport:
        """Explain this failed capture (what ran, the failing op, the boundary) in plain language (DOCUMENTED-UNSTABLE).

        Parameters
        ----------
        audience:
            Report style: ``"researcher"`` adds graph-pattern detail,
            ``"practitioner"`` emphasizes operational status, ``"auto"``
            balances the two.
        max_tokens:
            Optional token budget. Whole sections drop low-value-first until
            the report fits, every drop is disclosed, and the capture-status
            section never drops.

        Returns
        -------
        Any
            ``torchlens.report.ExplainReport``, a ``str`` of multi-section prose.
            The machine-readable projection of the same facts is
            ``trace.to_agent_json()``.
        """

        from ..report._explain import explain

        return explain(self, audience, max_tokens=max_tokens)

    def capability_card(self) -> Any:
        """The D24 capability card: which family members work on THIS object."""

        from ..report._registry import capability_card

        return capability_card(self)


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

    from torchlens._postprocess._materialize import materialize_from_events

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


def _display_label(layer: Op) -> str:
    """Return the public display label for a captured layer entry.

    Finalized records (the aborted-nonfinite prefix) display their canonical
    public label; un-finalized raw entries display their raw identity -- the
    only identity a dead partial has -- which is honest disclosure, never a
    stripped or arithmetically remapped public spelling (observe item 1).

    Parameters
    ----------
    layer:
        Layer pass log served by :attr:`PartialTrace.raw_layers`.

    Returns
    -------
    str
        Canonical public label when finalized, else the raw identity.
    """

    from .._capture._nonfinite_prefix import canonical_public_label

    label = canonical_public_label(layer)
    return label if label else _raw_label(layer)


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


__all__ = ["PartialTrace"]

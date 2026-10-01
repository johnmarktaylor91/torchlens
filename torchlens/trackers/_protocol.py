"""The DOCUMENTED sink protocol: capabilities, delivery states (memo 3.1/3.10).

A sink is a SERIALIZER over the shared emission view, never an engine.
The protocol is documented so third parties (Neptune, Comet, in-house) write
their own receivers -- TorchLens owns the engine, they own the sink. A
requested capability a sink does not advertise refuses BY NAME before any
partial emission; a sink failure latches and can never corrupt collection.

Delivery states (memo 3.10) are the honest vocabulary for "where did my
data go": ``computed`` (record produced), ``emitted`` (a named sink accepted
it and flush/close succeeded), ``relay_configured`` (best-effort
configuration evidence), ``relay_verified`` (downstream observer evidence).
Docs may say a TB emission "can feed" wandb/ClearML; runtime reports never
claim downstream presence above the evidence tier they hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ._errors import SinkProtocolError
from ._records import HistogramPoint, ScalarPoint, TextPoint

__tl_layer__ = "L8"

#: The closed capability vocabulary (memo 3.1). ``embedding`` is reserved
#: plumbing for the projector bridge (deferred table, section 9): declared
#: now so adding it later is a value change, not a protocol change.
CAPABILITIES = (
    "scalar",
    "raw_histogram",
    "text_manifest",
    "image",
    "graph",
    "run_metadata",
    "structured_event",
    "embedding",
)

#: The closed delivery-state vocabulary (memo 3.10).
DELIVERY_STATES = ("computed", "emitted", "relay_configured", "relay_verified")


@runtime_checkable
class Sink(Protocol):
    """The sink protocol every receiver implements (spellings unstable).

    Methods are only ever called for capabilities the sink advertises;
    ``require_capabilities`` runs before the first emission. ``flush`` and
    ``close`` must be safe to call once each at teardown; a failure in any
    method should raise (the engine latches the sink failed and preserves
    the partial report -- it never retries silently).
    """

    def capabilities(self) -> frozenset[str]:
        """The subset of :data:`CAPABILITIES` this sink can accept."""
        ...

    def emit_scalar(self, point: ScalarPoint) -> None:
        """Accept one scalar point."""
        ...

    def emit_histogram(self, point: HistogramPoint) -> None:
        """Accept one raw histogram (exact counts on explicit edges)."""
        ...

    def emit_text(self, point: TextPoint) -> None:
        """Accept one text payload (manifest, disclosure)."""
        ...

    def flush(self) -> None:
        """Force buffered emissions to durable/visible state."""
        ...

    def close(self) -> None:
        """Release the sink; idempotent."""
        ...


def sink_name(sink: Any) -> str:
    """Best-effort display name for one sink object."""

    return type(sink).__name__


def require_capabilities(sink: Any, needed: tuple[str, ...]) -> None:
    """Refuse by name when ``sink`` lacks a requested capability (3.1).

    Runs at attach, BEFORE any emission: a run that would lose one requested
    signal family must fail while the user is looking at the console, not
    after ten hours of partial panels.
    """

    capabilities_method = getattr(sink, "capabilities", None)
    if not callable(capabilities_method):
        raise SinkProtocolError(
            f"{sink!r} is not a tracker sink: it has no capabilities() "
            "method. Sinks are explicit receivers implementing the "
            "documented protocol; raw writer objects belong behind "
            "TensorBoardSink/WandbSink adapters.",
            code="tracker_sink_invalid",
            sink=repr(type(sink)),
            remedy=(
                "Wrap the writer: TensorBoardSink(writer_or_logdir), "
                "WandbSink(run), JSONLSink(path), or implement capabilities()."
            ),
        )
    advertised = frozenset(capabilities_method())
    unknown = advertised - frozenset(CAPABILITIES)
    if unknown:
        raise SinkProtocolError(
            f"{sink_name(sink)} advertises unknown capabilities "
            f"{sorted(unknown)}; the vocabulary is closed so engines and "
            "receivers can never silently disagree about what a word grants.",
            code="tracker_sink_invalid",
            sink=sink_name(sink),
            unknown=tuple(sorted(unknown)),
            remedy=f"Advertise only capabilities from {CAPABILITIES}.",
        )
    missing = [need for need in needed if need not in advertised]
    if missing:
        raise SinkProtocolError(
            f"{sink_name(sink)} does not support {missing} (advertised: "
            f"{sorted(advertised)}). Requested-but-unsupported refuses "
            "before partial emission: a run that silently drops one signal "
            "family is the empty-panel failure class this package exists to "
            "kill.",
            code="tracker_sink_capability_missing",
            sink=sink_name(sink),
            missing=tuple(missing),
            advertised=tuple(sorted(advertised)),
            remedy=(
                "Drop the unsupported signal, or use a sink that carries it "
                "(the JSONL sink accepts every non-image capability)."
            ),
        )


@dataclass
class SinkLedger:
    """Per-sink delivery accounting for the close report (memo 3.13).

    ``name`` is the DISPLAY name (the sink's class name, disambiguated with
    ``#2``, ``#3``, ... when several sinks of one class share a session);
    ``sink_id`` is the object identity the row is keyed on. ``emitted_data_points``
    counts DATA-family scalars and histograms only -- run-health, manifest,
    and check rows never count toward "the dashboard has data".
    """

    name: str
    sink_id: int = 0
    emitted_scalars: int = 0
    emitted_histograms: int = 0
    emitted_texts: int = 0
    emitted_data_points: int = 0
    failed: bool = False
    failure: str | None = None
    relay_state: str = "computed"
    relay_detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render the ledger row for reports."""

        return {
            "sink": self.name,
            "emitted_scalars": self.emitted_scalars,
            "emitted_histograms": self.emitted_histograms,
            "emitted_texts": self.emitted_texts,
            "emitted_data_points": self.emitted_data_points,
            "failed": self.failed,
            "failure": self.failure,
            "relay_state": self.relay_state,
            "relay_detail": self.relay_detail,
        }


@dataclass
class EmissionLedger:
    """Whole-run delivery accounting: one row per sink OBJECT, shared counters.

    Rows are keyed by ``id(sink)`` (the session holds every sink for its whole
    lifetime, so identities are stable), never by class name: two JSONL sinks
    (local + NFS is the realistic config) used to share ONE row, so a failure
    in either latched BOTH -- the healthy sink was silently starved and the
    counts merged (AUD-CODE 2.13).
    """

    sinks: dict[int, SinkLedger] = field(default_factory=dict)
    computed_scalars: int = 0
    computed_histograms: int = 0
    dropped_points: int = 0

    def row(self, sink: Any) -> SinkLedger:
        """The (created-on-first-use) ledger row for one sink object."""

        key = id(sink)
        row = self.sinks.get(key)
        if row is None:
            base = sink_name(sink)
            same_class = sum(
                1 for other in self.sinks.values() if other.name.split("#", 1)[0] == base
            )
            name = base if same_class == 0 else f"{base}#{same_class + 1}"
            row = SinkLedger(name=name, sink_id=key)
            self.sinks[key] = row
        return row


__all__ = [
    "CAPABILITIES",
    "DELIVERY_STATES",
    "EmissionLedger",
    "Sink",
    "SinkLedger",
    "require_capabilities",
    "sink_name",
]

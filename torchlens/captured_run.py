"""Shared public base types for TorchLens captured runs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable
from weakref import WeakKeyDictionary

from .ir import CaptureEvents

ActivationT = TypeVar("ActivationT")
_EVENT_STREAMS: WeakKeyDictionary[object, CaptureEvents] = WeakKeyDictionary()


def remember_event_stream(run: object, events: CaptureEvents) -> None:
    """Retain a raw event stream without adding public instance fields."""

    _EVENT_STREAMS[run] = events


@runtime_checkable
class ActivationLookup(Protocol[ActivationT]):
    """Protocol for raw-label/pass/address activation lookup consumers."""

    by_pass: dict[int, list[int]]
    orphan_records: list[dict[str, Any]]

    def activation_by_raw_label(
        self, raw_label: str, *, pass_index: int | None = None
    ) -> ActivationT:
        """Return one activation record by raw label and optional pass index."""

    def activations_by_pass(self, pass_index: int) -> list[ActivationT]:
        """Return activation records captured for one model-call pass."""

    def activations_by_address(self, address: str) -> list[ActivationT]:
        """Return activation records captured at a module/buffer address."""


class CapturedRun:
    """Small shared base for uncooked and cooked capture projections."""

    @property
    def event_stream(self) -> CaptureEvents | None:
        """Return the raw capture event bundle when retained."""

        events = getattr(self, "capture_events", None)
        if events is not None:
            return events
        events = _EVENT_STREAMS.get(self)
        if events is not None:
            return events
        return getattr(self, "_capture_events", None)

    @property
    def op_events(self) -> tuple[Any, ...]:
        """Return the raw operation event stream."""

        events = self.event_stream
        if events is None:
            return ()
        return tuple(events.op_events)

    @property
    def sibling_events(self) -> tuple[Any, ...]:
        """Return non-op sibling capture events in chronological groups."""

        events = self.event_stream
        if events is None:
            return ()
        return (
            *events.module_events,
            *events.module_prep_events,
            *events.module_enter_events,
            *events.module_exit_events,
            *events.conditional_events,
            *events.output_version_events,
        )

    @property
    def activation_payloads_by_raw_label(self) -> dict[str, Any]:
        """Return retained payloads keyed by raw operation label."""

        payloads: dict[str, Any] = {}
        for activation in self._iter_activation_records():
            raw_label = self._activation_raw_label(activation)
            if raw_label is None:
                continue
            payload = self._activation_payload(activation)
            if payload is not None:
                payloads[raw_label] = payload
        return payloads

    def activation_by_raw_label(self, raw_label: str, *, pass_index: int | None = None) -> Any:
        """Return one activation record by raw label and optional pass index.

        Parameters
        ----------
        raw_label:
            Raw operation label emitted during capture.
        pass_index:
            Optional 1-based model-call pass index.

        Returns
        -------
        Any
            Projection-specific activation record.

        Raises
        ------
        KeyError
            If no activation matches the supplied lookup.
        """

        matches = [
            activation
            for activation in self._iter_activation_records()
            if self._activation_raw_label(activation) == raw_label
            and (pass_index is None or self._activation_pass_index(activation) == pass_index)
        ]
        if not matches:
            pass_suffix = "" if pass_index is None else f" on pass {pass_index}"
            raise KeyError(f"No activation for raw label {raw_label!r}{pass_suffix}.")
        if len(matches) > 1 and pass_index is None:
            raise KeyError(
                f"Raw label {raw_label!r} matched {len(matches)} activations; pass pass_index."
            )
        return matches[0]

    def activations_by_pass(self, pass_index: int) -> list[Any]:
        """Return activation records captured for one model-call pass."""

        return [
            activation
            for activation in self._iter_activation_records()
            if self._activation_pass_index(activation) == pass_index
        ]

    def activations_by_address(self, address: str) -> list[Any]:
        """Return activation records captured at a module/buffer address."""

        return [
            activation
            for activation in self._iter_activation_records()
            if self._activation_address(activation) == address
        ]

    def _iter_activation_records(self) -> Iterable[Any]:
        """Iterate projection-specific retained activation records."""

        if hasattr(self, "records"):
            ensure_records = getattr(self, "_ensure_records", None)
            if ensure_records is not None:
                ensure_records()
            return tuple(getattr(self, "records", ()))
        return tuple(
            op
            for op in getattr(self, "layer_list", ())
            if getattr(op, "has_saved_activation", False)
        )

    def _activation_raw_label(self, activation: Any) -> str | None:
        """Return the raw label for one projection-specific activation."""

        ctx = getattr(activation, "ctx", None)
        if ctx is not None:
            return getattr(ctx, "raw_label", None) or getattr(ctx, "label", None)
        return getattr(activation, "_label_raw", None)

    def _activation_pass_index(self, activation: Any) -> int | None:
        """Return the pass index for one projection-specific activation."""

        ctx = getattr(activation, "ctx", None)
        if ctx is not None:
            return getattr(ctx, "pass_index", None)
        return getattr(activation, "pass_index", None)

    def _activation_address(self, activation: Any) -> str | None:
        """Return the address for one projection-specific activation."""

        ctx = getattr(activation, "ctx", None)
        if ctx is not None:
            return getattr(ctx, "address", None)
        module = getattr(activation, "module", None)
        return module if isinstance(module, str) else None

    def _activation_payload(self, activation: Any) -> Any:
        """Return the retained payload for one projection-specific activation."""

        if hasattr(activation, "ram_payload"):
            return getattr(activation, "ram_payload", None)
        return getattr(activation, "out", None)

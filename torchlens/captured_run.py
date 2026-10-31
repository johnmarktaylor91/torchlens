"""Shared public base types for TorchLens captured runs."""

from __future__ import annotations

import gc
from collections.abc import Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable

from torchlens._ir.capture_events import CaptureEvents

ActivationT = TypeVar("ActivationT")

#: Addresses listed in an ``activation_address_unmatched`` refusal before eliding.
_ADDRESS_LISTING_LIMIT = 20


def forget_event_stream(run: object) -> None:
    """Release the run-owned raw event stream for an explicitly cleaned run.

    The run (Trace/Recording) strongly owns its event stream through instance
    attributes -- ``_capture_events``, and on capture-time traces also the
    ``capture_events`` attribute holding the same stream; the former
    module-level ``_EVENT_STREAMS`` weak side registry is gone. Releasing pops
    the ``_capture_events`` instance attribute and clears the stream's working
    lanes, so a ``capture_events`` alias (or any outside holder) is left with
    an empty buffer rather than a live projection.
    """

    run_dict = getattr(run, "__dict__", None)
    events = run_dict.pop("_capture_events", None) if isinstance(run_dict, dict) else None
    if isinstance(events, CaptureEvents):
        events.release_working_projection()
        gc.collect(0)


@runtime_checkable
class _ActivationLookup(Protocol[ActivationT]):
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


class TraceLike:
    """Small shared base for uncooked and cooked capture projections."""

    @property
    def event_stream(self) -> CaptureEvents | None:
        """Return the raw capture event bundle when retained."""

        try:
            events = object.__getattribute__(self, "capture_events")
        except AttributeError:
            events = None
        if events is not None:
            return events
        try:
            events = object.__getattribute__(self, "_capture_events")
        except AttributeError:
            events = None
        return events

    @property
    def op_events(self) -> tuple[Any, ...]:
        """Return the raw operation event stream."""

        events = self.event_stream
        if events is None:
            return ()
        if hasattr(events, "amended_op_records"):
            return tuple(events.amended_op_records())
        return tuple(events.op_events)

    @property
    def backward_events(self) -> tuple[Any, ...]:
        """Return the runtime-only backward sidecar event stream."""

        events = self.event_stream
        if events is None:
            return ()
        return tuple(getattr(events, "backward_events", ()))

    @property
    def sibling_events(self) -> tuple[Any, ...]:
        """Return non-op sibling capture events in chronological groups."""

        events = self.event_stream
        if events is None:
            return ()
        return (
            *events.module_prep_events,
            *events.module_enter_events,
            *events.module_exit_events,
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
                existing_payload = payloads.get(raw_label)
                if existing_payload is None:
                    payloads[raw_label] = payload
                elif isinstance(existing_payload, list):
                    existing_payload.append(payload)
                else:
                    payloads[raw_label] = [existing_payload, payload]
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
        """Return activation records captured at a module/buffer address.

        Parameters
        ----------
        address:
            Module or buffer address, e.g. ``"layer1.0.conv1"``.

        Returns
        -------
        list[Any]
            Projection-specific activation records at ``address``, in capture order.

        Raises
        ------
        InvalidArgumentError
            ``activation_address_unmatched`` when no retained activation sits at
            ``address``; the error lists the addresses that do exist.
        """

        records = self._iter_activation_records()
        matches = [
            activation for activation in records if self._activation_address(activation) == address
        ]
        if matches:
            return matches
        from ._errors import InvalidArgumentError

        known = sorted(
            {
                str(found)
                for found in (self._activation_address(activation) for activation in records)
                if found is not None
            }
        )
        shown = ", ".join(repr(name) for name in known[:_ADDRESS_LISTING_LIMIT])
        more = (
            f" (and {len(known) - _ADDRESS_LISTING_LIMIT} more)"
            if len(known) > _ADDRESS_LISTING_LIMIT
            else ""
        )
        raise InvalidArgumentError(
            f"no retained activation at address {address!r}; addresses with retained "
            f"activations: {shown or 'none'}{more}",
            code="activation_address_unmatched",
            remedy=(
                f"pass one of the listed addresses, e.g. activations_by_address({known[0]!r})"
                if known
                else "capture with activations saved (the default save keeps every layer), "
                "then look the address up again"
            ),
            argument="address",
            known_addresses=tuple(known),
        )

    def _iter_activation_records(self) -> Iterable[Any]:
        """Iterate projection-specific retained activation records."""

        if hasattr(self, "records"):
            ensure_records = getattr(self, "_ensure_records", None)
            if ensure_records is not None:
                ensure_records()
            return tuple(getattr(self, "records", ()))
        return tuple(
            op
            for op in getattr(self, "_layer_list", ())
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
        """Return the module or buffer address for one projection-specific activation.

        Trace ops store their innermost module as a call label (``"fc:1"``) and
        buffer ops their buffer address; both report the bare address.
        """

        ctx = getattr(activation, "ctx", None)
        if ctx is not None:
            return getattr(ctx, "address", None)
        buffer_address = getattr(activation, "address", None)
        if getattr(activation, "is_buffer", False) and isinstance(buffer_address, str):
            return buffer_address
        module = getattr(activation, "module", None)
        if not isinstance(module, str):
            return None
        address, separator, call_index = module.rpartition(":")
        return address if separator and call_index.isdigit() else module

    def _activation_payload(self, activation: Any) -> Any:
        """Return the retained payload for one projection-specific activation."""

        if hasattr(activation, "ram_payload"):
            return getattr(activation, "ram_payload", None)
        return getattr(activation, "out", None)


__all__: list[str] = []

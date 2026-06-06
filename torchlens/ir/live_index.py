"""Event-backed live index for capture-time graph queries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .events import ModuleFrame, OpEvent


class LiveIndexWindowError(KeyError):
    """Raised when a live-index query targets an unavailable raw label."""


@dataclass(slots=True)
class LiveIndex:
    """Shared-engine-owned index over emitted operation events.

    The index is populated only from ``OpEvent`` objects and small sibling-event
    counters. It intentionally does not expose mutable ``Op`` field dictionaries.
    """

    by_raw_label: dict[str, OpEvent] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    children_by_parent: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    module_entry_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    module_entries_by_label: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def append(self, event: OpEvent) -> None:
        """Index one emitted operation event.

        Parameters
        ----------
        event
            Operation event emitted by the active backend.

        Returns
        -------
        None
            Mutates the live index in place.
        """

        self.by_raw_label[event.label_raw] = event
        self.labels.append(event.label_raw)
        for edge in event.parents:
            children = self.children_by_parent[edge.parent_label_raw]
            if event.label_raw not in children:
                children.append(event.label_raw)

    def replace(self, event: OpEvent) -> None:
        """Replace a previously indexed event.

        Parameters
        ----------
        event
            Updated operation event.

        Returns
        -------
        None
            Mutates event lookup and recomputes child edges.
        """

        self.by_raw_label[event.label_raw] = event
        self.rebuild_edges()

    def rebuild_edges(self) -> None:
        """Rebuild parent-to-child edges from indexed events.

        Returns
        -------
        None
            Mutates ``children_by_parent``.
        """

        self.children_by_parent.clear()
        for label in self.labels:
            event = self.by_raw_label[label]
            for edge in event.parents:
                children = self.children_by_parent[edge.parent_label_raw]
                if event.label_raw not in children:
                    children.append(event.label_raw)

    def require_event(self, label_raw: str) -> OpEvent:
        """Return an event or raise an explicit out-of-window error.

        Parameters
        ----------
        label_raw
            Raw operation label to resolve.

        Returns
        -------
        OpEvent
            Indexed event.
        """

        try:
            return self.by_raw_label[label_raw]
        except KeyError as exc:
            raise LiveIndexWindowError(
                f"{label_raw!r} is not available in the active live index."
            ) from exc

    def parents(self, label_raw: str) -> tuple[str, ...]:
        """Return raw parent labels for one event.

        Parameters
        ----------
        label_raw
            Raw operation label to inspect.

        Returns
        -------
        tuple[str, ...]
            Parent raw labels.
        """

        event = self.require_event(label_raw)
        return tuple(edge.parent_label_raw for edge in event.parents)

    def children(self, label_raw: str) -> tuple[str, ...]:
        """Return raw child labels for one indexed event.

        Parameters
        ----------
        label_raw
            Raw operation label to inspect.

        Returns
        -------
        tuple[str, ...]
            Child raw labels currently known in the active window.
        """

        self.require_event(label_raw)
        return tuple(self.children_by_parent.get(label_raw, ()))

    def has_payload(self, label_raw: str) -> bool:
        """Return whether an event carries a saved activation payload.

        Parameters
        ----------
        label_raw
            Raw operation label to inspect.

        Returns
        -------
        bool
            True when the event output has a saved activation.
        """

        return bool(self.require_event(label_raw).output.has_saved_activation)

    def payload(self, label_raw: str) -> Any:
        """Return an event payload or raise if unavailable.

        Parameters
        ----------
        label_raw
            Raw operation label to inspect.

        Returns
        -------
        Any
            Saved activation payload.
        """

        event = self.require_event(label_raw)
        if not event.output.has_saved_activation:
            raise LiveIndexWindowError(f"{label_raw!r} has no saved activation payload.")
        return event.output.tensor.payload

    def module_stack(self, label_raw: str) -> tuple[ModuleFrame, ...]:
        """Return the module stack carried by an event.

        Parameters
        ----------
        label_raw
            Raw operation label to inspect.

        Returns
        -------
        tuple[ModuleFrame, ...]
            Module-frame snapshot.
        """

        return self.require_event(label_raw).module_stack

    def note_module_entry(self, module_id: int, label_raw: str, address: str) -> None:
        """Record that a raw label entered a module during this pass.

        Parameters
        ----------
        module_id
            ``id(module)`` for the active module.
        label_raw
            Raw operation label entering the module.
        address
            TorchLens module address.

        Returns
        -------
        None
            Mutates module-entry counters only.
        """

        self.module_entry_counts[module_id] += 1
        self.module_entries_by_label[label_raw].append(address)

    def module_entry_count(self, module_id: int) -> int:
        """Return how many tensor labels have entered a module call.

        Parameters
        ----------
        module_id
            ``id(module)`` for the active module.

        Returns
        -------
        int
            Number of module-entry labels observed for the active session.
        """

        return self.module_entry_counts[module_id]

    def module_stack_membership(self, label_raw: str) -> tuple[str, ...]:
        """Return module addresses recorded for an input label at entry time.

        Parameters
        ----------
        label_raw
            Raw operation label to inspect.

        Returns
        -------
        tuple[str, ...]
            Module addresses in entry order.
        """

        return tuple(self.module_entries_by_label.get(label_raw, ()))

    def clear(self) -> None:
        """Clear all indexed events and sibling counters.

        Returns
        -------
        None
            Mutates the live index in place.
        """

        self.by_raw_label.clear()
        self.labels.clear()
        self.children_by_parent.clear()
        self.module_entry_counts.clear()
        self.module_entries_by_label.clear()

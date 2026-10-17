"""Event-backed live index for capture-time graph queries."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .events import ModuleFrame


class LiveIndexWindowError(KeyError):
    """Raised when a live-index query targets an unavailable raw label."""


@dataclass(slots=True)
class LiveIndex:
    """Shared-engine-owned index over emitted operation events.

    The index is populated only from journal op records (compat ``OpEvent``
    or decomposed ``OpRecord``, read through the shared flat-name protocol)
    and small sibling-event counters. It intentionally does not expose
    mutable ``Op`` field dictionaries.
    """

    by_raw_label: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    children_by_parent: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    module_entry_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    module_entries_by_label: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def append(self, event: Any) -> None:
        """Index one emitted operation event.

        A raw label re-seen at this boundary (a multi-pass ``Recorder`` folding
        each pass's journal in, or a failed-partial fastlog recording
        concatenating op-event streams) is treated as an atomic in-place
        replacement rather than a second insertion: the label is not
        double-listed and no stale parent->child edge from the prior event
        survives. This is identical to :meth:`replace` for a re-seen label.

        Parameters
        ----------
        event
            Operation event emitted by the active backend.

        Returns
        -------
        None
            Mutates the live index in place.
        """

        label = event.label_raw
        if label in self.by_raw_label:
            previous = self.by_raw_label[label]
            self.by_raw_label[label] = event
            self._move_child_edges(label, previous, event)
            return

        self.by_raw_label[label] = event
        self.labels.append(label)
        for edge in event.parents:
            children = self.children_by_parent[edge.parent_label_raw]
            if label not in children:
                children.append(label)

    def replace(self, event: Any) -> None:
        """Replace a previously indexed event.

        Only the child lists of parents the label gained or lost change, and
        they end exactly as :meth:`rebuild_edges` would leave them: each list
        in ``labels`` order, and no list a rebuild would not create. The
        iteration order of ``children_by_parent`` keys is not part of that
        contract.

        Parameters
        ----------
        event
            Updated operation event.

        Returns
        -------
        None
            Mutates event lookup and the affected child edges.
        """

        label = event.label_raw
        previous = self.by_raw_label.get(label)
        self.by_raw_label[label] = event
        self._move_child_edges(label, previous, event)

    def _move_child_edges(self, label: str, previous: Any, event: Any) -> None:
        """Move one label's child edges from ``previous``'s parents to ``event``'s.

        A label only sits in a parent's child list when it is in ``labels``,
        so a label missing from ``labels`` gains no edges, as in a rebuild.
        """

        old: dict[str, None] = {}
        if previous is not None:
            old = {edge.parent_label_raw: None for edge in previous.parents}
        new = {edge.parent_label_raw: None for edge in event.parents}
        if old == new:
            return
        for parent in old:
            if parent not in new:
                self._drop_child_edge(label, parent)
        added = [parent for parent in new if parent not in old]
        if added:
            self._add_child_edges(label, added)

    def _drop_child_edge(self, label: str, parent: str) -> None:
        """Remove ``label`` from ``parent``'s child list, dropping a list it empties."""

        children = self.children_by_parent.get(parent)
        if children is None or label not in children:
            return
        children.remove(label)
        if not children:
            del self.children_by_parent[parent]

    def _add_child_edges(self, label: str, parents: list[str]) -> None:
        """Insert ``label`` into each parent's child list at its ``labels`` position."""

        try:
            position = self.labels.index(label)
        except ValueError:
            return
        for parent in parents:
            children = self.children_by_parent[parent]
            children.insert(bisect_left(children, position, key=self.labels.index), label)

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

    def require_event(self, label_raw: str) -> Any:
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

    def copy(self) -> LiveIndex:
        """Return a structural copy holding the same event references.

        All container fields are duplicated into fresh objects (nested lists
        included) so that in-place mutation of the copy -- notably
        ``clear()`` during materialization -- cannot reach back and corrupt the
        source index. The indexed ``OpEvent`` objects are frozen and shared by
        reference; only the containers are new.

        Returns
        -------
        LiveIndex
            Independent index over the same events.
        """

        return LiveIndex(
            by_raw_label=dict(self.by_raw_label),
            labels=list(self.labels),
            children_by_parent=defaultdict(
                list, {key: list(value) for key, value in self.children_by_parent.items()}
            ),
            module_entry_counts=defaultdict(int, self.module_entry_counts),
            module_entries_by_label=defaultdict(
                list, {key: list(value) for key, value in self.module_entries_by_label.items()}
            ),
        )

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

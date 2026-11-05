"""Region instance connectivity and re-entry path search."""

from __future__ import annotations

from typing import Any


def _connected_instances(
    members: tuple[str, ...], internal_edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, ...], ...]:
    """Partition members into weakly-connected components (pass instances)."""

    neighbors: dict[str, set[str]] = {label: set() for label in members}
    for parent, child in internal_edges:
        neighbors[parent].add(child)
        neighbors[child].add(parent)
    seen: set[str] = set()
    components: list[tuple[str, ...]] = []
    for label in members:  # members are already execution-ordered
        if label in seen:
            continue
        stack, component = [label], []
        seen.add(label)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in neighbors[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        order = {member: position for position, member in enumerate(members)}
        components.append(tuple(sorted(component, key=order.__getitem__)))
    return tuple(components)


def _offending_reentry_path(graph: Any, members: frozenset[str], closure: frozenset[str]) -> str:
    """Find one leave-and-re-enter path witnessing a convexity violation."""

    outside = closure - members
    for start in sorted(outside):
        # walk backward to a member, forward to a member: start sits on a
        # member -> outside -> member path by construction of the closure.
        back: list[str] = [start]
        node = start
        while node not in members:
            inside_parents = [p for p in sorted(graph.parents.get(node, ())) if p in closure]
            if not inside_parents:
                break
            node = inside_parents[0]
            back.append(node)
        if node not in members:
            continue
        forward: list[str] = []
        node = start
        while node not in members:
            inside_children = [c for c in sorted(graph.children.get(node, ())) if c in closure]
            if not inside_children:
                break
            node = inside_children[0]
            forward.append(node)
        if node not in members:
            continue
        return " -> ".join(reversed(back)) + " -> " + " -> ".join(forward)
    return "(path reconstruction unavailable; offending ops: " + ", ".join(sorted(outside)) + ")"

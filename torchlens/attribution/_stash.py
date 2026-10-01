"""Forward/backward hook pairing store: the permanent LRP-adjacent surface.

Attrib memo D30: the ENTIRE permanent surface TorchLens ships for
relevance-rule recipes is a stash/fetch mechanism whose pairing is keyed on
stable site/call identity -- NO rule registry, NO composites, NO canonizers,
NO architecture support table (twice reaffirmed). Maintained research-rule
catalogs remain the LRP ecosystem's territory; TorchLens supplies the
mechanism, and the epsilon-LRP litmus recipe in
``docs/recipes/lrp_epsilon_litmus.md`` shows it in use.

Pairing rule: forward firings of one site push in execution order; backward
firings pop in REVERSE execution order (autograd visits a reused module's
calls last-first), so the store pairs a site's k-th forward firing with its
k-th-from-last backward firing via LIFO pop. The reused-module pairing is
pinned by test.

The incoming gradient tuple itself needs no accessor here: plain
``register_full_backward_hook`` hands it to the recipe directly. The
intervention-engine ``hook.stash`` spelling and its grad-output accessor ride
the externally-owned engine-defect conversation (attrib memo section 6:
F4/F5/S10/S11) and adopt THIS store's semantics when they land.
"""

from __future__ import annotations

from typing import Any

from torch.nn import Module

from torchlens.attribution._result import AttributionError


class SiteStash:
    """Pair forward-hook facts with backward-hook firings per site.

    One instance serves one forward+backward pass. Keys are the SITE (module
    object identity) and a fact name; values stack per firing so reused
    modules pair correctly.
    """

    def __init__(self) -> None:
        """Create an empty store."""

        self._store: dict[tuple[int, str], list[Any]] = {}
        self._site_names: dict[int, str] = {}
        self._firing_counts: dict[int, int] = {}

    def register_site(self, site: Module, label: str) -> None:
        """Record a human-readable label for one site (for diagnostics).

        Parameters
        ----------
        site
            Module whose firings will be stashed.
        label
            Dotted module path (or any stable human-readable name).
        """

        self._site_names[id(site)] = label

    def label_of(self, site: Module) -> str:
        """Return the registered label of ``site`` (or its class name)."""

        return self._site_names.get(id(site), type(site).__name__)

    def mark_firing(self, site: Module) -> int:
        """Record one forward firing of ``site`` and return its ordinal.

        Parameters
        ----------
        site
            Module that fired.

        Returns
        -------
        int
            Zero-based firing ordinal within this pass.
        """

        ordinal = self._firing_counts.get(id(site), 0)
        self._firing_counts[id(site)] = ordinal + 1
        return ordinal

    def stash(self, site: Module, name: str, value: Any) -> None:
        """Store one fact for the current forward firing of ``site``.

        Parameters
        ----------
        site
            Module the fact belongs to.
        name
            Fact name (``"input"``, ``"output"``, ...).
        value
            The fact.
        """

        self._store.setdefault((id(site), name), []).append(value)

    def fetch(self, site: Module, name: str) -> Any:
        """Pop the fact paired with the current BACKWARD firing of ``site``.

        Backward firings arrive in reverse execution order, so the LIFO pop
        pairs firing ``k`` of a reused module with its matching stash.

        Parameters
        ----------
        site
            Module whose backward hook is firing.
        name
            Fact name stashed during the paired forward firing.

        Returns
        -------
        Any
            The paired fact.

        Raises
        ------
        AttributionError
            If no stashed fact remains for this site/name (an unpaired
            backward firing, or a fact the forward hook never stashed).
        """

        stack = self._store.get((id(site), name))
        if not stack:
            raise AttributionError(
                f"no stashed fact {name!r} remains for site "
                f"{self.label_of(site)!r}: either the backward hook fired "
                "more often than the forward hook stashed, or the fact was "
                "never stashed. Remedy: stash the fact in the site's forward "
                "hook, once per firing.",
                code="lrp_stash_unpaired",
            )
        return stack.pop()

    def leftovers(self) -> dict[str, int]:
        """Return stashed facts never fetched (site label -> count).

        A nonzero leftover after a full backward pass means a forward firing
        whose backward never arrived (a site outside the scored path); the
        recipe DISCLOSES these rather than silently dropping them.
        """

        remaining: dict[str, int] = {}
        for (site_id, _name), stack in self._store.items():
            if stack:
                label = self._site_names.get(site_id, f"<site {site_id}>")
                remaining[label] = remaining.get(label, 0) + len(stack)
        return remaining


__all__ = ["SiteStash"]

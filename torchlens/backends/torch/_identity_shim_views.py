"""Wrapper-resolving views over torch's membership tables, for the identity shims.

``_resolve`` follows the wrapper ledger (``_state._decorated_to_orig``) from a
torchlens wrapper to the original torch callable. The three view classes wrap
the cached ``torch.overrides`` tables and the lazy-module method allowlist so a
membership test or lookup keyed by the live wrapper alias answers as it does for
the original, while iteration and contents stay exactly the upstream originals.
Installed and removed by :mod:`torchlens.backends.torch.identity_shims`.
"""

from __future__ import annotations

import collections
from typing import Any

from ... import _state


def _resolve(fn: Any) -> Any:
    """Follow the wrapper ledger from a torchlens wrapper to its original.

    Parameters
    ----------
    fn:
        Any object; non-wrappers resolve to themselves.

    Returns
    -------
    Any
        The original torch callable for a torchlens wrapper, else ``fn``.
    """

    seen: set[int] = set()
    while id(fn) in _state._decorated_to_orig and id(fn) not in seen:
        seen.add(id(fn))
        fn = _state._decorated_to_orig[id(fn)]
    return fn


class _LedgerResolvingTable(dict):
    """Dict view whose LOOKUPS resolve torchlens wrappers to originals.

    Iteration/keys stay exactly the underlying original-keyed contents (the
    wrapper-poisoning census gate keeps holding); only ``in``/``[]``/``get``
    additionally accept the live wrapper alias, so the documented
    ``func in torch.overrides.get_testing_overrides()`` membership check
    answers the same in both wrap epochs (grind-r5 b7 R55-A).
    """

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        original = _resolve(key)
        return original is not key and super().__contains__(original)

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            original = _resolve(key)
            if original is not key:
                return super().__getitem__(original)
            raise

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key`` (wrapper-resolving), else ``default``."""

        try:
            return self[key]
        except KeyError:
            return default


class _LedgerResolvingMembers(list):
    """List view whose membership test resolves torchlens wrappers."""

    def __contains__(self, item: Any) -> bool:
        if super().__contains__(item):
            return True
        original = _resolve(item)
        return original is not item and super().__contains__(original)


class _LedgerResolvingDefaultTable(collections.defaultdict):
    """Wrapper-resolving view that PRESERVES defaultdict semantics.

    ``get_overridable_functions()`` returns a ``defaultdict(list)``; the r5
    membership fix rebuilt it as a plain resolving dict, so a caller indexing
    a namespace with no recorded entries -- auto-vivification the upstream
    type guarantees -- started raising KeyError (grind-r6 b8 R56, opus
    wave-introduced residue). Ledger resolution runs BEFORE the default
    factory so a wrapper alias still finds its original's row rather than
    minting an empty one.
    """

    def __contains__(self, key: Any) -> bool:
        if dict.__contains__(self, key):
            return True
        original = _resolve(key)
        return original is not key and dict.__contains__(self, original)

    def __getitem__(self, key: Any) -> Any:
        if not dict.__contains__(self, key):
            original = _resolve(key)
            if original is not key and dict.__contains__(self, original):
                return dict.__getitem__(self, original)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key`` (wrapper-resolving), never auto-creating."""

        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        original = _resolve(key)
        if original is not key and dict.__contains__(self, original):
            return dict.__getitem__(self, original)
        return default

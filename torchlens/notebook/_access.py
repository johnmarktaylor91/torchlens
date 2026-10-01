"""The copy-expression root ladder (B2, F16; treescope memo section 4).

Every card header and graph-context link carries an EXECUTABLE access
expression (``log['conv2d_1_1']``) -- the rendered surface teaches the
access grammar. The expression's ROOT resolves through the converged
ladder:

1. an explicit root supplied to the display/export call wins;
2. treescope's own relative ``path`` argument for nested renders;
3. in a LIVE notebook only, an identity-based namespace scan -- no eval,
   no mutation, underscore/history names excluded (the F-G repair:
   without the exclusions the shortest-name tie-break picks ``_``,
   IPython's output slot, rebound by the next cell), shortest-then-
   lexicographic, with the resolved root DISPLAYED so a stale pick is
   visible;
4. no root -> copy the key literal only, copy-access disabled with a
   one-line reason.

Offline artifacts NEVER scan a namespace (the report passes an explicit
root or ships key literals).

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ["CopyRoot", "resolve_copy_root"]

#: Exact-name exclusions for the notebook identity scan (F-G): IPython's
#: output slot, history containers, and REPL builtins.
_EXCLUDED_NAMES = frozenset({"_", "In", "Out", "exit", "quit", "get_ipython"})

#: Pattern exclusions: ``_N`` output history, ``_i``/``_iN``/``_ii`` input
#: history, and the ``_ih``/``_oh``/``_dh`` history lists. A leading
#: underscore in general marks machine-managed names, so the scan skips
#: all of them (a strict superset of the memo list, never less safe).
_EXCLUDED_PATTERN = re.compile(r"^_")


@dataclass(frozen=True)
class CopyRoot:
    """Resolved copy-expression root.

    Attributes
    ----------
    expression:
        Root expression text (``"log"``), or ``None`` when disabled.
    source:
        Ladder rung that produced it: ``"explicit"``, ``"treescope_path"``,
        ``"namespace"``, or ``"disabled"``.
    reason:
        One-line reason when disabled (rendered on the card, never
        silent).
    """

    expression: str | None
    source: str
    reason: str | None = None

    def key_expression(self, key: str) -> str:
        """Compose the copyable expression for one lookup key."""

        if self.expression is None:
            return repr(key)
        return f"{self.expression}[{key!r}]"


def _name_excluded(name: str) -> bool:
    """Whether one namespace name is barred from the identity scan."""

    return name in _EXCLUDED_NAMES or bool(_EXCLUDED_PATTERN.match(name))


def _scan_live_namespace(target: Any) -> str | None:
    """Rung 3: identity-scan the live IPython user namespace.

    Returns the shortest (then lexicographically first) non-excluded name
    bound to ``target`` BY IDENTITY, or ``None`` outside a live kernel or
    when no binding survives the exclusions. Never evaluates or mutates
    anything.
    """

    try:
        from IPython import get_ipython
    except Exception:  # noqa: BLE001 - no IPython, no scan
        return None
    shell = get_ipython()
    if shell is None:
        return None
    user_ns = getattr(shell, "user_ns", None)
    if not isinstance(user_ns, dict):
        return None
    candidates = [
        name
        for name, value in user_ns.items()
        if value is target and isinstance(name, str) and not _name_excluded(name)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda name: (len(name), name))


def resolve_copy_root(
    target: Any,
    *,
    explicit_root: str | None = None,
    treescope_path: str | None = None,
    allow_namespace_scan: bool = True,
) -> CopyRoot:
    """Resolve the copy-expression root through the ladder.

    Parameters
    ----------
    target:
        Object whose access expressions are being rendered (the TRACE the
        keys index into).
    explicit_root:
        Rung 1: caller-supplied root; always wins.
    treescope_path:
        Rung 2: treescope's relative path for nested renders.
    allow_namespace_scan:
        Rung 3 gate. Offline artifact writers pass ``False`` -- an offline
        file never scans a namespace.

    Returns
    -------
    CopyRoot
        Resolved root with its rung; rung 4 carries the disabled reason.
    """

    if explicit_root:
        return CopyRoot(explicit_root, "explicit")
    if treescope_path:
        return CopyRoot(treescope_path, "treescope_path")
    if allow_namespace_scan:
        scanned = _scan_live_namespace(target)
        if scanned is not None:
            return CopyRoot(scanned, "namespace")
    return CopyRoot(None, "disabled", "no root name resolved; copy shows the key only")

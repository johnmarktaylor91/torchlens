"""Payload retention decided at JAX capture time.

The JAX interpreter asks :func:`retention_policy` once per capture for a keep
test. The test takes the capture index of a primitive equation (its position
in the flattened capture list, ``len(captures)`` when its capture is
appended) and answers whether that equation's concrete input and output
values must stay in its capture. When it answers ``False`` the interpreter
may free the values after their last use in the jaxpr; the capture keeps only
abstract shape and dtype placeholders.

A capture that never set a policy keeps every value, which is the plain
capture's behavior.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

KeepTest = Callable[[int], bool]


@dataclass(frozen=True)
class Retention:
    """Which equation captures keep concrete values.

    Parameters
    ----------
    keep
        Capture indexes whose values stay. ``None`` keeps every value.
    """

    keep: frozenset[int] | None

    def keeps(self, capture_index: int) -> bool:
        """Return whether the capture at ``capture_index`` keeps its values."""

        return self.keep is None or capture_index in self.keep


KEEP_ALL = Retention(keep=None)
KEEP_NONE = Retention(keep=frozenset())

_POLICIES: weakref.WeakKeyDictionary[Any, Retention] = weakref.WeakKeyDictionary()


def keep_only(indexes: Iterable[int]) -> Retention:
    """Return a retention keeping the values of the listed capture indexes."""

    return Retention(keep=frozenset(int(index) for index in indexes))


def set_retention(session: Any, retention: Retention) -> None:
    """Attach ``retention`` to one capture's equation session."""

    _POLICIES[session] = retention


def retention_of(session: Any) -> Retention:
    """Return the retention attached to ``session`` (keep all when none was set)."""

    return _POLICIES.get(session, KEEP_ALL)


def retention_policy(session: Any) -> KeepTest:
    """Return the keep test the interpreter applies to each primitive equation.

    Parameters
    ----------
    session
        The capture's open equation session (``JaxEquationSession``).

    Returns
    -------
    Callable[[int], bool]
        ``keep(capture_index)``: ``True`` when that equation's concrete values
        must stay in its capture.
    """

    return retention_of(session).keeps

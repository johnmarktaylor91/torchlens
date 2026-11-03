"""The ``selected_by`` relation kind: a control decision between two traced calls.

"writer selected_by router" records that the EARLIER member's output (a
router's route, a judge's verdict) decided whether the LATER member ran or
which input it got. The decider's text need not appear in the later prompt:
that would be ``fed_by`` (content provenance). Like every ordering kind the
row's ``from`` is the later member (the call the decision selected) and
``to`` the earlier member (the decider), so ``ordering_path`` and the
upstream/downstream walks read it in the same direction as ``successor_of``
and ``fed_by``. It records control, never that the decider caused the later
call's content.

Params (all optional, per-kind closed): ``chosen`` (one JSON scalar: what the
decider decided, such as route ``1`` or candidate ``"c2"``), ``candidates``
(a non-empty list of distinct JSON scalars the decider chose among; member
names or not, TorchLens never resolves them as members) and ``evidence``
(the graded evidence envelope ``successor_of`` admits, validated by
:mod:`torchlens.bundle._relations`). When both ``chosen`` and ``candidates``
are given, ``chosen`` must be one of the candidates. Scalars compare by type
and value, so the route ``1`` and the flag ``True`` are different decisions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SELECTED_BY_KIND = "selected_by"
#: Params whose values this module validates (``evidence`` is the shared envelope).
SELECTED_BY_DECISION_PARAMS = frozenset({"chosen", "candidates"})
_SCALAR_MAX_CHARS = 256


def _scalar_key(value: Any) -> tuple[str, Any]:
    """Return a type-aware identity for one decision scalar (``1`` is not ``True``)."""

    return (type(value).__name__, value)


def _validate_scalar(value: Any, *, what: str) -> Any:
    """Return ``value`` when it is a portable decision scalar, else refuse."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and value and len(value) <= _SCALAR_MAX_CHARS:
        return value
    raise ValueError(
        f"selected_by {what} must be a JSON scalar (a bool, an int, a finite float or a "
        f"non-empty string of at most {_SCALAR_MAX_CHARS} characters), got {value!r}"
    )


def validate_selected_by_param(key: str, value: Any) -> Any:
    """Validate one ``chosen``/``candidates`` param; return its persisted normal form.

    Raises
    ------
    ValueError
        On a non-scalar decision, a non-list or empty candidate list, or a
        repeated candidate (callers convert into ``bundle_relation_schema_invalid``).
    """

    if key == "chosen":
        return _validate_scalar(value, what="'chosen'")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(
            "selected_by 'candidates' must be a non-empty list of the options the "
            f"decider chose among, got {value!r}"
        )
    candidates = [
        _validate_scalar(item, what=f"candidate {index}") for index, item in enumerate(value)
    ]
    keys = [_scalar_key(item) for item in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError(f"selected_by 'candidates' repeats an option: {candidates!r}")
    return candidates


def check_selected_by_params(params: Mapping[str, Any]) -> None:
    """Cross-param rule: a decision named with its options must be one of them."""

    if "chosen" in params and "candidates" in params:
        options = {_scalar_key(item) for item in params["candidates"]}
        if _scalar_key(params["chosen"]) not in options:
            raise ValueError(
                f"selected_by 'chosen' {params['chosen']!r} is not one of its "
                f"'candidates' {params['candidates']!r}"
            )

"""The loaded-artifact field refusal, and the op graph's structural refusal built on it.

``_refuse`` is the one raise of a teaching ``ArtifactError`` for a persisted
field; ``forgery_validation`` and the load validators import it from here. The
format-3 reader rebuilds ``Layer.ops`` keyed by each op's ``pass_number`` before
any record's state hook runs, so a mistyped pass stamp reaches the reader first; the trace-level graph validator in ``_load_validators`` checks
the same stamps after the records exist. Both refuse with the one
``artifact_graph_structure_invalid`` teaching error defined here.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .format_errors import ArtifactError

GRAPH_REMEDY = (
    "re-save the artifact from its source capture with one current TorchLens "
    "version; do not hand-edit op rows, labels, pass stamps, or relation lists"
)


def _refuse(
    message: str,
    *,
    code: str,
    field: str,
    reason: str,
    remedy: str,
) -> NoReturn:
    """Raise one teaching portable-field refusal.

    Parameters
    ----------
    message:
        Human-readable description of the structural violation.
    code:
        Stable refusal code for the owning field family.
    field:
        Fully-qualified persisted field name.
    reason:
        Machine-readable structural failure class.
    remedy:
        Concrete recovery action.

    Raises
    ------
    ArtifactError
        Always.
    """

    raise ArtifactError(
        f"{field} is invalid in the loaded artifact: {message}. Remedy: {remedy}.",
        code=code,
        field=field,
        reason=reason,
        remedy=remedy,
    )


def graph_invalid(message: str, field: str, reason: str) -> NoReturn:
    """Raise the graph-structure refusal (AUD-CODE 3.0b)."""

    _refuse(
        message,
        code="artifact_graph_structure_invalid",
        field=field,
        reason=reason,
        remedy=GRAPH_REMEDY,
    )


def refuse_pass_key(label: Any, pass_index: Any) -> None:
    """Refuse an op whose ``pass_number`` is not an int before it keys ``Layer.ops``.

    The range check (``1 <= pass_index <= num_passes``) stays with the trace-level
    validator, which sees ``num_passes``; this is the type half of the same rule.
    """

    if isinstance(pass_index, bool) or not isinstance(pass_index, int):
        graph_invalid(
            f"op {label!r} carries pass_index={pass_index!r}; pass stamps are 1-based ints "
            "with 1 <= pass_index <= num_passes",
            "Op.pass_number",
            "pass_stamp_range",
        )

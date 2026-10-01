"""Typed refusals for the mech-interp kit (mikit memo; teaching refusals law).

Every user-reachable refusal in ``torchlens.mechinterp`` raises
:class:`MechInterpError` (or a documented sibling) carrying a stable
``fields["code"]`` and a non-empty ``fields["remedy"]``; callers branch on the
code, never on message text. The codes are contracted rows in
``docs/reference/error_refusal_contract.md`` (lockstep-gated).

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any, NoReturn

from ..errors._base import TorchLensError

__all__ = ["MechInterpError", "refuse"]


class MechInterpError(TorchLensError, RuntimeError):
    """Raised when a mech-interp kit function cannot honestly answer.

    Contracted refusals carry a stable ``fields["code"]`` and a structured
    ``fields["remedy"]``; the ``RuntimeError`` lineage keeps kit refusals
    catchable alongside the semantic package's existing error classes.
    """


def refuse(code: str, message: str, remedy: str, **payload: Any) -> NoReturn:
    """Raise a :class:`MechInterpError` with the contracted code and remedy.

    Parameters
    ----------
    code:
        Stable machine-readable refusal code (contract-doc row).
    message:
        Human-readable statement of what could not be done and why.
    remedy:
        The action that unblocks the caller; appended to the message so the
        refusal teaches at the point of failure.
    payload:
        Structured diagnostic fields (missing sites, frontier op labels,
        retention plans) surfaced on ``exc.fields``.
    """

    raise MechInterpError(f"{message} Remedy: {remedy}", code=code, remedy=remedy, **payload)

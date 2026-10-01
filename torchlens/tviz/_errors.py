"""Typed refusals for the transformer-pictures package (tviz memo).

Every user-reachable refusal in ``torchlens.tviz`` raises :class:`TvizError`
carrying a stable ``fields["code"]`` and a non-empty ``fields["remedy"]``;
callers branch on the code, never on message text (teaching-refusals law).
The codes are contracted rows in ``docs/reference/error_refusal_contract.md``
(lockstep-gated).

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any, NoReturn

from ..errors._base import TorchLensError

__all__ = ["TvizError", "refuse"]


class TvizError(TorchLensError, RuntimeError):
    """Raised when a transformer picture cannot be honestly produced.

    Contracted refusals carry a stable ``fields["code"]`` and a structured
    ``fields["remedy"]``; the ``RuntimeError`` lineage keeps tviz refusals
    catchable alongside the semantic and mechinterp packages' error classes.
    """


def refuse(code: str, message: str, remedy: str, **payload: Any) -> NoReturn:
    """Raise a :class:`TvizError` with the contracted code and remedy.

    Parameters
    ----------
    code:
        Stable machine-readable refusal code (contract-doc row).
    message:
        Human-readable statement of what could not be done and why.
    remedy:
        The action that unblocks the caller; appended to the message so the
        refusal teaches at the point of failure.
    **payload:
        Structured evidence attached to ``fields`` for programmatic callers.

    Raises
    ------
    TvizError
        Always.
    """

    raise TvizError(f"{message} Remedy: {remedy}", code=code, remedy=remedy, **payload)

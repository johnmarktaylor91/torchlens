"""Typed teaching refusals for the transforms substrate (transforms memo B1).

Every refusal raised by :mod:`torchlens.transforms` carries a stable
``fields["code"]`` and a concrete remedy, per the repo-wide error-refusal
contract (``docs/reference/error_refusal_contract.md``). Public code branches
on the code, never on message text.
"""

from __future__ import annotations

from typing import Any, cast

from .._errors import _actionable_message, _ActionableErrorMixin
from ..errors._base import ConfigurationError

__tl_layer__ = "L4"

__all__ = ["TransformContractError"]


class TransformContractError(_ActionableErrorMixin, ConfigurationError, RuntimeError):
    """Raised when a transform spec, chain, or plan violates its contract."""

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize an actionable transform-contract refusal.

        Parameters
        ----------
        problem:
            Description of the rejected object or operation and its cause.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )

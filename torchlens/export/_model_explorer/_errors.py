"""Typed refusals for the Model Explorer export family.

Every raise site carries a stable ``code`` and a concrete ``remedy`` per the
error-refusal contract (docs/reference/error_refusal_contract.md); consumers
branch on ``exc.fields["code"]``, never message text.
"""

from __future__ import annotations

from typing import Any, cast

from ..._errors import CaptureError, _actionable_message, _ActionableErrorMixin

__tl_layer__ = "L8"


class ModelExplorerExportError(_ActionableErrorMixin, CaptureError, RuntimeError):
    """Raised when a Model Explorer export cannot be emitted honestly.

    Covers exporter-integrity failures (an unresolvable parent reference,
    duplicate node/graph ids, an episode join mismatch): conditions that are
    TorchLens bugs or capture-integrity gaps rather than user errors, where
    silently continuing would ship a silently lossy artifact into a viewer
    that also drops bad rows silently.
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable export-integrity refusal.

        Parameters
        ----------
        problem:
            Description of the integrity failure.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves or reports the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )

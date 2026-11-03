"""Typed errors for the one-backward read surface (M(reads) lane F04).

One public error class with stable ``fields["code"]`` values -- consumers
branch on codes or structured fields, never message text. Spellings are
DOCUMENTED-UNSTABLE pending the naming sprint; the codes are the contract
(rows in ``docs/reference/error_refusal_contract.md``).
"""

from __future__ import annotations

from ...errors._base import TorchLensError

__all__ = ["ReadError", "ReadInternalError"]


class ReadError(TorchLensError, ValueError):
    """Raised for invalid or unserviceable one-backward read requests.

    Every raise site carries a stable ``code=`` plus structured fields; the
    closed reason vocabularies ride ``fields`` (for example
    ``fields["reason"]`` on addressing refusals).
    """


class ReadInternalError(TorchLensError, RuntimeError):
    """Raised when a read-side internal tripwire detects self-contamination.

    Fires only on TorchLens contract breaches (suppression leak, engine
    result-shape drift), never on user input.
    """

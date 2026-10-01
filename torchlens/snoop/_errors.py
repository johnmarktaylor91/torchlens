"""Typed refusal classes for live narration (lane F28, snoop memo D1-D5).

Every raise site in ``torchlens.snoop`` (and every echo refusal raised from
the entry layers) carries a stable ``fields["code"]`` plus a non-empty
``fields["remedy"]`` and is ledgered in
``docs/reference/error_refusal_contract.md``. Callers branch on the code,
never on message text.
"""

from __future__ import annotations

from ..errors._base import CaptureError, ConfigurationError

__tl_layer__ = "L5"


class EchoConfigError(ConfigurationError):
    """Raised for ``echo=`` configuration refusals at capture entry."""


class EchoStatsError(CaptureError):
    """Raised when an armed echo stats rung refuses a value computation.

    The exact rung's numel-budget refusal is a SEMANTIC refusal (the user
    asked for exact stats on a tensor the documented budget forbids), never
    an instrumentation failure: it propagates and fails the capture instead
    of stalling the forward for minutes or silently degrading to a sample.
    """


__all__ = ["EchoConfigError", "EchoStatsError"]

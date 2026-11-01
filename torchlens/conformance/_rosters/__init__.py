"""Per-backend conformance rosters; one module per backend.

Each module exposes ``roster() -> tuple[RosterModel, ...]``: its
``config_built`` rows always, and its ``ecosystem`` rows (pinned pretrained
checkpoints) only when :data:`ECOSYSTEM_ENV` is ``"1"`` (weights cached or
network allowed).
"""

from __future__ import annotations

import os

__tl_layer__ = "L9"

#: Environment switch for the ecosystem rows (pretrained weights).
ECOSYSTEM_ENV = "TORCHLENS_CONFORMANCE_ECOSYSTEM"


def ecosystem_enabled() -> bool:
    """Whether the ecosystem rows are switched on in this environment."""

    return os.environ.get(ECOSYSTEM_ENV, "") == "1"

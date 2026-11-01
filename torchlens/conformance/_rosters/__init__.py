"""Per-backend conformance rosters (``_rosters/<backend>.py``).

Each module exposes ``roster() -> tuple[RosterModel, ...]``: the backend's
``config_built`` rows (small models mirroring the cross-backend probe, with
independent op inventories and NumPy-computed intermediates) and its
``ecosystem`` rows (pinned pretrained checkpoints from the backend's own
ecosystem). Kept in its own package so a later move of the runner is one
``git mv``.
"""

from __future__ import annotations

__tl_layer__ = "L9"

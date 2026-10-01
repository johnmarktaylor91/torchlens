"""TorchLens ecosystem runtime: compat window, migration, providers.

The ecosystem surface (megasprint lane F32; ecosystem MEMO build items
B1-B3) packages the artifact-compatibility promise machinery for users:

- :func:`compat_window` renders the governed compatibility ledger
  (``torchlens._io.compat_ledger``) as a structured, dated report.
- :func:`migrate` is the transactional, upgrade-only artifact migration
  tool (v1: manifest-level steps; never executes a model, plugin, or
  foreign callable).
- :mod:`torchlens.ecosystem.plugins` is the entry-point provider surface:
  metadata-only discovery, explicit distribution-scoped activation.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification; the memo's ``tl.migrate`` / ``tl.compat_window`` /
``tl.plugins`` placeholders route through this subpackage until the naming
sprint and the surface lane (F35) assign facade rows.
"""

from __future__ import annotations

from . import plugins
from .compat import CompatWindowReport, compat_window
from .migrate import MigrationReport, MigrationStep, migrate

__all__ = [
    "CompatWindowReport",
    "MigrationReport",
    "MigrationStep",
    "compat_window",
    "migrate",
    "plugins",
]

__tl_layer__ = "L8"

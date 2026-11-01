"""Per-backend conformance rosters (``config_built`` and ``ecosystem`` rows).

One module per backend, ``torchlens/conformance/_rosters/<backend>.py``, each
exposing ``config_built_rows()`` and ``ecosystem_rows()``. New shared code
lives here so a later move of the runner out of the package is one ``git mv``.
"""

from __future__ import annotations

__tl_layer__ = "L2"

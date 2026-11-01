"""Per-backend conformance rosters (``<backend>.py`` exposes ``roster()``).

Each module holds that backend's ``config_built`` rows (the probe's small
models with op inventories and NumPy-computed expected intermediates) and its
``ecosystem`` rows (pinned pretrained families with checkpoint evidence). The
files move together with the conformance runner when it changes homes.
"""

from __future__ import annotations

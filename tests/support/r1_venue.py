"""The R1 offline-venue signature, snapshotted before any test module imports.

GATE-ID R1_OFFLINE_VENUE (``tests/real_model/r1/conftest.py``): R1 rows run only
when the preflighted environment exports both offline flags. Some test modules
``setdefault`` those flags at import time, so a check read at a test module's
import (or later) would mistake a plain developer box for the venue once such a
module loaded first. ``tests/conftest.py`` imports this module before any test
module, so :data:`IN_OFFLINE_VENUE` records the caller's environment only.
"""

from __future__ import annotations

import os

#: True when the process started with both offline flags set (the R1 venue).
IN_OFFLINE_VENUE: bool = (
    os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
)

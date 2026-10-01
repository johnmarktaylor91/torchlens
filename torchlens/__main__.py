"""``python -m torchlens``: the tiered read-only CLI (torchlens.agent.cli).

Tier-0 verbs (info/schema/guide/version/ls) stay torch-free: the package
``__init__`` defers its torch import, so ``python -m torchlens info run.tlspec``
reads manifest JSON only. See ``python -m torchlens --help``.
"""

from __future__ import annotations

import sys

from torchlens.agent.cli import main

if __name__ == "__main__":
    sys.exit(main())

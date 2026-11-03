"""Internal grouped capture configuration for spine orchestration.

``InternalCaptureConfig`` lives in ``torchlens._capture.stages``; this module keeps
the old import path (and the pickled module path of older traces) resolvable.
"""

from __future__ import annotations

from .stages import InternalCaptureConfig as InternalCaptureConfig

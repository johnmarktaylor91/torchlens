"""Live narration for TorchLens captures (the torchsnooper niche, better).

``torchlens._snoop`` implements the ``echo=`` option on ``tl.trace``,
``tl.record``, and ``fastlog.Recorder`` -- watch each op happen, one line
per event, with the last lines before a crash -- plus the post-hoc
``.narrate()`` renderer on ``Trace`` / ``PartialTrace`` / ``Recording``.
Narration is a DISPLAY of capture events, never a new kind of capture: the
package holds one read-only observer slot, one line grammar, and two crash
tails. All spellings are DOCUMENTED-UNSTABLE pending the naming sprint; the
grouped option class lives at ``tl.options.NarrationOptions``.

Rely on the partial (``exc.partial_log.narrate(20)``) when you expect an
exception; turn narration on (``echo=``) when you expect the process to DIE
-- file sinks flush per line precisely so bytes survive a hard death.
"""

from __future__ import annotations

from ._event import NarrationStats
from ._normalize import normalize_echo
from ._session import EchoSession, active_echo_session
from ._sink import EchoSink
from ._stats import EXACT_NUMEL_BUDGET, SAMPLED_BUDGET

__tl_layer__ = "L5"

__all__ = [
    "EXACT_NUMEL_BUDGET",
    "SAMPLED_BUDGET",
    "EchoSession",
    "EchoSink",
    "NarrationStats",
    "active_echo_session",
    "normalize_echo",
]

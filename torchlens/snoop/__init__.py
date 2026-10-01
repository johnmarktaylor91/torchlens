"""Live narration for TorchLens captures (the torchsnooper niche, better).

``torchlens.snoop`` implements the ``echo=`` option on ``tl.trace``,
``tl.record``, and ``fastlog.Recorder`` -- watch each op happen, one line
per event, with the last lines before a crash -- plus the post-hoc
``.narrate()`` renderer on ``Trace`` / ``PartialTrace`` / ``Recording``.
Narration is a DISPLAY of capture events, never a new kind of capture: the
package holds one read-only observer slot, one line grammar, and two crash
tails. All spellings are DOCUMENTED-UNSTABLE pending the naming sprint; the
grouped option class lives at ``tl.options.EchoOptions``.

Rely on the partial (``exc.partial_log.narrate(20)``) when you expect an
exception; turn narration on (``echo=``) when you expect the process to DIE
-- file sinks flush per line precisely so bytes survive a hard death.
"""

from __future__ import annotations

from ._errors import EchoConfigError, EchoStatsError
from ._event import NARRATION_KINDS, NarrationEvent, NarrationStats
from ._format import render_line, render_stats_segment
from ._narrate import narrate_partial, narrate_recording, narrate_trace
from ._normalize import normalize_echo, refuse_echo_shaped_predicate
from ._session import EchoSession, active_echo_session
from ._sink import EchoSink
from ._stats import EXACT_NUMEL_BUDGET, SAMPLED_BUDGET

__tl_layer__ = "L5"

__all__ = [
    "EXACT_NUMEL_BUDGET",
    "NARRATION_KINDS",
    "SAMPLED_BUDGET",
    "EchoConfigError",
    "EchoSession",
    "EchoSink",
    "EchoStatsError",
    "NarrationEvent",
    "NarrationStats",
    "active_echo_session",
    "narrate_partial",
    "narrate_recording",
    "narrate_trace",
    "normalize_echo",
    "refuse_echo_shaped_predicate",
    "render_line",
    "render_stats_segment",
]

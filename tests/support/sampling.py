"""Read the realized draws saved on a trace's fire records (``FireRecord.sampling``)."""

from __future__ import annotations

from typing import Any

from torchlens.intervention._types import SamplingRecord


def sampling_draws(trace: Any) -> tuple[SamplingRecord, ...]:
    """Return every ``SamplingRecord`` on the trace's fire records, in fire order.

    Fire records live on each op's ``edits``; the ONE builder stamps a monotonic
    timestamp on each, so draws are ordered by record timestamp, then by their
    position inside the record.
    """

    keyed: list[tuple[float, int, int, SamplingRecord]] = []
    for op_index, label in enumerate(trace.op_labels):
        for record in trace.ops[label].edits or ():
            stamp = record.timestamp if record.timestamp is not None else 0.0
            for draw_index, draw in enumerate(record.sampling):
                keyed.append((stamp, op_index, draw_index, draw))
    return tuple(draw for _, _, _, draw in sorted(keyed, key=lambda item: item[:3]))

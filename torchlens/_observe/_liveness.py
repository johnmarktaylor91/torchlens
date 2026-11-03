"""The estimated logical-value liveness view (observe item 13).

Opt-in, ESTIMATED, and scoped: production through last RECORDED consumer, for
ordinary non-retained activation values only. The per-category death rules
are FIXED and PRINTED beside the curve -- the 4.87x measurement showed the
rules ARE the answer, so they are disclosure, never a tunable. The
autograd-saved band is EXCLUDED WHOLESALE (it receives no inferred death in
any phase; its gross monotone series is its whole story -- panel ruling
10.3). Never part of the default render; never stacked with or reconciled
against allocator numbers; lifetime facts are never presented as observed
allocation/free events.

The justifying case: inference-mode traces, where cumulative-produced and
live diverge ~50x and the serving footprint decomposes as params +
persistent buffers + live-activation peak + held/returned tensors.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DEATH_RULES", "estimated_liveness"]

#: The FIXED death rules, printed beside every curve.
DEATH_RULES = {
    "parameter": "persistent (never dies)",
    "buffer": "persistent (never dies)",
    "input": "held by the caller; alive through phase end",
    "activation": "dies at its last RECORDED consumer",
    "consumerless_activation": (
        "conservatively held to phase end -- exactly right for returned "
        "KV-cache tensors, because the caller really holds them"
    ),
    "autograd_saved": (
        "EXCLUDED wholesale: no inferred death in any phase (its gross "
        "monotone series is its whole story here)"
    ),
    "op_gradient": "out of scope for the v1 estimate (forward-phase view)",
    "parameter_gradient": "out of scope for the v1 estimate (forward-phase view)",
}


def estimated_liveness(artifact: dict[str, Any]) -> dict[str, Any]:
    """Derive the opt-in estimated liveness series from one timeline artifact.

    Parameters
    ----------
    artifact:
        A ``torchlens.memory_timeline.v2`` artifact.

    Returns
    -------
    dict[str, Any]
        ``{"evidence": "estimated", "death_rules": ..., "series": [(ordinal,
        live_bytes), ...], "peak_bytes", "peak_ordinal",
        "persistent_baseline_bytes", "excluded"}``. The series never changes
        the artifact's observed category totals.
    """

    persistent = int(artifact["persistent_baseline_bytes"])
    max_ordinal = 0
    value_rows: list[tuple[int, int, int]] = []  # (birth, death, bytes)
    for row in artifact["events"]:
        if row["phase"] != "forward":
            continue
        ordinal = int(row["ordinal"] or 0)
        max_ordinal = max(max_ordinal, ordinal)
    for row in artifact["events"]:
        if row["phase"] != "forward" or row["category"] not in ("input", "activation"):
            continue
        ordinal = int(row["ordinal"] or 0)
        if row["category"] == "input":
            death = max_ordinal
        else:
            last_consumer = row.get("last_consumer_ordinal")
            death = int(last_consumer) if last_consumer is not None else max_ordinal
        value_rows.append((ordinal, death, int(row["bytes"])))

    series: list[tuple[int, int]] = []
    peak_bytes = persistent
    peak_ordinal = 0
    for ordinal in range(1, max_ordinal + 1):
        live = persistent + sum(
            value_bytes for birth, death, value_bytes in value_rows if birth <= ordinal <= death
        )
        series.append((ordinal, live))
        if live > peak_bytes:
            peak_bytes = live
            peak_ordinal = ordinal
    return {
        "evidence": "estimated",
        "death_rules": dict(DEATH_RULES),
        "series": series,
        "peak_bytes": peak_bytes,
        "peak_ordinal": peak_ordinal,
        "persistent_baseline_bytes": persistent,
        "excluded": {
            "autograd_saved": DEATH_RULES["autograd_saved"],
            "backward_phases": "the v1 estimate is a forward-phase view",
        },
        "caption": (
            "estimated logical-value liveness (hypothesis, not observed "
            "allocation/free events); never stacked with allocator numbers"
        ),
    }

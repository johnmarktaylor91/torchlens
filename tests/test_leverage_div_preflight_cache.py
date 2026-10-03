"""B11 (cone preflight, exact-membership oracle) + B12 (generated cache key).

B11's oracle is EXACT (memo D-12 / composition row "preflight x lane"):
predicted replay membership EQUALS the engine's actual recomputed cone.
B12 pins the generated-universe property the warm-cache key now has: every
capture option is in the key or reviewed cache-irrelevant, and the reviewed
ledgers cannot go stale.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

import torch
from test_leverage_div_fixtures import ReusedReluNet

import torchlens as tl
from torchlens.capture.preflight import cone_preflight


def _armed_trace():
    torch.manual_seed(0)
    model = ReusedReluNet()
    x = torch.rand(1, 2, 3, 3)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


# ---------------------------------------------------------------------------
# B11: cone preflight.
# ---------------------------------------------------------------------------


def test_cone_preflight_membership_equals_actual_replay_cone():
    """The exact oracle: predicted replay membership EQUALS actual."""

    trace = _armed_trace()
    plan = cone_preflight(trace, trace["relu_1_2"].ops)
    assert plan["replay_ready"]
    fork = trace.fork()
    fork.do("relu_1_2", tl.zero_ablate())
    assert fork.last_run["engine"] == "replay"
    # Replay disclosures spell single-pass ops bare; the preflight is
    # pass-qualified throughout. Compare in bare-label space for this
    # single-pass fixture.
    planned_bare = tuple(label.rsplit(":", 1)[0] for label in plan["planned_replay_labels"])
    assert planned_bare == tuple(fork.last_run["cone"])


def test_cone_preflight_discloses_engine_assumptions():
    trace = _armed_trace()
    plan = cone_preflight(trace, trace["relu_2_4"].ops)
    assert plan["cone_size"] < plan["total_ops"]
    assert plan["live_lane_ops"] == plan["total_ops"]  # live always runs the full forward
    notes = " ".join(plan["notes"])
    assert "intervention_ready" in notes
    assert "never numeric change" in notes


def test_cone_preflight_unarmed_capture_reports_not_replay_ready():
    torch.manual_seed(0)
    model = ReusedReluNet()
    trace = tl.trace(model, torch.rand(1, 2, 3, 3))
    plan = cone_preflight(trace, trace["relu_1_2"].ops)
    assert plan["replay_ready"] is False


# ---------------------------------------------------------------------------
# B12: the warm-cache key is GENERATED from the options universe.
# ---------------------------------------------------------------------------


def test_cache_key_ledgers_never_go_stale():
    """Every curated/neutral ledger name is a REAL CaptureOptions field."""

    from torchlens.user_funcs import CAPTURE_CACHE_KEY_CURATED, CAPTURE_CACHE_KEY_NEUTRAL

    option_names = {field.name for field in dataclass_fields(tl.options.CaptureOptions)}
    trace_kwarg_ledger_names = {
        # Curated/neutral rows may also name trace() kwargs that never became
        # CaptureOptions fields; those are pinned here explicitly so a rename
        # on either side goes red.
        "cache",
        "cache_dir",
        "unwrap_when_done",
        "verbose",
        "batch_render",
    }
    for name in set(CAPTURE_CACHE_KEY_CURATED) | set(CAPTURE_CACHE_KEY_NEUTRAL):
        assert name in option_names or name in trace_kwarg_ledger_names, (
            f"stale cache-key ledger row {name!r}: it names neither a "
            "CaptureOptions field nor a pinned trace() kwarg"
        )


def test_every_capture_option_is_keyed_or_reviewed():
    """The generated-universe property: no option can silently skip the key."""

    from torchlens.user_funcs import (
        CAPTURE_CACHE_KEY_CURATED,
        CAPTURE_CACHE_KEY_NEUTRAL,
        _sweep_option_fields_into_cache_config,
    )

    config: dict[str, object] = {}
    _sweep_option_fields_into_cache_config(
        config,
        tl.options.CaptureOptions(),
        prefix="capture_option",
        curated=CAPTURE_CACHE_KEY_CURATED,
        neutral=CAPTURE_CACHE_KEY_NEUTRAL,
    )
    swept = {key.split(":", 1)[1] for key in config}
    for field in dataclass_fields(tl.options.CaptureOptions):
        if field.name.startswith("_"):
            continue
        assert (
            field.name in swept
            or field.name in CAPTURE_CACHE_KEY_CURATED
            or field.name in CAPTURE_CACHE_KEY_NEUTRAL
        ), f"CaptureOptions.{field.name} is neither keyed nor reviewed cache-irrelevant"


def test_swept_option_change_misses_the_cache_key():
    """Two configs differing in ONE swept semantic option produce different keys."""

    from torchlens._capture_state_helpers import _capture_cache_key
    from torchlens.user_funcs import (
        CAPTURE_CACHE_KEY_CURATED,
        CAPTURE_CACHE_KEY_NEUTRAL,
        _sweep_option_fields_into_cache_config,
    )

    model = torch.nn.Linear(2, 2)
    x = torch.zeros(1, 2)

    def key_for(options: tl.options.CaptureOptions) -> str:
        config: dict[str, object] = {}
        _sweep_option_fields_into_cache_config(
            config,
            options,
            prefix="capture_option",
            curated=CAPTURE_CACHE_KEY_CURATED,
            neutral=CAPTURE_CACHE_KEY_NEUTRAL,
        )
        return _capture_cache_key(model, (x,), {}, config)

    baseline = key_for(tl.options.CaptureOptions())
    changed = key_for(tl.options.CaptureOptions(track_nonfinite=True))
    assert baseline != changed

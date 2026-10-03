"""grouping= knob entry legality + the grouping_policy_v1 stamp: writer
content, loader coherence rules C1-C8, tamper degradation, the canonical
degraded settlement's monotonic round trip, and the G-gates for the stamp
family (S3 switch discipline).

All stamp vocabulary and refusal spellings are PROVISIONAL pending the S2
amendment; these tests pin the shipped behavior, not ratified names.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from _oracle_env import expect_bundle_minor_version_mismatch

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._io import ArtifactSchemaAgeWarning
from torchlens._io.prerelease import activate_prerelease_fields
from torchlens.errors._base import TorchLensWarning
from torchlens.postprocess._grouping_stamp import (
    GROUPING_POLICY_SCHEMA,
    build_grouping_policy_stamp,
    degraded_grouping_policy_stamp,
    validate_grouping_policy_stamp,
)


def _tiny_trace(**kwargs) -> tl.Trace:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    return tl.trace(model, torch.randn(1, 4), **kwargs)


def _healthy() -> dict:
    return build_grouping_policy_stamp(ran_recurrence_grouping=True, requested="structural")


# ---------------------------------------------------------------------------
# grouping= entry legality (closed vocabulary; only "structural" proceeds)
# ---------------------------------------------------------------------------


def test_grouping_kwarg_is_keyword_only_and_structural_proceeds() -> None:
    log = _tiny_trace(grouping="structural")
    assert log.grouping == "structural"
    assert log.grouping_policy == _healthy()


@pytest.mark.smoke
def test_grouping_entry_refusals() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    for value, code in (
        ("strict_shapes", "grouping_policy_unavailable"),
        ("fold_sites", "grouping_policy_unavailable"),
        ("bogus", "grouping_invalid"),
        (None, "grouping_invalid"),
    ):
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.trace(model, torch.randn(1, 4), grouping=value)
        assert excinfo.value.fields["code"] == code


# ---------------------------------------------------------------------------
# Writer content (policy records what step 7 actually ran)
# ---------------------------------------------------------------------------


def test_stamp_written_on_default_and_degraded_paths() -> None:
    default_log = _tiny_trace()
    assert default_log.grouping_policy == {
        "schema": GROUPING_POLICY_SCHEMA,
        "policy": "structural",
        "requested": "structural",
        "folded_sites": "none",
        "site_join": "none",
        "detector": "grouper_v1",
        "effective": True,
        "settlement_note": None,
    }
    degraded_path_log = _tiny_trace(capture=tl.options.CaptureOptions(recurrence_detection=False))
    stamp = degraded_path_log.grouping_policy
    assert stamp["policy"] == "params_only"
    assert stamp["detector"] == "unknown"
    assert stamp["effective"] is True
    assert stamp["settlement_note"] is None
    # C5 coherence holds live on both paths.
    assert (
        validate_grouping_policy_stamp(
            stamp,
            recurrence_detection=degraded_path_log.recurrence_detection,
            grouping=degraded_path_log.grouping,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Loader validation rule matrix (exact-key schema + C1-C8)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_validation_rule_matrix() -> None:
    assert validate_grouping_policy_stamp(_healthy()) is None
    assert validate_grouping_policy_stamp("not-a-dict") == "malformed"
    assert validate_grouping_policy_stamp({**_healthy(), "schema": "v2"}) == "malformed"
    assert validate_grouping_policy_stamp({**_healthy(), "extra": 1}) == "unknown_key"
    missing = _healthy()
    missing.pop("detector")
    assert validate_grouping_policy_stamp(missing) == "malformed"
    assert validate_grouping_policy_stamp({**_healthy(), "policy": "greedy"}) == "vocabulary"
    assert validate_grouping_policy_stamp({**_healthy(), "requested": "params_only"}) == (
        "vocabulary"  # params_only is a POLICY value, never a requested value
    )
    assert validate_grouping_policy_stamp({**_healthy(), "detector": "grouper_v2"}) == "vocabulary"
    assert validate_grouping_policy_stamp({**_healthy(), "effective": "yes"}) == "vocabulary"
    assert validate_grouping_policy_stamp({**_healthy(), "settlement_note": "oops"}) == (
        "vocabulary"  # settlement tokens carry the grouping_stamp_ prefix
    )
    # C1/C2: fold state matches the claiming policy.
    fold = {**_healthy(), "policy": "fold_sites"}
    assert validate_grouping_policy_stamp(fold) == "C1"
    assert validate_grouping_policy_stamp({**_healthy(), "folded_sites": "all"}) == "C2"
    # C3: ineffective grouping never claims a policy it did not run.
    assert validate_grouping_policy_stamp({**_healthy(), "effective": False}) == "C3"
    # C4: site lists parse as s1| keys, sorted, unique.
    fold_ok_shape = {**_healthy(), "policy": "fold_sites"}
    assert (
        validate_grouping_policy_stamp({**fold_ok_shape, "folded_sites": ["s1|a|relu||1"]}) is None
    )
    assert validate_grouping_policy_stamp({**fold_ok_shape, "folded_sites": ["bad-key"]}) == "C4"
    assert (
        validate_grouping_policy_stamp(
            {**fold_ok_shape, "folded_sites": ["s1|b|relu||1", "s1|a|relu||1"]}
        )
        == "C4"
    )
    assert (
        validate_grouping_policy_stamp(
            {**fold_ok_shape, "folded_sites": ["s1|a|relu||1", "s1|a|relu||1"]}
        )
        == "C4"
    )
    # C5: stamp/recurrence_detection coherence.
    assert validate_grouping_policy_stamp(_healthy(), recurrence_detection=False) == "C5"
    params_only = build_grouping_policy_stamp(ran_recurrence_grouping=False, requested="structural")
    assert validate_grouping_policy_stamp(params_only, recurrence_detection=True) == "C5"
    # C6: mirror coherence with trace.grouping.
    assert validate_grouping_policy_stamp(_healthy(), grouping="strict_shapes") == "C6"
    # C7: a healthy stamp never carries a settlement.
    assert (
        validate_grouping_policy_stamp({**_healthy(), "settlement_note": "grouping_stamp_x"})
        == "C7"
    )
    # C8: a product-layer join is fail-closed until the S2 amendment wires
    # L2's capture-kind predicate.
    assert validate_grouping_policy_stamp({**_healthy(), "site_join": "all"}) == "C8"
    # The canonical degraded settlement is LEGAL under the exact schema
    # (monotonic: it validates clean and stays degraded).
    assert validate_grouping_policy_stamp(degraded_grouping_policy_stamp("legacy")) is None


# ---------------------------------------------------------------------------
# G1: stamp + mirror round-trip under the switch
# ---------------------------------------------------------------------------


def test_g1_stamp_roundtrip_under_switch(tmp_path) -> None:
    trace = _tiny_trace()
    path = tmp_path / "stamp.tlspec"
    with activate_prerelease_fields():
        tl.save(trace, str(path))
        with warnings.catch_warnings():
            warnings.simplefilter("error", TorchLensWarning)
            loaded = tl.load(str(path))
    assert loaded.grouping == "structural"
    assert loaded.grouping_policy == trace.grouping_policy


# ---------------------------------------------------------------------------
# G3: tamper degrades typed; degrade round-trips byte-stable (monotonic)
# ---------------------------------------------------------------------------


def _tampered_reload(tmp_path, mutate, name: str) -> tl.Trace:
    trace = _tiny_trace()
    path = tmp_path / f"{name}.tlspec"
    with activate_prerelease_fields():
        tl.save(trace, str(path))
        loaded = tl.load(str(path))
        mutate(loaded)
        repath = tmp_path / f"{name}_tampered.tlspec"
        tl.save(loaded, str(repath))
        with pytest.warns(TorchLensWarning, match="grouping_policy stamp is invalid"):
            return tl.load(str(repath))


def test_g3_tampered_unknown_key_refuses_and_settles(tmp_path) -> None:
    def plant_unknown_key(trace: tl.Trace) -> None:
        trace.grouping_policy = {**trace.grouping_policy, "forged": True}

    reloaded = _tampered_reload(tmp_path, plant_unknown_key, "unknown_key")
    assert reloaded.grouping_policy == degraded_grouping_policy_stamp("unknown_key")


def test_g3_forged_policy_degrades_with_rule_name(tmp_path) -> None:
    def forge_policy(trace: tl.Trace) -> None:
        trace.grouping_policy = {**trace.grouping_policy, "policy": "greedy"}

    reloaded = _tampered_reload(tmp_path, forge_policy, "forged_policy")
    assert reloaded.grouping_policy == degraded_grouping_policy_stamp("vocabulary")


def test_g3_mirror_incoherence_degrades_c6(tmp_path) -> None:
    def forge_requested(trace: tl.Trace) -> None:
        trace.grouping_policy = {**trace.grouping_policy, "requested": "fold_sites"}

    reloaded = _tampered_reload(tmp_path, forge_requested, "mirror")
    assert reloaded.grouping_policy == degraded_grouping_policy_stamp("c6")


def test_g3_degraded_settlement_roundtrips_byte_stable(tmp_path) -> None:
    def forge_policy(trace: tl.Trace) -> None:
        trace.grouping_policy = {**trace.grouping_policy, "policy": "greedy"}

    settled = _tampered_reload(tmp_path, forge_policy, "settle_once")
    settled_payload = dict(settled.grouping_policy)
    # Re-save the SETTLED trace: the settlement is legal under the exact
    # writer schema, so the reload adopts it verbatim with NO further
    # warning and stays degraded (verdicts only worsen, never heal).
    path = tmp_path / "settled_again.tlspec"
    with activate_prerelease_fields():
        tl.save(settled, str(path))
        with warnings.catch_warnings():
            warnings.simplefilter("error", TorchLensWarning)
            resettled = tl.load(str(path))
    assert resettled.grouping_policy == settled_payload


# ---------------------------------------------------------------------------
# G5: legacy artifacts settle silently to the legacy settlement
# ---------------------------------------------------------------------------


def test_g5_legacy_v7_artifact_settles_silently() -> None:
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "tlspec_v7" / "tiny_v7.tlspec"
    with warnings.catch_warnings():
        # Silent wrt TorchLensWarning; the version-age advisory is expected.
        warnings.simplefilter("error", TorchLensWarning)
        warnings.simplefilter("default", ArtifactSchemaAgeWarning)
        # This fixture was recorded on a torch 2.13 CUDA build (R6): the
        # minor-mismatch advisory is also expected on any other torch minor,
        # narrowly, same as the age advisory above.
        with expect_bundle_minor_version_mismatch():
            loaded = tl.load(str(fixture))  # real v7 write: the stamp was DROPped
    assert loaded.grouping_policy == degraded_grouping_policy_stamp("legacy")
    assert loaded.grouping_policy["settlement_note"] == "grouping_stamp_legacy"
    # The mirror field restores its default.
    assert loaded.grouping == "structural"


def test_healthy_stamp_persists_on_plain_v8_round_trip(tmp_path) -> None:
    trace = _tiny_trace()
    path = tmp_path / "fresh.tlspec"
    tl.save(trace, str(path))  # tlspec v8: the stamp persists plainly
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        loaded = tl.load(str(path))
    assert loaded.grouping_policy == trace.grouping_policy
    assert loaded.grouping == "structural"

"""Pre-release field registrar mechanism (persistence-schema seam S3).

New portable fields land declared ``FieldPolicy.DROP`` under the current
tlspec version and register with :mod:`torchlens._io.prerelease`. This module
proves the three mechanism guarantees:

(a) registered (DROP-declared) fields do NOT persist by default;
(b) the activation switch only activates under test;
(c) switch-on writes carry the pre-release marker and are therefore
    DISTINGUISHABLE from real current-version artifacts -- loading one
    without the switch refuses typed, fail-closed.

The registry is process-global; every planting here restores it (same
discipline as the ``HAS_*`` capability-probe caches).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import FieldPolicy, PreReleaseArtifactError
from torchlens._io.prerelease import (
    PRERELEASE_MARKER,
    PRERELEASE_STATE_KEY,
    activate_prerelease_fields,
    prerelease_fields_active,
    register_prerelease_field,
    registered_prerelease_fields,
    unregister_prerelease_field,
    validate_prerelease_state,
)
from torchlens._io.scrub import scrub_for_save
from torchlens.data_classes.trace import Trace

pytestmark = pytest.mark.smoke

#: Existing declared-DROP scalar Trace field used as the planted gated field.
#: Any declared-DROP field works; this one is a plain session-time int.
_PLANT_FIELD = "_tl_save_selector_fire_count"

#: STANDING lane registrations: real sprint-gated fields registered at import
#: time and retired only at the coordinated tlspec bump. Inventory assertions
#: are made RELATIVE to this ledger so each new writer lane lands here as a
#: reviewed one-line diff (registrar keeps the live inventory). The tlspec v8
#: coordinated bump (2026-08-17) retired the entire feature-megasprint
#: inventory (FireRecord/HelperSpec edge families, the AtenOp primitive
#: profile constellation, kernel telemetry, Op site_key + edge stores, the
#: eight Trace markers, and the episode/_kernel_telemetry annotation keys):
#: the registry is EMPTY again until the next sprint gates a new family.
# The C01 sidecar seam's "sidecar" annotations sub-key persisted plainly at
# the C07 tlspec v9 write; the registry is empty again (import kept so the
# inventory pins stay import-order-deterministic either way).
import torchlens._io.sidecar  # noqa: F401,E402

_STANDING_REGISTRATIONS: dict[str, tuple[str, ...]] = {}


@pytest.fixture
def planted_field():
    register_prerelease_field(Trace, _PLANT_FIELD)
    try:
        yield _PLANT_FIELD
    finally:
        unregister_prerelease_field(Trace, _PLANT_FIELD)


def _tiny_trace() -> Trace:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    return tl.trace(model, torch.randn(1, 4))


# ---------------------------------------------------------------------------
# Registration discipline
# ---------------------------------------------------------------------------


def test_registration_requires_declared_drop_policy() -> None:
    keep_field = next(
        name for name, policy in Trace.PORTABLE_STATE_SPEC.items() if policy is FieldPolicy.KEEP
    )
    with pytest.raises(ValueError, match="not 'drop'"):
        register_prerelease_field(Trace, keep_field)
    with pytest.raises(ValueError, match="not a declared portable field"):
        register_prerelease_field(Trace, "_no_such_field_anywhere")
    with pytest.raises(ValueError, match="no-op"):
        register_prerelease_field(Trace, _PLANT_FIELD, persisted_policy=FieldPolicy.DROP)
    # No refused registration may have landed; only STANDING lane
    # registrations (real sprint-gated fields awaiting the coordinated bump)
    # are present.
    assert registered_prerelease_fields() == _STANDING_REGISTRATIONS


def test_registry_inventory_and_unregister(planted_field: str) -> None:
    inventory = registered_prerelease_fields()
    assert planted_field in inventory["Trace"]
    unregister_prerelease_field(Trace, planted_field)
    assert registered_prerelease_fields() == _STANDING_REGISTRATIONS
    # Fixture teardown unregisters again; must be idempotent.


def test_annotations_keys_retired_at_the_bump() -> None:
    # The S7 episode-ledger home and the L3 telemetry relation persisted
    # plainly at the tlspec v8 bump; the C01 sidecar namespace persisted
    # plainly at the tlspec v9 bump (C07). No gated annotation sub-keys
    # remain until a future sprint registers a new family.
    assert registered_prerelease_fields().get("Trace.annotations", ()) == ()


# ---------------------------------------------------------------------------
# (a) DROP fields do not persist by default
# ---------------------------------------------------------------------------


def test_gated_field_does_not_persist_by_default(planted_field: str) -> None:
    assert not prerelease_fields_active()
    trace = _tiny_trace()
    setattr(trace, planted_field, 7)
    scrubbed_state, _, _ = scrub_for_save(trace)
    assert scrubbed_state[planted_field] is None
    assert PRERELEASE_STATE_KEY not in scrubbed_state
    assert scrubbed_state["tlspec_version"] == tl._io.TLSPEC_VERSION


# ---------------------------------------------------------------------------
# (b) the switch only activates under test
# ---------------------------------------------------------------------------


def test_switch_is_test_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not prerelease_fields_active()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(RuntimeError, match="TEST-ONLY"), activate_prerelease_fields():
        pass  # pragma: no cover - refused before entry
    assert not prerelease_fields_active()


def test_switch_scopes_and_restores() -> None:
    with activate_prerelease_fields():
        assert prerelease_fields_active()
        with activate_prerelease_fields():
            assert prerelease_fields_active()
        assert prerelease_fields_active()
    assert not prerelease_fields_active()


# ---------------------------------------------------------------------------
# (c) switch-on writes persist the field, carry the marker, and are
#     distinguishable from real current-version artifacts
# ---------------------------------------------------------------------------


def test_switched_write_persists_field_and_carries_marker(planted_field: str) -> None:
    trace = _tiny_trace()
    setattr(trace, planted_field, 7)
    with activate_prerelease_fields():
        scrubbed_state, _, _ = scrub_for_save(trace)
    assert scrubbed_state[planted_field] == 7
    marker = scrubbed_state[PRERELEASE_STATE_KEY]
    assert marker["marker"] == PRERELEASE_MARKER
    assert f"Trace.{planted_field}" in marker["fields"]


def test_marked_artifact_refuses_to_load_as_real_current_version(
    planted_field: str, tmp_path
) -> None:
    trace = _tiny_trace()
    setattr(trace, planted_field, 7)
    marked_path = tmp_path / "marked.tlspec"
    with activate_prerelease_fields():
        tl.save(trace, str(marked_path))
    # Without the switch the marked artifact refuses typed.
    with pytest.raises(PreReleaseArtifactError, match="pre-release field marker"):
        tl.load(str(marked_path))
    # Under the switch the exit-gate round trip works and the field survives.
    with activate_prerelease_fields():
        loaded = tl.load(str(marked_path))
    assert getattr(loaded, planted_field) == 7


def test_unmarked_artifact_loads_clean_with_no_marker(planted_field: str, tmp_path) -> None:
    trace = _tiny_trace()
    setattr(trace, planted_field, 7)
    plain_path = tmp_path / "plain.tlspec"
    tl.save(trace, str(plain_path))
    loaded = tl.load(str(plain_path))
    # Default write: field dropped, no marker, loads with or without switch.
    assert getattr(loaded, planted_field, None) in (None, 0)
    with activate_prerelease_fields():
        tl.load(str(plain_path))


def test_malformed_marker_refuses_even_under_switch() -> None:
    tampered = {PRERELEASE_STATE_KEY: {"marker": "wrong-marker"}}
    with activate_prerelease_fields(), pytest.raises(PreReleaseArtifactError, match="malformed"):
        validate_prerelease_state(dict(tampered), cls_name="Trace")
    with pytest.raises(PreReleaseArtifactError, match="pre-release field marker"):
        validate_prerelease_state(dict(tampered), cls_name="Trace")
    # Absent marker is a no-op on the load chokepoint.
    validate_prerelease_state({"tlspec_version": 7}, cls_name="Trace")

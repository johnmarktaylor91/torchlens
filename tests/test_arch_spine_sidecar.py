"""SIDECAR seam gates (C01 item 14; architecture memo 6.3 seam 1).

Namespaced typed sidecar families: registration law, payload validation,
size budget, analysis-only behavior when the provider is absent, and plain
persistence as of the coordinated tlspec v9 write (C07): real saves
round-trip the ``"sidecar"`` annotations namespace and loads validate the
envelope shape fail-closed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.prerelease import gated_annotations_keys
from torchlens._io.sidecar import SIDECAR_ANNOTATIONS_KEY
from torchlens.io import (
    AnalysisOnlySidecar,
    SidecarError,
    SidecarFamily,
    attach_sidecar,
    list_sidecar_families,
    read_sidecar,
    register_sidecar_family,
    unregister_sidecar_family,
)

pytestmark = pytest.mark.smoke

FAMILY = SidecarFamily(
    family_id="testorg.saliency",
    schema_id="testorg_saliency_v1",
    version=2,
    owner="testorg",
    size_budget_bytes=4096,
)


@pytest.fixture
def registered_family():
    register_sidecar_family(FAMILY)
    try:
        yield FAMILY
    finally:
        if FAMILY.family_id in list_sidecar_families():
            unregister_sidecar_family(FAMILY.family_id)


@pytest.fixture(scope="module")
def traced():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


class TestRegistrationLaw:
    def test_family_id_must_be_namespaced(self) -> None:
        for bad_id in ("saliency", "", "Bad.Case", ".leading"):
            with pytest.raises(SidecarError) as excinfo:
                register_sidecar_family(
                    SidecarFamily(family_id=bad_id, schema_id="s", version=1, owner="o")
                )
            assert excinfo.value.fields["code"] == "sidecar_family_id_invalid"

    def test_schema_contract_validated(self) -> None:
        with pytest.raises(SidecarError) as excinfo:
            register_sidecar_family(
                SidecarFamily(family_id="a.b", schema_id="", version=1, owner="o")
            )
        assert excinfo.value.fields["code"] == "sidecar_schema_invalid"
        with pytest.raises(SidecarError) as excinfo:
            register_sidecar_family(
                SidecarFamily(family_id="a.b", schema_id="s", version=0, owner="o")
            )
        assert excinfo.value.fields["code"] == "sidecar_schema_invalid"
        with pytest.raises(SidecarError) as excinfo:
            register_sidecar_family(
                SidecarFamily(
                    family_id="a.b", schema_id="s", version=1, owner="o", size_budget_bytes=0
                )
            )
        assert excinfo.value.fields["code"] == "sidecar_schema_invalid"

    def test_non_family_registration_refuses(self) -> None:
        with pytest.raises(SidecarError) as excinfo:
            register_sidecar_family({"family_id": "a.b"})  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "sidecar_family_type_invalid"

    def test_collision_refusal_rides_the_kernel(self, registered_family) -> None:
        from torchlens._registry import RegistryError

        with pytest.raises(RegistryError) as excinfo:
            register_sidecar_family(FAMILY)
        assert excinfo.value.fields["code"] == "registry_entry_duplicate"


class TestAttachAndRead:
    def test_round_trip_on_a_real_trace(self, registered_family, traced) -> None:
        attach_sidecar(traced, FAMILY.family_id, {"scores": [0.5, 0.25]})
        assert read_sidecar(traced, FAMILY.family_id) == {"scores": [0.5, 0.25]}
        envelope = traced.annotations[SIDECAR_ANNOTATIONS_KEY][FAMILY.family_id]
        assert envelope["schema_id"] == FAMILY.schema_id
        assert envelope["version"] == FAMILY.version
        assert envelope["owner"] == FAMILY.owner

    def test_write_requires_registration(self, traced) -> None:
        from torchlens._registry import RegistryError

        with pytest.raises(RegistryError) as excinfo:
            attach_sidecar(traced, "nobody.registered", {"x": 1})
        assert excinfo.value.fields["code"] == "registry_entry_unknown"

    def test_non_trace_target_refuses(self, registered_family) -> None:
        with pytest.raises(SidecarError) as excinfo:
            attach_sidecar(object(), FAMILY.family_id, {"x": 1})
        assert excinfo.value.fields["code"] == "sidecar_trace_invalid"

    def test_non_json_payload_refuses(self, registered_family, traced) -> None:
        with pytest.raises(SidecarError) as excinfo:
            attach_sidecar(traced, FAMILY.family_id, {"tensor": torch.randn(2)})
        assert excinfo.value.fields["code"] == "sidecar_payload_invalid"

    def test_size_budget_enforced(self, registered_family, traced) -> None:
        with pytest.raises(SidecarError) as excinfo:
            attach_sidecar(traced, FAMILY.family_id, {"big": "x" * 8192})
        assert excinfo.value.fields["code"] == "sidecar_budget_exceeded"

    def test_absent_sidecar_read_refuses(self, registered_family, traced) -> None:
        fresh_model = nn.Sequential(nn.Linear(4, 3))
        fresh = tl.trace(fresh_model, torch.randn(2, 4))
        with pytest.raises(SidecarError) as excinfo:
            read_sidecar(fresh, FAMILY.family_id)
        assert excinfo.value.fields["code"] == "sidecar_absent"

    def test_malformed_envelope_refuses(self, registered_family, traced) -> None:
        traced.annotations.setdefault(SIDECAR_ANNOTATIONS_KEY, {})["testorg.saliency"] = "junk"
        try:
            with pytest.raises(SidecarError) as excinfo:
                read_sidecar(traced, FAMILY.family_id)
            assert excinfo.value.fields["code"] == "sidecar_envelope_invalid"
        finally:
            traced.annotations[SIDECAR_ANNOTATIONS_KEY].pop(FAMILY.family_id, None)

    def test_newer_envelope_version_refuses_typed(self, registered_family, traced) -> None:
        attach_sidecar(traced, FAMILY.family_id, {"v": 1})
        envelope = traced.annotations[SIDECAR_ANNOTATIONS_KEY][FAMILY.family_id]
        envelope["version"] = FAMILY.version + 1
        try:
            with pytest.raises(SidecarError) as excinfo:
                read_sidecar(traced, FAMILY.family_id)
            assert excinfo.value.fields["code"] == "sidecar_version_unsupported"
        finally:
            traced.annotations[SIDECAR_ANNOTATIONS_KEY].pop(FAMILY.family_id, None)


class TestMissingProviderAnalysisOnly:
    def test_provider_absent_read_degrades_analysis_only(self, traced) -> None:
        register_sidecar_family(FAMILY)
        attach_sidecar(traced, FAMILY.family_id, {"scores": [1.0]})
        unregister_sidecar_family(FAMILY.family_id)
        try:
            view = read_sidecar(traced, FAMILY.family_id)
            assert isinstance(view, AnalysisOnlySidecar)
            assert view.analysis_only is True
            assert view.payload == {"scores": [1.0]}
            assert view.schema_id == FAMILY.schema_id
        finally:
            traced.annotations[SIDECAR_ANNOTATIONS_KEY].pop(FAMILY.family_id, None)


class TestPersistedAtV9:
    def test_sidecar_key_is_no_longer_gated(self) -> None:
        # The C01-era pre-release registration retired at the C07 v9 bump.
        assert SIDECAR_ANNOTATIONS_KEY not in gated_annotations_keys()

    def test_real_save_round_trips_the_sidecar(self, registered_family, tmp_path) -> None:
        model = nn.Sequential(nn.Linear(4, 3))
        trace = tl.trace(model, torch.randn(2, 4))
        attach_sidecar(trace, FAMILY.family_id, {"scores": [0.5, 1.5]})
        path = tmp_path / "plain.tlspec"
        tl.save(trace, path)
        loaded = tl.load(path)
        assert read_sidecar(loaded, FAMILY.family_id) == {"scores": [0.5, 1.5]}

"""tlspec v9 annotation-family admissions (C07 coordinated schema write).

Three reserved ``Trace.annotations`` families settle at v9:

- ``"sidecar"`` (C01 seam): flips from the pre-release registrar to plain
  persistence; loads validate the namespace/envelope SHAPE fail-closed while
  payload semantics stay with the owning family's read-time validator.
- ``"capture_advisories"`` (A09): capture-time honesty disclosures persist;
  loads validate the row shape fail-closed (a malformed row would crash the
  report preambles that render it).
- ``"health_facts"`` (C02): admitted with its shipped consumer contract as
  the validation row -- fail-closed DEGRADATION, never a false clean: a
  malformed or tampered payload renders NOT-CHECKED with the
  ``persisted_invalid`` source basis, and a well-formed record survives the
  round trip as the load-time authority.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import ArtifactError


def _traced():
    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), torch.randn(2, 4))


def _reload_with_annotation(tmp_path, key, value, name):
    trace = _traced()
    trace.annotations[key] = value
    path = tmp_path / f"{name}.tlspec"
    trace.save(path)
    return tl.load(path)


# ---------------------------------------------------------------------------
# sidecar: plain persistence + fail-closed envelope shape
# ---------------------------------------------------------------------------


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "namespace",
    [
        "not-a-mapping",
        {"NotNamespaced": {"schema_id": "s", "version": 1, "owner": "o", "payload": {}}},
        {"myorg.saliency": {"schema_id": "", "version": 1, "owner": "o", "payload": {}}},
        {"myorg.saliency": {"schema_id": "s", "version": 0, "owner": "o", "payload": {}}},
        {"myorg.saliency": {"schema_id": "s", "version": 1, "owner": "", "payload": {}}},
        {
            "myorg.saliency": {
                "schema_id": "s",
                "version": 1,
                "owner": "o",
                "payload": {},
                "extra": 1,
            }
        },
    ],
)
def test_malformed_sidecar_namespace_refuses_at_load(tmp_path, namespace):
    with pytest.raises(ArtifactError) as excinfo:
        _reload_with_annotation(tmp_path, "sidecar", namespace, "sidecar_bad")
    assert getattr(excinfo.value, "fields", {}).get("code") == "artifact_sidecar_invalid"


# ---------------------------------------------------------------------------
# capture_advisories: persists; fail-closed row shape
# ---------------------------------------------------------------------------


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "rows",
    [
        "not-a-list",
        [{"kind": "k", "count": 1, "message": "m"}],
        [{"kind": "", "count": 1, "first_location": None, "message": "m"}],
        [{"kind": "k", "count": 0, "first_location": None, "message": "m"}],
        [{"kind": "k", "count": 1, "first_location": 4, "message": "m"}],
        [{"kind": "k", "count": 1, "first_location": None, "message": 9}],
    ],
)
def test_malformed_capture_advisories_refuse_at_load(tmp_path, rows):
    with pytest.raises(ArtifactError) as excinfo:
        _reload_with_annotation(tmp_path, "capture_advisories", rows, "advisories_bad")
    assert getattr(excinfo.value, "fields", {}).get("code") == "artifact_capture_advisories_invalid"


# ---------------------------------------------------------------------------
# health_facts: consumer-contract validation row (degrade, never a false clean)
# ---------------------------------------------------------------------------


def test_tampered_health_facts_are_never_the_authority(tmp_path):
    """A malformed persisted record is rejected fail-closed, never trusted.

    With retained payloads the loaded trace re-derives an HONEST verdict
    from real bytes (that is the designed degradation, not a false clean);
    the guarantee under test is that the tampered record itself never
    becomes the served ``persisted``-basis authority.
    """

    from torchlens.report._health import health_facts

    trace = _traced()
    health_facts(trace)
    trace.annotations["health_facts"] = {"schema_version": "corrupt"}
    path = tmp_path / "health_bad.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    served = health_facts(loaded)
    assert served.basis != "persisted", (
        "a malformed persisted health record must never be served as the persisted-basis authority"
    )
    assert served.source_basis is None

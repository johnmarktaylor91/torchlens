"""tlspec v9 audit-row admissions (C07 coordinated schema write).

Three families land in the ONE closed ``intervention_audit`` row grammar:

1. ACT site rows admit the OPTIONAL string ``source`` disclosure the shipped
   selection-door writer already emits (A04 dense-subspace point-of-use
   stamp). Pre-v9 the validator refused it, so a saved selection-intervened
   artifact refused ITS OWN load (C03 measured defect).
2. PARAM rows (param-substitution ``do(tl.params(...), edit)``) are admitted
   with their shipped ``params[]``/``disclosure`` payload. Same defect class.
3. EVENT rows -- the ``intervention_event_v2`` transaction envelope as a
   first-class audit row kind, with the OPTIONAL hash-chain extension
   (``seq``/``prev_event_digest``) the persistent experiment ledger (F03)
   writes against without another version bump.

The validator stays a tripwire: every admission is closed-vocabulary and
fail-closed; malformed variants refuse typed.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io.forgery_validation import _validate_audit_row

_DIGEST = hashlib.sha256(b"schema-v9-audit").hexdigest()
_CHAIN_DIGEST = hashlib.sha256(b"schema-v9-chain").hexdigest()


class _TwoConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))))


@pytest.fixture(scope="module")
def log():
    torch.manual_seed(0)
    trace = tl.trace(
        _TwoConv(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        yield trace
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# End-to-end: the two measured self-load-refusal defects flip to green.
# ---------------------------------------------------------------------------


def test_selection_intervened_artifact_loads_its_own_save(log, tmp_path):
    """ACT rows with per-site source disclosures round-trip through load."""

    fork = log.fork()
    fork.do(tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(fork), tl.zero_ablate())
    act_rows = [row for row in fork.intervention_audit if row.get("kind") == "ACT"]
    assert any("source" in site for row in act_rows for site in row.get("sites", ())), (
        "fixture must exercise the shipped per-site source disclosure"
    )
    path = tmp_path / "selection_intervened.tlspec"
    tl.save(fork, path)
    loaded = tl.load(path)
    loaded_rows = [row for row in loaded.intervention_audit if row.get("kind") == "ACT"]
    assert any("source" in site for row in loaded_rows for site in row.get("sites", ()))


@pytest.mark.smoke
def test_param_intervened_artifact_loads_its_own_save(log, tmp_path):
    """PARAM rows (parameter substitution) round-trip through load."""

    fork = log.fork()
    fork.do(tl.params("c1.weight"), tl.scale(0.5))
    assert any(row.get("kind") == "PARAM" for row in fork.intervention_audit)
    path = tmp_path / "param_intervened.tlspec"
    tl.save(fork, path)
    loaded = tl.load(path)
    loaded_rows = [row for row in loaded.intervention_audit if row.get("kind") == "PARAM"]
    assert loaded_rows, "PARAM audit row must survive the round trip"
    for row in loaded_rows:
        assert row["disclosure"].startswith("parameter values substituted")
        assert all(param["param_address"] for param in row["params"])


# ---------------------------------------------------------------------------
# Unit grammar: admissions are closed and fail-closed.
# ---------------------------------------------------------------------------


def _act_row(**site_overrides):
    site = {"relation": "exact", "selected": 1, "site_key": "('s1|a',)"}
    site.update(site_overrides)
    return {
        "kind": "ACT",
        "edit": "zero_ablate",
        "selection_repr": "units(...)",
        "resolve_digest": _DIGEST,
        "sites": [site],
    }


def _param_row(**overrides):
    row = {
        "kind": "PARAM",
        "edit": "scale",
        "selection_repr": "params('c1.weight')",
        "resolve_digest": _DIGEST,
        "disclosure": "parameter values substituted at consumption for replay",
        "params": [
            {
                "param_address": "c1.weight",
                "consumers": ["conv2d_1_1:1"],
                "occurrences": [
                    {
                        "edge_address": "('conv2d_1_1:1', 'arg', '(1,)')",
                        "consumer": "conv2d_1_1:1",
                        "value_digest": _DIGEST,
                    }
                ],
            }
        ],
    }
    row.update(overrides)
    return row


def _event_row(**overrides):
    row = {
        "kind": "EVENT",
        "schema": "intervention_event_v2",
        "event_id": "lineage:1",
        "transaction_id": "lineage:1",
        "parent_event_id": None,
        "lane": "replay",
        "door": "do",
        "edit_names": ["zero_ablate"],
        "selection_repr": "units(...)",
        "status": "fired",
        "fire_count": 1,
        "site_keys": ["s1|a"],
        "rules": [{"rule_id": "r1"}],
        "zero_fire_rule_ids": [],
        "error": None,
        "event_digest": _DIGEST,
    }
    row.update(overrides)
    return row


@pytest.mark.smoke
def test_act_site_source_is_optional_and_string_typed():
    assert _validate_audit_row(0, _act_row()) is not None
    assert _validate_audit_row(0, _act_row(source="subspace basis sha256:ab")) is not None
    with pytest.raises(TorchLensIOError):
        _validate_audit_row(0, _act_row(source=7))
    with pytest.raises(TorchLensIOError):
        _validate_audit_row(0, _act_row(unexpected="x"))


@pytest.mark.smoke
def test_param_row_grammar_is_closed():
    digest, edit, targets = _validate_audit_row(0, _param_row())
    assert digest == _DIGEST and edit == "scale"
    assert ("PARAM", "c1.weight") in targets
    with pytest.raises(TorchLensIOError):
        _validate_audit_row(0, _param_row(disclosure=None))
    bad = _param_row()
    bad["params"][0]["param_address"] = ""
    with pytest.raises(TorchLensIOError):
        _validate_audit_row(0, bad)
    bad = _param_row()
    bad["params"][0]["occurrences"][0]["value_digest"] = "nope"
    with pytest.raises(TorchLensIOError):
        _validate_audit_row(0, bad)
    bad = _param_row()
    bad["params"][0]["extra"] = 1
    with pytest.raises(TorchLensIOError):
        _validate_audit_row(0, bad)


@pytest.mark.smoke
def test_event_row_grammar_and_chain_extension():
    assert _validate_audit_row(0, _event_row()) is None, (
        "EVENT rows are transaction envelopes: they stay outside the "
        "helper-recipe resolve-digest relation"
    )
    assert _validate_audit_row(0, _event_row(seq=0, prev_event_digest=None)) is None
    assert _validate_audit_row(0, _event_row(seq=3, prev_event_digest=_CHAIN_DIGEST)) is None
    for mutation in (
        {"schema": "intervention_event_v1"},
        {"lane": "warp"},
        {"status": "maybe"},
        {"fire_count": -1},
        {"fire_count": True},
        {"edit_names": [1]},
        {"rules": [["not-a-mapping"]]},
        {"error": 7},
        {"event_digest": "short"},
        {"seq": -1},
        {"prev_event_digest": "short"},
        {"surprise": "field"},
    ):
        with pytest.raises(TorchLensIOError):
            _validate_audit_row(0, _event_row(**mutation))


@pytest.mark.smoke
def test_param_selection_recipe_form_is_admitted_and_closed():
    """The shipped param-substitution FireRecord recipe loads; malformed refuses.

    Third instance of the C03 defect class, found by this lane's end-to-end
    round trip: per-occurrence PARAM recipes ({resolve_digest, edge_address,
    param_address, note}) were outside the pre-v9 recipe grammar.
    """

    from torchlens._io.forgery_validation import _validate_selection_recipe

    recipe = {
        "resolve_digest": _DIGEST,
        "edge_address": "('conv2d_1_1:1', 'arg', '(1,)')",
        "param_address": "c1.weight",
        "note": "parameter substituted at consumption for replay; live parameter unchanged",
    }
    assert _validate_selection_recipe(0, recipe) == (_DIGEST, "PARAM", "c1.weight")
    for mutation in (
        {"param_address": ""},
        {"note": 3},
        {"edge_address": "not a tuple"},
    ):
        with pytest.raises(TorchLensIOError):
            _validate_selection_recipe(0, {**recipe, **mutation})


def test_unknown_kind_still_refuses_typed():
    with pytest.raises(TorchLensIOError) as excinfo:
        _validate_audit_row(0, {"kind": "MYSTERY"})
    assert "'ACT', 'EDGE', 'PARAM', or 'EVENT'" in str(excinfo.value)

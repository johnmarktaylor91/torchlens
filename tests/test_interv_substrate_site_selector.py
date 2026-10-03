"""C03: tl.site() structural selector + live site-key minting (Build 0a/0c).

The address law's structural lane: a ``site`` WHERE term is valid in EVERY
lane -- post hoc against ``op.site_key``, and during a live capture against
the streaming minter armed by the ``tl.trace`` entry. The parity pin is the
verification surgery risk 1 demands: the SAME streaming minter, fed retained
ops in execution order, reproduces the postprocess keys byte-identically.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import ArgumentTypeError
from torchlens.intervention import site
from torchlens.intervention.errors import SelectorCapabilityError
from torchlens.intervention.site_keys import mint_keys_in_execution_order


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _ReusedModule(nn.Module):
    """One relu INSTANCE called twice in one block (the ResNet reuse idiom)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc2(self.act(self.fc1(x))))


class _Recurrent(nn.Module):
    """One cell module applied for three timesteps (pass-qualified sites)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.cell(x))
        return x


def _ready(model: nn.Module) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_site_selector_construction_refusals() -> None:
    with pytest.raises(ArgumentTypeError) as excinfo:
        site()
    assert excinfo.value.fields["code"] == "site_selector_empty"
    with pytest.raises(ArgumentTypeError) as excinfo:
        site("s1||relu||1", op_type="relu")
    assert excinfo.value.fields["code"] == "site_selector_key_conflict"
    with pytest.raises(ArgumentTypeError) as excinfo:
        site("not-a-site-key")
    assert excinfo.value.fields["code"] == "site_selector_key_invalid"


def test_site_classifies_structural() -> None:
    from torchlens.intervention.spec import classify_where

    assert classify_where(site(op_type="relu")) == "structural"


# ---------------------------------------------------------------------------
# Post-hoc addressing (site lifecycle)
# ---------------------------------------------------------------------------


def test_post_hoc_resolution_by_components_and_key() -> None:
    log = _ready(_TinyModel())
    by_type = log.resolve_sites(site(op_type="relu"))
    assert by_type.labels() == ("relu_1_2",)
    key = log["relu_1_2"].site_key
    assert log.resolve_sites(site(key)).labels() == ("relu_1_2",)
    by_module = log.resolve_sites(site(module_path="fc1"))
    assert "linear_1_1" in by_module.labels()


def test_reused_module_calls_share_one_site_key_family() -> None:
    """Module reuse COLLIDES the bare structural key (the documented L1 fact).

    Ordinals restart per pass-qualified innermost call instance, so a reused
    relu INSTANCE mints the same key for each call (measured on resnet18:
    10.6% of sites). site(key) therefore addresses the position FAMILY --
    both calls -- and cross-trace per-occurrence claims must ride the guarded
    join, never the bare key (leverage D-1/D-2; ordinals are display-only).
    """

    log = _ready(_ReusedModule())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fan-out disclosure, expected here
        labels = log.resolve_sites(site(op_type="relu")).labels()
        assert len(labels) == 2
        keys = {log.ops[label].site_key for label in labels}
        assert len(keys) == 1, "reused-module calls share ONE structural position key"
        (shared_key,) = keys
        assert set(log.resolve_sites(site(shared_key)).labels()) == set(labels)


# ---------------------------------------------------------------------------
# Live lane (capture): minting parity + intervene= addressing
# ---------------------------------------------------------------------------


def test_live_site_intervention_fires_at_the_right_site() -> None:
    model = _TinyModel()
    log = tl.trace(model, torch.randn(2, 4), intervene=tl.when(site(op_type="relu"), tl.scale(0.0)))
    assert torch.count_nonzero(log["relu_1_2"].out) == 0
    events = [
        row
        for row in log.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]
    assert events[-1]["site_keys"] == [log["relu_1_2"].site_key]


@pytest.mark.smoke
def test_live_minted_keys_match_posthoc_keys() -> None:
    """PARITY PIN (surgery risk 1): live-minted key == postprocess key."""

    model = _TinyModel()
    log = tl.trace(
        model,
        torch.randn(2, 4),
        intervene=tl.when(site(op_type="relu"), tl.scale(1.0)),
    )
    events = [
        row
        for row in log.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]
    assert events[-1]["site_keys"] == [log["relu_1_2"].site_key]


@pytest.mark.parametrize("model_cls", [_TinyModel, _ReusedModule, _Recurrent])
def test_streaming_minter_reproduces_postprocess_keys(model_cls: type[nn.Module]) -> None:
    """The streaming minter, fed retained ops in execution order, reproduces
    every postprocess key byte-identically (orphan-free fixtures)."""

    log = _ready(model_cls())
    reminted = mint_keys_in_execution_order(log)
    mismatches = {
        label: (reminted[label], log.ops[label].site_key)
        for label in reminted
        if log.ops[label].site_key != reminted[label]
    }
    assert not mismatches, f"streaming/postprocess key drift: {mismatches}"


def test_recurrent_pass_qualified_ordinals_stay_distinct() -> None:
    """Pass-qualified call instances restart ordinals: three timestep relus
    (root instance) count 1..3, and each key resolves exactly one op."""

    log = _ready(_Recurrent())
    relu_labels = [label for label in log.op_labels if log.ops[label].layer_type == "relu"]
    keys = [log.ops[label].site_key for label in relu_labels]
    assert len(set(keys)) == len(keys) == 3


# ---------------------------------------------------------------------------
# Typed boundaries
# ---------------------------------------------------------------------------


def test_unarmed_capture_surface_refuses_typed() -> None:
    """tl.record does not arm the live minter: site() there refuses typed.

    Fastlog tolerates per-op predicate failures by design ("auto" mode), so
    the typed capability refusal is asserted under fail-fast, where the
    SelectorCapabilityError propagates from the first evaluated op.
    """

    model = _TinyModel()
    with pytest.raises(SelectorCapabilityError) as excinfo:
        tl.record(
            model,
            torch.randn(2, 4),
            save=site(op_type="relu"),
            on_predicate_error="fail-fast",
        )
    assert "site" in str(excinfo.value)


def test_do_accepts_site_spec_on_the_replay_engine() -> None:
    log = _ready(_TinyModel())
    fork = log.fork()
    fork.do(tl.when(site(op_type="relu"), tl.scale(0.0)))
    assert torch.count_nonzero(fork["relu_1_2"].out) == 0


def test_rerun_with_sticky_site_hook_refuses_at_preflight() -> None:
    """Live-forward engines cannot serve structural site targets mid-forward:
    the refusal fires at the rerun PREFLIGHT, before any forward runs."""

    model = _TinyModel()
    log = _ready(model)
    fork = log.fork()
    fork.attach_hooks(site(op_type="relu"), tl.scale(0.0), confirm_mutation=True)
    with pytest.raises(SelectorCapabilityError):
        fork.run(model, torch.randn(2, 4))

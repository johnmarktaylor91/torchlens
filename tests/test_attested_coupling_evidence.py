"""Coupling evidence machinery (lane F42): digests, fire counts, segments.

Pre-flip unit coverage of the settlement writers and derived reads: the
intervention digest is deterministic over ordered fire facts, per-step fire
counts distinguish measured zero from no-opportunity, segments split at
measured break joins and never span one, the capture-digest attestation
binds a ledger to ITS product and refuses foreign/tampered ledgers, and both
replay doors (run() and do()) honor the verdict's fresh-ledger-or-refuse
law. End-to-end coupled captures land with the REFUSED -> COUPLED flip.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._capture._episode_coupling import (
    CouplingFire,
    CouplingSession,
    attest_coupling,
    coupling_segments,
    refuse_coupled_replay,
    settlement_fire_counts,
)
from torchlens.errors._episode import EpisodeCaptureError
from torchlens.options import EpisodeOptions


class Step(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.inner = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.inner(x))


class Root(nn.Module):
    def __init__(self, n_steps: int = 3):
        super().__init__()
        self.step = Step()
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self.step(x) + 0.0
        return x


def _episode_trace(model: Root, x: torch.Tensor, **kwargs) -> tl.Trace:
    return tl.trace(
        model,
        x,
        episode=EpisodeOptions(
            stepped_module=model.step, num_steps=model.n_steps, step_output_kind="digest"
        ),
        **kwargs,
    )


def _fire(step, site="relu_1_2", replaced=True):
    return CouplingFire(
        step=step,
        site=site,
        helper="zero_ablate",
        timing="post",
        signal="forward",
        replaced=replaced,
    )


# ---------------------------------------------------------------------------
# Session facts and the intervention digest
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_coupling.py.
def test_outside_step_fires_are_bucketed_not_guessed():
    """Fires outside every stepped call never land in a row's count."""

    session = CouplingSession()
    session.fires.extend([_fire(None), _fire(1), _fire(None)])
    assert session.outside_step_fires == 2
    assert settlement_fire_counts(session, started=2, n_total=2) == [0, 1]


# ---------------------------------------------------------------------------
# Per-segment facts across broken joins (pure derivation)
# ---------------------------------------------------------------------------


def _payload(grades, statuses, fire_counts=None, envelope_present=True):
    n = len(statuses)
    counts = fire_counts or [None] * n
    header = {"stepped_module": "step", "capture_digest": "x" * 64}
    if envelope_present:
        header["step_join"] = {
            "schema": "episode_step_join_v1",
            "claim": "measured",
            "grades": grades,
            "basis": {},
            "break_step": next(
                (i for i, g in enumerate(grades) if g in ("exogenous", "declared")), None
            ),
            "live_break_step": None,
            "exogenous_positions": {},
            "reasons": {},
        }
    rows = [{"episode_step": i, "status": statuses[i], "fire_count": counts[i]} for i in range(n)]
    return {"header": header, "rows": rows}


class _Subject:
    def __init__(self, payload):
        self.annotations = {"episode": payload}


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_coupling.py.
def test_segments_disclose_unchecked_joins_and_unmeasured_basis():
    """Unchecked interior joins and absent envelopes are disclosed, not attested."""

    measured = coupling_segments(
        _Subject(_payload(grades=[None, "unchecked", "continuous"], statuses=["complete"] * 3))
    )
    assert len(measured) == 1 and measured[0].unchecked_joins == 1
    assert measured[0].fire_count_total is None  # no counts -> no total claim

    unmeasured = coupling_segments(
        _Subject(_payload(grades=[], statuses=["complete"] * 3, envelope_present=False))
    )
    assert len(unmeasured) == 1
    assert unmeasured[0].join_basis == "unmeasured"
    assert unmeasured[0].unchecked_joins == 2  # every interior join unmeasured


# ---------------------------------------------------------------------------
# Attestation: the capture digest consumed (the positive binding claim)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_coupling.py.
def test_attestation_refuses_a_pre_binding_artifact():
    """A ledger with no capture digest cannot be attested (re-capture)."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    log.annotations["episode"]["header"]["capture_digest"] = None
    with pytest.raises(EpisodeCaptureError) as excinfo:
        attest_coupling(log)
    assert excinfo.value.fields["code"] == "episode_coupling_unmintable"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_coupling.py.
def test_attestation_none_for_plain_captures():
    """Plain products mirror Trace.episode: None, never a fake verdict."""

    log = tl.trace(Root(), torch.randn(2, 4))
    assert log.episode_coupling is None
    assert attest_coupling(log) is None
    assert coupling_segments(log) == ()


# ---------------------------------------------------------------------------
# Replay doors: derive a fresh ledger or refuse; never a foreign ledger
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_coupling.py, intervention/_trace_run.py.
def test_run_refuses_a_coupled_product_before_execution():
    """The run gate fires on intervention evidence, both digests present."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    # Forge the coupled marker: the gate reads the header slot, and the flip
    # commit adds real end-to-end coupled products.
    log.annotations["episode"]["header"]["intervention_digest"] = "f" * 64
    with pytest.raises(EpisodeCaptureError) as excinfo:
        refuse_coupled_replay(log)
    assert excinfo.value.fields["code"] == "episode_coupled_replay_underivable"
    with pytest.raises(EpisodeCaptureError):
        log.run(inputs=torch.randn(2, 4))


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_coupling.py, _selection/do_plan.py, _selection/selection.py.
def test_do_quarantines_inherited_episode_evidence():
    """A do() edit on an episode fork quarantines the stale step evidence."""

    model = Root()
    log = _episode_trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(replayable=True, save_mode="reference"),
    )
    fork = log.fork()
    assert fork.annotations["episode"].get("rows") is not None  # inherited
    fork.do(fork[tl.units("relu_1_2:2", [(0, 0)])], tl.zero_ablate())
    quarantined = fork.annotations["episode"]
    assert quarantined.get("quarantined") is True
    assert quarantined.get("code") == "episode_evidence_dropped_perturbed_replay"
    # The source product keeps its evidence untouched.
    assert log.annotations["episode"].get("rows") is not None
    assert log.episode_coupling.bound is True

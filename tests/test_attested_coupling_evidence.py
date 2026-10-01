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

import dataclasses

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_coupling import (
    CouplingFire,
    CouplingSession,
    attest_coupling,
    coupling_segments,
    mint_intervention_digest,
    quarantine_episode_after_perturbed_replay,
    refuse_coupled_replay,
    settlement_fire_counts,
)
from torchlens.errors.episode import EpisodeCaptureError
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke


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
        episode=EpisodeSpec(
            stepped_module=model.step, n_steps=model.n_steps, step_output_kind="digest"
        ),
        **kwargs,
    )


def _fire(step, site="relu_1_2", replaced=True):
    return CouplingFire(
        step=step,
        site=site,
        helper="zero_ablate",
        timing="post",
        direction="forward",
        replaced=replaced,
    )


# ---------------------------------------------------------------------------
# Session facts and the intervention digest
# ---------------------------------------------------------------------------


def test_fire_counts_distinguish_zero_from_no_opportunity():
    """Started steps read measured counts (0 included); unstarted read None."""

    session = CouplingSession()
    session.fires.extend([_fire(0), _fire(0), _fire(2)])
    assert settlement_fire_counts(session, started=3, n_total=5) == [2, 0, 1, None, None]
    assert settlement_fire_counts(None, started=3, n_total=5) is None


def test_outside_step_fires_are_bucketed_not_guessed():
    """Fires outside every stepped call never land in a row's count."""

    session = CouplingSession()
    session.fires.extend([_fire(None), _fire(1), _fire(None)])
    assert session.outside_step_fires == 2
    assert settlement_fire_counts(session, started=2, n_total=2) == [0, 1]


def test_intervention_digest_is_deterministic_and_order_sensitive():
    """Identical fire evidence mints the identical digest; order matters."""

    a = CouplingSession(rule_identity=(("rule-1", "tl.func('relu')"),))
    b = CouplingSession(rule_identity=(("rule-1", "tl.func('relu')"),))
    for session in (a, b):
        session.fires.extend([_fire(0), _fire(1)])
    assert mint_intervention_digest(a) == mint_intervention_digest(b)

    swapped = CouplingSession(rule_identity=(("rule-1", "tl.func('relu')"),))
    swapped.fires.extend([_fire(1), _fire(0)])
    assert mint_intervention_digest(swapped) != mint_intervention_digest(a)

    zero = CouplingSession(rule_identity=(("rule-1", "tl.func('relu')"),))
    assert mint_intervention_digest(zero) != mint_intervention_digest(a)


def test_digest_covers_replaced_flag_and_outside_fires():
    """Observing (replaced=False) and outside-step fires change the digest."""

    replaced = CouplingSession()
    replaced.fires.append(_fire(0, replaced=True))
    observing = CouplingSession()
    observing.fires.append(_fire(0, replaced=False))
    assert mint_intervention_digest(replaced) != mint_intervention_digest(observing)

    inside = CouplingSession()
    inside.fires.append(_fire(0))
    outside = CouplingSession()
    outside.fires.append(_fire(None))
    assert mint_intervention_digest(inside) != mint_intervention_digest(outside)


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


def test_segments_split_at_break_joins_and_never_span_one():
    """exogenous/declared joins divide segments; facts stay per segment."""

    payload = _payload(
        grades=[None, "continuous", "exogenous", "continuous", "declared"],
        statuses=["complete"] * 5,
        fire_counts=[1, 0, 2, 0, 3],
    )
    segments = coupling_segments(_Subject(payload))
    assert [(s.start_step, s.end_step) for s in segments] == [(0, 1), (2, 3), (4, 4)]
    assert [s.fire_count_total for s in segments] == [1, 2, 3]
    assert all(s.join_basis == "measured" for s in segments)
    assert all(s.unchecked_joins == 0 for s in segments)


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


def test_segments_close_at_the_started_prefix():
    """Absent rows never join a segment (no opportunity, no claim)."""

    payload = _payload(
        grades=[None, "continuous", None, None],
        statuses=["complete", "complete", "absent", "absent"],
    )
    segments = coupling_segments(_Subject(payload))
    assert [(s.start_step, s.end_step) for s in segments] == [(0, 1)]


# ---------------------------------------------------------------------------
# Attestation: the capture digest consumed (the positive binding claim)
# ---------------------------------------------------------------------------


def test_attestation_binds_a_real_uncoupled_episode_product():
    """A real capture attests bound=True, coupled=False, entry-dark counts."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    attestation = log.episode_coupling
    assert attestation is not None
    assert attestation.bound is True
    assert attestation.coupled is False
    assert attestation.intervention_digest is None
    assert attestation.fire_counts == (None, None, None)
    assert len(attestation.segments) == 1
    assert attestation.segments[0].join_basis == "measured"


def test_attestation_refuses_a_tampered_ledger():
    """Editing the persisted evidence breaks the binding, typed."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    header = log.annotations["episode"]["header"]
    header["capture_digest"] = "0" * 64
    with pytest.raises(EpisodeCaptureError) as excinfo:
        attest_coupling(log)
    assert excinfo.value.fields["code"] == "episode_coupling_unbound"
    assert excinfo.value.fields["persisted_digest"] == "0" * 64


def test_attestation_refuses_a_pre_binding_artifact():
    """A ledger with no capture digest cannot be attested (re-capture)."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    log.annotations["episode"]["header"]["capture_digest"] = None
    with pytest.raises(EpisodeCaptureError) as excinfo:
        attest_coupling(log)
    assert excinfo.value.fields["code"] == "episode_coupling_unmintable"


def test_attestation_none_for_plain_captures():
    """Plain products mirror Trace.episode: None, never a fake verdict."""

    log = tl.trace(Root(), torch.randn(2, 4))
    assert log.episode_coupling is None
    assert attest_coupling(log) is None
    assert coupling_segments(log) == ()


def test_attestation_survives_save_load(tmp_path):
    """The binding recomputes identically from the loaded product."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    target = tmp_path / "episode.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    attestation = loaded.episode_coupling
    assert attestation is not None and attestation.bound is True
    assert attestation.capture_digest == log.episode_coupling.capture_digest


# ---------------------------------------------------------------------------
# Replay doors: derive a fresh ledger or refuse; never a foreign ledger
# ---------------------------------------------------------------------------


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


def test_run_keeps_uncoupled_episode_replay_shipped_behavior():
    """Uncoupled episode products still run; evidence quarantines (F40a)."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    refuse_coupled_replay(log)  # no-op: uncoupled
    result = log.run(inputs=torch.randn(2, 4))
    episode_key = result.trace.annotations.get("episode")
    assert episode_key is not None and episode_key.get("quarantined") is True
    assert episode_key.get("code") == "episode_evidence_dropped_fresh_execution"


def test_do_quarantines_inherited_episode_evidence():
    """A do() edit on an episode fork quarantines the stale step evidence."""

    model = Root()
    log = _episode_trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
        save_mode="reference",
    )
    fork = log.fork()
    assert fork.annotations["episode"].get("rows") is not None  # inherited
    fork.do(tl.units("relu_1_2:2", [(0, 0)]).resolve(fork), tl.zero_ablate())
    quarantined = fork.annotations["episode"]
    assert quarantined.get("quarantined") is True
    assert quarantined.get("code") == "episode_evidence_dropped_perturbed_replay"
    # The source product keeps its evidence untouched.
    assert log.annotations["episode"].get("rows") is not None
    assert log.episode_coupling.bound is True


def test_quarantine_is_idempotent_and_plain_safe():
    """The quarantine helper never double-wraps and ignores plain traces."""

    plain = tl.trace(Root(), torch.randn(2, 4))
    quarantine_episode_after_perturbed_replay(plain)  # no episode key: no-op
    assert "episode" not in plain.annotations

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    quarantine_episode_after_perturbed_replay(log)
    first = dict(log.annotations["episode"])
    quarantine_episode_after_perturbed_replay(log)
    assert log.annotations["episode"] == first


def test_coupling_attestation_is_frozen():
    """The verdict object is immutable evidence, never a mutable report."""

    model = Root()
    log = _episode_trace(model, torch.randn(2, 4))
    attestation = log.episode_coupling
    with pytest.raises(dataclasses.FrozenInstanceError):
        attestation.coupled = True

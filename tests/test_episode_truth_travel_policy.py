"""Annotations TRAVEL POLICY red tests (lane F40a, foldA D6).

The foldA round-3 headline receipts, as permanent regressions:

* R11 -- a fresh ``run(inputs=...)`` product carried the ORIGINAL episode
  ledger (``provenance_tier=exact``) under ``path_faithfulness=VERIFIED``
  with zero warnings.
* R12 -- the same class generalized: capture-time observer values
  (``annotations["logged_values"]``) rode a fresh execution the same way.

The fix under test: the per-sub-key travel registry
(``torchlens/capture/_annotations_travel.py``) applied at the ONE provider
settlement finalizer. VERIFIED may never coexist with foreign step evidence.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._annotations_travel import (
    TRAVEL_CAPTURE_EVIDENCE,
    register_travel_policy,
    registered_travel_policies,
    scrub_fresh_execution_annotations,
    travel_policy_for,
)
from torchlens.capture._episode_ledger import (
    capture_kind_for,
    episode_ledger_for,
    validate_loaded_episode_annotations,
)
from torchlens.options import EpisodeSpec
from torchlens.runnable import PathFaithfulness

pytestmark = pytest.mark.smoke

V = 16
N_STEPS = 3


class _Step(nn.Module):
    """Deterministic stepped model: next token = (last token + 1) % V.

    The permutation-matrix construction (the R11 probe fixture) guarantees
    different inputs produce different tokens, so foreign evidence is
    unambiguous.
    """

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(V))
            self.out.weight.copy_(torch.roll(torch.eye(V), shifts=1, dims=0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


class _Greedy(nn.Module):
    """Greedy generation loop: the episode root."""

    def __init__(self, n: int = N_STEPS) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids[:, -self.n :]


def _episode_trace(model: _Greedy, **extra) -> tl.Trace:
    """Capture one episode; the CALLER keeps the model alive for run()."""

    return tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
        **extra,
    )


def test_fresh_run_product_never_carries_foreign_episode_ledger():
    """R11 verbatim: run(inputs=x2) on an episode capture of x1.

    The product's output is correct for x2, the report says VERIFIED, and the
    product must NOT present x1's per-step tokens as its own evidence.
    """

    model = _Greedy()
    trace = _episode_trace(model)
    capture_ledger = episode_ledger_for(trace)
    assert capture_ledger is not None
    assert [row.step_output for row in capture_ledger.rows] == [(2,), (3,), (4,)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = trace.run(inputs=torch.tensor([[9]]))

    assert result.output.tolist() == [[10, 11, 12]]
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    # The headline invariant: VERIFIED never coexists with foreign step
    # evidence -- the product parses to NO episode ledger at all.
    assert episode_ledger_for(result.trace) is None
    note = result.trace.annotations.get("episode")
    assert note == {
        "quarantined": True,
        "code": "episode_evidence_dropped_fresh_execution",
        "detail": note["detail"],
    }
    assert "fresh re-execution" in note["detail"]
    assert capture_ledger.header.episode_id in note["detail"]
    # The drop is a disclosure on the product, not warning noise.
    assert [str(w.message) for w in caught] == []
    # The SOURCE trace keeps its own honest ledger untouched.
    assert episode_ledger_for(trace) is not None


def test_fresh_run_product_never_carries_foreign_logged_values():
    """R12 verbatim: observer values keyed to x1 never ride the x2 product."""

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x) * 2

    model = _M()
    x1 = torch.ones(1, 4)
    trace = tl.trace(model, x1)
    trace.annotations.setdefault("logged_values", {})["input_sum"] = float(x1.sum())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = trace.run(inputs=torch.ones(1, 4) * 7)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert "logged_values" not in result.trace.annotations
    assert result.trace.logged_values == {}
    assert [str(w.message) for w in caught] == []
    # The source trace keeps its capture-time observation.
    assert trace.logged_values == {"input_sum": 4.0}


def test_fresh_execution_scrubs_every_registered_sub_key():
    """The generalized composition row (foldA s7): fresh execution x EVERY
    registered annotations sub-key, asserted per key from the registry."""

    registry = registered_travel_policies()
    assert set(registry) >= {"episode", "logged_values"}
    assert "sidecar" not in registry  # C01's owner decides that key

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    for key, row in registry.items():
        assert row.policy == TRAVEL_CAPTURE_EVIDENCE
        model = _M()  # keep alive: the trace holds its source model weakly
        trace = tl.trace(model, torch.ones(1, 4))
        if key not in trace.annotations:
            # Seed a payload so the drop is observable. The episode key uses
            # its real declared-marker grammar; other keys take a plain dict.
            if key == "episode":
                trace.annotations[key] = {"declared": {"episode_id": "ep-test"}}
            else:
                trace.annotations[key] = {"probe": 1.0}
        result = trace.run(inputs=torch.ones(1, 4) * 3)
        product_annotations = result.trace.annotations
        if row.drop_mode == "note":
            assert product_annotations[key]["quarantined"] is True
            assert product_annotations[key]["code"] == row.note_code
        else:
            assert key not in product_annotations


def test_guarded_fast_rerun_never_exposes_foreign_ledger():
    """Gate row: guarded-fast (run(fast=True)) never exposes a foreign ledger."""

    model = _Greedy()
    trace = _episode_trace(
        model,
        save=tl.in_module("step") | tl.func("cat") | tl.func("argmax") | tl.func("getitem"),
    )
    ledger_before = episode_ledger_for(trace)
    assert ledger_before is not None
    result = trace.run(inputs=torch.tensor([[9]]), fast=True)
    # The fast-live product IS the live trace, whose payloads were refreshed
    # in place: its episode evidence no longer describes what it presents.
    assert episode_ledger_for(result.trace) is None
    assert capture_kind_for(result.trace) == "plain"
    # The pre-read ledger object stays valid (the GAP-9 cross-tier pin reads
    # it before re-running); only the product stops claiming it.
    assert [row.step_output for row in ledger_before.rows] == [(2,), (3,), (4,)]


def test_plain_fork_is_not_a_fresh_execution_and_carries_the_ledger():
    """A fork's records hold the capture's own values; evidence travels."""

    trace = _episode_trace(_Greedy())
    fork = trace.fork()
    forked_ledger = episode_ledger_for(fork)
    assert forked_ledger is not None
    assert [row.step_output for row in forked_ledger.rows] == [(2,), (3,), (4,)]


def test_travel_drop_note_round_trips_load_validation_silently():
    """The in-key note is inert at load: no warning, no quarantine rewrite."""

    model = _Greedy()
    trace = _episode_trace(model)
    result = trace.run(inputs=torch.tensor([[5]]))
    note = dict(result.trace.annotations["episode"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_loaded_episode_annotations(result.trace)
    assert [str(w.message) for w in caught] == []
    assert result.trace.annotations["episode"] == note


def test_scrub_is_idempotent_and_never_rewrites_prior_diagnostics():
    """A second settlement pass keeps the first drop's note untouched."""

    model = _Greedy()
    trace = _episode_trace(model)
    result = trace.run(inputs=torch.tensor([[5]]))
    first_note = dict(result.trace.annotations["episode"])
    assert scrub_fresh_execution_annotations(result.trace) == ()
    assert result.trace.annotations["episode"] == first_note


def test_registry_refuses_malformed_rows():
    """The registry validates its own vocabulary fail-closed."""

    with pytest.raises(ValueError, match="non-empty string"):
        register_travel_policy("", owner="x")
    with pytest.raises(ValueError, match="unknown annotations travel policy"):
        register_travel_policy("probe_key", policy="mystery", owner="x")
    with pytest.raises(ValueError, match="note_code"):
        register_travel_policy("probe_key", drop_mode="note", owner="x")
    with pytest.raises(ValueError, match="note_code"):
        register_travel_policy("probe_key", drop_mode="remove", note_code="c", owner="x")
    assert travel_policy_for("probe_key") is None

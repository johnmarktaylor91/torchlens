"""The REFUSED -> COUPLED flip (lane F42's LAST commit).

The trace-verb verdict's item-7 test list, verbatim: interventions before,
inside, and after a stepped call; zero and multiple fires; changed output
and step count; same/new-input replay; ``forced_feed``; ``halt=``; selective
save; save/load; and failed or exogenous joins. Only then does the
compatibility cell change from REFUSED to COUPLED -- this file IS that gate,
landing in the same commit as the flip.

Digest identity is the assertion law: coupled products are compared by
capture/intervention digest, never by output equality alone.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors.episode import EpisodeCaptureError
from torchlens.intervention import at_step
from torchlens.options import EpisodeSpec

SEED = 1234


class TinyLM(nn.Module):
    """Minimal stepped model: embedding -> mean -> head logits."""

    def __init__(self, vocab: int = 16, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class GreedyRunner(nn.Module):
    """Append feed with root-loop work BEFORE and AFTER the stepped calls."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.pre = nn.Identity()
        self.post = nn.Identity()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        current = torch.abs(ids)  # BEFORE any stepped call (outside-step op)
        tokens = []
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        out = torch.cat(tokens, dim=1)
        return out * 1  # AFTER the last stepped call (outside-step op)


class EosRunner(nn.Module):
    """Value-dependent loop: stops when the model emits the EOS token."""

    def __init__(self, model: nn.Module, eos: int, max_steps: int = 6):
        super().__init__()
        self.model = model
        self.eos = eos
        self.max_steps = max_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.max_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
            if int(next_token.reshape(-1)[0]) == self.eos:
                break
        return torch.cat(tokens, dim=1)


class ToolInjectRunner(nn.Module):
    """The tool-call shape: exogenous tokens enter the context mid-loop."""

    def __init__(self, model: nn.Module, n_steps: int, inject_at: int = 2, n_inject: int = 2):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.inject_at = inject_at
        self.n_inject = n_inject

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for step in range(self.n_steps):
            if step == self.inject_at:
                tool = torch.full((ids.shape[0], self.n_inject), 9, dtype=ids.dtype)
                current = torch.cat([current, tool], dim=1)
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


def _ids() -> torch.Tensor:
    return torch.tensor([[1, 2, 3]])


def _coupled(runner: nn.Module, ids: torch.Tensor, intervene: object, **spec_kwargs):
    spec = EpisodeSpec(stepped_module=runner.model, **spec_kwargs)
    return tl.trace(
        runner,
        ids,
        episode=spec,
        intervene=intervene,
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )


def _header(log):
    return log.annotations["episode"]["header"]


def _fire_counts(log):
    return [row["fire_count"] for row in log.annotations["episode"]["rows"]]


# ---------------------------------------------------------------------------
# The cell itself: REFUSED became COUPLED
# ---------------------------------------------------------------------------


def test_episode_x_intervene_cell_is_coupled():
    """The combination runs, and the product carries the full evidence bar."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=4)
    log = _coupled(runner, _ids(), tl.when(tl.func("tanh"), tl.scale(0.5)), n_steps=4)
    attestation = log.episode_coupling
    assert attestation.coupled is True
    assert attestation.bound is True  # product binding: the capture digest
    assert attestation.intervention_digest is not None
    assert attestation.fidelity_basis == "perturbed"
    assert attestation.fire_counts == (1, 1, 1, 1)
    assert log.outcome.status.name == "COMPLETE"


# ---------------------------------------------------------------------------
# Item 7, clause by clause: before / inside / after a stepped call
# ---------------------------------------------------------------------------


def test_fires_before_inside_and_after_a_stepped_call():
    """Inside-step fires land in rows; before/after fires stay outside."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=3)
    # abs runs BEFORE step 0; mul runs AFTER the last step; tanh runs INSIDE
    # every step. One spec fires at all three positions (add is dtype-safe
    # on the integer token sites where scale would refuse).
    spec = tl.when(tl.func("abs") | tl.func("mul") | tl.func("tanh"), tl.add(1))
    log = _coupled(runner, _ids(), spec, n_steps=3)
    assert _fire_counts(log) == [1, 1, 1]  # tanh only: one per step
    # The before/after fires are outside-step evidence: folded into the
    # digest, never guessed into a row. A tanh-only capture (no
    # before/after fires) must therefore mint a DIFFERENT digest.
    torch.manual_seed(0)
    tanh_only = _coupled(
        GreedyRunner(TinyLM(), n_steps=3),
        _ids(),
        tl.when(tl.func("tanh"), tl.scale(1.0)),
        n_steps=3,
    )
    assert _fire_counts(tanh_only) == [1, 1, 1]
    assert _header(log)["intervention_digest"] != _header(tanh_only)["intervention_digest"]


def test_zero_and_multiple_fires_are_first_class():
    """Zero fires stamp measured 0s; multiple fires per step count exactly."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=3)
    with pytest.warns(UserWarning, match="matched zero sites"):
        zero = _coupled(
            runner, _ids(), tl.when(tl.func("nonexistent_func"), tl.zero_ablate()), n_steps=3
        )
    assert _fire_counts(zero) == [0, 0, 0]  # measured zeros, never None
    attestation = zero.episode_coupling
    assert attestation.coupled is True  # zero fires is still coupled evidence
    assert attestation.fidelity_basis != "perturbed"  # nothing replaced

    torch.manual_seed(0)
    multi = _coupled(
        GreedyRunner(TinyLM(), n_steps=3),
        _ids(),
        tl.when(tl.func("tanh") | tl.func("mean"), tl.scale(1.0)),
        n_steps=3,
    )
    assert _fire_counts(multi) == [2, 2, 2]


def test_step_qualified_intervention_fires_at_exactly_one_step():
    """The evidence-bar selector: at_step(k) narrows the fire to step k."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=4)
    log = _coupled(runner, _ids(), tl.when(tl.func("tanh") & at_step(2), tl.scale(0.5)), n_steps=4)
    assert _fire_counts(log) == [0, 0, 1, 0]
    assert _header(log)["fidelity_basis"] == "perturbed"


def test_changed_output_and_changed_step_count():
    """Perturbation may change emissions AND how many steps run; both settle."""

    torch.manual_seed(0)
    model = TinyLM()
    ids = _ids()
    baseline = tl.trace(
        EosRunner(model, eos=int(model(ids).argmax(-1)[0]), max_steps=5),
        ids,
        episode=EpisodeSpec(stepped_module=model),
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )
    baseline_steps = len(baseline.annotations["episode"]["rows"])

    # Zero-ablating the hidden state changes the logits argmax away from the
    # baseline EOS, so the value-dependent loop runs a different number of
    # steps and emits different tokens.
    torch.manual_seed(0)
    model2 = TinyLM()
    eos2 = int(model2(ids).argmax(-1)[0])
    perturbed = tl.trace(
        EosRunner(model2, eos=eos2, max_steps=5),
        ids,
        episode=EpisodeSpec(stepped_module=model2),
        intervene=tl.when(tl.func("tanh"), tl.zero_ablate()),
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )
    perturbed_steps = len(perturbed.annotations["episode"]["rows"])
    assert perturbed.outcome.status.name == "COMPLETE"
    assert _header(perturbed)["fidelity_basis"] == "perturbed"
    assert all(count == 1 for count in _fire_counts(perturbed))
    # Digest identity is the law: the perturbed product's digests differ from
    # the baseline's (never output equality alone).
    assert _header(perturbed)["capture_digest"] != _header(baseline)["capture_digest"]
    assert _header(baseline)["intervention_digest"] is None
    assert (perturbed_steps != baseline_steps) or (
        [r["step_output"] for r in perturbed.annotations["episode"]["rows"]]
        != [r["step_output"] for r in baseline.annotations["episode"]["rows"]]
    )


def test_identical_coupled_reruns_mint_identical_digests():
    """Digest identity: same model/input/seed/spec -> the same two digests."""

    def capture():
        torch.manual_seed(0)
        runner = GreedyRunner(TinyLM(), n_steps=3)
        return _coupled(
            runner, _ids(), tl.when(tl.func("tanh") & at_step(1), tl.scale(0.5)), n_steps=3
        )

    first, second = capture(), capture()
    assert _header(first)["capture_digest"] == _header(second)["capture_digest"]
    assert _header(first)["intervention_digest"] == _header(second)["intervention_digest"]


# ---------------------------------------------------------------------------
# Same/new-input replay on both engines
# ---------------------------------------------------------------------------


def test_replay_refuses_coupled_products_on_both_engines():
    """run() on a coupled product refuses typed: same input, new input,
    default and guarded-fast engines alike (the verdict's refusal arm)."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=3)
    ids = _ids()
    log = _coupled(runner, ids, tl.when(tl.func("tanh"), tl.scale(0.5)), n_steps=3)
    for inputs in (ids, torch.tensor([[4, 5, 6]])):
        for fast in (False, True):
            with pytest.raises(EpisodeCaptureError) as excinfo:
                log.run(inputs=inputs, fast=fast)
            assert excinfo.value.fields["code"] == "episode_coupled_replay_underivable"
    # The refusal preserved the product: the ledger is still bound.
    assert log.episode_coupling.bound is True


def test_uncoupled_episode_replay_never_carries_the_ledger():
    """The no-foreign-ledger regression: replay output has no live ledger."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=3)
    log = tl.trace(
        runner,
        _ids(),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=3),
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )
    # A growing-context refresh legitimately warns on multi-pass shape
    # drift; the regression under test is the LEDGER, not the shapes.
    with pytest.warns(UserWarning, match="Tensor shape changed"):
        result = log.run(inputs=torch.tensor([[4, 5, 6]]))
    episode_key = result.trace.annotations.get("episode")
    assert episode_key is not None and episode_key.get("quarantined") is True


# ---------------------------------------------------------------------------
# forced_feed
# ---------------------------------------------------------------------------


def test_forced_feed_couples_with_perturbation_disclosed():
    """forced_tokens + intervene: forced feed disclosed, perturbed fidelity."""

    torch.manual_seed(0)
    model = TinyLM()

    class ForcedRunner(nn.Module):
        def __init__(self, inner: nn.Module, forced: tuple[int, ...]):
            super().__init__()
            self.model = inner
            self.forced = forced

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            tokens = []
            current = ids
            for forced_token in self.forced:
                self.model(current)  # the model runs; its emission is ignored
                next_token = torch.full((ids.shape[0], 1), forced_token, dtype=ids.dtype)
                tokens.append(next_token)
                current = torch.cat([current, next_token], dim=1)
            return torch.cat(tokens, dim=1)

    forced = (5, 7, 9)
    runner = ForcedRunner(model, forced)
    log = tl.trace(
        runner,
        _ids(),
        episode=EpisodeSpec(stepped_module=model, forced_tokens=forced),
        intervene=tl.when(tl.func("tanh"), tl.scale(0.5)),
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )
    header = _header(log)
    assert header["token_feed"] == "forced"
    # Perturbation outranks the forced disclosure on the ONE fidelity slot;
    # the forced feed stays visible on token_feed.
    assert header["fidelity_basis"] == "perturbed"
    assert _fire_counts(log) == [1, 1, 1]


# ---------------------------------------------------------------------------
# halt=
# ---------------------------------------------------------------------------


def test_halt_mid_episode_keeps_measured_fire_counts():
    """A halted coupled episode settles with counts on the started prefix."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=4)
    calls = {"n": 0}

    def stop_at_third_step(ctx) -> bool:
        if getattr(ctx, "func_name", "") == "tanh":
            calls["n"] += 1
            return calls["n"] >= 3
        return False

    log = tl.trace(
        runner,
        _ids(),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=4),
        intervene=tl.when(tl.func("tanh"), tl.scale(0.5)),
        halt=stop_at_third_step,
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )
    assert log.outcome.status.name == "HALTED"
    rows = log.annotations["episode"]["rows"]
    statuses = [row["status"] for row in rows]
    assert statuses[-1] == "absent"  # the never-started tail is absent
    started = [row for row in rows if row["status"] != "absent"]
    assert all(row["fire_count"] is not None for row in started)
    assert sum(row["fire_count"] for row in started) >= 2
    assert _header(log)["fidelity_basis"] == "perturbed"
    assert log.episode_coupling.bound is True


# ---------------------------------------------------------------------------
# Selective save
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_selective_save_couples_and_keeps_evidence():
    """save= narrows retention; the coupling evidence still settles whole."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=3)
    log = tl.trace(
        runner,
        _ids(),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=3),
        intervene=tl.when(tl.func("tanh") & at_step(1), tl.scale(0.5)),
        save=tl.func("tanh"),
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )
    assert _fire_counts(log) == [0, 1, 0]
    assert log.episode_coupling.coupled is True
    assert [row["step_output"] is not None for row in log.annotations["episode"]["rows"]] == [
        True,
        True,
        True,
    ]


# ---------------------------------------------------------------------------
# Save/load
# ---------------------------------------------------------------------------


def test_coupled_product_survives_save_load_and_loaded_replay_refuses(tmp_path):
    """The coupled slots persist (reserved C07X slots); the gate holds loaded."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=3)
    log = _coupled(runner, _ids(), tl.when(tl.func("tanh") & at_step(1), tl.scale(0.5)), n_steps=3)
    target = tmp_path / "coupled.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    header = _header(loaded)
    assert header["intervention_digest"] == _header(log)["intervention_digest"]
    assert header["fidelity_basis"] == "perturbed"
    assert _fire_counts(loaded) == [0, 1, 0]
    attestation = loaded.episode_coupling
    assert attestation.coupled is True and attestation.bound is True
    with pytest.raises(EpisodeCaptureError) as excinfo:
        loaded.run(inputs=_ids())
    assert excinfo.value.fields["code"] == "episode_coupled_replay_underivable"


# ---------------------------------------------------------------------------
# Failed or exogenous joins
# ---------------------------------------------------------------------------


def test_exogenous_join_couples_with_per_segment_facts():
    """A broken join splits the coupled facts per segment, never across."""

    torch.manual_seed(0)
    runner = ToolInjectRunner(TinyLM(), n_steps=5, inject_at=2)
    log = tl.trace(
        runner,
        _ids(),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=5),
        intervene=tl.when(tl.func("tanh"), tl.scale(0.5)),
        capture=tl.options.CaptureOptions(random_seed=SEED),
    )
    envelope = _header(log)["step_join"]
    assert envelope is not None and envelope["break_step"] == 2
    attestation = log.episode_coupling
    assert attestation.coupled is True
    segments = attestation.segments
    assert [(s.start_step, s.end_step) for s in segments] == [(0, 1), (2, 4)]
    assert [s.fire_count_total for s in segments] == [2, 3]


def test_failed_forward_coupled_episode_attaches_partial_evidence():
    """A mid-episode crash still lands fire counts on the partial ledger."""

    torch.manual_seed(0)
    model = TinyLM()

    class CrashRunner(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.model = inner

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            current = ids
            for step in range(4):
                if step == 2:
                    raise RuntimeError("mid-episode crash")
                next_token = self.model(current).argmax(-1, keepdim=True)
                current = torch.cat([current, next_token], dim=1)
            return current

    runner = CrashRunner(model)
    with pytest.raises(RuntimeError, match="mid-episode crash"), pytest.warns(Warning):
        tl.trace(
            runner,
            _ids(),
            episode=EpisodeSpec(stepped_module=model, n_steps=4),
            intervene=tl.when(tl.func("tanh"), tl.scale(0.5)),
            capture=tl.options.CaptureOptions(random_seed=SEED),
        )


def test_the_old_refusal_site_is_retired():
    """The D5 pre-execution refusal no longer fires; the cell is COUPLED."""

    torch.manual_seed(0)
    runner = GreedyRunner(TinyLM(), n_steps=2)
    log = _coupled(runner, _ids(), tl.when(tl.func("tanh"), tl.scale(0.5)), n_steps=2)
    assert log.outcome.status.name == "COMPLETE"
    assert log.episode_coupling.coupled is True

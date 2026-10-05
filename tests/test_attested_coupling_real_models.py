"""Attested coupling on REAL model classes (lane F42; foldA s6 F42 arms).

The realism rule: toy-only validation is disqualifying. The R0 roster's real
distilgpt2 and Llama-family classes (vendored configs, zero network) carry
the coupled arms: interventions before/inside/after generation steps,
changed tokens, zero and multiple fires, step-qualified selectors,
same/new-input replay refusals on both engines, forced feed, halt, selective
save, save/load, exogenous joins, ``feed="closed"``, and one
diffusion-shaped perturbed episode. Digest identity is the assertion law —
never output equality alone.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.errors.episode import EpisodeCaptureError, EpisodeDeclarationError, EpisodeJoinError
from torchlens.intervention import at_step
from torchlens.options import EpisodeSpec

pytest.importorskip("transformers")

from tests.real_model.r0.families import SEED, VOCAB, build_distilgpt2, build_llama

# heavy, not smoke: real 2-layer transformer episodes measure over the 5 s
# smoke budget on the dev box (tiered honestly, like the F40b real arm).
pytestmark = [pytest.mark.heavy, pytest.mark.real_model]

N_STEPS = 3
CAPTURE_SEED = 777


class _Generate(nn.Module):
    """Append-all greedy feed returning FULL ids: real generate()'s shape."""

    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            logits = self.model(input_ids=ids).logits
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids


class _ToolInjectGenerate(nn.Module):
    """The tool-call shape on the real class: exogenous ids enter mid-loop."""

    def __init__(self, model: nn.Module, n_steps: int, inject_at: int = 1) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.inject_at = inject_at

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for step in range(self.n_steps):
            if step == self.inject_at:
                tool = torch.full((ids.shape[0], 2), 7, dtype=ids.dtype)
                ids = torch.cat([ids, tool], dim=1)
            logits = self.model(input_ids=ids).logits
            ids = torch.cat([ids, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        return ids


class _Denoiser(nn.Module):
    """Float stepped model (the diffusion/fixed-point shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - 0.1 * self.lin(x)


class _DenoiseLoop(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self.model(x)
        return x


def _prompt() -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(0, VOCAB, (1, 8), generator=generator)


def _coupled_trace(runner: nn.Module, intervene, **kwargs) -> tl.Trace:
    return tl.trace(
        runner,
        _prompt(),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=runner.n_steps),
        intervene=intervene,
        capture=tl.options.CaptureOptions(random_seed=CAPTURE_SEED),
        **kwargs,
    )


def _header(log: tl.Trace) -> dict:
    return log.annotations["episode"]["header"]


def _fire_counts(log: tl.Trace) -> list:
    return [row["fire_count"] for row in log.annotations["episode"]["rows"]]


@contextlib.contextmanager
def _plain_layer_norm_zeroed() -> Iterator[None]:
    """Plain torch: every ``F.layer_norm`` returns zeros (no TorchLens involved)."""

    original = F.layer_norm
    F.layer_norm = lambda *args, **kwargs: torch.zeros_like(original(*args, **kwargs))
    try:
        yield
    finally:
        F.layer_norm = original


def _plain_greedy_tokens(perturbed: bool) -> list[int]:
    """The N_STEPS greedy tokens of plain-torch generation on the fixed weights."""

    runner = _Generate(build_distilgpt2("eager"), N_STEPS)
    context = _plain_layer_norm_zeroed() if perturbed else contextlib.nullcontext()
    with torch.no_grad(), context:
        return runner(_prompt())[0, -N_STEPS:].tolist()


def _baseline_and_coupled():
    runner = _Generate(build_distilgpt2("eager"), N_STEPS)
    baseline = tl.trace(
        runner,
        _prompt(),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=N_STEPS),
        capture=tl.options.CaptureOptions(random_seed=CAPTURE_SEED),
    )
    runner2 = _Generate(build_distilgpt2("eager"), N_STEPS)
    # Zeroing every layer_norm zeroes ln_f, so the bias-free lm_head emits all-zero
    # logits and greedy argmax picks token 0 at every step, whatever the weights.
    coupled = _coupled_trace(runner2, tl.when(tl.func("layer_norm"), tl.zero_ablate()))
    return baseline, coupled


def _step_tokens(log: tl.Trace) -> list[int]:
    """The token each episode step emitted (step_output is the appended id)."""

    rows = log.annotations["episode"]["rows"]
    return [int(torch.as_tensor(row["step_output"]).reshape(-1)[-1]) for row in rows]


def test_distilgpt2_coupled_evidence_bar():
    """Coupled distilgpt2: bound binding, perturbed fidelity, exact counts."""

    # The premise, checked in plain torch first: the perturbation moves greedy argmax
    # for these fixed weights. If it ever stops, this fails here, before TorchLens.
    plain_base, plain_perturbed = _plain_greedy_tokens(False), _plain_greedy_tokens(True)
    assert plain_perturbed != plain_base, (
        f"premise broken: zeroing layer_norm leaves greedy tokens {plain_base} unchanged"
    )
    baseline, coupled = _baseline_and_coupled()
    attestation = coupled.episode_coupling
    assert attestation.coupled is True and attestation.bound is True
    assert attestation.fidelity_basis == "perturbed"
    counts = _fire_counts(coupled)
    assert len(counts) == N_STEPS and all(isinstance(c, int) and c >= 1 for c in counts)
    # Digest identity, never output equality alone: the coupled product's
    # evidence column may or may not visibly differ, but its digests MUST
    # differ from the baseline's uncoupled header.
    assert _header(baseline)["intervention_digest"] is None
    assert _header(coupled)["intervention_digest"] is not None
    assert _step_tokens(baseline) == plain_base
    assert _step_tokens(coupled) == plain_perturbed


def test_distilgpt2_identical_coupled_reruns_mint_identical_digests():
    """Digest identity on the real class: same seed/spec -> same digests."""

    def capture() -> tl.Trace:
        runner = _Generate(build_distilgpt2("eager"), N_STEPS)
        return _coupled_trace(runner, tl.when(tl.func("softmax") & at_step(1), tl.scale(0.5)))

    first, second = capture(), capture()
    assert _header(first)["capture_digest"] == _header(second)["capture_digest"]
    assert _header(first)["intervention_digest"] == _header(second)["intervention_digest"]
    assert _fire_counts(first) == _fire_counts(second)


def test_distilgpt2_step_qualified_fire_lands_on_one_step():
    """at_step(1) on the real class: step 1 fires, steps 0/2 measure zero."""

    runner = _Generate(build_distilgpt2("eager"), N_STEPS)
    log = _coupled_trace(runner, tl.when(tl.func("softmax") & at_step(1), tl.scale(0.5)))
    counts = _fire_counts(log)
    assert counts[1] >= 1
    assert counts[0] == 0 and counts[2] == 0


def test_distilgpt2_replay_refuses_both_engines_and_survives_save_load(tmp_path):
    """Same/new-input replay refuses on both engines, live and loaded."""

    runner = _Generate(build_distilgpt2("eager"), N_STEPS)
    log = _coupled_trace(runner, tl.when(tl.func("softmax"), tl.scale(0.5)))
    prompt = _prompt()
    new_prompt = torch.randint(0, VOCAB, (1, 8), generator=torch.Generator().manual_seed(99))
    for inputs in (prompt, new_prompt):
        for fast in (False, True):
            with pytest.raises(EpisodeCaptureError) as excinfo:
                log.run(inputs=inputs, fast=fast)
            assert excinfo.value.fields["code"] == "episode_coupled_replay_underivable"
    target = tmp_path / "coupled_distilgpt2.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    assert loaded.episode_coupling.coupled is True and loaded.episode_coupling.bound is True
    assert _fire_counts(loaded) == _fire_counts(log)
    with pytest.raises(EpisodeCaptureError):
        loaded.run(inputs=prompt)


def test_distilgpt2_declared_crossing_scopes_facts_per_segment():
    """The tool-call shape: coupled facts split at the declared crossing.

    The root returns the FULL fed ids (prompt + emissions + the two injected
    tool tokens), so the evidence column derives under the declared-crossing
    chain arm (W051 FIX2): each step's emission sits right after its
    MEASURED entry, the positions read are disclosed in the header, and the
    join into step 1 grades ``declared``.
    """

    runner = _ToolInjectGenerate(build_distilgpt2("eager"), N_STEPS, inject_at=1)
    log = tl.trace(
        runner,
        _prompt(),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=N_STEPS, crossings=(1,)),
        intervene=tl.when(tl.func("softmax"), tl.scale(0.5)),
        capture=tl.options.CaptureOptions(random_seed=CAPTURE_SEED),
    )
    header = _header(log)
    envelope = header["step_join"]
    assert envelope is not None and envelope["break_step"] == 1
    assert envelope["grades"] == [None, "declared", "continuous"]
    # 8-token prompt; step 1's entry carries the emission plus 2 tool tokens.
    assert header["step_output_positions"] == [8, 11, 12]
    segments = log.episode_coupling.segments
    assert [(s.start_step, s.end_step) for s in segments] == [(0, 0), (1, 2)]
    assert all(s.fire_count_total is not None for s in segments)


def test_distilgpt2_undeclared_injection_refuses_ambiguous_column():
    """Undeclared mid-loop injection on a full-ids root: refuse, teach crossings.

    Measured alone, two surplus positions in step 1's entry are
    indistinguishable from multi-token decoding, so tail alignment would
    misattribute the tool token to step 0 (the pre-license behaviour); the
    license refuses typed and names the ``crossings`` declaration.
    """

    runner = _ToolInjectGenerate(build_distilgpt2("eager"), N_STEPS, inject_at=1)
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        _coupled_trace(runner, tl.when(tl.func("softmax"), tl.scale(0.5)))
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "crossings=(k, ...)" in str(excinfo.value)


def test_distilgpt2_feed_closed_strict_arm_halts_coupled_capture():
    """feed='closed' x intervene=: the strict arm still halts typed."""

    runner = _ToolInjectGenerate(build_distilgpt2("eager"), N_STEPS, inject_at=1)
    with pytest.raises(EpisodeJoinError) as excinfo, pytest.warns(Warning):
        tl.trace(
            runner,
            _prompt(),
            episode=EpisodeSpec(stepped_module=runner.model, n_steps=N_STEPS, feed="closed"),
            intervene=tl.when(tl.func("softmax"), tl.scale(0.5)),
            capture=tl.options.CaptureOptions(random_seed=CAPTURE_SEED),
        )
    assert excinfo.value.fields["code"] == "episode_feed_closed_violation"


def test_llama_family_coupled_evidence_bar():
    """The second real family (Llama class): the same coupled evidence bar."""

    runner = _Generate(build_llama("eager"), N_STEPS)
    log = _coupled_trace(runner, tl.when(tl.func("softmax") & at_step(2), tl.scale(0.5)))
    attestation = log.episode_coupling
    assert attestation.coupled is True and attestation.bound is True
    assert attestation.fidelity_basis == "perturbed"
    counts = _fire_counts(log)
    assert counts[2] >= 1 and counts[0] == 0 and counts[1] == 0


def test_diffusion_shaped_perturbed_episode():
    """One diffusion-shaped perturbed episode: float root, digest kind."""

    torch.manual_seed(0)
    runner = _DenoiseLoop(_Denoiser(), n_steps=4)
    log = tl.trace(
        runner,
        torch.randn(1, 6),
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=4, step_output_kind="digest"),
        intervene=tl.when(tl.func("linear") & at_step(2), tl.zero_ablate()),
        capture=tl.options.CaptureOptions(random_seed=CAPTURE_SEED),
    )
    assert _fire_counts(log) == [0, 0, 1, 0]
    header = _header(log)
    assert header["fidelity_basis"] == "perturbed"
    assert header["step_output_kind"] == "digest"
    rows = log.annotations["episode"]["rows"]
    assert all(isinstance(row["step_output"], str) for row in rows)
    assert log.episode_coupling.bound is True

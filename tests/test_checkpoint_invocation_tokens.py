"""Checkpoint invocation token pins (L9 memo 2.3, wave-2 subset).

Tokens are the WITNESS, site keys the GROUPER: one classified non-reentrant
checkpoint enter mints exactly one per-trace ordinal token; pack evidence is
count-only (the forward-side slot->op binding is NOT claimed); unpack
evidence points are backward-derived (fire brackets -> shipped user-op
pairing -> read-only L1 site keys). The projected summary lands on the
DROP-gated ``Trace.checkpoint_invocation_witness`` field. The typed
ambiguity REFUSAL awaits a pending contract amendment and is deliberately NOT shipped here;
no test below exercises an identity-read refusal. All spellings
DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import backward as backward_mod
from torchlens.backends.torch._aten_capture import _activate_aten_recording_for_tests
from torchlens.ir.events import CheckpointInvocationObserved


class _OneCheckpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.a(x))
        h = checkpoint(self.b, h, use_reentrant=False)
        return self.c(torch.relu(h))


class _TwoIdenticalCheckpoints(nn.Module):
    """Two structurally identical checkpointed invocations (the ruling's core case)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b1 = nn.Linear(8, 8)
        self.b2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.a(x))
        h = checkpoint(self.b1, h, use_reentrant=False)
        return checkpoint(self.b2, h, use_reentrant=False)


def _captured(model: nn.Module, x: torch.Tensor, *, backward: bool = True) -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        model,
        x,
        save_mode="reference",
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    if backward:
        trace.log_backward(trace.output_ops[0].out.sum())
    return trace


# ---------------------------------------------------------------------------
# Minting: one token per logical invocation; recompute never mints.
# ---------------------------------------------------------------------------


def test_plain_single_checkpoint_mints_exactly_one_token() -> None:
    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 1
    assert witness["degrade_flags"] == []
    assert witness["verdict"] == "checkpoint_invocations_observed"
    (record,) = witness["tokens"].values()
    # Recompute non-minting: the backward ran (unpack evidence exists), the
    # _recomputation_hook entered, and the count stayed 1.
    assert record["pack_count"] > 0
    assert record["unpack_evidence_count"] > 0
    minted = [
        event for event in trace.backward_events if isinstance(event, CheckpointInvocationObserved)
    ]
    assert len(minted) == 1


def test_two_identical_invocations_mint_two_tokens() -> None:
    trace = _captured(_TwoIdenticalCheckpoints(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 2
    assert sorted(witness["tokens"]) == [1, 2]
    for record in witness["tokens"].values():
        assert record["unpack_evidence_count"] > 0


def test_forward_only_capture_has_token_with_zero_unpack_evidence() -> None:
    trace = _captured(_OneCheckpoint(), torch.randn(3, 4), backward=False)
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 1
    (record,) = witness["tokens"].values()
    assert record["unpack_evidence_count"] == 0
    assert record["site_key_candidates"] == []


@pytest.mark.smoke
def test_backward_derived_site_candidates_resolve_under_armed_brackets() -> None:
    torch.manual_seed(0)
    with _activate_aten_recording_for_tests():
        trace = tl.trace(
            _OneCheckpoint(),
            torch.randn(3, 4),
            save_mode="reference",
            capture=tl.options.CaptureOptions(backward_ready=True),
        )
        trace.log_backward(trace.output_ops[0].out.sum())
    witness = trace.checkpoint_invocation_witness
    (record,) = witness["tokens"].values()
    # Unpack evidence inside witnessed fire brackets resolves through the
    # shipped user-op pairing to the checkpointed module's site key.
    assert record["site_key_candidates"], "armed brackets produced no site candidates"
    assert all(key.startswith("s1|") for key in record["site_key_candidates"])
    assert any(label is not None for _, label, _ in record["window_evidence"])


# ---------------------------------------------------------------------------
# Negative control + degrade classes.
# ---------------------------------------------------------------------------


class _PlainHooks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            h = self.fc(x)
        with torch.autograd.graph.save_on_cpu():
            return h * 2


def test_negative_control_plain_hooks_and_save_on_cpu() -> None:
    trace = _captured(_PlainHooks(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert witness["degrade_flags"] == []
    assert witness["verdict"] == "no_checkpoint_invocation_observed"


class _Reentrant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(checkpoint(lambda y: torch.relu(self.a(y)), x, use_reentrant=True))


def test_reentrant_sets_d5_and_never_reaches_affirmative_verdict() -> None:
    trace = _captured(_Reentrant(), torch.randn(3, 4, requires_grad=True))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert "reentrant_node_discovered" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


class _PausedCheckpoint(nn.Module):
    """A checkpoint entered under paused logging: classifier condition (2) fails."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.a(x))
        with _state.pause_logging():
            h = checkpoint(self.b, h, use_reentrant=False)
        # Traced tail: the paused region must not produce the model output
        # (an all-paused tail would be an unattributable output by design).
        return self.c(h)


@pytest.mark.smoke
def test_unwitnessed_enter_sets_d6_instead_of_staying_silent() -> None:
    torch.manual_seed(0)
    # The paused-region output enters module c without graph provenance;
    # TorchLens discloses that honestly and the disclosure is expected here.
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(
            _PausedCheckpoint(),
            torch.randn(3, 4),
            save_mode="reference",
            capture=tl.options.CaptureOptions(backward_ready=True),
        )
    trace.log_backward(trace.output_ops[0].out.sum())
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert "unwitnessed_checkpoint_enter" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


def test_classifier_unavailable_sets_d1_and_no_tokens(monkeypatch) -> None:
    monkeypatch.setattr(backward_mod, "_resolve_checkpoint_hook_cls", lambda: None)
    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert "classifier_unavailable" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


@pytest.mark.smoke
def test_unmatched_backward_warn_sets_d4_and_warn_once_preserved() -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        nn.Linear(4, 2),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    foreign = torch.randn(3, requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with trace.recording_backward():
            (foreign * 2).sum().backward()
    unmatched = [w for w in caught if "did not" in str(w.message)]
    assert len(unmatched) == 1
    witness = trace.checkpoint_invocation_witness
    assert "unmatched_backward_warn" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


@pytest.mark.smoke
def test_carry_foreign_hook_attrs_preserves_non_tl_state() -> None:
    """The attribute-carry helper copies foreign state, never TorchLens's own.

    torch >= 2.14's checkpoint internals stash a private ``_user_hooks``
    attribute directly on the pack-hook callable (see
    ``backward_mod._carry_foreign_hook_attrs``'s docstring); the token
    wrapper replacing that callable must carry it forward verbatim.
    """

    def old_hook(x: int) -> int:
        return x

    old_hook._checkpoint_internal = True  # type: ignore[attr-defined]
    old_hook._user_hooks = ("sentinel", "value")  # type: ignore[attr-defined]
    old_hook.__tl_saved_tensors_hook_scoped__ = True  # type: ignore[attr-defined]

    def new_hook(x: int) -> int:
        return x

    assert backward_mod._carry_foreign_hook_attrs(new_hook, old_hook) is True
    assert new_hook._checkpoint_internal is True  # type: ignore[attr-defined]
    assert new_hook._user_hooks == ("sentinel", "value")  # type: ignore[attr-defined]
    assert not hasattr(new_hook, "__tl_saved_tensors_hook_scoped__")


@pytest.mark.smoke
def test_hook_identity_attrs_preserved_when_simulated_in_play(monkeypatch) -> None:
    """Forcing the torch-2.14 attribute-carry path engaged mints a token normally.

    Simulates ``HAS_CHECKPOINT_INTERNAL_HOOK_CLASS`` being True on whatever
    torch is actually installed, using the REAL ``_carry_foreign_hook_attrs``:
    the generic copy-forward is a no-op when there is nothing foreign to
    carry, so the capture must behave identically to the unsimulated path.
    """

    monkeypatch.setattr(backward_mod, "_checkpoint_hook_identity_attrs_in_play", lambda: True)
    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 1
    assert "hook_identity_unpreserved" not in witness["degrade_flags"]
    assert witness["verdict"] == "checkpoint_invocations_observed"


@pytest.mark.smoke
def test_failed_hook_identity_preserve_sets_d7_and_does_not_leak(monkeypatch) -> None:
    """A forced attribute-carry failure degrades D7 and never leaks into later captures.

    Regression pin for the torch 2.14 cascade (261 ``CheckpointError`` failures):
    installing a token wrapper that drops torch's private ``_user_hooks``
    attribute makes torch's own ``__exit__`` raise ``AttributeError`` BEFORE it
    can pop the hook off its global stack, permanently corrupting every later
    checkpoint (and, eventually, every later saved-tensor capture) in the
    process. The fix skips the swap instead of installing it, so this capture
    must complete cleanly (no token, D7 flagged) and a later, unmocked capture
    must mint normally -- proof the failure never escaped this one enter.
    """

    monkeypatch.setattr(backward_mod, "_checkpoint_hook_identity_attrs_in_play", lambda: True)
    monkeypatch.setattr(backward_mod, "_carry_foreign_hook_attrs", lambda new, old: False)

    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert "hook_identity_unpreserved" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"

    # Prove no leak: undo the forced failure and capture again. A corrupted
    # global hook stack would make this (or a later) capture raise
    # CheckpointError/AttributeError instead of minting a clean token.
    monkeypatch.undo()
    clean_trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    clean_witness = clean_trace.checkpoint_invocation_witness
    assert clean_witness["token_count"] == 1
    assert "hook_identity_unpreserved" not in clean_witness["degrade_flags"]
    assert clean_witness["verdict"] == "checkpoint_invocations_observed"


def test_patch_unavailable_sets_d2(monkeypatch) -> None:
    trace = _captured(nn.Linear(4, 2), torch.randn(3, 4))
    monkeypatch.setattr(backward_mod, "_SAVED_TENSORS_HOOKS_INIT_PATCHED", False)
    backward_mod._refresh_checkpoint_witness(trace)
    witness = trace.checkpoint_invocation_witness
    assert "patch_unavailable" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


# ---------------------------------------------------------------------------
# Fresh-wrapper discipline: re-minting never stacks token layers.
# ---------------------------------------------------------------------------


def test_token_wrappers_unwrap_prior_layer_instead_of_stacking() -> None:
    trace = tl.trace(nn.Linear(4, 2), torch.randn(3, 4))
    state = backward_mod._checkpoint_token_state(trace)
    state["tokens"][1] = {"pack_count": 0, "unpack_evidence": []}
    state["tokens"][2] = {"pack_count": 0, "unpack_evidence": []}
    calls: list[str] = []

    def base(value):
        calls.append("base")
        return value

    first = backward_mod._token_bearing_pack_hook(trace, 1, base)
    second = backward_mod._token_bearing_pack_hook(
        trace, 2, getattr(first, "__tl_token_inner__", first)
    )
    assert second.__tl_token_inner__ is base
    second(torch.zeros(1))
    assert state["tokens"][1]["pack_count"] == 0, "stale token layer still counting"
    assert state["tokens"][2]["pack_count"] == 1
    assert calls == ["base"]


# ---------------------------------------------------------------------------
# Wave-2 schema pins: the witness never rides ordinary v7 saves.
# ---------------------------------------------------------------------------


def test_witness_rides_ordinary_save_at_v8(tmp_path) -> None:
    """tlspec v8: the checkpoint witness persists on a plain save/load."""

    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    assert trace.checkpoint_invocation_witness["token_count"] == 1
    path = tmp_path / "ckpt_plain.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    assert loaded.checkpoint_invocation_witness is not None
    assert loaded.checkpoint_invocation_witness["token_count"] == 1

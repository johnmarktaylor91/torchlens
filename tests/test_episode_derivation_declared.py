"""Episode evidence derivation on the DECLARED source and kind (lane F40b).

foldA D8: the derivation is declaration-driven, never root-output-shape
guessing. The four historical typed refusals (exactly-one output tensor,
retained output, integer dtype, axis length == row count) are this lane's
red tests: real ``generate()`` shapes (prompt+completion) and float-emitting
stepped roots must STOP refusing under the right declaration, while the
declaration-mismatch arms keep refusing WITH teaching messages. Plus: the
minted capture digest (foldA D7 middle third), the kind-conditional save
rule, and the cross-grammar quarantine rows (C07X item (v))."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._capture._episode_ledger import (
    EPISODE_LEDGER_VERSION,
    EpisodeLedger,
    validate_loaded_episode_annotations,
)
from torchlens.errors.episode import EpisodeDeclarationError
from torchlens.options import EpisodeOptions

V = 16
N_STEPS = 3


class _Step(nn.Module):
    """Deterministic stepped model (next token = last token + 1 mod V)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(V))
            self.out.weight.copy_(torch.roll(torch.eye(V), shifts=1, dims=0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


class _Base(nn.Module):
    def __init__(self, n: int = N_STEPS) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def _run(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids


class _FloatRoot(_Base):
    """Float-emitting stepped root: per-step hidden-state-shaped output."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._run(ids)[:, -self.n :].float() / V


class _DictRoot(_Base):
    """dict/ModelOutput-shaped root: prompt+completion under a named slot."""

    def forward(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        full = self._run(ids)
        return {"sequences": full, "scores": full.float() * 0.5}


class _LastTokenOnlyRoot(_Base):
    """Root returning FEWER positions than steps ran (axis too short)."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._run(ids)[:, -1:]


def _prompt() -> torch.Tensor:
    return torch.tensor([[1, 2]])


def _episode(model: nn.Module, **spec_kwargs) -> tl.Trace:
    spec = EpisodeOptions(stepped_module=model.step, num_steps=N_STEPS, **spec_kwargs)
    return tl.trace(model, _prompt(), episode=spec)


# ---------------------------------------------------------------------------
# The D8 acceptance arms (the red tests: these shapes STOP refusing)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_derivation.py.
def test_dict_root_with_declared_source_derives_tokens():
    """A dict/ModelOutput root resolves through the declared slot path."""

    trace = _episode(_DictRoot(), step_output_from="sequences")
    ledger = trace.episode
    assert ledger is not None
    assert ledger.header.step_output_from == "sequences"
    column = [row.step_output for row in ledger.rows]
    assert all(isinstance(entry, tuple) for entry in column)
    # Tail-aligned: the prompt prefix stays out of the evidence column
    # (next token = last + 1 under the deterministic stepped model).
    assert column == [((2 + k + 1) % V,) for k in range(N_STEPS)]


# ---------------------------------------------------------------------------
# Declaration-mismatch arms (keep refusing, WITH teaching)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_derivation.py, campaign mutant kills.
def test_multi_output_root_without_declared_source_teaches_slots():
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        _episode(_DictRoot())
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    message = str(excinfo.value)
    assert "step_output_from" in message
    assert "sequences" in message and "scores" in message


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_derivation.py.
def test_declared_source_naming_no_slot_teaches_available():
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        _episode(_DictRoot(), step_output_from="logits")
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    message = str(excinfo.value)
    assert "logits" in message and "sequences" in message


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_derivation.py.
def test_axis_shorter_than_steps_refuses_with_tail_teaching():
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        _episode(_LastTokenOnlyRoot())
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "LAST 3 positions" in str(excinfo.value)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_ledger.py.
def test_kind_vocabulary_and_source_contradiction_refuse_at_declaration():
    model = _FloatRoot()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            model,
            _prompt(),
            episode=EpisodeOptions(stepped_module=model.step, step_output_kind="logits"),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "'digest'" in str(excinfo.value)

    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            model,
            _prompt(),
            episode=EpisodeOptions(
                stepped_module=model.step,
                step_output_kind="none",
                step_output_from="sequences",
            ),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "contradicts" in str(excinfo.value)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/door.py.
def test_value_free_save_still_refuses_for_evidence_kinds():
    model = _FloatRoot()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            model,
            _prompt(),
            episode=EpisodeOptions(
                stepped_module=model.step, num_steps=N_STEPS, step_output_kind="digest"
            ),
            save="none",
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "step_output_kind='none'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Capture digest: minted at settlement, bound to THIS product (foldA D7)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-grammar quarantine (C07X item (v)): consumed version, never normalize
# ---------------------------------------------------------------------------


def _v1_payload() -> dict:
    """A pre-rename (grammar v1) episode payload: tokens/cache_len rows, no
    family version field. No released torchlens ever wrote this to a public
    artifact; the row exists so an internal pre-amendment payload QUARANTINES
    typed rather than normalizing."""

    return {
        "header": {
            "episode_id": "ep-v1",
            "capture_kind": "episode",
            "stepped_module": "step",
            "n_steps_declared": 1,
            "entry_seed": 7,
            "token_feed": "free",
            "provenance_tier": "exact",
            "structure_only": False,
            "escalated_from": None,
            "reason": None,
            "fidelity_basis": None,
        },
        "rows": [
            {
                "episode_step": 0,
                "role": "prefill",
                "status": "complete",
                "coord": {"member_call_index": 1, "pass_range": None},
                "tokens": [3],
                "cache_len": 2,
                "frontier": None,
                "rng_digest": None,
                "escalation": None,
            }
        ],
    }


class _Plain(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def _quarantine_of(payload: dict) -> tuple[dict, list]:
    trace = tl.trace(_Plain(), torch.randn(2, 2))
    trace.annotations["episode"] = payload
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_loaded_episode_annotations(trace)
    return trace.annotations["episode"], caught


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_ledger.py.
def test_future_family_version_quarantines_typed():
    payload = _v1_payload()
    payload["header"]["episode_ledger_version"] = EPISODE_LEDGER_VERSION + 1
    quarantined, _ = _quarantine_of(payload)
    assert quarantined["quarantined"] is True
    assert quarantined["code"] == "episode_ledger_incoherent"
    assert f"grammar version {EPISODE_LEDGER_VERSION + 1}" in quarantined["detail"]


# ---------------------------------------------------------------------------
# Presence rule: the digest kind's direction (load-validated both ways)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_ledger.py.
def test_all_complete_digest_ledger_requires_evidence():
    model = _FloatRoot()
    trace = _episode(model, step_output_kind="digest", step_axis=1)
    payload = {
        "header": dict(trace.annotations["episode"]["header"]),
        "rows": [dict(row) for row in trace.annotations["episode"]["rows"]],
    }
    payload["rows"][0]["step_output"] = None
    with pytest.raises(ValueError, match="missing"):
        EpisodeLedger.from_payload(payload)

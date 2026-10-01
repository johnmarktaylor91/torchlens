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

import io
import pickle
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_ledger import (
    _HEADER_PAYLOAD_KEYS,
    _ROW_PAYLOAD_KEYS,
    EPISODE_LEDGER_VERSION,
    EpisodeLedger,
    ResolvedEpisode,
    mint_capture_digest,
    validate_loaded_episode_annotations,
)
from torchlens.errors import TorchLensWarning
from torchlens.errors.episode import EpisodeDeclarationError
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke

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


class _DiffusionShapedRoot(_Base):
    """Root whose output has NO per-step structure (one final image)."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emitted = self._run(ids)[:, -self.n :]
        return emitted.float().sum(dim=-1, keepdim=True).expand(-1, 4).contiguous()


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
    spec = EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS, **spec_kwargs)
    return tl.trace(model, _prompt(), episode=spec)


def _pickle_roundtrip(log):
    buffer = io.BytesIO()
    pickle.dump(log, buffer)
    buffer.seek(0)
    return pickle.load(buffer)


# ---------------------------------------------------------------------------
# The D8 acceptance arms (the red tests: these shapes STOP refusing)
# ---------------------------------------------------------------------------


def test_float_root_digest_kind_stops_refusing():
    """R13-C2 acceptance: a float root captures under kind='digest'."""

    model = _FloatRoot()
    trace = _episode(model, step_output_kind="digest", step_axis=1)
    ledger = trace.episode
    assert ledger is not None
    assert ledger.header.step_output_kind == "digest"
    assert ledger.header.step_output_from == "output"
    assert ledger.header.step_axis == 1
    assert [row.status for row in ledger.rows] == ["complete"] * N_STEPS
    for row in ledger.rows:
        assert isinstance(row.step_output, str)
        assert row.step_output.startswith("sha256:")
    # Deterministic evidence: the same capture yields the same digests.
    again = _episode(_FloatRoot(), step_output_kind="digest", step_axis=1)
    assert [r.step_output for r in again.episode.rows] == [r.step_output for r in ledger.rows]


def test_none_kind_status_only_ledger_admits_value_free_save():
    """A root with no per-step output structure captures under kind='none',
    and the kind-conditional save rule admits a value-free save policy."""

    model = _DiffusionShapedRoot()
    trace = tl.trace(
        model,
        _prompt(),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS, step_output_kind="none"),
        capture=tl.options.CaptureOptions(layers_to_save="none"),
    )
    ledger = trace.episode
    assert ledger is not None
    assert ledger.header.step_output_kind == "none"
    assert ledger.header.step_output_from is None
    assert ledger.header.step_axis is None
    assert [row.status for row in ledger.rows] == ["complete"] * N_STEPS
    assert all(row.step_output is None for row in ledger.rows)


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


def test_multi_output_root_without_declared_source_teaches_slots():
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        _episode(_DictRoot())
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    message = str(excinfo.value)
    assert "step_output_from" in message
    assert "sequences" in message and "scores" in message


def test_declared_source_naming_no_slot_teaches_available():
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        _episode(_DictRoot(), step_output_from="logits")
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    message = str(excinfo.value)
    assert "logits" in message and "sequences" in message


def test_axis_shorter_than_steps_refuses_with_tail_teaching():
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        _episode(_LastTokenOnlyRoot())
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "LAST 3 positions" in str(excinfo.value)


def test_kind_vocabulary_and_source_contradiction_refuse_at_declaration():
    model = _FloatRoot()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            model,
            _prompt(),
            episode=EpisodeSpec(stepped_module=model.step, step_output_kind="logits"),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "'digest'" in str(excinfo.value)

    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            model,
            _prompt(),
            episode=EpisodeSpec(
                stepped_module=model.step,
                step_output_kind="none",
                step_output_from="sequences",
            ),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "contradicts" in str(excinfo.value)


def test_value_free_save_still_refuses_for_evidence_kinds():
    model = _FloatRoot()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            model,
            _prompt(),
            episode=EpisodeSpec(
                stepped_module=model.step, n_steps=N_STEPS, step_output_kind="digest"
            ),
            capture=tl.options.CaptureOptions(layers_to_save="none"),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "step_output_kind='none'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Capture digest: minted at settlement, bound to THIS product (foldA D7)
# ---------------------------------------------------------------------------


def test_capture_digest_minted_and_recomputable_from_the_product():
    class _EmittedOnly(_Base):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self._run(ids)[:, -self.n :]

    model = _EmittedOnly()
    trace = tl.trace(
        model,
        _prompt(),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
        capture=tl.options.CaptureOptions(random_seed=1234),
    )
    ledger = trace.episode
    assert ledger is not None
    digest = ledger.header.capture_digest
    assert isinstance(digest, str) and len(digest) == 64
    # Recomputable from the product alone (the F42 consumption contract):
    # every input persists on the product.
    resolved = ResolvedEpisode(
        episode_id="ep-recompute",  # deliberately different: NOT a digest input
        address="step",
        n_steps=N_STEPS,
        step_axis=-1,
        step_output_kind="tokens",
        step_output_from=None,
        forced_tokens=None,
        escalated_from=None,
        reason=None,
    )
    assert mint_capture_digest(trace, resolved.address, N_STEPS) == digest


def test_capture_digest_differs_across_seeds():
    class _EmittedOnly(_Base):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self._run(ids)[:, -self.n :]

    def capture(seed: int):
        model = _EmittedOnly()
        return tl.trace(
            model,
            _prompt(),
            episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
            capture=tl.options.CaptureOptions(random_seed=seed),
        )

    first = capture(1234).episode.header.capture_digest
    second = capture(5678).episode.header.capture_digest
    same = capture(1234).episode.header.capture_digest
    assert first != second
    assert first == same


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


def test_grammar_v1_payload_quarantines_typed_never_normalizes():
    quarantined, caught = _quarantine_of(_v1_payload())
    assert quarantined["quarantined"] is True
    assert quarantined["code"] == "episode_ledger_incoherent"
    assert "grammar version None" in quarantined["detail"]
    assert "never normalize" in quarantined["detail"]
    assert any(
        isinstance(w.message, TorchLensWarning) and "quarantined" in str(w.message) for w in caught
    )


def test_future_family_version_quarantines_typed():
    payload = _v1_payload()
    payload["header"]["episode_ledger_version"] = EPISODE_LEDGER_VERSION + 1
    quarantined, _ = _quarantine_of(payload)
    assert quarantined["quarantined"] is True
    assert quarantined["code"] == "episode_ledger_incoherent"
    assert f"grammar version {EPISODE_LEDGER_VERSION + 1}" in quarantined["detail"]


def test_cross_grammar_quarantine_through_the_real_load_path():
    """The quarantine fires on the artifact load hook, not just the helper."""

    class _EmittedOnly(_Base):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self._run(ids)[:, -self.n :]

    model = _EmittedOnly()
    trace = tl.trace(
        model,
        _prompt(),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
    )
    trace.annotations["episode"] = _v1_payload()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = _pickle_roundtrip(trace)
    note = restored.annotations["episode"]
    assert note["quarantined"] is True
    assert note["code"] == "episode_ledger_incoherent"
    assert any(isinstance(w.message, TorchLensWarning) for w in caught)
    # A quarantined record is not claims: the public accessor serves None,
    # while the capture-kind marker still reads the declaration as episode.
    assert restored.episode is None
    assert restored.capture_kind == "episode"


def test_cache_len_never_reappears_in_the_grammar():
    """cache_len was an arithmetic guess, never a measured cache fact; the
    field is DELETED end-to-end and the closed key sets pin it out."""

    assert "cache_len" not in _ROW_PAYLOAD_KEYS
    assert "cache_len" not in _HEADER_PAYLOAD_KEYS
    assert "tokens" not in _ROW_PAYLOAD_KEYS  # renamed to generic step_output

    class _EmittedOnly(_Base):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self._run(ids)[:, -self.n :]

    model = _EmittedOnly()
    trace = tl.trace(
        model,
        _prompt(),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
    )
    payload = trace.annotations["episode"]
    ledger = EpisodeLedger.from_payload(payload)
    assert ledger.header.episode_ledger_version == EPISODE_LEDGER_VERSION
    for row_payload in payload["rows"]:
        assert "cache_len" not in row_payload
        assert "tokens" not in row_payload


# ---------------------------------------------------------------------------
# Presence rule: the digest kind's direction (load-validated both ways)
# ---------------------------------------------------------------------------


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

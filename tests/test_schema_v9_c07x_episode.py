"""C07X amendment, payload A: episode row grammar v2 + root/step identity facts.

Covers foldA s5 item 3 (i)-(vi) + foldB s4.4 items 3, 4, 5 riding the open
tlspec-v9 window (SV-6 decided GENERIC; TLSPEC_VERSION stays 9):

- episode ledger grammar v2 (family version 2): declared step-output kind +
  source (``step_output_kind``/``step_output_from``), generic
  ``step_output``/``step_axis`` names, arithmetic ``cache_len`` DELETED and
  replaced by the measured generic carried-state witness slots
  (``entry_state_digest``/``exit_state_digest``, channel-keyed, None =
  NOT MEASURED — no field may imply an unmeasured cache fact);
- the capture-digest slot and the reserved coupling slots (``perturbed``
  fidelity basis, ``intervention_digest``, per-step ``fire_count``);
- the optional ``step_join`` slot (default absent);
- the cross-grammar quarantine row: a v1 payload (cache_len/tokens) loads
  QUARANTINED, never silently normalized;
- the Trace-level root entry-point fact, written UNCONDITIONALLY on every
  capture, with fail-closed descriptor-grammar load validation;
- the entry-dark Op facts ``episode_step`` (with the stamp-without-
  declaration coherence row) and ``tl_authored_root`` (with the
  marker-without-bound-method-root coherence row);
- save/load round trip of a real episode capture under grammar v2.

Every protective claim gets a test that TRIES the forbidden thing (D18).
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.capture._episode_ledger import (
    EPISODE_LEDGER_VERSION,
    EpisodeLedger,
    EpisodeLedgerHeader,
    EpisodeLedgerRow,
    episode_ledger_for,
)
from torchlens.errors import TorchLensWarning

pytestmark = pytest.mark.smoke


class _Greedy(torch.nn.Module):
    """Minimal stepped generator: embeds, projects, argmaxes, appends."""

    def __init__(self) -> None:
        super().__init__()
        self.step = torch.nn.Linear(4, 4)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.step(state)


class _EpisodeRoot(torch.nn.Module):
    def __init__(self, n_steps: int = 3) -> None:
        super().__init__()
        self.block = _Greedy()
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        state = torch.nn.functional.one_hot(ids, num_classes=4).float()
        tokens = []
        for _ in range(self.n_steps):
            state = self.block(state)
            next_token = state[:, -1:].argmax(dim=-1)
            tokens.append(next_token)
            state = torch.nn.functional.one_hot(
                torch.cat([ids, *tokens], dim=1), num_classes=4
            ).float()
        return torch.cat(tokens, dim=1)


def _capture_episode(n_steps: int = 3):
    root = _EpisodeRoot(n_steps=n_steps)
    ids = torch.tensor([[0, 1, 2]])
    return tl.trace(
        root,
        ids,
        episode=tl.options.EpisodeSpec(stepped_module=root.block, n_steps=n_steps),
    )


def _header(**overrides) -> EpisodeLedgerHeader:
    kwargs: dict = {
        "episode_id": "ep",
        "stepped_module": "block",
        "entry_seed": 0,
        "n_steps_declared": 2,
    }
    kwargs.update(overrides)
    return EpisodeLedgerHeader(**kwargs)


def _row(step: int, **overrides) -> EpisodeLedgerRow:
    kwargs: dict = {
        "episode_step": step,
        "role": "prefill" if step == 0 else "decode",
        "status": "complete",
        "coord": {"member_call_index": step + 1, "pass_range": None},
        "step_output": (step,),
    }
    kwargs.update(overrides)
    return EpisodeLedgerRow(**kwargs)


# ---------------------------------------------------------------------------
# Grammar v2: header slots.
# ---------------------------------------------------------------------------


def test_header_carries_family_version_2() -> None:
    assert EPISODE_LEDGER_VERSION == 2
    header = _header()
    assert header.episode_ledger_version == 2
    assert header.to_payload()["episode_ledger_version"] == 2


def test_wrong_family_version_refuses() -> None:
    """Future grammar changes are family-local; this build validates v2 only."""

    with pytest.raises(ValueError, match="family version"):
        _header(episode_ledger_version=3)


def test_step_output_kind_default_tokens_and_closed() -> None:
    assert _header().step_output_kind == "tokens"
    with pytest.raises(ValueError, match="out of vocabulary"):
        _header(step_output_kind="frames")


def test_entry_dark_header_slots_default_none() -> None:
    """step_output_from/step_join/capture_digest/intervention_digest
    DEFAULT to absent (None): only their writers (F40b/F40c/F42) fill them,
    so a bare header constructs entry-dark."""

    header = _header()
    assert header.step_output_from is None
    assert header.step_join is None
    assert header.capture_digest is None
    assert header.intervention_digest is None


def test_perturbed_fidelity_basis_admitted() -> None:
    header = _header(escalated_from="sha:1", reason="requested", fidelity_basis="perturbed")
    rebuilt = EpisodeLedgerHeader.from_payload(header.to_payload())
    assert rebuilt.fidelity_basis == "perturbed"


def test_step_join_envelope_shape_validated() -> None:
    header = _header(step_join={"claim": "unmeasured"})
    assert EpisodeLedgerHeader.from_payload(header.to_payload()).step_join == {
        "claim": "unmeasured"
    }
    with pytest.raises(ValueError, match="step_join"):
        _header(step_join="measured")


# ---------------------------------------------------------------------------
# Grammar v2: row slots — cache_len deleted, carried-state witness reserved.
# ---------------------------------------------------------------------------


def test_cache_len_is_deleted_from_the_grammar() -> None:
    """The arithmetic cache guess is gone in BOTH directions: the dataclass
    has no slot and a payload carrying the key refuses (unknown key)."""

    with pytest.raises(TypeError):
        _row(0, cache_len=3)
    payload = _row(0).to_payload()
    payload["cache_len"] = 3
    with pytest.raises(ValueError, match="unknown keys"):
        EpisodeLedgerRow.from_payload(payload)


def test_carried_state_witness_slots_default_not_measured() -> None:
    """SV-6 = GENERIC: the witness is measured-or-absent; None = NOT
    MEASURED, and no shipped writer fabricates a value."""

    row = _row(0)
    assert row.entry_state_digest is None
    assert row.exit_state_digest is None


def test_carried_state_witness_channel_keyed_shape() -> None:
    row = _row(
        0, entry_state_digest={"kv_cache": "sha256:aa"}, exit_state_digest={"kv_cache": "sha256:bb"}
    )
    rebuilt = EpisodeLedgerRow.from_payload(row.to_payload())
    assert rebuilt.entry_state_digest == {"kv_cache": "sha256:aa"}
    with pytest.raises(ValueError, match="digests must be non-empty strings"):
        _row(0, entry_state_digest={"kv_cache": ""})
    with pytest.raises(ValueError, match="channels must be non-empty strings"):
        _row(0, exit_state_digest={"": "sha256:aa"})


def test_fire_count_reserved_slot() -> None:
    assert _row(0).fire_count is None
    rebuilt = EpisodeLedgerRow.from_payload(_row(0, fire_count=2).to_payload())
    assert rebuilt.fire_count == 2
    with pytest.raises(ValueError, match="fire_count"):
        _row(0, fire_count=-1)


def test_step_output_kind_coherence() -> None:
    """tokens rows carry int tuples; digest rows strings; none rows nothing."""

    with pytest.raises(ValueError, match="step_output_kind='tokens'"):
        EpisodeLedger(_header(), [_row(0, step_output="sha256:aa"), _row(1)])
    with pytest.raises(ValueError, match="step_output_kind='digest'"):
        EpisodeLedger(_header(step_output_kind="digest"), [_row(0), _row(1, step_output=None)])
    with pytest.raises(ValueError, match="step_output_kind='none'"):
        EpisodeLedger(
            _header(step_output_kind="none"),
            [_row(0), _row(1)],
        )
    # The legal digest-kind and none-kind shapes assemble.
    EpisodeLedger(
        _header(step_output_kind="digest"),
        [_row(0, step_output="sha256:aa"), _row(1, step_output="sha256:bb")],
    )
    EpisodeLedger(
        _header(step_output_kind="none"),
        [_row(0, step_output=None), _row(1, step_output=None)],
    )


# ---------------------------------------------------------------------------
# Cross-grammar quarantine: v1 payloads never silently normalize.
# ---------------------------------------------------------------------------


def _v1_payload() -> dict:
    """A pre-amendment (grammar v1) episode annotations payload."""

    return {
        "header": {
            "episode_id": "ep",
            "capture_kind": "episode",
            "stepped_module": "block",
            "n_steps_declared": 2,
            "entry_seed": 0,
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
                "tokens": [2],
                "cache_len": 3,
                "frontier": None,
                "rng_digest": None,
                "escalation": None,
            }
        ],
    }


def test_v1_payload_quarantines_at_load_validation() -> None:
    """The old-grammar payload is a quarantine ROW, never a normalization
    ladder (foldA D9): validate_loaded_episode_annotations replaces it with
    the typed diagnostic record under ONE warning."""

    from torchlens.capture._episode_ledger import validate_loaded_episode_annotations

    log = _capture_episode()
    log.annotations["episode"] = _v1_payload()
    with pytest.warns(TorchLensWarning, match="quarantined"):
        validate_loaded_episode_annotations(log)
    record = log.annotations["episode"]
    assert record["quarantined"] is True
    assert record["code"] == "episode_ledger_incoherent"
    assert episode_ledger_for(log) is None


# ---------------------------------------------------------------------------
# Live writer: grammar v2 end to end + save/load round trip.
# ---------------------------------------------------------------------------


def test_live_episode_writes_grammar_v2() -> None:
    log = _capture_episode()
    ledger = episode_ledger_for(log)
    assert ledger is not None
    header = ledger.header
    assert header.episode_ledger_version == 2
    assert header.step_output_kind == "tokens"
    assert header.step_axis == -1  # the declared step axis, generic spelling
    # Lane F40b activated the grammar's derivation carriers: the source
    # disclosure is written whenever a kind consumes one, and the capture
    # digest binds the ledger to this product (64-hex, foldA D7).
    assert header.step_output_from == "output"
    assert isinstance(header.capture_digest, str)
    assert len(header.capture_digest) == 64
    # Lane F40c landed the join measurement: the reserved slot carries the
    # measured envelope on live captures. intervention_digest is F42's
    # coupling slot: None here because THIS capture is uncoupled.
    assert header.step_join is not None
    assert header.step_join["schema"] == "episode_step_join_v1"
    assert header.step_join["claim"] == "measured"
    assert header.intervention_digest is None
    for row in ledger.rows:
        assert isinstance(row.step_output, tuple)
        assert row.entry_state_digest is None
        assert row.exit_state_digest is None
        assert row.fire_count is None
    payload = log.annotations["episode"]
    assert "cache_len" not in payload["rows"][0]
    assert "tokens" not in payload["rows"][0]


def test_episode_round_trip_under_v9(tmp_path) -> None:
    log = _capture_episode()
    target = tmp_path / "episode.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    ledger = episode_ledger_for(loaded)
    assert ledger is not None
    assert ledger.header.episode_ledger_version == 2
    assert [row.step_output for row in ledger.rows] == [
        row.step_output for row in episode_ledger_for(log).rows
    ]


# ---------------------------------------------------------------------------
# Root entry-point fact (item (iv)) + Op entry-dark facts.
# ---------------------------------------------------------------------------


def test_root_entry_point_written_unconditionally() -> None:
    """foldA D10: every capture writes the fact; plain and episode alike."""

    model = _Greedy()
    plain = tl.trace(model, torch.randn(2, 4))
    assert plain.root_entry_point is not None
    assert plain.root_entry_point.startswith("module_call:")
    assert plain.root_entry_point.endswith("_Greedy.forward")
    episode_log = _capture_episode()
    assert episode_log.root_entry_point is not None
    assert episode_log.root_entry_point.endswith("_EpisodeRoot.forward")


def test_root_entry_point_round_trips(tmp_path) -> None:
    model = _Greedy()
    log = tl.trace(model, torch.randn(2, 4))
    target = tmp_path / "plain.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    assert loaded.root_entry_point == log.root_entry_point
    assert loaded.root_entry_point.startswith("module_call:")


def _roundtrip(log, tmp_path, name: str):
    target = tmp_path / f"{name}.tlspec"
    tl.save(log, target)
    return tl.load(target)


def test_root_entry_point_grammar_refuses_at_load(tmp_path) -> None:
    from torchlens._io.format_errors import TorchLensIOError

    model = _Greedy()
    log = tl.trace(model, torch.randn(2, 4))
    log.root_entry_point = "teleport:_Greedy.forward"
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(log, tmp_path, "root_bad")
    assert excinfo.value.fields["code"] == "artifact_root_entry_point_invalid"


def test_op_entry_dark_facts_default_none() -> None:
    model = _Greedy()
    log = tl.trace(model, torch.randn(2, 4))
    for op in log.ops:
        assert op.episode_step is None
        assert op.tl_authored_root is None


def test_episode_step_stamp_without_declaration_refuses(tmp_path) -> None:
    """A stamped op on a plain capture is a forged/drifted artifact."""

    from torchlens._io.format_errors import TorchLensIOError

    model = _Greedy()
    log = tl.trace(model, torch.randn(2, 4))
    log.ops[0].episode_step = 0
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(log, tmp_path, "stamp_bad")
    assert excinfo.value.fields["code"] == "artifact_episode_step_invalid"


def test_malformed_episode_step_refuses(tmp_path) -> None:
    from torchlens._io.format_errors import TorchLensIOError

    log = _capture_episode()
    log.ops[0].episode_step = -1
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(log, tmp_path, "stamp_type_bad")
    assert excinfo.value.fields["code"] == "artifact_episode_step_invalid"


def test_tl_authored_root_without_bound_method_root_refuses(tmp_path) -> None:
    """The marker discloses a wrapper root the capture must also declare."""

    from torchlens._io.format_errors import TorchLensIOError

    model = _Greedy()
    log = tl.trace(model, torch.randn(2, 4))
    log.ops[0].tl_authored_root = True
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(log, tmp_path, "marker_bad")
    assert excinfo.value.fields["code"] == "artifact_tl_authored_root_invalid"


def test_stamped_episode_capture_round_trips(tmp_path) -> None:
    """The legal direction: a stamp on a real episode capture validates."""

    log = _capture_episode()
    log.ops[0].episode_step = 0
    loaded = _roundtrip(log, tmp_path, "stamp_ok")
    assert loaded.ops[0].episode_step == 0

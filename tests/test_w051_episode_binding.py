"""W051 regressions for the ledger-to-product binding (audit 2.19 and 3.2).

* 2.19 -- the episode ledger had NO structural anchor: any grammatically
  valid ledger grafted into a plain artifact's ``annotations["episode"]``
  loaded with zero warnings and made the product an "episode". Loads now
  anchor the header's stepped module and every row's call index to the
  product's recorded module calls, recompute the capture digest, and
  re-derive the evidence column from the retained root output; any failure
  quarantines ``episode_ledger_incoherent`` with the reason named.
* 3.2 -- ``capture_digest`` hashed program identity only, so two
  equal-length prompts minted identical digests and a rewritten ledger still
  attested ``bound=True``. The digest is now the content-binding v2 form
  (identity + every persisted ledger fact), attestation re-derives the
  evidence column, and pre-v2 digests read as pre-binding, never foreign.
"""

from __future__ import annotations

import copy
import glob
import pickle
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_derivation import (
    CAPTURE_DIGEST_SCHEMA,
    mint_capture_digest,
    recompute_capture_digests,
    rederived_evidence_matches,
)
from torchlens.capture._episode_ledger import episode_ledger_for
from torchlens.errors import TorchLensError, TorchLensWarning
from torchlens.options import CaptureOptions, EpisodeSpec

V = 16
N_STEPS = 3


class _Step(nn.Module):
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
    def __init__(self, n: int = N_STEPS) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids[:, -self.n :]


def _episode(model: _Greedy, prompt: int, **extra) -> tl.Trace:
    return tl.trace(
        model,
        torch.tensor([[prompt]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
        capture=CaptureOptions(random_seed=0),
        **extra,
    )


def _save(trace: tl.Trace, path) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.save(trace, path)
    return glob.glob(str(path) + "/**/metadata.pkl", recursive=True)[0]


def _rewrite_metadata(metadata_path: str, mutate) -> None:
    with open(metadata_path, "rb") as handle:
        state = pickle.load(handle)
    mutate(state)
    with open(metadata_path, "wb") as handle:
        pickle.dump(state, handle)


def _load_quarantined(path):
    """Load; return (trace, quarantine detail) -- exactly one warning expected."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = tl.load(path)
    messages = [str(w.message) for w in caught if issubclass(w.category, TorchLensWarning)]
    assert len(messages) == 1, messages
    note = loaded.annotations["episode"]
    assert note["quarantined"] is True
    assert note["code"] == "episode_ledger_incoherent"
    return loaded, note["detail"]


# ---------------------------------------------------------------------------
# 3.2 -- the content-binding digest.
# ---------------------------------------------------------------------------


def test_equal_length_prompts_mint_different_digests_under_one_seed() -> None:
    model = _Greedy()
    a = _episode(model, 1)
    b = _episode(model, 9)
    header_a = a.annotations["episode"]["header"]
    header_b = b.annotations["episode"]["header"]
    assert header_a["capture_digest"] != header_b["capture_digest"]
    # Identical re-runs (same prompt, same seed) still mint IDENTICAL digests:
    # the episode_id is excluded from the bound content.
    again = _episode(model, 1)
    assert again.annotations["episode"]["header"]["capture_digest"] == header_a["capture_digest"]
    assert CAPTURE_DIGEST_SCHEMA == "episode_capture_digest_v2"


def test_swapped_ledger_between_equal_length_prompts_is_unbound() -> None:
    model = _Greedy()
    a = _episode(model, 1)
    b = _episode(model, 9)
    a.annotations["episode"] = copy.deepcopy(b.annotations["episode"])
    with pytest.raises(TorchLensError) as info:
        _ = a.episode_coupling
    assert info.value.fields["code"] == "episode_coupling_unbound"


def test_in_memory_rewrite_of_rows_or_grades_is_unbound() -> None:
    model = _Greedy()
    log = _episode(model, 1)
    assert log.episode_coupling.bound is True
    assert log.episode_coupling.evidence_rederived is True

    rewritten = _episode(model, 1)
    rewritten.annotations["episode"]["rows"][1]["step_output"] = [999]
    with pytest.raises(TorchLensError) as info:
        _ = rewritten.episode_coupling
    assert info.value.fields["code"] == "episode_coupling_unbound"

    regraded = _episode(model, 1)
    regraded.annotations["episode"]["header"]["step_join"]["grades"][1] = "exogenous"
    regraded.annotations["episode"]["header"]["step_join"]["break_step"] = 1
    with pytest.raises(TorchLensError) as info:
        _ = regraded.episode_coupling
    assert info.value.fields["code"] == "episode_coupling_unbound"


def test_re_minted_foreign_ledger_is_caught_by_evidence_rederivation() -> None:
    """A forger who re-mints the digest over swapped content is still caught."""

    model = _Greedy()
    a = _episode(model, 1)
    b = _episode(model, 9)
    foreign = copy.deepcopy(b.annotations["episode"])
    foreign["header"]["capture_digest"] = mint_capture_digest(a, "step", N_STEPS, foreign)
    a.annotations["episode"] = foreign
    bound, _legacy = recompute_capture_digests(a, foreign)
    assert bound == foreign["header"]["capture_digest"]  # the digest alone is fooled
    assert rederived_evidence_matches(a, foreign) is False  # the product's values are not
    with pytest.raises(TorchLensError) as info:
        _ = a.episode_coupling
    assert info.value.fields["code"] == "episode_coupling_unbound"
    assert "does not re-derive" in str(info.value)


def test_value_free_ledger_binds_with_the_basis_disclosed() -> None:
    """kind='none' derives no column: bound on identity + content, disclosed."""

    model = _Greedy()
    log = tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS, step_output_kind="none"),
        save=tl.in_module("step.emb"),
    )
    attestation = log.episode_coupling
    assert attestation.bound is True
    assert attestation.evidence_rederived is None  # nothing to compare, disclosed
    assert rederived_evidence_matches(log, log.annotations["episode"]) is None


def test_legacy_identity_digest_reads_pre_binding_not_foreign() -> None:
    model = _Greedy()
    log = _episode(model, 1)
    payload = log.annotations["episode"]
    payload["header"]["capture_digest"] = mint_capture_digest(log, "step", N_STEPS)
    with pytest.raises(TorchLensError) as info:
        _ = log.episode_coupling
    assert info.value.fields["code"] == "episode_coupling_unmintable"


# ---------------------------------------------------------------------------
# 2.19 -- loads anchor the ledger to the product.
# ---------------------------------------------------------------------------


def test_genuine_artifact_round_trips_bound(tmp_path) -> None:
    model = _Greedy()
    log = _episode(model, 1)
    path = tmp_path / "genuine.tlspec"
    _save(log, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        loaded = tl.load(path)
    ledger = episode_ledger_for(loaded)
    assert ledger is not None
    assert ledger.step_output_series() == ((2,), (3,), (4,))
    assert loaded.episode_coupling.bound is True
    assert loaded.episode_coupling.evidence_rederived is True


def test_grafted_ledger_quarantines_at_load(tmp_path) -> None:
    model = _Greedy()
    donor = _episode(model, 1)
    ledger = copy.deepcopy(donor.annotations["episode"])
    plain = tl.trace(nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU()), torch.randn(1, 1, 8, 8))
    path = tmp_path / "plain.tlspec"
    metadata = _save(plain, path)

    def graft(state):
        state["annotations"] = dict(state.get("annotations") or {})
        state["annotations"]["episode"] = ledger

    _rewrite_metadata(metadata, graft)
    loaded, detail = _load_quarantined(path)
    assert "recorded no call on this product" in detail
    assert episode_ledger_for(loaded) is None
    assert loaded.episode_coupling is None  # no ledger to attest


def test_tampered_rows_quarantine_at_load(tmp_path) -> None:
    model = _Greedy()
    log = _episode(model, 1)
    path = tmp_path / "tampered.tlspec"
    metadata = _save(log, path)

    def tamper(state):
        state["annotations"]["episode"]["rows"][1]["step_output"] = [9]

    _rewrite_metadata(metadata, tamper)
    _loaded, detail = _load_quarantined(path)
    assert "capture_digest does not match" in detail


def test_call_index_outside_recorded_calls_quarantines_at_load(tmp_path) -> None:
    model = _Greedy()
    log = _episode(model, 1)
    path = tmp_path / "coords.tlspec"
    metadata = _save(log, path)

    def drift(state):
        payload = state["annotations"]["episode"]
        payload["rows"][2]["coord"]["member_call_index"] = 7
        payload["header"]["capture_digest"] = mint_capture_digest(log, "step", N_STEPS, payload)

    _rewrite_metadata(metadata, drift)
    _loaded, detail = _load_quarantined(path)
    assert "member_call_index 7" in detail


def test_re_minted_swapped_ledger_is_refuted_by_the_loaded_payload(tmp_path) -> None:
    """Persisted twin of the re-mint case: the LOAD refutes the rows (W051-CAPT3).

    Inside ``__setstate__`` the root output is still an unmaterialized blob
    handle, so the in-setstate anchor has nothing to compare and admits the
    ledger; the bundle loader re-runs the evidence anchor after blob attach,
    so the loaded product arrives already quarantined -- it never reaches the
    first ``episode_coupling`` read as an admitted ledger.
    """

    model = _Greedy()
    a = _episode(model, 1)
    b = _episode(model, 9)
    path = tmp_path / "swapped.tlspec"
    metadata = _save(a, path)
    foreign = copy.deepcopy(b.annotations["episode"])
    foreign["header"]["capture_digest"] = mint_capture_digest(a, "step", N_STEPS, foreign)

    def swap(state):
        state["annotations"]["episode"] = foreign

    _rewrite_metadata(metadata, swap)
    loaded, detail = _load_quarantined(path)
    assert "does not re-derive" in detail
    assert episode_ledger_for(loaded) is None  # refuted at load, not on first read
    assert loaded.episode_coupling is None  # no admitted ledger survives to attest


def test_genuine_episode_artifact_survives_the_post_attach_anchor(tmp_path) -> None:
    """The second anchor pass admits the product's own ledger (no false refutation)."""

    log = _episode(_Greedy(), 1)
    path = tmp_path / "genuine.tlspec"
    _save(log, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        loaded = tl.load(path)
    assert episode_ledger_for(loaded) is not None
    assert loaded.episode_coupling.evidence_rederived is True

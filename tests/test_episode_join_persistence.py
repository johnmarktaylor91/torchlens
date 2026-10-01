"""Episode join envelope persistence (lane F40c): save/load x arms.

The ``episode_step_join_v1`` envelope rides the C07X-reserved header slot
(no grammar bump); loads validate the envelope FAIL-CLOSED against the rows
(geometry violations quarantine ``episode_ledger_incoherent``), claims keep
refusing after a round trip, and an absent envelope reads UNMEASURED on
every surface -- the composition row `episode=` x `step_join` x save/load.
"""

from __future__ import annotations

import json
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_ledger import EpisodeLedger, episode_ledger_for
from torchlens.errors import EpisodeJoinError, TorchLensWarning
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class GreedyRunner(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


class ToolInjectRunner(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int, inject_at: int = 2):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.inject_at = inject_at

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for step in range(self.n_steps):
            if step == self.inject_at:
                tool = torch.full((ids.shape[0], 2), 9, dtype=ids.dtype)
                current = torch.cat([current, tool], dim=1)
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


def _prompt() -> torch.Tensor:
    return torch.tensor([[1, 2, 3]])


def _capture(runner_cls, **spec_kwargs):
    model = TinyLM()
    return tl.trace(
        runner_cls(model, 4),
        _prompt(),
        episode=EpisodeSpec(stepped_module=model, n_steps=4, **spec_kwargs),
    )


def _round_trip(log, tmp_path):
    path = tmp_path / "episode.tlspec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.save(log, str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        return tl.load(str(path))


def test_continuous_envelope_round_trips(tmp_path) -> None:
    log = _capture(GreedyRunner)
    loaded = _round_trip(log, tmp_path)
    envelope = loaded.annotations["episode"]["header"]["step_join"]
    assert envelope == log.annotations["episode"]["header"]["step_join"]
    assert envelope["grades"] == [None, "continuous", "continuous", "continuous"]
    series = episode_ledger_for(loaded).step_output_series()
    assert len(series) == 4


def test_broken_envelope_round_trips_and_claims_still_refuse(tmp_path) -> None:
    log = _capture(ToolInjectRunner)
    loaded = _round_trip(log, tmp_path)
    envelope = loaded.annotations["episode"]["header"]["step_join"]
    assert envelope["break_step"] == 2
    assert envelope["grades"][2] == "exogenous"
    with pytest.raises(EpisodeJoinError) as excinfo:
        episode_ledger_for(loaded).step_output_series()
    assert excinfo.value.fields["code"] == "episode_feed_break_exogenous"
    with pytest.raises(EpisodeJoinError):
        loaded.run()


def test_declared_crossing_envelope_round_trips(tmp_path) -> None:
    log = _capture(ToolInjectRunner, crossings=(2,))
    loaded = _round_trip(log, tmp_path)
    envelope = loaded.annotations["episode"]["header"]["step_join"]
    assert envelope["grades"][2] == "declared"
    with pytest.raises(EpisodeJoinError) as excinfo:
        episode_ledger_for(loaded).step_output_series()
    assert excinfo.value.fields["code"] == "episode_join_declared_crossing"


def _tampered_payload(log, mutate) -> dict:
    payload = json.loads(json.dumps(log.annotations["episode"]))
    mutate(payload["header"]["step_join"])
    return payload


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda env: env.__setitem__("schema", "step_join_v0"), "schema"),
        (lambda env: env.__setitem__("claim", "unmeasured"), "claim"),
        (lambda env: env["grades"].__setitem__(2, "vibes"), "closed vocabulary"),
        (lambda env: env["grades"].pop(), "one entry per ledger row"),
        (lambda env: env["grades"].__setitem__(0, "continuous"), "prefill"),
        (lambda env: env.__setitem__("break_step", 1), "does not match"),
        (lambda env: env.__setitem__("extra", 1), "exactly the keys"),
        (lambda env: env["exogenous_positions"].__setitem__("99", 1), "in-range"),
    ],
)
def test_forged_envelope_geometry_refuses_fail_closed(mutate, match) -> None:
    log = _capture(GreedyRunner)
    payload = _tampered_payload(log, mutate)
    with pytest.raises(ValueError, match=match):
        EpisodeLedger.from_payload(payload)


def test_forged_envelope_quarantines_at_the_load_seam() -> None:
    """A tampered envelope quarantines typed at the ONE load-validation seam
    every artifact load routes through, never loading as claims (the
    episode x step_join x save/load composition row)."""

    from torchlens.capture._episode_ledger import validate_loaded_episode_annotations

    log = _capture(GreedyRunner)
    payload = json.loads(json.dumps(log.annotations["episode"]))
    payload["header"]["step_join"]["grades"][1] = "vibes"
    log.annotations["episode"] = payload
    with pytest.warns(TorchLensWarning, match="episode_ledger_incoherent"):
        validate_loaded_episode_annotations(log)
    assert log.annotations["episode"]["quarantined"] is True
    assert episode_ledger_for(log) is None


def test_absent_envelope_reads_unmeasured_and_series_refuses() -> None:
    """A ledger without the envelope (pre-F40c artifact shape) stays legal,
    reads unmeasured, and refuses the series claim typed."""

    log = _capture(GreedyRunner)
    payload = json.loads(json.dumps(log.annotations["episode"]))
    payload["header"]["step_join"] = None
    ledger = EpisodeLedger.from_payload(payload)  # legal: absent = unmeasured
    from torchlens._capture_honesty import episode_facts

    log.annotations["episode"] = payload
    assert episode_facts(log)["step_join"] == "unmeasured"
    with pytest.raises(EpisodeJoinError) as excinfo:
        ledger.step_output_series()
    assert excinfo.value.fields["code"] == "episode_join_unmeasured"
    # The narrow gates (replay, fold) deliberately keep shipped behavior on
    # unmeasured artifacts: no measured break, no new refusal.
    from torchlens.capture._episode_join import refuse_broken_join_claim

    refuse_broken_join_claim(log)  # does not raise

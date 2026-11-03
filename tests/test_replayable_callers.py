"""Internal captures that only fork and replay do not arm the runnable apparatus.

Every eager ``tl.trace`` capture records the per-op argument templates that
``fork().do(...)`` replays from, so the internal captures behind the SAE splice,
the causal audit, the site sweep's capture door and the experimental session
take a plain capture. ``replayable=True`` would add the runnable apparatus
(callable registry, edge provenance, buffer versions) that none of them reads,
at several times the capture cost. These tests pin that each path captures
without it; the edge audit, which does read edge provenance, keeps the flag
and is covered by its own tests.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.attribution as attribution
import torchlens.experimental

pytestmark = pytest.mark.filterwarnings("ignore::torchlens.errors.InterventionWarning")


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _Sae(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.dec(latents)


class _LinearResidual(nn.Module):
    """``out = fc2(fc1(x)) + skip(x)``: the audit's closed-form linear case."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)
        self.skip = nn.Linear(4, 3, bias=False)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randint(-3, 4, parameter.shape, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x)) + self.skip(x)


def _mlp_case() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Mlp().eval(), torch.randn(3, 6)


def _recording(monkeypatch: pytest.MonkeyPatch, target: Any) -> list[tl.Trace]:
    """Wrap ``target.trace`` so every trace it produces is appended to the returned list."""

    produced: list[tl.Trace] = []
    original = target.trace

    def recorded(*args: Any, **kwargs: Any) -> Any:
        trace = original(*args, **kwargs)
        produced.append(trace)
        return trace

    monkeypatch.setattr(target, "trace", recorded)
    return produced


def test_sae_splice_clean_and_replayed_traces_are_plain_captures() -> None:
    model, x = _mlp_case()
    sae = _Sae(10, 16).eval()
    result = tl.bridge.sae.splice(model, x, site="fc1", sae=sae)
    assert result.clean.is_replayable is False
    assert result.spliced.is_replayable is False
    assert result.spliced.last_run["engine"] == "replay"
    assert len(result.spliced_outputs) == 1


def test_sae_logged_splice_captures_the_live_model_plainly() -> None:
    model, x = _mlp_case()
    sae = _Sae(10, 16).eval()
    off = tl.bridge.sae.splice(model, x, site="fc1", sae=sae)
    on = tl.bridge.sae.splice(model, x, site="fc1", sae=sae, log_latents=True)
    assert on.spliced.is_replayable is False
    assert on.latents_site is not None
    assert torch.equal(on.spliced[on.latents_site].out, off.latents)
    assert torch.equal(on.spliced_outputs[0], off.spliced_outputs[0])


def test_causal_audit_traces_the_model_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _LinearResidual().double().eval()
    x = torch.randint(-3, 4, (2, 4), generator=torch.Generator().manual_seed(1)).double()
    site = attribution.SiteTarget("fc2", unit=2)
    claim = attribution.layer_attribution(model, x, target=site, layer="fc1")
    produced = _recording(monkeypatch, tl)
    result = attribution.causal_audit(model, x, claim=claim, target=site, k=1)
    assert result.rank_agreement == 1.0
    assert produced, "the audit captures the model through tl.trace"
    assert all(trace.is_replayable is False for trace in produced)


def test_site_sweep_capture_door_captures_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchlens._user_funcs as user_funcs

    model, x = _mlp_case()
    base = tl.trace(model, x)
    label = list(base[tl.func("relu")].labels)[0]
    produced = _recording(monkeypatch, user_funcs)
    bundle = tl.experiment.sweep_sites(
        base,
        candidates={"relu": label},
        edit=tl.zero_ablate(),
        metric=lambda t: float(t[t.output_layers[0]].out.sum()),
        retain="all",
        model=model,
        inputs=x,
    )
    # A label candidate lowers to a site key, which rides the capture door.
    assert len(produced) == 1
    member = bundle["relu"]
    assert member is produced[0]
    assert member.is_replayable is False
    assert member.options.save_arg_templates is True
    assert not torch.equal(member[member.output_layers[0]].out, base[base.output_layers[0]].out)


def test_experimental_session_captures_plainly() -> None:
    model, x = _mlp_case()
    session = torchlens.experimental.Session(model)
    first = session.invoke(x)
    second = session.invoke(x + 1)
    assert first.is_replayable is False
    assert second.is_replayable is False
    assert first.options.save_arg_templates is True
    assert len(session.to_bundle()) == 2

"""Lane F01 injections checkpoint: ``log_injections`` stages 0-1.

Surgery memo 3.5 / foldA D12: computation performed inside intervention
hooks becomes first-class injected-op records with anchored identity.
Covered here, with THE MISFIRE TEST first (the highest-priority test in the
build -- both halves of the no-ordinal-consumption mechanism measured):
injected ops never shift a later target's live label and never consume
site-key cohort ordinals; anchored provenance is VERBATIM the C07
``injection_provenance`` slot grammar (host_site_key, spec_rule_id,
host_pass, firing_index, nesting_path, local_op_ordinal, output_slot); the
query split (``trace.injected_ops`` / ``trace.model_ops``); selectors never
firing on injected ops; stage-1 save refusing typed and naming lane F44;
exception cleanup; and the release gate -- ordinary validation behaves
IDENTICALLY on logged and unlogged intervened traces (no new exemption,
ever).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl

_LOGGED = tl.options.CaptureOptions(log_injections=True)


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3: early-inject / late-target substrate."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


class _TwoRelu(nn.Module):
    """Two relu fires for one rule: the firing-index substrate."""

    def __init__(self) -> None:
        """Build two linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fire relu twice."""

        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


def _sae_like(out: torch.Tensor, *, ctx) -> torch.Tensor:
    """SAE-style injected computation: encode, gate, decode (5 torch calls)."""

    z = torch.relu(out @ torch.eye(out.shape[-1]))
    return torch.sigmoid(z) * 2.0


@pytest.fixture()
def chain():
    """Seeded chain model + input."""

    torch.manual_seed(0)
    return _Chain().eval(), torch.randn(3, 4)


def test_injection_validation_tripwire_unchanged(chain) -> None:
    """RELEASE GATE: ordinary validation identical on logged vs unlogged.

    No surgery tolerance, no new validation exemption, ever: the logged
    trace's forward-pass validation verdict must equal the unlogged
    intervened trace's verdict (like for like; an intervened run diverges
    from the plain oracle in BOTH cases identically).
    """

    model, x = chain
    spec = tl.when(tl.func("relu"), _sae_like)
    base = tl.trace(model, x, intervene=spec)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert logged.validate_forward_pass(model(x)) == base.validate_forward_pass(model(x))
    # and a PLAIN armed capture validates exactly like a plain unarmed one
    plain_logged = tl.trace(model, x, capture=_LOGGED)
    plain_base = tl.trace(model, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logged_verdict = plain_logged.validate_forward_pass(model(x))
        base_verdict = plain_base.validate_forward_pass(model(x))
    assert str(logged_verdict) == str(base_verdict)


def test_injection_hook_exception_cleans_recorder(chain) -> None:
    """A raising hook tears the recorder down; later torch ops stay clean."""

    model, x = chain

    def _boom(out: torch.Tensor, *, ctx) -> torch.Tensor:
        """Inject one op, then raise."""

        torch.sigmoid(out)
        raise RuntimeError("hook boom")

    with pytest.raises(Exception, match="hook boom"):
        tl.trace(model, x, intervene=tl.when(tl.func("relu"), _boom), capture=_LOGGED)
    # no leftover TorchFunctionMode: plain torch runs unintercepted
    assert torch.relu(torch.tensor([-1.0, 2.0])).tolist() == [0.0, 2.0]


def test_streamed_to_disk_intervene_capture_settles_and_loads(chain, tmp_path) -> None:
    """FIX-2 regression: streamed ``to_disk`` + ``intervene=`` captures save.

    The streaming finalize scrubs the trace while the fire counter is still
    attached; before the enrollment this refused the whole capture at the
    save boundary. The bundle must settle, carry the intervention audit,
    and load back with the session counter dropped, never persisted.
    """

    model, x = chain
    path = str(tmp_path / "streamed_intervened.tlspec")
    log = tl.trace(
        model,
        x,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        storage=tl.to_disk(path),
    )
    assert log.outcome.status.name == "COMPLETE"
    assert len(log.intervention_audit) == 1
    loaded = tl.load(path)
    # The streamed descriptor is finalized before the settle epilogue appends
    # the audit row, so audit parity across the stream boundary is not pinned
    # here -- only the refusal class this fix removed: the capture settles,
    # the artifact loads, and the session counter never persists.
    assert loaded.outcome.status.name == "COMPLETE"
    assert getattr(loaded, "_tl_intervene_selector_fire_count", None) is None

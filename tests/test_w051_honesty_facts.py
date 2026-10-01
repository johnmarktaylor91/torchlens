"""Lane W051-HONESTY: the honesty fact block carries interventions and honest witness wording.

AUD-HONESTY M5: ``tl.trace(m, x, intervene=...)`` zero-ablates a site while the
capture settles ``complete``; ``to_agent_json()``, ``to_pandas().attrs``, the export
preamble, and ``explain`` carried NO intervention fact, so an agent reading the dump
could not know the numbers were counterfactual. H4/L10: every surface rendered the
default ``capture_verified=None`` as "no ceiling recorded" -- a clean bill the code
cannot back because the default capture never arms the completeness witness.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import (
    capture_honesty_facts,
    honesty_preamble_lines,
    intervention_facts,
)
from torchlens.options import CaptureOptions

pytestmark = pytest.mark.smoke


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


@pytest.fixture(scope="module")
def model_and_x() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _MLP().eval(), torch.randn(3, 8)


def test_untouched_capture_carries_no_intervention_block(model_and_x) -> None:
    model, x = model_and_x
    trace = tl.trace(model, x)
    assert intervention_facts(trace) is None
    facts = capture_honesty_facts(trace)
    assert "interventions" not in facts
    assert "interventions" not in trace.to_agent_json()["capture"]


def test_capture_time_intervene_is_disclosed_on_every_surface(model_and_x) -> None:
    """M5: the zero-ablated capture stays ``complete`` but every honesty surface
    now says the relu payload is counterfactual."""

    model, x = model_and_x
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    assert bool((trace["relu_1_2"].out == 0).all())
    facts = capture_honesty_facts(trace)
    block = facts["interventions"]
    assert block["intervened"] is True
    assert block["replaced_ops"] == ["relu_1_2"]
    assert block["replaced_op_count"] == 1
    assert block["fire_count"] == 1
    assert block["audit_row_count"] == 1
    assert block["audit_kinds"] == ["ACT"]
    # JSON-primitive: the block round-trips through json.
    assert json.loads(json.dumps(block)) == block
    dump = trace.to_agent_json()
    assert dump["capture"]["interventions"] == block
    assert "interven" in json.dumps(dump["guide"]).lower()
    preamble = "\n".join(honesty_preamble_lines(trace))
    assert "INTERVENED capture" in preamble and "relu_1_2" in preamble
    explained = tl.report.explain(trace)
    assert "INTERVENED capture" in explained and "counterfactual" in explained
    attrs = trace.to_pandas().attrs["torchlens_capture_honesty"]
    assert attrs["interventions"]["replaced_ops"] == ["relu_1_2"]


def test_intervention_block_survives_save_load(model_and_x, tmp_path) -> None:
    model, x = model_and_x
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    tl.save(trace, tmp_path / "iv.tlspec")
    loaded = tl.load(tmp_path / "iv.tlspec")
    block = capture_honesty_facts(loaded)["interventions"]
    assert block["replaced_ops"] == ["relu_1_2"]
    assert block["fire_count"] == 1


def test_fork_do_edits_are_disclosed(model_and_x) -> None:
    """Fork-side ``do()`` (ACT and PARAM kinds) is disclosed through the same block."""

    model, x = model_and_x
    trace = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    fork = trace.fork()
    fork.do(tl.units("relu_1_2", [(0, 0)]), tl.zero_ablate())
    block = intervention_facts(fork)
    assert block is not None
    assert block["replaced_ops"] == ["relu_1_2"]
    assert "ACT" in block["audit_kinds"]
    param_fork = trace.fork()
    param_fork.do(tl.params("b.weight"), tl.scale(0.5))
    param_block = intervention_facts(param_fork)
    assert param_block is not None
    assert param_block["audit_kinds"] == ["PARAM"]
    assert intervention_facts(trace) is None  # the source trace is untouched


def test_default_capture_verification_reads_not_recorded(model_and_x) -> None:
    """H4/L10: ``capture_verified=None`` renders as NOT RECORDED with the arming
    instruction, never as a clean bill."""

    model, x = model_and_x
    trace = tl.trace(model, x)
    assert trace.capture_verified is None
    explained = tl.report.explain(trace)
    assert "no ceiling recorded" not in explained
    assert "Capture verification: not recorded" in explained
    assert "completeness_witness=True" in explained
    preamble = honesty_preamble_lines(trace)
    assert preamble[0].endswith("verified=not_recorded")
    assert any("not recorded" in line and "completeness witness" in line for line in preamble)
    guide = json.dumps(trace.to_agent_json()["guide"])
    assert "not a clean bill" in guide

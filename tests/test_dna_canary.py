"""L6 DNA CANARY (D7 evidence) — lands immediately after stage 2.

Two live-trace demos judging the do()-verb lock (design memo sec 5):

DEMO 1 — zero-ablation via ``do()``: downstream deltas real; FireRecords +
state ``REPLAY_PROPAGATED`` + ``last_run`` honest; the nonfinite-gate
(ABORTED_NONFINITE settlement) behavior untouched.

DEMO 2 — SAME-RUN patch via ``do()`` with ``tl.patch_from``: THREE runs,
because identity alone cannot distinguish "patched with its own values"
from "silently did nothing":
  (a) IDENTITY — patching the trace's own values gives exact-identity
      output under the deterministic demo model, provenance present, and
      corroborated validation green end to end;
  (b) DISCRIMINATING TWIN — one masked element perturbed by a documented
      epsilon propagates a real nonzero downstream delta, cross-checked
      against direct recompute;
  (c) TAMPER TWIN — the same values presented captured-native (provenance
      stripped) FAIL validation.

The D7 rubric artifact (ergonomics / honesty / composability) is filed at
the sprint results dir; this suite is the executable half of that evidence.
The canary must NOT be descoped to pass — a failing demo IS the evidence.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.io import TraceState

_replay_module = importlib.import_module("torchlens.intervention.replay")

#: Demo 2 discriminating-twin perturbation (documented epsilon).
EPSILON = 10.0

#: Demo 2 committed index set (the memo's ``idx``).
IDX = ((0, 0, 1, 1), (0, 1, 2, 2))


class _CanaryNet(nn.Module):
    """Tiny deterministic CPU convnet (seeded; exact-geometry ops only)."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))))


def _fresh_capture() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    torch.manual_seed(0)
    model = _CanaryNet().double()
    x = torch.randn(1, 1, 12, 12, dtype=torch.float64)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    return model, x, trace


def test_demo_1_zero_ablation_via_do():
    """DEMO 1: one-verb zero ablation; honest state/records; N-gate untouched."""

    model, x, trace = _fresh_capture()
    baseline_out = trace["output_1"].out.clone()
    baseline_relu = trace["relu_1_2"].out.clone()

    fork = trace.fork()
    fork.do(trace["relu_1_2"].__selection__(), tl.zero_ablate())

    # downstream deltas real
    assert bool((fork["relu_1_2"].out == 0).all())
    assert not torch.equal(fork["output_1"].out, baseline_out)
    # source trace untouched (capture truth)
    assert torch.equal(trace["relu_1_2"].out, baseline_relu)

    # FireRecords + state + last_run honest
    records = list(fork["relu_1_2"].interventions)
    assert records and records[-1].helper_name == "zero_ablate"
    assert records[-1].replaced is True
    assert records[-1].engine == "replay"
    assert fork["relu_1_2"].intervention_replaced
    assert fork.state is TraceState.REPLAY_PROPAGATED
    assert isinstance(fork.last_run, dict) and fork.last_run.get("engine") == "replay"
    # audit record present (Selection-targeted disclosure)
    assert fork.intervention_audit[-1]["edit"] == "zero_ablate"

    # nonfinite-gate behavior untouched: a fresh raise_on_nan capture of a
    # nan-producing forward still settles ABORTED_NONFINITE and raises.
    class _NanNet(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs / (inputs - inputs)

    with pytest.raises(Exception) as excinfo:
        tl.trace(_NanNet(), torch.ones(2, 2), capture=tl.options.CaptureOptions(raise_on_nan=True))
    assert "non-finite" in str(excinfo.value).lower()


def test_demo_2_same_run_patch_identity():
    """DEMO 2a: patching the trace's OWN values is exact identity, with
    provenance present and corroborated validation green end to end."""

    model, x, trace = _fresh_capture()
    selection = tl.units("relu_1_2", IDX)
    resolved = selection.resolve(trace)

    patched = trace.fork()
    del resolved  # the resolve() spelling below is the memo's exact demo shape
    patched.do(selection.resolve(patched), tl.patch_from(trace))

    # exact-identity output under the deterministic demo model
    assert torch.equal(patched["output_1"].out, trace["output_1"].out)
    # provenance present: FireRecords at the patched site
    records = list(patched["relu_1_2"].interventions)
    assert records and records[-1].helper_name == "patch_from"
    assert patched.intervention_audit[-1]["patch_source"]["source_model_class"].endswith(
        "_CanaryNet"
    )
    # corroborated validation green END TO END (the identity patch reproduces
    # ground truth, and the intervention is corroborated, not hidden)
    assert patched.validate_forward_pass(model(x)) is True


def test_demo_2_same_run_patch_discriminating_twin():
    """DEMO 2b: a perturbed source value IS consumed (engine really patches),
    cross-checked against direct recompute."""

    model, x, trace = _fresh_capture()
    selection = tl.units("relu_1_2", IDX)
    mask = selection.resolve(trace)[0].mask

    source = trace.fork()
    perturbed = source["relu_1_2"].out.clone()
    perturbed[0, 0, 1, 1] += EPSILON
    _replay_module._commit_replay_updates(source, {"relu_1_2": perturbed}, {})

    patched = trace.fork()
    patched.do(selection.resolve(patched), tl.patch_from(source))

    # the patched element carries the perturbed value; the rest is intact
    assert torch.isclose(patched["relu_1_2"].out[0, 0, 1, 1], perturbed[0, 0, 1, 1])
    assert torch.equal(patched["relu_1_2"].out[~mask], trace["relu_1_2"].out[~mask])

    # expected nonzero downstream delta, cross-checked against DIRECT recompute
    assert not torch.equal(patched["output_1"].out, trace["output_1"].out)
    expected = torch.relu(model.c2(patched["relu_1_2"].out))
    assert torch.allclose(patched["output_1"].out, expected)


def test_demo_2_same_run_patch_tamper_twin():
    """DEMO 2c: the SAME values presented captured-native (provenance
    stripped) FAIL validation — the tripwire distinguishes an honest patch
    from silent tampering."""

    model, x, trace = _fresh_capture()

    tampered = trace.fork()
    forged = tampered["relu_1_2"].out.clone()
    forged[0, 0, 1, 1] += EPSILON
    # captured-native presentation: values written into the record with NO
    # FireRecords, NO spec revision, NO intervention_replaced flag.
    _replay_module._commit_replay_updates(tampered, {"relu_1_2": forged}, {})
    assert not tampered["relu_1_2"].intervention_replaced

    result = tampered.validate_forward_pass(model(x))
    assert result is False

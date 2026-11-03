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

import torch
import torch.nn as nn

import torchlens as tl

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
        capture=tl.options.CaptureOptions(replayable=True, save_args=True),
    )
    return model, x, trace


def test_demo_2_same_run_patch_identity():
    """DEMO 2a: patching the trace's OWN values is exact identity, with
    provenance present and corroborated validation green end to end."""

    model, x, trace = _fresh_capture()
    selection = tl.units("relu_1_2", IDX)
    resolved = trace[selection]

    patched = trace.fork()
    del resolved  # the resolve() spelling below is the memo's exact demo shape
    patched.do(patched[selection], tl.patch_from(trace))

    # exact-identity output under the deterministic demo model
    assert torch.equal(patched["output_1"].out, trace["output_1"].out)
    # provenance present: FireRecords at the patched site
    records = list(patched["relu_1_2"].edits)
    assert records and records[-1].edit_name == "patch_from"
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
    mask = trace[selection][0].mask

    source = trace.fork()
    perturbed = source["relu_1_2"].out.clone()
    perturbed[0, 0, 1, 1] += EPSILON
    _replay_module._commit_replay_updates(source, {"relu_1_2": perturbed}, {})

    patched = trace.fork()
    patched.do(patched[selection], tl.patch_from(source))

    # the patched element carries the perturbed value; the rest is intact
    assert torch.isclose(patched["relu_1_2"].out[0, 0, 1, 1], perturbed[0, 0, 1, 1])
    assert torch.equal(patched["relu_1_2"].out[~mask], trace["relu_1_2"].out[~mask])

    # expected nonzero downstream delta, cross-checked against DIRECT recompute
    assert not torch.equal(patched["output_1"].out, trace["output_1"].out)
    expected = torch.relu(model.c2(patched["relu_1_2"].out))
    assert torch.allclose(patched["output_1"].out, expected)

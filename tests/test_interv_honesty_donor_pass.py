"""Intervention honesty: ``patch_from`` donor resolution is PASS-QUALIFIED
(edits memo row 4, decision D2 -- the sprint's worst silent wrongness).

The historical hook resolved the donor via the bare layer label, which the
lookup table binds to the LAST pass: on recurrent/weight-reused models a
patch addressed at pass k silently delivered pass N's donor value. Donor
resolution now goes through the pass-qualified spelling; a bare lookup on a
multi-pass donor site refuses typed instead of guessing.

Row gate: the recurrent-model donor pin runs green on a real weight-shared
transformer encoder (ALBERT-pattern cross-layer sharing, stdlib form); the
R0 ALBERT-family pin ran as a Tier-B xfail while FIX-A's replay-cone crash
(lane A05) was open and was flipped to a plain test once FIX-A landed
(verified XPASS on the merged tree; edits memo B4 row honored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import HookValueError

R0_DIR = str(Path(__file__).resolve().parent / "real_model" / "r0")


class _TwoPass(nn.Module):
    """Minimal weight-reused module: one Linear called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.lin(x))
        x = torch.relu(self.lin(x))
        return x


def _pair() -> tuple[tl.Trace, tl.Trace]:
    torch.manual_seed(0)
    model = _TwoPass().eval()
    xa, xb = torch.randn(2, 4), torch.randn(2, 4)
    capture = tl.options.CaptureOptions(intervention_ready=True)
    donor = tl.trace(model, xa, capture=capture)
    base = tl.trace(model, xb, capture=capture)
    return donor, base


@pytest.mark.smoke
@pytest.mark.parametrize("pass_index", [1, 2])
def test_patch_from_delivers_the_addressed_pass(pass_index: int) -> None:
    """Each addressed pass receives ITS donor value, never the last pass's."""

    donor, base = _pair()
    other = 3 - pass_index
    fork = base.fork()
    fork.do(f"relu_1_2:{pass_index}", tl.patch_from(donor))
    patched = fork[f"relu_1_2:{pass_index}"].out
    assert torch.equal(patched, donor[f"relu_1_2:{pass_index}"].out)
    assert not torch.equal(patched, donor[f"relu_1_2:{other}"].out)


@pytest.mark.smoke
def test_bare_multipass_donor_lookup_refuses_instead_of_guessing() -> None:
    """A fire context without a pass on a multi-pass donor site refuses typed
    and teaches every pass-qualified spelling."""

    from torchlens.intervention.hooks import make_hook_context

    donor, base = _pair()
    hook_callable = tl.patch_from(donor).factory()
    context = make_hook_context(
        name="patch_from",
        layer_log={"layer_label": "relu_1_2", "label": "relu_1_2"},
        run_ctx={},
    )
    with pytest.raises(HookValueError) as excinfo:
        hook_callable(base["relu_1_2:1"].out, hook=context)
    message = str(excinfo.value)
    assert "multi-pass" in message
    assert "'relu_1_2:1'" in message
    assert "'relu_1_2:2'" in message
    assert excinfo.value.fields["code"] == "patch_donor_pass_ambiguous"
    remedy = excinfo.value.fields["remedy"]
    assert remedy and message.rstrip(".").endswith(remedy)


@pytest.mark.smoke
def test_single_pass_donor_still_resolves() -> None:
    """Single-pass sites keep resolving (bare or qualified spellings)."""

    donor, base = _pair()
    fork = base.fork()
    fork.do("linear_1_1:1", tl.patch_from(donor))
    assert torch.equal(fork["linear_1_1:1"].out, donor["linear_1_1:1"].out)


class _SharedEncoder(nn.Module):
    """ALBERT-pattern cross-layer sharing in stdlib form: one real
    ``nn.TransformerEncoderLayer`` applied twice with SHARED weights."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.layer(x)
        return x


@pytest.mark.smoke
@pytest.mark.real_model
def test_recurrent_model_donor_pin_shared_transformer_encoder() -> None:
    """ROW GATE (recurrent-model donor pin): pass-correct donors on a real
    weight-shared transformer encoder, at every multi-pass linear site."""

    torch.manual_seed(0)
    model = _SharedEncoder().eval()
    xa, xb = torch.randn(5, 2, 16), torch.randn(5, 2, 16)
    capture = tl.options.CaptureOptions(intervention_ready=True)
    donor = tl.trace(model, xa, capture=capture)
    base = tl.trace(model, xb, capture=capture)

    sites = [
        label
        for label in donor.layer_labels
        if label.startswith("linear") and getattr(donor[label], "num_passes", 1) > 1
    ]
    assert sites, "shared encoder must produce multi-pass linear sites"
    for site in sites:
        for pass_index in (1, 2):
            other = 3 - pass_index
            fork = base.fork()
            fork.do(f"{site}:{pass_index}", tl.patch_from(donor))
            patched = fork[f"{site}:{pass_index}"].out
            assert torch.equal(patched, donor[f"{site}:{pass_index}"].out), (
                f"{site} pass {pass_index}: donor is not pass-correct"
            )
            assert not torch.equal(patched, donor[f"{site}:{other}"].out), (
                f"{site} pass {pass_index}: received the other pass's donor"
            )


@pytest.mark.real_model
@pytest.mark.heavy
def test_recurrent_model_donor_pin_albert_family() -> None:
    """Tier-B row (edits memo B4): pass-correct donors on the R0 ALBERT family
    fixture (real cross-layer weight sharing -- every encoder layer reuses ONE
    module)."""

    sys.path.insert(0, R0_DIR)
    try:
        from families import FAMILIES
    finally:
        sys.path.remove(R0_DIR)
    spec = next(family for family in FAMILIES if family.name == "albert")

    model = spec.build("eager")
    input_ids = spec.input_kwargs()["input_ids"]
    torch.manual_seed(1)
    other_ids = torch.randint_like(input_ids, low=5, high=int(input_ids.max()) + 1)

    capture = tl.options.CaptureOptions(intervention_ready=True)
    donor = tl.trace(model, input_ids, capture=capture)
    base = tl.trace(model, other_ids, capture=capture)

    site = next(
        label
        for label in donor.layer_labels
        if label.startswith("linear") and getattr(donor[label], "num_passes", 1) > 1
    )
    donor_pass_1 = donor[f"{site}:1"].out
    donor_pass_2 = donor[f"{site}:2"].out
    assert not torch.equal(donor_pass_1, donor_pass_2), "degenerate donor site"

    for pass_index, same, other in [
        (1, donor_pass_1, donor_pass_2),
        (2, donor_pass_2, donor_pass_1),
    ]:
        fork = base.fork()
        fork.do(f"{site}:{pass_index}", tl.patch_from(donor))
        patched = fork[f"{site}:{pass_index}"].out
        assert torch.equal(patched, same), f"pass {pass_index} donor is not pass-correct"
        assert not torch.equal(patched, other), f"pass {pass_index} received the other pass"

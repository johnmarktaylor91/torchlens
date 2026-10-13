"""``detach_saved_activations`` holds on every save path, in both directions.

The matrix crosses the three save forms (the default save-all, a ``tl.module``
union and a ``tl.in_module`` union) with the four places a saved activation
comes back (the source trace, the legacy ``run(model, x)`` rerun, the
transactional ``run(inputs=...)`` and the guarded ``run(inputs=..., fast=True)``)
and both settings of the knob. ``False`` (the default) returns graph-connected
payloads whose backward reaches the parameters upstream of the site; ``True``
returns detached payloads. The model reuses one ``ReLU`` per residual block and
adds the shortcut in place, the two shapes that used to refuse the fast path
(``conditional_arm_divergence`` and ``run_capability_unavailable``).

Repeated optimiser steps through the fast run's truncated output must equal the
same steps on a hand-truncated eager model, and the ``tl.extract`` and
``tl.record`` knobs are tested directly.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TrainingModeConfigError
from torchlens.options import CaptureOptions

_SITE = "layer2"


class _Block(nn.Module):
    """Residual block that calls one ``ReLU`` twice, as torchvision's BasicBlock does."""

    def __init__(self, channels: int) -> None:
        """Build the two convolutions and their norms.

        Parameters
        ----------
        channels:
            Input and output channel count.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Block output.
        """

        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class _Net(nn.Module):
    """Stem, two two-block stages and a linear head."""

    def __init__(self) -> None:
        """Build the stages."""

        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(_Block(8), _Block(8))
        self.layer2 = nn.Sequential(_Block(8), _Block(8))
        self.fc = nn.Linear(8, 5)

    def stem_to_layer2(self, x: torch.Tensor) -> torch.Tensor:
        """Run the hand-truncated prefix that ends at ``layer2``.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            ``layer2`` output.
        """

        return self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the whole network.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Logits.
        """

        return self.fc(torch.flatten(self.stem_to_layer2(x).mean(dim=(2, 3)), 1))


def _model() -> _Net:
    """Return a seeded eval-mode network."""

    torch.manual_seed(0)
    return _Net().eval()


def _inputs(seed: int) -> torch.Tensor:
    """Return a seeded image batch.

    Parameters
    ----------
    seed:
        Generator seed.

    Returns
    -------
    torch.Tensor
        Batch of shape ``(2, 3, 8, 8)``.
    """

    return torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(seed))


_SAVES: dict[str, Callable[[], Any]] = {
    "save_all": lambda: None,
    "module": lambda: tl.module("layer1") | tl.module(_SITE),
    "in_module": lambda: tl.in_module("layer1") | tl.in_module(_SITE),
}


def _site_after(path: str, model: _Net, trace: Any, x2: torch.Tensor) -> torch.Tensor:
    """Return the ``layer2`` payload a save path hands back.

    Parameters
    ----------
    path:
        One of ``trace``, ``legacy_run``, ``run_inputs`` and ``run_inputs_fast``.
    model:
        Source model.
    trace:
        Fresh capture of ``model``.
    x2:
        New input for the reruns.

    Returns
    -------
    torch.Tensor
        The saved ``layer2`` out.
    """

    if path == "trace":
        return trace[_SITE].out
    if path == "legacy_run":
        return trace.run(model, x2)[_SITE].out
    if path == "run_inputs":
        # carry_state=True keeps the declared-state restore out of this
        # attachment check; the restore's own backward contract is pinned with
        # the until= work.
        return trace.run(inputs=x2, carry_state=True).trace[_SITE].out
    return trace.run(inputs=x2, fast=True).trace[_SITE].out


@pytest.mark.parametrize("detach", [False, True], ids=["attached", "detached"])
@pytest.mark.parametrize("path", ["trace", "legacy_run", "run_inputs", "run_inputs_fast"])
@pytest.mark.parametrize("save_name", list(_SAVES))
def test_detach_choice_holds_on_every_save_path(save_name: str, path: str, detach: bool) -> None:
    """Every save form and every return path honours the knob in both directions."""

    model = _model()
    trace = tl.trace(
        model,
        _inputs(1),
        save=_SAVES[save_name](),
        capture=CaptureOptions(detach_saved_activations=detach),
    )
    site = _site_after(path, model, trace, _inputs(2))

    assert site.requires_grad is (not detach)
    assert (site.grad_fn is not None) is (not detach)
    if detach:
        return
    for param in model.parameters():
        param.grad = None
    site.pow(2).mean().backward()
    upstream = [model.conv1.weight, *model.layer1.parameters(), *model.layer2.parameters()]
    assert all(param.grad is not None for param in upstream)
    assert model.fc.weight.grad is None


def test_fast_optimiser_steps_match_hand_truncated_eager_model() -> None:
    """Training through the fast run's ``layer2`` output equals the eager prefix, step for step."""

    traced = _model()
    eager = copy.deepcopy(traced)
    trace = tl.trace(traced, _inputs(1), save=tl.module(_SITE))
    traced_opt = torch.optim.SGD(traced.parameters(), lr=0.1)
    eager_opt = torch.optim.SGD(eager.parameters(), lr=0.1)
    for step in range(4):
        batch = _inputs(10 + step)
        traced_opt.zero_grad()
        trace.run(inputs=batch, fast=True).trace[_SITE].out.pow(2).mean().backward()
        traced_opt.step()
        eager_opt.zero_grad()
        eager.stem_to_layer2(batch).pow(2).mean().backward()
        eager_opt.step()

    for (name, after_fast), after_eager in zip(
        traced.named_parameters(), eager.parameters(), strict=True
    ):
        assert torch.equal(after_fast, after_eager), name


def test_extract_knob_returns_attached_outs_on_request() -> None:
    """``tl.extract`` keeps its detached default and attaches with the knob."""

    model = _model()
    x = _inputs(3)
    default = tl.extract(model, x, [_SITE])[_SITE]
    attached = tl.extract(model, x, [_SITE], detach_saved_activations=False)[_SITE]

    assert default.grad_fn is None and not default.requires_grad
    assert attached.grad_fn is not None
    assert torch.equal(default, attached.detach())
    attached.sum().backward()
    assert model.layer1[0].conv1.weight.grad is not None


def test_record_knob_returns_attached_payloads_on_request() -> None:
    """``tl.record`` keeps its detached default and attaches with the knob."""

    model = _model()
    x = _inputs(4)
    default = [record.ram_payload for record in tl.record(model, x, save=tl.module(_SITE))]
    attached = [
        record.ram_payload
        for record in tl.record(model, x, save=tl.module(_SITE), detach_saved_activations=False)
    ]

    assert default and all(payload.grad_fn is None for payload in default)
    assert attached and all(payload.grad_fn is not None for payload in attached)
    attached[0].sum().backward()
    assert model.conv1.weight.grad is not None


def test_record_knob_refuses_detaching_a_backward_ready_record() -> None:
    """An explicit ``True`` contradicts ``backward_ready=True`` and refuses typed."""

    with pytest.raises(TrainingModeConfigError) as refused:
        tl.record(
            _model(),
            _inputs(5),
            save=tl.module(_SITE),
            backward_ready=True,
            detach_saved_activations=True,
        )
    assert refused.value.fields["code"] == "backward_ready_conflict"

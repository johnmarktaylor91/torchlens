"""``run(inputs=..., until=..., fast=True)`` stops the native forward and trains the prefix (U26.3).

The fast live tier cuts its native forward right after the ``until=`` site's
output is collected, so nothing downstream runs, and the saved prefix comes back
graph-connected under the default save. Acceptance: on torchvision resnet18 the
``layer2`` output's gradients equal a hand-truncated eager model's on every
upstream parameter, downstream parameters stay untouched, and the stages after
``layer2`` never execute. The cut is disclosed (``report.truncated``,
``report.stopped_at``, ``RunTruncation`` with the ``live_stop_after`` regime), the
sites that did not run read ``None`` for that run, and the next full run restores
them. Repeated fast training steps no longer report a pending value edit.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.intervention.errors import PendingValueEditsWarning

torchvision = pytest.importorskip("torchvision")

_UPSTREAM = ("conv1", "bn1", "layer1", "layer2")
_DOWNSTREAM = ("layer3", "layer4", "fc")


def _resnet18() -> nn.Module:
    """Return a seeded, weight-free, eval-mode resnet18."""

    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None).eval()


def _batch(seed: int) -> torch.Tensor:
    """Return a seeded small image batch.

    Parameters
    ----------
    seed:
        Generator seed.

    Returns
    -------
    torch.Tensor
        Batch of shape ``(2, 3, 32, 32)``.
    """

    return torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def _eager_layer2(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run resnet18's stem through ``layer2`` by hand.

    Parameters
    ----------
    model:
        A torchvision resnet.
    x:
        Image batch.

    Returns
    -------
    torch.Tensor
        ``layer2`` output.
    """

    stem = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer2(model.layer1(stem))


def _grads(model: nn.Module, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor | None]:
    """Collect parameter gradients under the given top-level prefixes.

    Parameters
    ----------
    model:
        Model whose parameters were differentiated.
    prefixes:
        Top-level module names to include.

    Returns
    -------
    dict[str, torch.Tensor | None]
        Gradient (or ``None``) by parameter name.
    """

    return {
        name: param.grad
        for name, param in model.named_parameters()
        if name.split(".", 1)[0] in prefixes
    }


@pytest.mark.parametrize(
    "save",
    [None, tl.module("layer1") | tl.module("layer2"), tl.in_module("layer2")],
    ids=["save_all", "module", "in_module"],
)
def test_resnet18_until_fast_gradients_equal_hand_truncated_eager(save: Any) -> None:
    """Every upstream gradient equals the eager prefix's; downstream never runs."""

    model = _resnet18()
    eager = copy.deepcopy(model)
    trace = tl.trace(model, _batch(0), save=save)
    calls: dict[str, int] = dict.fromkeys(_DOWNSTREAM, 0)

    def _count(name: str) -> Any:
        def hook(_module: nn.Module, _args: Any, _out: Any) -> None:
            calls[name] += 1

        return hook

    handles = [getattr(model, name).register_forward_hook(_count(name)) for name in _DOWNSTREAM]
    try:
        x = _batch(1)
        result = trace.run(inputs=x, until="layer2", fast=True)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == dict.fromkeys(_DOWNSTREAM, 0)
    assert result.output is None
    assert result.report.truncated is True
    assert result.report.truncation is not None
    assert result.report.truncation.regime == "live_stop_after"
    site = result.trace["layer2"].out
    assert site.grad_fn is not None
    site.pow(2).mean().backward()
    _eager_layer2(eager, x).pow(2).mean().backward()

    fast_grads = _grads(model, _UPSTREAM)
    eager_grads = _grads(eager, _UPSTREAM)
    assert fast_grads and all(grad is not None for grad in fast_grads.values())
    for name, grad in fast_grads.items():
        assert torch.equal(grad, eager_grads[name]), name
    assert all(grad is None for grad in _grads(model, _DOWNSTREAM).values())


def test_until_fast_clears_skipped_sites_and_a_full_run_restores_them() -> None:
    """Sites after the stop read ``None`` for the cut run; the next full run refreshes them."""

    model = _resnet18()
    trace = tl.trace(model, _batch(0), save=tl.module("layer2") | tl.module("layer3"))
    cut = trace.run(inputs=_batch(1), until="layer2", fast=True)
    assert cut.trace["layer3"].out is None
    assert cut.trace["layer2"].out is not None

    full = trace.run(inputs=_batch(1), fast=True)
    assert full.output is not None
    assert full.trace["layer3"].out is not None


def test_until_fast_refuses_an_unsaved_site_typed() -> None:
    """A site the capture did not save has no fast hook to stop at."""

    model = _resnet18()
    trace = tl.trace(model, _batch(0), save=tl.module("layer1"))
    with pytest.raises(RunCapabilityUnavailableError) as refused:
        trace.run(inputs=_batch(1), until="layer3", fast=True)
    assert refused.value.fields["detection_stage"] == "fast_until_site_unsaved"


def test_repeated_fast_training_steps_report_no_pending_value_edit() -> None:
    """A previous fast run's own trail is a fresh execution, not a value edit."""

    model = _resnet18()
    trace = tl.trace(model, _batch(0), save=tl.module("layer2"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for step in range(3):
            optimizer.zero_grad()
            out = trace.run(inputs=_batch(10 + step), until="layer2", fast=True).trace["layer2"]
            out.out.pow(2).mean().backward()
            optimizer.step()
    assert not [w for w in caught if issubclass(w.category, PendingValueEditsWarning)]

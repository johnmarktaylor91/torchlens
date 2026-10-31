"""U31.2: the checks session beside an LR scheduler, in both registration orders.

The incumbent's mount replaced ``optimizer.step``, which on current torch
crashes or makes the scheduler warn that ``optimizer.step()`` "has been
overridden after learning rate scheduler initialization". The TorchLens
``ChecksSession`` mounts through the public optimizer step hooks, so a real
model training a few steps with a scheduler created before OR after
``attach()`` must emit neither scheduler warning, must not crash, and must
still count every accepted step and decay the learning rate.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torchvision
from torch import nn

from torchlens.checks import ChecksSession

pytestmark = [pytest.mark.real_model, pytest.mark.real_class]

_STEPS = 3
_SCHEDULER_WARNINGS = ("overridden", "lr_scheduler.step()", "optimizer.step()")


def _train(order: str) -> tuple[ChecksSession, torch.optim.Optimizer, list[str]]:
    """Run a short real-model loop; return the session, optimizer and warnings."""

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None, num_classes=10).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    session = ChecksSession(model, optimizer)
    session.register_change_check()
    session.register_nonfinite_grad_check()
    images, labels = torch.randn(2, 3, 32, 32), torch.tensor([1, 7])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if order == "scheduler_first":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            session.attach()
        else:
            session.attach()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        try:
            for _ in range(_STEPS):
                optimizer.zero_grad()
                nn.functional.cross_entropy(model(images), labels).backward()
                optimizer.step()
                scheduler.step()
        finally:
            session.detach()
    messages = [str(w.message) for w in caught]
    return session, optimizer, messages


@pytest.mark.parametrize("order", ["scheduler_first", "attach_first"])
def test_checks_session_coexists_with_an_lr_scheduler(order: str) -> None:
    """No scheduler warning, no crash, every step counted, the LR decays."""

    session, optimizer, messages = _train(order)
    scheduler_warnings = [
        message
        for message in messages
        if any(fragment in message for fragment in _SCHEDULER_WARNINGS)
    ]
    assert scheduler_warnings == [], scheduler_warnings
    assert session.accepted_step_id == _STEPS
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**_STEPS)
    report = session.report()
    assert report.checks_run, "the registered checks never ran"

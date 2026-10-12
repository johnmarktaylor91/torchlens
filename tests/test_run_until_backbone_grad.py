"""Trainable truncated backbones through ``trace.run`` (U26.3 fixes 1 and 2).

A trace taken once must support training a backbone through ``trace.run``:

- ``run(until=...)`` runs on BatchNorm models in eval mode (the prefix
  buffer-sink evidence used to compare as ORDERED tuples, and the halted
  refresh discovers its sinks from the stop frontier rather than the output,
  so equal evidence in a different order refused ``buffer_sink_routing_mutable``).
- ``backward()`` through a default (``carry_state=False``) run's activations
  works: the declared-state restore used to ``copy_`` every slot back, bumping
  the version counter of parameters and eval-mode running statistics autograd
  had saved, so backward raised torch's in-place modification error.

Gradients are checked against an independently run, hand-truncated eager copy
of the model, parameter by parameter, and the cut-off part of the model is
proven never to execute.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import torchlens as tl


class _BnStack(nn.Module):
    """Three conv-BN-ReLU blocks and a head: two BNs before ``block2``, one after."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.block1 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block2(self.block1(self.block0(x)))
        return self.head(h.mean(dim=(2, 3)))


def _eval_stack() -> _BnStack:
    torch.manual_seed(0)
    model = _BnStack()
    # Non-trivial running statistics so eval-mode BN is not the identity.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.5, 1.5)
    return model.eval()


def _state_bytes(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _module_call_counter(model: nn.Module, names: tuple[str, ...]) -> dict[str, int]:
    counts = dict.fromkeys(names, 0)
    modules = dict(model.named_modules())
    for name in names:

        def _hook(_module, _inputs, _output, name=name):
            counts[name] += 1

        modules[name].register_forward_hook(_hook)
    return counts


def test_until_runs_on_eval_batchnorm_stack():
    """Bug 1: until= on a multi-BN eval model runs instead of refusing typed."""

    model = _eval_stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    result = log.run(inputs=torch.randn(2, 3, 8, 8), until="block1")
    assert result.report.truncated is True
    assert result.output is None
    assert result.trace["block1"].out is not None


def test_until_backward_without_carry_state_matches_hand_truncated_eager():
    """Bugs 1+2: until= backward works by default and equals a hand-truncated eager model.

    Upstream of the ``block1`` stop: block0 and block1 each carry conv weight,
    conv bias, BN weight and BN bias, so exactly 8 parameters receive gradients.
    block2 and the head never execute and keep ``grad is None``.
    """

    model = _eval_stack()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 8, 8)
    x2 = torch.randn(2, 3, 8, 8)
    counts = _module_call_counter(model, ("block2", "head"))
    log = tl.trace(model, x, save=tl.in_module("block1"))
    counts.update(dict.fromkeys(counts, 0))
    before = _state_bytes(model)

    result = log.run(inputs=x2, until="block1")
    activation = result.trace["block1"].out
    assert activation.grad_fn is not None
    activation.pow(2).mean().backward()

    assert counts == {"block2": 0, "head": 0}
    got = _grads(model)
    assert len(got) == 8
    assert all(name.startswith(("block0.", "block1.")) for name in got)
    assert all(param.grad is None for param in model.block2.parameters())
    assert all(param.grad is None for param in model.head.parameters())

    reference.block1(reference.block0(x2)).pow(2).mean().backward()
    expected = _grads(reference)
    assert sorted(got) == sorted(expected)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=0, atol=0, msg=name)

    after = _state_bytes(model)
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_full_run_backward_without_carry_state_matches_eager():
    """Bug 2 on the full path: default run() backward equals the eager full model."""

    model = _eval_stack()
    reference = copy.deepcopy(model)
    x2 = torch.randn(2, 3, 8, 8)
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    result = log.run(inputs=x2)
    result.output.pow(2).mean().backward()
    reference(x2).pow(2).mean().backward()

    got, expected = _grads(model), _grads(reference)
    assert len(expected) == len(list(model.parameters())) == 14
    assert sorted(got) == sorted(expected)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=0, atol=0, msg=name)


def test_restore_rewrites_only_changed_declared_state():
    """The restore bracket leaves untouched slots' version counters alone."""

    model = _eval_stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    versions = {name: value._version for name, value in model.state_dict(keep_vars=True).items()}
    log.run(inputs=torch.randn(2, 3, 8, 8))
    log.run(inputs=torch.randn(2, 3, 8, 8), until="block1")
    after = {name: value._version for name, value in model.state_dict(keep_vars=True).items()}
    assert after == versions


def _resnet18_eval() -> nn.Module:
    torchvision_models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    return torchvision_models.resnet18(weights=None, num_classes=10).eval()


_LAYER2_UPSTREAM_PARAMS = 30  # stem 3 + layer1 12 + layer2 15 (incl. downsample conv and BN)
_RESNET18_PARAMS = 62


def test_resnet18_until_layer2_grads_match_hand_truncated_eager():
    """The RERUN-GRAD probe on stock resnet18: until= trains the stem..layer2 backbone."""

    model = _resnet18_eval()
    reference = copy.deepcopy(model)
    x, x2 = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    counts = _module_call_counter(model, ("layer3", "layer4", "fc"))
    log = tl.trace(model, x, save=tl.in_module("layer1") | tl.in_module("layer2"))
    counts.update(dict.fromkeys(counts, 0))

    result = log.run(inputs=x2, until="layer2")
    assert result.report.truncated is True
    result.trace["layer2"].out.pow(2).mean().backward()

    assert counts == {"layer3": 0, "layer4": 0, "fc": 0}
    got = _grads(model)
    assert len(got) == _LAYER2_UPSTREAM_PARAMS
    stem = nn.Sequential(
        reference.conv1,
        reference.bn1,
        reference.relu,
        reference.maxpool,
        reference.layer1,
        reference.layer2,
    )
    stem(x2).pow(2).mean().backward()
    expected = _grads(reference)
    assert sorted(got) == sorted(expected)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=0, atol=0, msg=name)
    for name, param in model.named_parameters():
        if name.startswith(("layer3.", "layer4.", "fc.")):
            assert param.grad is None, name


def test_resnet18_full_run_grads_match_eager():
    """Full default run(): backward reaches all 62 parameters, equal to eager."""

    model = _resnet18_eval()
    reference = copy.deepcopy(model)
    x2 = torch.randn(2, 3, 32, 32)
    log = tl.trace(model, torch.randn(2, 3, 32, 32))
    log.run(inputs=x2).output.pow(2).mean().backward()
    reference(x2).pow(2).mean().backward()
    got, expected = _grads(model), _grads(reference)
    assert len(expected) == _RESNET18_PARAMS
    assert sorted(got) == sorted(expected)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=0, atol=0, msg=name)

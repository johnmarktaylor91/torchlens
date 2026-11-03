"""The README's unit-influence line on torchvision ResNet-18 (random weights, CPU).

``trace["conv1"].projective_field.of(patch, within=["layer2", "layer3"])`` must
run exactly as the README spells it: ``"conv1"`` is a module address, and
``"layer2"`` / ``"layer3"`` are module addresses that stand for each block's
output op. The answer is checked against the union of the shipped per-unit
``.at()`` lifts.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl

torchvision = pytest.importorskip("torchvision")


@pytest.fixture(scope="module")
def resnet_trace():
    """ResNet-18, random weights, eval mode, one 224x224 image."""

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).eval()
    trace = tl.trace(model, torch.randn(1, 3, 224, 224))
    try:
        yield trace
    finally:
        trace.cleanup()


def _module_output(trace, address: str):
    """Return the only output op of the module at ``address``."""

    (label,) = trace.modules[address].calls[0].output_ops
    return trace.ops[label]


def test_readme_projective_field_of_line(resnet_trace) -> None:
    """The README line returns one partial region per named block, matching ``.at()``."""

    trace = resnet_trace
    conv1 = _module_output(trace, "conv1")
    patch = torch.zeros(tuple(conv1.shape), dtype=torch.bool)
    patch[0, :, 50:52, 60:62] = True

    region = trace["conv1"].projective_field.of(patch, within=["layer2", "layer3"])

    assert isinstance(region, tl.Selection)
    targets = [_module_output(trace, "layer2"), _module_output(trace, "layer3")]
    keys = [f"{entry.key[0]}:{entry.key[1]}" for entry in region]
    assert keys == [op.label for op in targets]
    units = [(row, col) for row in (50, 51) for col in (60, 61)]
    for entry, target in zip(region, targets, strict=True):
        assert 0 < entry.num_selected < entry.mask.numel()
        assert entry.provenance.basis == "geometric"
        expected = torch.zeros(tuple(target.shape), dtype=torch.bool)
        for unit in units:
            box = conv1.projective_field.at(unit, target=target)
            (row,) = [
                lifted
                for lifted in trace[box.__selection__()]
                if f"{lifted.key[0]}:{lifted.key[1]}" == target.label
            ]
            expected |= row.mask
        assert torch.equal(entry.mask, expected), target.label


def test_readme_receptive_field_of_mirror(resnet_trace) -> None:
    """The receptive direction from layer3 back to conv1 is a partial conv1 region."""

    trace = resnet_trace
    layer3 = _module_output(trace, "layer3")
    units = torch.zeros(tuple(layer3.shape), dtype=torch.bool)
    units[0, :, 3:5, 3:5] = True
    region = trace["layer3"].receptive_field.of(units, within="conv1")
    (entry,) = list(region)
    assert f"{entry.key[0]}:{entry.key[1]}" == _module_output(trace, "conv1").label
    assert 0 < entry.num_selected < entry.mask.numel()

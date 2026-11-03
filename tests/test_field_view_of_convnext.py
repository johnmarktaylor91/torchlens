"""The README's unit-influence line in its ConvNeXt form (timm ``convnext_tiny``, random weights).

``trace["stem"].projective_field.of(patch, within=["stages.1", "stages.2"])``:
``"stem"`` is the module whose output op seeds the query, and ``"stages.1"`` /
``"stages.2"`` are plain dotted module-address strings that ``within=`` reads
as each stage's output op. Geometric rules through ConvNeXt's permute, linear
and layer-norm ops are separate work; until they exist the geometric method
refuses typed with ``influence_geometry_unavailable``. Either way the module
strings must resolve: they never refuse as unknown sites.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.errors import SelectionError

timm = pytest.importorskip("timm")


@pytest.fixture(scope="module")
def convnext_trace():
    """convnext_tiny, random weights, eval mode, one 224x224 image."""

    torch.manual_seed(0)
    model = timm.create_model("convnext_tiny", pretrained=False).eval()
    trace = tl.trace(model, torch.randn(1, 3, 224, 224))
    try:
        yield trace
    finally:
        trace.cleanup()


def _module_output(trace, address: str):
    """Return the only output op of the module at ``address``."""

    (label,) = trace.modules[address].calls[0].output_ops
    return trace.ops[label]


def test_readme_convnext_line_reads_module_strings(convnext_trace) -> None:
    """The stage strings resolve; the answer is per-stage regions or the typed geometry refusal."""

    trace = convnext_trace
    stem = _module_output(trace, "stem")
    patch = torch.zeros(tuple(stem.shape), dtype=torch.bool)
    patch[0, :, 20:22, 24:26] = True
    try:
        region = trace["stem"].projective_field.of(patch, within=["stages.1", "stages.2"])
    except SelectionError as exc:
        assert exc.fields["code"] == "influence_geometry_unavailable", exc.fields
        return
    targets = [_module_output(trace, "stages.1"), _module_output(trace, "stages.2")]
    assert [f"{entry.key[0]}:{entry.key[1]}" for entry in region] == [op.label for op in targets]
    assert all(0 < entry.num_selected < entry.mask.numel() for entry in region)

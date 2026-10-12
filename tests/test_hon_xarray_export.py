"""U28.2: ``tl.export.xarray`` on real-model traces, the three August legs.

August found the exporter unusable on every real-model leg: (a) xarray in no
extra, (b) a partial trace crashed reading an unsaved placeholder's ``.out``,
(c) a save-everything trace of a buffer-bearing model refused with an untyped
row-count error. Now (b) skips unsaved layers and lists them in
``attrs["skipped_unsaved_layers"]``; (c) refuses typed
(``xarray_buffer_layer_unsupported``) with remedies that run; a leading-size
mismatch refuses typed (``xarray_presentation_mismatch``). Leg (a) is the
``neuro`` extra (pyproject, the integrator's file); these tests need xarray
installed and run the legs on torchvision's real resnet18 class with
constructed weights.
"""

from __future__ import annotations

import pytest
import torch
import torchvision
from torch import nn

import torchlens as tl
from torchlens.errors import InvalidArgumentError

pytest.importorskip("xarray")


def _resnet_and_images() -> tuple[nn.Module, torch.Tensor]:
    """torchvision's resnet18 (constructed weights) and eight small images."""

    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None).eval(), torch.randn(8, 3, 32, 32)


def test_partial_resnet_trace_exports_and_discloses_skipped_layers() -> None:
    """Leg (b): saved sites export exactly; every unsaved layer is disclosed."""

    model, images = _resnet_and_images()
    # August's leg (b) spelling: module names save the module OUTPUTS.
    trace = tl.trace(
        model,
        images,
        capture=tl.options.CaptureOptions(layers_to_save=["layer4", "avgpool"]),
    )
    array = tl.export.xarray(trace)

    skipped = array.attrs["skipped_unsaved_layers"]
    skipped_buffers = array.attrs["skipped_buffer_layers"]
    saved = [op for op in trace.layer_list if op.has_saved_activation and not op.is_buffer]
    buffers = [op for op in trace.layer_list if op.has_saved_activation and op.is_buffer]
    assert skipped and buffers, "the leg needs both unsaved layers and retained buffers"
    assert sorted(skipped_buffers) == sorted(op.layer_label for op in buffers)
    assert len(skipped) + len(buffers) + len(saved) == len(trace.layer_list)
    assert array.sizes["presentation"] == 8
    for op in saved:
        if not isinstance(op.out, torch.Tensor):
            continue
        mask = (array.coords["layer"] == op.layer_label).values
        columns = array.values[:, mask]
        expected = op.out.detach().reshape(8, -1).numpy()
        assert columns.shape == expected.shape
        assert (columns == expected).all()


def test_save_all_resnet_refuses_buffers_typed_and_the_remedies_run() -> None:
    """Leg (c): a saved buffer refuses typed; both remedies it names work."""

    model, images = _resnet_and_images()
    trace = tl.trace(model, images, capture=tl.options.CaptureOptions(layers_to_save="all"))
    with pytest.raises(InvalidArgumentError) as info:
        tl.export.xarray(trace)
    assert info.value.fields["code"] == "xarray_buffer_layer_unsupported"
    assert "tl.bridge.xarray.data_array" in info.value.fields["remedy"]

    selective = tl.trace(
        model, images, capture=tl.options.CaptureOptions(layers_to_save=["avgpool"])
    )
    assert tl.export.xarray(selective).sizes["presentation"] == 8
    site = trace.modules["avgpool"].output_layers[0]
    per_site = tl.bridge.xarray.data_array(trace, site)
    assert per_site.sizes["presentation"] == 8


class _Transposer(nn.Module):
    """Saves an activation whose leading axis is not the batch."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).t().contiguous()


def test_mismatched_leading_size_refuses_typed_and_the_remedy_runs() -> None:
    """A leading-size mismatch names the layer, typed; per-site export works."""

    trace = tl.trace(_Transposer(), torch.randn(3, 4))
    with pytest.raises(InvalidArgumentError) as info:
        tl.export.xarray(trace)
    assert info.value.fields["code"] == "xarray_presentation_mismatch"
    assert info.value.fields["layer"] in str(info.value)

    site = trace.modules["lin"].output_layers[0]
    assert tl.bridge.xarray.data_array(trace, site).sizes["presentation"] == 3

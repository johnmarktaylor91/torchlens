"""U28.2 + the batch-2 buffer ruling: ``tl.export.xarray`` on real-model traces.

August found the exporter unusable on every real-model leg: (a) xarray in no
extra, (b) a partial trace crashed reading an unsaved placeholder's ``.out``,
(c) a save-everything trace of a buffer-bearing model refused. Now (b) skips
unsaved layers and lists them in ``attrs["skipped_unsaved_layers"]``; (c) the
export is an ``xarray.Dataset`` whose ``out`` variable is the NeuroidAssembly
and whose every saved buffer layer is its own module-state variable (own dims,
no presentation axis), on save-all and partial traces alike (buffer ruling,
option c; the former buffer-layer refusal is removed); a leading-size mismatch refuses typed
(``xarray_presentation_mismatch``). Leg (a) is the ``neuro`` extra; these
tests need xarray installed and run the legs on torchvision's real resnet18
class with constructed weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


def _assert_buffers_are_module_state(dataset: Any, trace: Any, model: nn.Module) -> None:
    """Every saved buffer layer is its own variable: own dims, exact values, marked."""

    buffers = [op for op in trace._layer_list if op.has_saved_activation and op.is_buffer]
    assert buffers, "the leg needs retained buffers"
    labels = [op.layer_label for op in buffers]
    assert dataset.attrs["module_state_variables"] == labels
    assert set(dataset.data_vars) == {"out", *labels}
    for op in buffers:
        variable = dataset[op.layer_label]
        expected = model.get_buffer(op.address).detach().numpy()
        assert variable.dims == tuple(f"{op.layer_label}_dim_{i}" for i in range(expected.ndim))
        assert "presentation" not in variable.dims
        assert variable.dtype == expected.dtype
        assert variable.values.tobytes() == expected.tobytes()
        assert variable.values.tobytes() == op.out.detach().numpy().tobytes()
        assert variable.attrs["torchlens_kind"] == "module_state"
        assert variable.attrs["layer_label"] == op.layer_label
        assert variable.attrs["buffer_address"] == op.address
    assert "torchlens_kind" not in dataset["out"].attrs


def test_partial_resnet_trace_exports_buffers_and_discloses_skipped_layers() -> None:
    """Leg (b): saved sites export exactly; unsaved layers are disclosed; retained
    buffers export as module-state variables, the same as on a save-all trace."""

    model, images = _resnet_and_images()
    # August's leg (b) spelling: module names save the module OUTPUTS.
    trace = tl.trace(
        model,
        images,
        capture=tl.options.CaptureOptions(layers_to_save=["layer4", "avgpool"]),
    )
    dataset = tl.export.xarray(trace)
    array = dataset["out"]

    skipped = array.attrs["skipped_unsaved_layers"]
    assert dataset.attrs["skipped_unsaved_layers"] == skipped
    saved = [op for op in trace._layer_list if op.has_saved_activation and not op.is_buffer]
    buffers = [op for op in trace._layer_list if op.has_saved_activation and op.is_buffer]
    assert skipped and buffers, "the leg needs both unsaved layers and retained buffers"
    assert len(skipped) + len(buffers) + len(saved) == len(trace._layer_list)
    assert array.sizes["presentation"] == 8
    assert set(array.coords["layer"].values) == {op.layer_label for op in saved}
    _assert_buffers_are_module_state(dataset, trace, model)
    for op in saved:
        if not isinstance(op.out, torch.Tensor):
            continue
        mask = (array.coords["layer"] == op.layer_label).values
        columns = array.values[:, mask]
        expected = op.out.detach().reshape(8, -1).numpy()
        assert columns.shape == expected.shape
        assert (columns == expected).all()


def test_save_all_batchnorm_trace_exports_everything() -> None:
    """Leg (c): a save-all BatchNorm trace no longer refuses. Activations keep the
    presentation axis; every buffer round-trips bit-exact as module state."""

    model, images = _resnet_and_images()
    trace = tl.trace(model, images, capture=tl.options.CaptureOptions(layers_to_save="all"))
    dataset = tl.export.xarray(trace)
    array = dataset["out"]

    assert array.dims == ("presentation", "neuroid")
    assert array.sizes["presentation"] == 8
    assert array.attrs["skipped_unsaved_layers"] == []
    activations = [
        op
        for op in trace._layer_list
        if not op.is_buffer and isinstance(op.out, torch.Tensor) and op.out.ndim > 0
    ]
    assert {op.layer_label for op in activations} <= set(array.coords["layer"].values)
    for op in activations:
        mask = (array.coords["layer"] == op.layer_label).values
        expected = op.out.detach().reshape(8, -1).numpy()
        assert array.values[:, mask].tobytes() == np.ascontiguousarray(expected).tobytes()
    _assert_buffers_are_module_state(dataset, trace, model)


def test_buffer_layer_refusal_code_is_removed() -> None:
    """The removed buffer-layer refusal code is gone from source and contract."""

    root = Path(tl.__file__).resolve().parent
    code = "xarray_buffer_layer_" + "unsupported"
    assert code not in (root / "export" / "__init__.py").read_text(encoding="utf-8")
    contract = root.parent / "docs" / "reference" / "error_refusal_contract.md"
    if contract.exists():
        assert code not in contract.read_text(encoding="utf-8")


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
    assert "tl.bridge.xarray.data_array" in info.value.fields["remedy"]
    assert info.value.fields["layer"] in str(info.value)

    site = trace.modules["lin"].output_layers[0]
    assert tl.bridge.xarray.data_array(trace, site).sizes["presentation"] == 3

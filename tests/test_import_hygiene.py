"""Import-time hygiene regression tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest
import torch


def test_import_torchlens_does_not_import_torchvision_when_installed() -> None:
    """Bare TorchLens import should not import torchvision even when installed."""

    if importlib.util.find_spec("torchvision") is None:
        pytest.skip("torchvision is not installed")

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torchlens, sys; assert 'torchvision' not in sys.modules",
        ],
        check=True,
    )


def test_import_torchlens_does_not_import_heavy_torch_submodules() -> None:
    """Bare TorchLens import should not force deferred torch internals."""

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torchlens, sys; "
                "assert 'torch._dynamo' not in sys.modules; "
                "assert 'torch._dynamo.eval_frame' not in sys.modules"
            ),
        ],
        check=True,
    )


@pytest.mark.heavy
@pytest.mark.optional
def test_torchvision_model_trace_still_covers_torchvision_ops() -> None:
    """First wrapper use should still include torchvision ops and trace torchvision models."""

    torchvision_models = pytest.importorskip("torchvision.models")

    import torchlens as tl
    from torchlens.constants import TORCHVISION_FUNCS, get_orig_torch_funcs

    wrapped_targets = set(get_orig_torch_funcs())
    assert set(TORCHVISION_FUNCS).issubset(wrapped_targets)

    model = torchvision_models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 32, 32)
    trace = tl.trace(model, x, save=tl.func("conv2d"))

    assert trace.num_layers > 0
    assert any(op.func_name == "conv2d" for op in trace.ops)


@pytest.mark.optional
def test_torchvision_cpp_ops_record_real_func_name() -> None:
    """Torchvision PyCapsule ops keep their public op name in TorchLens metadata."""

    torchvision_ops = pytest.importorskip("torchvision.ops")

    import torchlens as tl

    class NmsModel(torch.nn.Module):
        """Tiny module that calls torchvision NMS."""

        def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            """Run torchvision NMS on boxes and scores."""

            boxes, scores = inputs
            return torchvision_ops.nms(boxes, scores, 0.5)

    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1], [3.0, 3.0, 4.0, 4.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    trace = tl.trace(NmsModel(), (boxes, scores), layers_to_save="all")

    assert any(op.func_name == "nms" for op in trace.ops)
    assert any(op.has_saved_activation for op in trace.ops if op.func_name == "nms")

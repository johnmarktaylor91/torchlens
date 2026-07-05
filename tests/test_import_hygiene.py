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

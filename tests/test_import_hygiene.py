"""Import-time hygiene regression tests."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_LAZY_MODULE_CASES = (
    ("fastlog", "record"),
    ("intervention", "func"),
    ("user_funcs", "trace"),
    ("data_classes", "Buffer"),
)


def _run_import_script(script: str) -> None:
    """Run an import assertion against this checkout in a fresh interpreter.

    Parameters
    ----------
    script:
        Python source containing assertions for one import pattern.
    """

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root)
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


@pytest.mark.parametrize(("module_name", "member_name"), _LAZY_MODULE_CASES)
def test_lazified_module_top_level_access_patterns(module_name: str, member_name: str) -> None:
    """Top-level attributes and from-imports should load each deferred module."""

    _run_import_script(
        "import torchlens as tl; "
        f"module = tl.{module_name}; "
        f"from torchlens import {module_name} as imported_module; "
        "assert module is imported_module; "
        f"assert hasattr(module, {member_name!r})"
    )


@pytest.mark.parametrize(("module_name", "member_name"), _LAZY_MODULE_CASES)
def test_lazified_module_direct_import_patterns(module_name: str, member_name: str) -> None:
    """Direct submodule and member imports should work from a bare package import."""

    _run_import_script(
        "import importlib; "
        "import torchlens; "
        f"module = importlib.import_module('torchlens.{module_name}'); "
        f"from torchlens.{module_name} import {member_name}; "
        f"assert getattr(module, {member_name!r}) is {member_name}"
    )


def test_bare_import_defers_lazified_feature_modules() -> None:
    """Bare imports must not eagerly initialize the deferred feature islands."""

    _run_import_script(
        "import sys; import torchlens; "
        "blocked = ('torchlens.fastlog', 'torchlens.intervention', "
        "'torchlens.user_funcs', 'torchlens.data_classes'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for prefix in blocked for name in sys.modules)"
    )


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

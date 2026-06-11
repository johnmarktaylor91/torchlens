"""Tests for torch.func transform boundary capture."""

import pytest
import torch
import torch.nn as nn
from typing import cast

import torchlens as tl
from torchlens import Recording


_HAS_TORCH_FUNC = hasattr(torch, "func")


class VmapMaskModel(nn.Module):
    """Build a mask with vmap and consume it in a downstream operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` with a vmapped row mask applied.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Masked tensor.
        """

        mask = torch.vmap(lambda row: row > 0)(x)
        return x.masked_fill(mask, 0.0)


class GradInnerModel(nn.Module):
    """Compute gradients through ``torch.func.grad`` inside forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a transform-computed gradient plus a normal eager op.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Gradient result shifted by one.
        """

        grad_fn = torch.func.grad(lambda z: (z * z).sum())
        return grad_fn(x) + 1.0


class GradOverModuleModel(nn.Module):
    """Reproduce grad-over-module capture without module-hook crashes."""

    def __init__(self) -> None:
        """Initialize the inner module."""

        super().__init__()
        self.inner = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a gradient computed through an inner module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transform-computed gradient.
        """

        def loss_fn(z: torch.Tensor) -> torch.Tensor:
            """Return a scalar inner-module loss.

            Parameters
            ----------
            z:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Scalar loss.
            """

            return self.inner(z).sum()

        return torch.func.grad(loss_fn)(x)


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_vmap_boundary_node_has_clean_parent_edge() -> None:
    """Instrumented vmap emits one boundary node consumed by downstream ops."""

    x = torch.randn(3, 4)
    log = tl.trace(VmapMaskModel().eval(), x, layers_to_save="all")
    vmap_ops = [op for op in log.ops if op.type == "vmap"]

    assert len(vmap_ops) == 1
    assert vmap_ops[0].label == "vmap_1_1:1"
    assert vmap_ops[0].parents == ["input_1"]
    assert vmap_ops[0].is_transform is True
    assert vmap_ops[0].transform_kind == "vmap"
    assert vmap_ops[0].transform_chain == ("vmap",)
    assert vmap_ops[0].transform_config["in_dims"] == 0
    assert vmap_ops[0].transform_fn_name == "<lambda>"
    assert log.transforms == (vmap_ops[0],)
    assert any(op.type == "maskedfill" and "vmap_1_1" in op.parents for op in log.ops)


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_grad_boundary_node_has_clean_parent_edge() -> None:
    """Instrumented grad emits one boundary node with the input as parent."""

    x = torch.randn(4)
    log = tl.trace(GradInnerModel().eval(), x, layers_to_save="all")
    grad_ops = [op for op in log.ops if op.type == "grad"]

    assert len(grad_ops) == 1
    assert grad_ops[0].parents == ["input_1"]
    assert grad_ops[0].is_transform is True
    assert grad_ops[0].transform_kind == "grad"
    assert grad_ops[0].transform_chain == ("grad",)
    assert grad_ops[0].transform_config["argnums"] == 0
    assert any(op.type == "add" and "grad_1_1" in op.parents for op in log.ops)


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_grad_over_module_boundary_does_not_crash() -> None:
    """Paused inner execution prevents module-entry tracking crashes."""

    x = torch.randn(4)
    log = tl.trace(GradOverModuleModel().eval(), x, layers_to_save="all")

    assert [op for op in log.ops if op.type == "grad"]


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_func_transform_selector_matches_trace_and_record() -> None:
    """Transform predicates match both Trace and Recording capture paths."""

    x = torch.randn(3, 4)
    log = tl.trace(VmapMaskModel().eval(), x, save=tl.func_transform("vmap"))
    recording = cast(Recording, tl.record(VmapMaskModel().eval(), x, save=tl.func_transform()))

    assert [op.label for op in log.transforms] == ["vmap_1_1:1"]
    assert len(recording.records) == 1
    assert recording.records[0].ctx.is_transform is True
    assert recording.records[0].ctx.transform_kind == "vmap"

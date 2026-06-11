"""Tests for torch.func transform boundary capture."""

from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import _state
from torchlens import Recording, Trace
from torchlens.backends.torch.wrappers import wrap_torch


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


class RawGradOverModuleModel(nn.Module):
    """Use an explicitly uninstrumented raw grad builder over a module."""

    def __init__(self) -> None:
        """Initialize the raw builder and inner module."""

        super().__init__()
        wrap_torch()
        self.raw_grad = _state._decorated_to_orig[id(torch.func.grad)]
        self.inner = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a raw-transform gradient through an inner module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Raw-transform-computed gradient.
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

        return self.raw_grad(loss_fn)(x)


class AutogradJacobianModel(nn.Module):
    """Use legacy ``torch.autograd.functional.jacobian`` in forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a jacobian reduced to the input shape.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reduced jacobian.
        """

        jac = torch.autograd.functional.jacobian(lambda z: z * z, x)
        return jac.diagonal().sum()


class AutogradHessianModel(nn.Module):
    """Use legacy ``torch.autograd.functional.hessian`` in forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a hessian reduction.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reduced hessian.
        """

        hessian = torch.autograd.functional.hessian(lambda z: (z * z).sum(), x)
        return hessian.sum()


class FunctionalCallBasicModel(nn.Module):
    """Use ``torch.func.functional_call`` with substituted parameters."""

    def __init__(self) -> None:
        """Initialize the inner module."""

        super().__init__()
        self.inner = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a functional_call with tensor substitutions.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Inner module output.
        """

        params = {name: value + 1.0 for name, value in self.inner.named_parameters()}
        return torch.func.functional_call(self.inner, params, (x,))


class FunctionalCallBufferModel(nn.Module):
    """Use ``functional_call`` with a substituted buffer."""

    def __init__(self) -> None:
        """Initialize the inner module with a buffer."""

        super().__init__()
        self.inner = nn.Module()
        self.inner.register_buffer("offset", torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a functional_call with a buffer substitution.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted tensor.
        """

        offset = cast(torch.Tensor, self.inner.offset)
        buffers = {"offset": offset + 2.0}

        def forward(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
            """Apply the module buffer.

            Parameters
            ----------
            module:
                Module carrying the buffer.
            value:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Shifted tensor.
            """

            return value + cast(torch.Tensor, module.offset)

        self.inner.forward = forward.__get__(self.inner, nn.Module)  # type: ignore[method-assign]
        return torch.func.functional_call(self.inner, buffers, (x,))


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
def test_raw_grad_over_module_wrapper_tensor_leak_does_not_crash() -> None:
    """Hardening guards tolerate wrapper tensors from uninstrumented transforms."""

    x = torch.randn(4)
    log = tl.trace(RawGradOverModuleModel().eval(), x, layers_to_save="all")

    assert log.output_layers


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


@pytest.mark.parametrize(
    ("model", "op_type", "kind"),
    [
        (AutogradJacobianModel(), "autogradjacobian", "autograd.jacobian"),
        (AutogradHessianModel(), "autogradhessian", "autograd.hessian"),
    ],
)
def test_autograd_functional_direct_call_boundary(
    model: nn.Module,
    op_type: str,
    kind: str,
) -> None:
    """Legacy autograd.functional transforms emit direct-call boundary nodes."""

    x = torch.randn(3)
    log = tl.trace(model.eval(), x, layers_to_save="all")
    transform_ops = [op for op in log.ops if op.type == op_type]

    assert len(transform_ops) == 1
    assert transform_ops[0].parents == ["input_1"]
    assert transform_ops[0].is_transform is True
    assert transform_ops[0].transform_kind == kind


def test_functional_call_substituted_params_are_tensor_parents() -> None:
    """Substituted params are graph parents, not registered Param claims."""

    log = tl.trace(FunctionalCallBasicModel().eval(), torch.randn(3), layers_to_save="all")
    linear = next(op for op in log.ops if op.type == "linear")

    assert linear.module == "inner:1"
    assert len(linear.parents) == 3
    assert linear.parents[0] == "input_1"
    assert linear.parent_params == []


def test_functional_call_substituted_buffers_are_tensor_parents() -> None:
    """Substituted buffers are graph parents under functional_call."""

    log = tl.trace(FunctionalCallBufferModel().eval(), torch.randn(3), layers_to_save="all")
    add_ops = [op for op in log.ops if op.type == "add"]

    assert any("input_1" in op.parents and len(op.parents) == 2 for op in add_ops)


def test_transform_metadata_pandas_and_tlspec_round_trip(tmp_path: Path) -> None:
    """Transform metadata is tabular and portable."""

    x = torch.randn(3, 4)
    log = tl.trace(VmapMaskModel().eval(), x, layers_to_save="all")
    dataframe = log.to_pandas()
    path = tmp_path / "vmap.tlspec"

    tl.save(log, path)
    loaded = cast(Trace, tl.load(path))

    assert bool(dataframe.loc[dataframe["type"] == "vmap", "is_transform"].iloc[0])
    assert dataframe.loc[dataframe["type"] == "vmap", "transform_kind"].iloc[0] == "vmap"
    assert loaded.transforms[0].transform_kind == "vmap"
    assert loaded.transforms[0].transform_config["in_dims"] == 0


def test_transform_selector_intervention_no_crash() -> None:
    """Transform selector can be used as an intervention predicate."""

    x = torch.randn(3, 4)
    log = tl.trace(
        VmapMaskModel().eval(),
        x,
        intervene=tl.when(tl.func_transform("vmap"), tl.zero_ablate()),
    )

    assert [op.transform_kind for op in log.transforms] == ["vmap"]

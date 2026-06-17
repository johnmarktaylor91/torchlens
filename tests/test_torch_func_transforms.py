"""Tests for torch.func transform boundary capture."""

import warnings
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import _state
from torchlens import Recording, Trace
from torchlens.validation import validate_forward_pass
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


class ForeignArgTensorModel(nn.Module):
    """Consume a tensor held outside the module as an op argument."""

    def __init__(self, foreign: torch.Tensor) -> None:
        """Store a foreign tensor supplied by the test."""

        super().__init__()
        self._foreign = foreign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the foreign tensor to the input.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted tensor.
        """

        return x + self._foreign


class ModuleAttrTensorModel(nn.Module):
    """Consume a plain tensor module attribute as an op argument."""

    def __init__(self) -> None:
        """Initialize the module tensor attribute."""

        super().__init__()
        self.offset = torch.ones(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the module tensor attribute to the input.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted tensor.
        """

        return x + self.offset


class PreBuiltTransformModel(nn.Module):
    """Use a transform callable built before ``forward`` runs."""

    def __init__(self) -> None:
        """Build the transform callable after explicit wrapping."""

        super().__init__()
        wrap_torch()
        self.vmapped = torch.vmap(lambda row: row * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Invoke the prebuilt transform callable.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """

        return self.vmapped(x)


class RawPreBuiltTransformModel(nn.Module):
    """Use a raw prebuilt transform callable to exercise retained warnings."""

    def __init__(self) -> None:
        """Build a raw transform callable via ``_decorated_to_orig``."""

        super().__init__()
        wrap_torch()
        raw_vmap = _state._decorated_to_orig[id(torch.vmap)]
        self.vmapped = raw_vmap(lambda row: row * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Invoke the raw prebuilt transform callable.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """

        return self.vmapped(x)


class RaisingInnerFnModel(nn.Module):
    """Raise from inside a transform callable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise while a transform boundary is active.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This method always raises.
        """

        def raise_inner(row: torch.Tensor) -> torch.Tensor:
            """Raise from the transform body."""

            raise RuntimeError("inner transform failure")

        return torch.vmap(raise_inner)(x)


class HFStyleMaskModel(nn.Module):
    """Reproduce runtime attribute-style vmap mask construction."""

    def _mask_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Build a row mask through attribute lookup.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Boolean mask.
        """

        vmap = torch.vmap
        return vmap(lambda row: row > 0)(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the runtime-built mask.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Masked tensor.
        """

        return x.masked_fill(self._mask_rows(x), 0.0)


class TwoVmapBoundaryModel(nn.Module):
    """Invoke two separate vmap boundaries in one forward pass."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sum of two separately vmapped computations.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Combined transform outputs.
        """

        first = torch.vmap(lambda row: row * 2.0)(x)
        second = torch.vmap(lambda row: row + 1.0)(x)
        return first + second


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
    assert vmap_ops[0].transform_fn_source is not None
    assert log.transforms == (vmap_ops[0],)
    assert any(op.type == "maskedfill" and "vmap_1_1" in op.parents for op in log.ops)


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_transform_boundary_warning_fires_for_each_collapsed_region() -> None:
    """Every collapsed transform boundary should warn, even within one trace."""

    x = torch.randn(3, 4)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        log = tl.trace(TwoVmapBoundaryModel().eval(), x, layers_to_save="all")

    boundary_warnings = [
        record
        for record in records
        if issubclass(record.category, UserWarning)
        and "captured a vmap transform as a boundary op" in str(record.message)
    ]
    assert len(boundary_warnings) == 2
    assert [op.transform_kind for op in log.transforms] == ["vmap", "vmap"]


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


def test_provenance_warning_foreign_tensor_contract() -> None:
    """Foreign arg tensors warn; module tensor attributes do not."""

    x = torch.randn(3)
    with pytest.warns(UserWarning, match="no graph/source provenance") as caught:
        foreign_log = tl.trace(ForeignArgTensorModel(torch.ones(3)).eval(), x)

    assert len(caught) == 1
    assert "arg1" in str(caught[0].message)
    add_op = next(op for op in foreign_log.ops if op.type == "add")
    assert add_op.unattributed_tensor_args == ("arg1",)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        module_log = tl.trace(ModuleAttrTensorModel().eval(), x)

    provenance_warnings = [
        record for record in records if "no graph/source provenance" in str(record.message)
    ]
    assert provenance_warnings == []
    module_add = next(op for op in module_log.ops if op.type == "add")
    assert module_add.unattributed_tensor_args == ()


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_prebuilt_transform_wrap_order_and_raw_warning_contract() -> None:
    """Prebuilt decorated transforms capture; raw prebuilt transforms retain warning."""

    x = torch.randn(3, 4)
    decorated_log = tl.trace(PreBuiltTransformModel().eval(), x, layers_to_save="all")

    assert [op.transform_kind for op in decorated_log.transforms] == ["vmap"]
    assert decorated_log.transforms[0].parents == ["input_1"]

    with pytest.warns(UserWarning, match="functorch"):
        raw_log = tl.trace(RawPreBuiltTransformModel().eval(), x, layers_to_save="all")

    assert raw_log.transforms == ()
    assert torch.allclose(RawPreBuiltTransformModel().eval()(x), x * 2.0)


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_raising_inner_transform_cleans_up_logging_state() -> None:
    """Exceptions from transform bodies leave logging state sane."""

    with pytest.raises(RuntimeError, match="inner transform failure"):
        tl.trace(RaisingInnerFnModel().eval(), torch.randn(3, 4))

    assert _state._logging_enabled is False
    log = tl.trace(VmapMaskModel().eval(), torch.randn(3, 4))
    assert [op.transform_kind for op in log.transforms] == ["vmap"]


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_hf_style_runtime_vmap_mask_regression() -> None:
    """Attribute-style runtime vmap lookup captures a clean boundary node."""

    x = torch.randn(3, 4)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        log = tl.trace(HFStyleMaskModel().eval(), x, layers_to_save="all")

    assert [op.transform_kind for op in log.transforms] == ["vmap"]
    assert log.transforms[0].parents == ["input_1"]
    assert any("functorch" in str(record.message).lower() for record in records)
    assert not any("no graph/source provenance" in str(record.message) for record in records)


@pytest.mark.skipif(not _HAS_TORCH_FUNC, reason="torch.func not available")
def test_transform_matrix_completion_rows(tmp_path: Path) -> None:
    """Cross-cutting matrix rows work for vmap transform boundary capture."""

    x = torch.randn(3, 4)
    model = VmapMaskModel().eval()
    expected = model(x)
    assert torch.allclose(model(x), expected)

    log = tl.trace(model, x, layers_to_save=["vmap_1_1"])
    transform = log.transforms[0]
    assert transform.has_saved_activation is True
    assert isinstance(transform.out, torch.Tensor)
    assert torch.equal(transform.out, x > 0)
    assert validate_forward_pass(VmapMaskModel().eval(), x) is True

    predicate_log = tl.trace(VmapMaskModel().eval(), x, save=tl.func_transform("vmap"))
    assert [op.label for op in predicate_log.transforms] == ["vmap_1_1:1"]

    recording = cast(Recording, tl.record(VmapMaskModel().eval(), x, save=tl.func_transform()))
    assert recording.records[0].ctx.transform_kind == "vmap"

    path = tmp_path / "matrix-vmap.tlspec"
    tl.save(predicate_log, path)
    loaded = cast(Trace, tl.load(path))
    assert loaded.transforms[0].transform_kind == "vmap"

    intervened = tl.trace(
        VmapMaskModel().eval(),
        x,
        layers_to_save="all",
        intervene=tl.when(tl.func_transform("vmap"), tl.zero_ablate()),
    )
    assert intervened.transforms[0].intervention_replaced is True

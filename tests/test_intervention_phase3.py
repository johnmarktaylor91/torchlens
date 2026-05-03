"""Smoke tests for intervention Phase 3 hook contracts and helpers."""

from __future__ import annotations

from types import MappingProxyType

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import (
    HookSignatureError,
    HookSiteCoverageError,
    HookValueError,
    SpliceModuleDtypeError,
)
from torchlens.intervention.hooks import HookContext, make_hook_context, normalize_hook_plan
from torchlens.intervention.runtime import _execute_hook
from torchlens.intervention.types import HelperSpec


def _good_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
    """Return the out unchanged.

    Parameters
    ----------
    out:
        Activation tensor.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Input out.
    """

    return out


def _context() -> HookContext:
    """Return a representative hook context.

    Returns
    -------
    HookContext
        Context with mapping-proxy layer metadata.
    """

    return make_hook_context(
        name="test",
        layer_log={
            "layer_label": "relu_1_1",
            "layer_type": "relu",
            "shape": (2, 3),
            "dtype": torch.float32,
            "tensor_device": torch.device("cpu"),
            "address": "block",
            "call_index": 1,
        },
    )


@pytest.mark.smoke
def test_phase3_helpers_import_and_return_specs() -> None:
    """All Phase 3 helpers are importable from the top-level namespace."""

    helper_specs = [
        tl.zero_ablate(),
        tl.mean_ablate(),
        tl.resample_ablate(),
        tl.steer(torch.ones(3), feature_axis=-1),
        tl.scale(2.0),
        tl.clamp(min=-1.0, max=1.0),
        tl.noise(0.1, seed=1),
        tl.project_onto(torch.ones(3), feature_axis=-1),
        tl.project_off(torch.ones(3), feature_axis=-1),
        tl.swap_with(torch.ones(2, 3)),
        tl.splice_module(torch.nn.Identity()),
        tl.bwd_hook(_good_hook),
        tl.grad_zero(),
        tl.grad_scale(0.5),
    ]

    assert all(isinstance(spec, HelperSpec) for spec in helper_specs)
    assert helper_specs[0].name == "zero_ablate"
    assert helper_specs[-1].kind == "backward"
    assert dict(helper_specs[-1].metadata)["live_rerun_only"] is True


@pytest.mark.smoke
def test_hook_context_uses_mapping_proxy_and_frozen_fields() -> None:
    """HookContext exposes metadata as snapshots rather than live logs."""

    context = _context()

    assert isinstance(context.layer_log, MappingProxyType)
    assert context.layer_log["layer_label"] == "relu_1_1"
    with pytest.raises(TypeError):
        context.layer_log["layer_label"] = "mutated"  # type: ignore[index]
    with pytest.raises(AttributeError):
        context.name = "mutated"  # type: ignore[misc]


@pytest.mark.smoke
def test_normalizer_accepts_supported_shapes_in_order() -> None:
    """Hook normalization covers callable, helper, mapping, list, and pair shapes."""

    default_entries = normalize_hook_plan(_good_hook, default_site_target=tl.label("x"))
    helper_entries = normalize_hook_plan(tl.zero_ablate(), default_site_target=tl.label("x"))
    mapping_entries = normalize_hook_plan({tl.label("x"): _good_hook, tl.func("relu"): tl.scale(2)})
    list_entries = normalize_hook_plan([(tl.label("x"), _good_hook), (tl.label("y"), tl.scale(2))])
    pair_entries = normalize_hook_plan((tl.label("x"), _good_hook))

    assert len(default_entries) == 1
    assert helper_entries[0].helper_spec is not None
    assert [entry.metadata["attach_order"] for entry in mapping_entries] == [0, 1]
    assert [entry.metadata["attach_order"] for entry in list_entries] == [0, 1]
    assert pair_entries[0].site_target == tl.label("x")


@pytest.mark.smoke
def test_normalizer_rejects_missing_site_and_bad_signature() -> None:
    """Normalizer fails closed on ambiguous bare hooks and bad signatures."""

    def bad_hook(out: torch.Tensor) -> torch.Tensor:
        """Return out with a deliberately invalid signature."""

        return out

    with pytest.raises(HookSiteCoverageError):
        normalize_hook_plan(_good_hook)
    with pytest.raises(HookSignatureError, match="hook"):
        normalize_hook_plan((tl.label("x"), bad_hook))


@pytest.mark.smoke
def test_execute_hook_rejects_none_type_shape_dtype_and_device() -> None:
    """Hook execution validates return values by default."""

    out = torch.ones(2, 3)
    context = _context()

    def none_hook(out: torch.Tensor, *, hook: HookContext) -> None:
        """Return None, which Phase 3 rejects."""

        return None

    def list_hook(out: torch.Tensor, *, hook: HookContext) -> list[torch.Tensor]:
        """Return the wrong type."""

        return [out]

    def shape_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return the wrong shape."""

        return torch.ones(3, 2)

    def dtype_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return the wrong dtype."""

        return out.to(torch.float64)

    def device_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return a tensor on the wrong device."""

        return torch.empty(out.shape, dtype=out.dtype, device="meta")

    with pytest.raises(HookValueError, match="None"):
        _execute_hook(none_hook, out, context)
    with pytest.raises(HookValueError, match="list"):
        _execute_hook(list_hook, out, context)
    with pytest.raises(HookValueError, match="shape"):
        _execute_hook(shape_hook, out, context)
    with pytest.raises(HookValueError, match="dtype"):
        _execute_hook(dtype_hook, out, context)
    with pytest.raises(HookValueError, match="device"):
        _execute_hook(device_hook, out, context)


@pytest.mark.smoke
def test_force_shape_change_allows_metadata_changes() -> None:
    """The escape hatch byops dtype/device/shape checks."""

    out = torch.ones(2, 3)

    def shape_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return a shape-changed tensor."""

        return torch.ones(3, 2, dtype=torch.float64)

    result = _execute_hook(shape_hook, out, _context(), force_shape_change=True)

    assert result.shape == (3, 2)
    assert result.dtype == torch.float64


@pytest.mark.smoke
def test_seeded_noise_is_deterministic_and_unseeded_records_note() -> None:
    """Stochastic helper RNG follows the Phase 3 policy."""

    out = torch.zeros(2, 3)
    context = _context()

    seeded_a = tl.noise(1.0, seed=123)()
    seeded_b = tl.noise(1.0, seed=123)()
    unseeded = tl.noise(1.0)()

    assert torch.equal(seeded_a(out, hook=context), seeded_b(out, hook=context))
    _ = unseeded(out, hook=context)
    assert any("noise used unseeded" in note for note in context.run_ctx["operation_history_notes"])


@pytest.mark.smoke
def test_splice_module_dtype_error_is_specific() -> None:
    """splice_module reports dtype mismatches with its specific error type."""

    class _DoubleModule(torch.nn.Module):
        """Module that deliberately changes dtype."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a float64 tensor.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Converted tensor.
            """

            return x.to(torch.float64)

    hook = tl.splice_module(_DoubleModule())()

    with pytest.raises(SpliceModuleDtypeError):
        hook(torch.ones(2, 3), hook=_context())

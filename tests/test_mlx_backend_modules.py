"""MLX object-module hierarchy tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.validation import MetadataInvariantError  # noqa: E402
from torchlens.validation import check_metadata_invariants  # noqa: E402


class MLXParameterless(nn.Module):
    """Parameterless MLX module used by hierarchy tests."""

    def __call__(self, x: mx.array) -> mx.array:
        """Apply a parameterless MLX op.

        Parameters
        ----------
        x
            Input array.

        Returns
        -------
        mx.array
            Rectified output.
        """

        return nn.relu(x)


class MLXNested(nn.Module):
    """Nested MLX module with a parameterless child and a linear head."""

    def __init__(self) -> None:
        """Initialize nested children."""

        super().__init__()
        self.encoder = MLXParameterless()
        self.head = nn.Linear(4, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run nested forward.

        Parameters
        ----------
        x
            Input array.

        Returns
        -------
        mx.array
            Model output.
        """

        return self.head(self.encoder(x))


class MLXShared(nn.Module):
    """MLX module with one shared child registered under two aliases."""

    def __init__(self) -> None:
        """Initialize shared aliases."""

        super().__init__()
        shared = nn.Linear(4, 4)
        self.left = shared
        self.right = shared

    def __call__(self, x: mx.array) -> mx.array:
        """Run both shared aliases.

        Parameters
        ----------
        x
            Input array.

        Returns
        -------
        mx.array
            Combined output.
        """

        return mx.add(self.left(x), self.right(x))


class MLXListContainer(nn.Module):
    """MLX module with submodules inside a Python list."""

    def __init__(self) -> None:
        """Initialize list-contained children."""

        super().__init__()
        self.layers = [nn.Linear(4, 4), MLXParameterless(), nn.Linear(4, 2)]

    def __call__(self, x: mx.array) -> mx.array:
        """Run list-contained modules sequentially.

        Parameters
        ----------
        x
            Input array.

        Returns
        -------
        mx.array
            Model output.
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MLXSimpleMLP(nn.Module):
    """Simple MLX MLP used for surface tests."""

    def __init__(self) -> None:
        """Initialize linear children."""

        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 4)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP.

        Parameters
        ----------
        x
            Input array.

        Returns
        -------
        mx.array
            Model output.
        """

        hidden = self.l1(x)
        hidden = nn.relu(hidden)
        return self.l2(hidden)


def _input() -> mx.array:
    """Return a deterministic MLX input.

    Returns
    -------
    mx.array
        Input array.
    """

    return mx.ones((2, 4))


@pytest.mark.optional
def test_mlx_nested_modules_preserve_object_module_attribution() -> None:
    """Nested MLX modules should populate module logs and selectors."""

    trace = tl.trace(MLXNested(), _input(), backend="mlx")

    assert trace.module_identity_mode == "object_module"
    assert set(module.address for module in trace.modules) == {"self", "encoder", "head"}
    assert set(trace.modules["self"].call_children) == {"encoder", "head"}
    assert {"encoder:1", "head:1"} <= set(trace.module_calls.keys())
    encoder_labels = trace.resolve_sites(tl.in_module("encoder"), max_fanout=16).labels()
    head_labels = trace.resolve_sites(tl.in_module("head"), max_fanout=16).labels()
    assert encoder_labels
    assert head_labels
    assert all("encoder:1" in trace[label].modules for label in encoder_labels)
    assert all("head:1" in trace[label].modules for label in head_labels)
    assert trace.modules["head"].params
    assert check_metadata_invariants(trace) is True


@pytest.mark.optional
def test_mlx_shared_submodule_aliases_use_primary_address() -> None:
    """Shared MLX modules should expose aliases and repeated call labels."""

    trace = tl.trace(MLXShared(), _input(), backend="mlx")

    shared = trace.modules["left"]
    assert trace.modules["right"] is shared
    assert shared.address == "left"
    assert shared.all_addresses == ["left", "right"]
    assert shared.num_calls == 2
    assert shared.call_labels == ["left:1", "left:2"]
    left_labels = trace.resolve_sites(tl.in_module("left"), max_fanout=32).labels()
    assert any("left:1" in trace[label].modules for label in left_labels)
    assert any("left:2" in trace[label].modules for label in left_labels)
    assert {tuple(param.all_module_addresses) for param in shared.params} == {("left", "right")}
    assert check_metadata_invariants(trace) is True


@pytest.mark.optional
def test_mlx_list_container_and_parameterless_modules_are_discovered() -> None:
    """MLX named_modules traversal should find list and parameterless modules."""

    trace = tl.trace(MLXListContainer(), _input(), backend="mlx")

    assert {"layers.0", "layers.1", "layers.2"} <= {module.address for module in trace.modules}
    assert trace.resolve_sites(tl.in_module("layers.1"), max_fanout=16).labels()
    assert trace.modules["layers.0"].params
    assert trace.modules["layers.2"].params
    assert check_metadata_invariants(trace) is True

    parameterless = tl.trace(MLXParameterless(), _input(), backend="mlx")
    assert [module.address for module in parameterless.modules] == ["self"]
    assert parameterless.resolve_sites(tl.in_module("self"), max_fanout=16).labels()
    assert check_metadata_invariants(parameterless) is True


@pytest.mark.optional
def test_mlx_object_module_public_surface_matrix(tmp_path: Path) -> None:
    """Assert public surfaces on a real MLX object-module trace."""

    trace = tl.trace(MLXSimpleMLP(), _input(), backend="mlx")

    assert trace.module_identity_mode == "object_module"
    assert trace.modules["l1"].params
    assert trace.module_calls["l1:1"].forward_kwargs == {}
    assert isinstance(trace.summary(), str)
    assert len(trace.to_pandas()) == len(trace.layer_list)
    assert len(trace.modules.to_pandas()) == len(trace.modules)
    assert len(trace.module_calls["l1:1"].to_pandas()) == 1
    dot = trace.draw(
        vis_outpath=str(tmp_path / "mlx_object_graph"),
        vis_save_only=True,
        vis_fileformat="dot",
    )
    assert isinstance(dot, str)

    bundle_path = tmp_path / "mlx_object_portable.tlspec"
    expected = np.asarray(trace[trace.output_layers[0]].out)
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)
    loaded_out = loaded[loaded.output_layers[0]].out
    assert loaded.backend == "mlx"
    assert loaded.module_identity_mode == "object_module"
    assert isinstance(loaded_out, mx.array)
    np.testing.assert_allclose(np.asarray(loaded_out), expected)


@pytest.mark.optional
def test_mlx_module_selectors_filter_static_save() -> None:
    """MLX save selectors should allow module and in_module after hierarchy support."""

    module_trace = tl.trace(MLXSimpleMLP(), _input(), backend="mlx", save=tl.module("l1"))
    in_module_trace = tl.trace(MLXSimpleMLP(), _input(), backend="mlx", save=tl.in_module("l1"))

    assert list(module_trace.saved_ops.keys()) == ["linear_1_2_raw:1"]
    assert list(in_module_trace.saved_ops.keys()) == ["linear_1_2_raw:1"]


@pytest.mark.optional
def test_mlx_function_root_mode_remains_available() -> None:
    """Explicit function_root should preserve root-only module mode."""

    trace = tl.trace(
        MLXSimpleMLP(),
        _input(),
        backend="mlx",
        module_identity_mode="function_root",
    )

    assert trace.module_identity_mode == "function_root"
    assert [module.address for module in trace.modules] == ["self"]
    assert trace.param_source == "none"
    assert check_metadata_invariants(trace) is True


@pytest.mark.optional
def test_mlx_object_module_corruption_invariant_fails() -> None:
    """Dropping MLX object-module attribution should fail metadata invariants."""

    trace = tl.trace(MLXSimpleMLP(), _input(), backend="mlx")
    victim = next(layer for layer in trace.layer_list if layer.module and not layer.is_input)
    victim.module = None
    victim.modules = []
    victim.module_call_stack = []
    victim.output_of_modules = []
    victim.output_of_module_calls = []
    victim.atomic_module_call = None

    with pytest.raises(MetadataInvariantError, match="module_attribution"):
        check_metadata_invariants(trace)

"""tinygrad object-module backend tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import torchlens as tl
from torchlens.backends import BackendPayloadUnsupportedError, BackendUnsupportedError
from torchlens.validation import check_metadata_invariants
from torchlens.validation.invariants import MetadataInvariantError

tinygrad = pytest.importorskip("tinygrad")
Tensor = pytest.importorskip("tinygrad").Tensor
nn = pytest.importorskip("tinygrad.nn")


pytestmark = pytest.mark.backend_tinygrad


class TinyLinearModel:
    """Simple tinygrad object model with one Linear child."""

    def __init__(self) -> None:
        """Initialize the model."""

        self.fc = nn.Linear(3, 4)
        self.fc.weight = Tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
        self.fc.bias = Tensor([0.0, 0.0, 0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the model.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.fc(x).relu()


class TinyBlock:
    """Nested tinygrad object-model block."""

    def __init__(self) -> None:
        """Initialize the block."""

        self.proj = nn.Linear(3, 4)
        self.proj.weight = Tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
        self.proj.bias = Tensor([0.0, 0.0, 0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the block.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.proj(x).relu()


class TinyNestedModel:
    """Nested tinygrad model with a compound encoder and leaf head."""

    def __init__(self) -> None:
        """Initialize the model."""

        self.encoder = TinyBlock()
        self.head = nn.Linear(4, 2)
        self.head.weight = Tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        self.head.bias = Tensor([0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the model.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.head(self.encoder(x))


class TinySharedModel:
    """tinygrad model reusing one Linear instance under two addresses."""

    def __init__(self) -> None:
        """Initialize the model."""

        shared = nn.Linear(3, 3)
        shared.weight = Tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        shared.bias = Tensor([0.0, 0.0, 0.0])
        self.left = shared
        self.right = shared

    def __call__(self, x: Any) -> Any:
        """Run the shared module twice with distinct UOps.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.left(x) + self.right(x + 1.0)


class TinyReusedTensorModel:
    """tinygrad model that reuses an already-constructed UOp."""

    def __init__(self) -> None:
        """Initialize the model."""

        self.fc = nn.Linear(3, 3)
        self.fc.weight = Tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        self.fc.bias = Tensor([0.0, 0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the model.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        y = self.fc(x)
        return y + y


def test_tinygrad_simple_linear_uses_object_module_hierarchy() -> None:
    """Simple tinygrad Linear objects should produce object-module logs."""

    model = TinyLinearModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    assert trace.module_identity_mode == "object_module"
    assert [module.address for module in trace.modules] == ["self", "fc"]
    assert trace.modules["self"].address_children == ["fc"]
    assert trace.modules["fc"].address_parent == "self"
    assert trace.modules["fc"].params
    assert {param.module_address for param in trace.modules["fc"].params} == {"fc"}
    fc_labels = trace.resolve_sites(tl.in_module("fc"), max_fanout=16).labels()
    assert fc_labels
    assert all("fc:1" in trace[label].modules for label in fc_labels)
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_nested_modules_preserve_address_tree_and_selectors() -> None:
    """Nested tinygrad objects should preserve parent and child addresses."""

    model = TinyNestedModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    assert trace.module_identity_mode == "object_module"
    assert set(module.address for module in trace.modules) == {
        "self",
        "encoder",
        "encoder.proj",
        "head",
    }
    assert set(trace.modules["self"].address_children) == {"encoder", "head"}
    assert trace.modules["encoder"].address_parent == "self"
    assert trace.modules["encoder"].address_children == ["encoder.proj"]
    assert trace.modules["encoder.proj"].address_parent == "encoder"
    proj_labels = trace.resolve_sites(tl.in_module("encoder.proj"), max_fanout=16).labels()
    encoder_labels = trace.resolve_sites(tl.in_module("encoder"), max_fanout=32).labels()
    assert set(proj_labels) < set(encoder_labels)
    assert all("encoder.proj:1" in trace[label].modules for label in proj_labels)
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_shared_submodule_aliases_and_multicall() -> None:
    """Shared tinygrad objects should mirror primary-address alias semantics."""

    model = TinySharedModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    shared = trace.modules["left"]
    assert trace.modules["right"] is shared
    assert shared.address == "left"
    assert shared.all_addresses == ["left", "right"]
    assert shared.num_calls == 2
    assert set(shared.ops.keys()) == {1, 2}
    assert shared.call_labels == ["left:1", "left:2"]
    assert trace.modules["self"].address_children == ["left"]
    assert {param.module_address for param in shared.params} == {"left"}
    assert {tuple(param.all_module_addresses) for param in shared.params} == {("left", "right")}
    left_labels = trace.resolve_sites(tl.in_module("left"), max_fanout=32).labels()
    assert any("left:1" in trace[label].modules for label in left_labels)
    assert any("left:2" in trace[label].modules for label in left_labels)
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_reused_uop_keeps_first_construction_attribution() -> None:
    """Reused tinygrad UOps should not invent a second module-call attribution."""

    model = TinyReusedTensorModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    fc_labels = trace.resolve_sites(tl.in_module("fc"), max_fanout=16).labels()
    assert fc_labels
    assert trace.modules["fc"].num_calls == 1
    assert all("fc:1" in trace[label].modules for label in fc_labels)
    root_only = [
        op.label
        for op in trace.layer_list
        if op.module == "self:1" and op.layer_type == "add" and not op.is_input
    ]
    assert root_only
    assert check_metadata_invariants(trace) is True


def test_tinygrad_raw_function_traces_remain_function_root() -> None:
    """Raw tinygrad callables should keep the existing function-root module mode."""

    def raw_fn(x: Any) -> Any:
        """Return a raw tinygrad function output.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return (x + 1.0).relu()

    trace = tl.trace(raw_fn, Tensor.ones(3), backend="tinygrad")

    assert trace.module_identity_mode == "function_root"
    assert [module.address for module in trace.modules] == ["self"]
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_object_model_can_force_function_root() -> None:
    """Explicit function_root should preserve the old root-only behavior."""

    trace = tl.trace(
        TinyLinearModel(),
        Tensor.ones(1, 3),
        backend="tinygrad",
        module_identity_mode="function_root",
    )

    assert trace.module_identity_mode == "function_root"
    assert [module.address for module in trace.modules] == ["self"]
    assert trace.param_source == "none"
    assert check_metadata_invariants(trace) is True


def test_tinygrad_object_module_requires_discoverable_object() -> None:
    """Explicit object_module should reject raw tinygrad functions."""

    def raw_fn(x: Any) -> Any:
        """Return a raw tinygrad function output.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return x + 1.0

    with pytest.raises(BackendUnsupportedError, match="object_module.*callable object"):
        tl.trace(
            raw_fn,
            Tensor.ones(3),
            backend="tinygrad",
            module_identity_mode="object_module",
        )


def test_tinygrad_object_module_public_surface_matrix(tmp_path: Path) -> None:
    """Assert executable public surfaces on a real tinygrad object-module trace."""

    trace = tl.trace(TinyLinearModel(), Tensor.ones(1, 3), backend="tinygrad")

    assert trace.module_identity_mode == "object_module"
    assert trace.modules["fc"].params
    assert trace.modules["fc"].forward_args is not None
    assert trace.module_calls["fc:1"].forward_kwargs == {}
    assert isinstance(trace.summary(), str)
    assert len(trace.to_pandas()) == len(trace.layer_list)
    assert len(trace.modules.to_pandas()) == len(trace.modules)
    assert len(trace.module_calls["fc:1"].to_pandas()) == 1
    dot = trace.draw(
        vis_outpath=str(tmp_path / "tinygrad_object_graph"),
        vis_save_only=True,
        vis_fileformat="dot",
    )
    assert isinstance(dot, str)
    audit_path = tmp_path / "tinygrad_object_audit.tlspec"
    trace.save(audit_path, level="audit")
    loaded = tl.load(audit_path)
    assert loaded.backend == "tinygrad"
    assert loaded.module_identity_mode == "object_module"
    assert all(op.out is None for op in loaded.layer_list)
    with pytest.raises(
        BackendPayloadUnsupportedError, match="audit-only|materialized payloads"
    ) as exc_info:
        trace.save(tmp_path / "tinygrad_object_portable.tlspec")
    assert "expected a tensor for portable blobification" not in str(exc_info.value)


def test_tinygrad_object_module_corruption_invariant_fails() -> None:
    """Dropping object-module op attribution should fail metadata invariants."""

    trace = tl.trace(TinyLinearModel(), Tensor.ones(1, 3), backend="tinygrad")
    victim = next(layer for layer in trace.layer_list if layer.module and not layer.is_input)
    victim.module = None
    victim.modules = []
    victim.module_call_stack = []
    victim.output_of_modules = []
    victim.output_of_module_calls = []
    victim.atomic_module_call = None

    with pytest.raises(MetadataInvariantError, match="module_attribution"):
        check_metadata_invariants(trace)

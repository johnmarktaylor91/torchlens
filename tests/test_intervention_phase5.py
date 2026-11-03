"""Phase 5 postprocess preservation tests for intervention metadata."""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens._utils.hashing import _hashable_path_component
from torchlens.errors import MetadataInvariantError
from torchlens.intervention.types import ParentRef
from torchlens.recording import RecordContext
from torchlens.validation._invariants_entry import check_func_call_id_invariant


class _SplitModel(torch.nn.Module):
    """Model with a multi-output torch operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split and consume both outputs.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum of split chunks.
        """

        left, right = torch.split(x, 1, dim=0)
        return left + right


def _save_only_outputs(ctx: RecordContext) -> bool:
    """Select synthetic output records for predicate selective-save tests."""

    return ctx.kind == "synthetic_output" or ctx.layer_type == "output"


class _LinearSplitModel(torch.nn.Module):
    """Module-containing split fixture for graph hash tests."""

    def __init__(self) -> None:
        """Create a single linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a linear layer, split its output, and recombine.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Recombined split outputs.
        """

        left, right = torch.split(self.linear(x), 1, dim=0)
        return left + right


class _DifferentGraphModel(torch.nn.Module):
    """Different graph fixture for hash sensitivity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a different operation sequence.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output scaled by two.
        """

        return torch.relu(x) * 2


class _RenamedLinearSplitModel(torch.nn.Module):
    """Same operations as ``_LinearSplitModel`` under a different module address."""

    def __init__(self) -> None:
        """Create a single linear layer under a different attribute name."""

        super().__init__()
        self.proj = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a linear layer, split its output, and recombine.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Recombined split outputs.
        """

        left, right = torch.split(self.proj(x), 1, dim=0)
        return left + right


class _RecurrentReluModel(torch.nn.Module):
    """Small recurrent-style fixture with repeated operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same operation in a Python loop.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Recurrent relu result.
        """

        out = x
        for _ in range(3):
            out = torch.relu(out)
        return out


def _parent_refs(value: object) -> list[ParentRef]:
    """Collect ``ParentRef`` leaves from a nested template.

    Parameters
    ----------
    value:
        Template value to traverse.

    Returns
    -------
    list[ParentRef]
        Parent references found in the template.
    """

    if isinstance(value, ParentRef):
        return [value]
    if isinstance(value, tuple | list):
        refs: list[ParentRef] = []
        for item in value:
            refs.extend(_parent_refs(item))
        return refs
    if hasattr(value, "__dataclass_fields__"):
        refs = []
        for field_name in value.__dataclass_fields__:
            refs.extend(_parent_refs(getattr(value, field_name)))
        return refs
    return []


def test_graph_shape_hash_is_stable_and_sensitive() -> None:
    """Graph hash matches same graph/shape and changes for a different graph."""

    log1 = tl.trace(
        _LinearSplitModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    log2 = tl.trace(
        _LinearSplitModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    log3 = tl.trace(
        _DifferentGraphModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True),
    )

    assert log1.provenance.fingerprint is not None
    assert len(log1.provenance.fingerprint) == 64
    assert log1.provenance.fingerprint == log2.provenance.fingerprint
    assert log3.provenance.fingerprint != log1.provenance.fingerprint


def test_func_call_id_invariant_catches_duplicate_container_paths() -> None:
    """Invariant S fails when same-call outputs reuse an output path."""

    log = tl.trace(
        _SplitModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    split_layers = [layer for layer in log._layer_list if layer.func_name == "split"]
    split_layers[1].container_path = split_layers[0].container_path

    try:
        check_func_call_id_invariant(log)
    except MetadataInvariantError as exc:
        assert exc.check_name == "func_call_id_consistency"
    else:
        raise AssertionError("Invariant S did not catch duplicate container_path")


def test_hashable_path_component_plain_str_uses_value_not_bound_method() -> None:
    """Plain string path components must hash by value rather than ``str.index``."""

    assert _hashable_path_component("logits") == {"type": "str", "value": "'logits'"}

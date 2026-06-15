"""Runtime container view tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.container import TupleIndex


class DemoModelOutput(dict):
    """Minimal HuggingFace ``ModelOutput`` stand-in."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a mapping output with attribute access.

        Parameters
        ----------
        **kwargs:
            Output values.
        """

        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class HFLikeModel(nn.Module):
    """Return a nested HF-like output."""

    def forward(self, x: torch.Tensor) -> DemoModelOutput:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        DemoModelOutput
            Nested output container.
        """

        return DemoModelOutput(
            logits=x + 1,
            past_key_values=((x + 2, x + 3),),
        )


@dataclass
class PairBox:
    """Registered custom output container."""

    left: torch.Tensor
    right: torch.Tensor


class PairBoxModel(nn.Module):
    """Return a registered custom container."""

    def forward(self, x: torch.Tensor) -> PairBox:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        PairBox
            Custom container.
        """

        return PairBox(x + 1, x + 2)


class TupleModel(nn.Module):
    """Return a simple tuple output."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple output.
        """

        return x + 1, x + 2


def _register_pair_box() -> None:
    """Register the PairBox fixture type."""

    tl.register_container(
        PairBox,
        lambda value: ([value.left, value.right], None),
        lambda _aux, children: PairBox(children[0], children[1]),
    )


def test_hf_model_output_container_reconstructs_round_trip() -> None:
    """HF-like final output reconstructs to the original container type."""

    trace = tl.trace(HFLikeModel(), torch.tensor([1.0]), intervention_ready=True)
    rebuilt = trace.reconstruct_output()

    assert isinstance(rebuilt, DemoModelOutput)
    assert torch.equal(rebuilt["logits"], torch.tensor([2.0]))
    assert torch.equal(rebuilt["past_key_values"][0][1], torch.tensor([4.0]))

    output_container = trace.ops[trace.output_layers[0]].container
    assert isinstance(output_container, tl.Container)
    assert output_container.reconstructable is True
    assert output_container["past_key_values"][0][1].layer_label in trace.output_layers


def test_container_repr_summary_uses_output_cross_references() -> None:
    """Container repr and summary expose an indented tree with output refs."""

    trace = tl.trace(HFLikeModel(), torch.tensor([1.0]), intervention_ready=True)
    output_container = trace.ops[trace.output_layers[0]].container

    assert isinstance(output_container, tl.Container)
    tree = repr(output_container)
    assert "logits: -> output_1" in tree
    assert "past_key_values: tuple" in tree
    assert "[0]: tuple" in tree
    assert "[1]: -> output_3" in tree
    assert output_container.summary() == tree


def test_trace_to_pandas_adds_output_role_column() -> None:
    """Trace.to_pandas exports the leaf role within output containers."""

    trace = tl.trace(HFLikeModel(), torch.tensor([1.0]), intervention_ready=True)
    frame = trace.to_pandas()
    roles = dict(zip(frame["layer_label"], frame["output_role"], strict=True))

    assert "output_role" in frame.columns
    assert roles["output_1"] == "logits"
    assert roles["output_2"] == "0"
    assert roles["output_3"] == "1"


def test_capture_container_structure_reconstructs_without_intervention_ready() -> None:
    """Opt-in final-output structure reconstructs without intervention metadata."""

    trace = tl.trace(HFLikeModel(), torch.tensor([1.0]), capture_container_structure=True)
    rebuilt = trace.reconstruct_output()

    assert trace.intervention_ready is False
    assert isinstance(rebuilt, DemoModelOutput)
    assert torch.equal(rebuilt["logits"], torch.tensor([2.0]))
    assert torch.equal(rebuilt["past_key_values"][0][1], torch.tensor([4.0]))


def test_capture_container_structure_default_off_preserves_output_shape_metadata() -> None:
    """Default OFF matches explicit False and does not capture final output specs."""

    model = HFLikeModel()
    x = torch.tensor([1.0])
    default_trace = tl.trace(model, x, random_seed=0)
    explicit_false_trace = tl.trace(model, x, random_seed=0, capture_container_structure=False)

    assert default_trace.graph_shape_hash == explicit_false_trace.graph_shape_hash
    assert (
        [default_trace.ops[label].container_spec for label in default_trace.output_layers]
        == [
            explicit_false_trace.ops[label].container_spec
            for label in explicit_false_trace.output_layers
        ]
        == [None, None, None]
    )
    with pytest.raises(ValueError, match="No reconstructable final-output container"):
        default_trace.reconstruct_output()


def test_custom_registered_container_reconstructs() -> None:
    """Registered custom containers are captured and reconstructable."""

    _register_pair_box()
    trace = tl.trace(PairBoxModel(), torch.tensor([2.0]), intervention_ready=True)
    rebuilt = trace.reconstruct_output()

    assert isinstance(rebuilt, PairBox)
    assert torch.equal(rebuilt.left, torch.tensor([3.0]))
    assert torch.equal(rebuilt.right, torch.tensor([4.0]))


def test_output_at_selects_nested_container_path() -> None:
    """Nested output selector resolves typed captured container paths."""

    trace = tl.trace(HFLikeModel(), torch.tensor([1.0]), intervention_ready=True)
    site = trace.resolve_sites(tl.output_at(("past_key_values", 0, 1))).first()

    assert site.layer_label in trace.output_layers
    assert torch.equal(site.out, torch.tensor([4.0]))


def test_post_load_reconstruct_output_round_trips(tmp_path: Path) -> None:
    """Portable save/load preserves enough output structure to reconstruct."""

    trace = tl.trace(HFLikeModel(), torch.tensor([1.0]), intervention_ready=True)
    path = tmp_path / "container.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    rebuilt = loaded.reconstruct_output()

    assert isinstance(rebuilt, DemoModelOutput)
    assert torch.equal(rebuilt["logits"], torch.tensor([2.0]))
    assert torch.equal(rebuilt["past_key_values"][0][0], torch.tensor([3.0]))


def test_path_only_container_view_degrades_without_reconstruction() -> None:
    """An op with only path metadata exposes a non-reconstructable view."""

    trace = tl.trace(TupleModel(), torch.tensor([1.0]), intervention_ready=True)
    op = trace.ops[trace.output_layers[0]].copy()
    op.source_trace = trace
    op.container_spec = None
    op.container_path = (TupleIndex(0),)

    container = op.container

    assert isinstance(container, tl.Container)
    assert container.reconstructable is False
    with pytest.raises(ValueError, match="Path-only"):
        container.reconstruct()


def test_paths_only_backend_registry_view_does_not_fake_reconstruct() -> None:
    """A paths-only backend degrades registry-backed views to path-only by role."""

    trace = tl.trace(
        StackTupleOutputModel(),
        (torch.tensor([1.0]), torch.tensor([2.0])),
        capture_container_structure=True,
    )
    trace.backend = "jax"

    container = trace.ops[trace.output_layers[0]].container
    stack_op = next(op for op in trace.ops if op.func_name == "stack")
    input_container = stack_op.input_containers[0]

    assert isinstance(container, tl.Container)
    assert container.kind is None
    assert container.reconstructable is False
    assert input_container.kind is None
    assert input_container.reconstructable is False
    with pytest.raises(ValueError, match="Path-only"):
        container.reconstruct()
    with pytest.raises(ValueError, match="Path-only"):
        input_container.reconstruct()


def test_backend_none_container_capability_returns_no_false_view() -> None:
    """A backend declaring no structure does not expose false registry views."""

    trace = tl.trace(
        StackTupleOutputModel(),
        (torch.tensor([1.0]), torch.tensor([2.0])),
        capture_container_structure=True,
    )
    trace.backend = "mlx"

    output_op = trace.ops[trace.output_layers[0]]
    stack_op = next(op for op in trace.ops if op.func_name == "stack")

    assert output_op.container is None
    assert stack_op.input_containers == ()


class StackTupleOutputModel(nn.Module):
    """Consume a list at an op boundary and return a tuple output."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack two tensors and return a tuple."""

        stacked = torch.stack([left, right])
        return stacked, right - left

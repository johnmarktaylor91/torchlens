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

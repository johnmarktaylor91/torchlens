"""Regression tests for multi-output module handling."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import MultiOutputModuleError, TorchLensError
from torchlens.intervention.types import (
    ContainerSpec,
    DataclassField,
    DictKey,
    TupleIndex,
    rebuild_container_from_spec,
)


class LSTMModel(nn.Module):
    """Small model exposing all three LSTM outputs."""

    def __init__(self) -> None:
        """Initialize the LSTM fixture."""

        super().__init__()
        self.lstm = nn.LSTM(5, 10)
        self.label = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Sequence-major input tensor.

        Returns
        -------
        torch.Tensor
            Linear projection of the final hidden state.
        """

        batch_size = x.shape[1]
        h_0 = torch.zeros(1, batch_size, 10)
        c_0 = torch.zeros(1, batch_size, 10)
        _output, (h_n, _c_n) = self.lstm(x, (h_0, c_0))
        return self.label(h_n[-1])


class GRUModel(nn.Module):
    """Small model exposing GRU outputs."""

    def __init__(self) -> None:
        """Initialize the GRU fixture."""

        super().__init__()
        self.gru = nn.GRU(5, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Sequence-major input tensor.

        Returns
        -------
        torch.Tensor
            Tensor depending on both GRU outputs.
        """

        h_0 = torch.zeros(1, x.shape[1], 10)
        output, h_n = self.gru(x, h_0)
        return output + h_n[0:1].expand_as(output)


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM fixture."""

    def __init__(self) -> None:
        """Initialize the bidirectional LSTM fixture."""

        super().__init__()
        self.lstm = nn.LSTM(8, 4, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Batch-major input tensor.

        Returns
        -------
        torch.Tensor
            Mean pooled sequence output.
        """

        output, _state = self.lstm(x)
        return output.mean(1)


class MHAModel(nn.Module):
    """MultiheadAttention fixture."""

    def __init__(self) -> None:
        """Initialize the attention fixture."""

        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=16, num_heads=4)
        self.out = nn.Linear(16, 4)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        q:
            Query sequence.
        k:
            Key sequence.
        v:
            Value sequence.

        Returns
        -------
        torch.Tensor
            Projected attention output.
        """

        attn_output, _attn_weights = self.mha(q, k, v)
        return self.out(attn_output.mean(0))


class DictOutputModule(nn.Module):
    """Module returning a dict of tensors."""

    def __init__(self) -> None:
        """Initialize dict-output layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return two named tensors.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Logits and hidden tensors.
        """

        return {"logits": self.fc1(x), "hidden": self.fc2(x)}


class DictWrapper(nn.Module):
    """Wrapper consuming a dict-returning module."""

    def __init__(self) -> None:
        """Initialize the wrapper."""

        super().__init__()
        self.inner = DictOutputModule()
        self.last = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output depending on both dict values.
        """

        out = self.inner(x)
        return self.last(out["logits"] + out["hidden"])


@dataclass
class PairBox:
    """Simple dataclass container for reconstruction tests."""

    first: Any
    second: Any


def _lstm_trace() -> Any:
    """Return a traced LSTM fixture.

    Returns
    -------
    Any
        TorchLens trace for the LSTM fixture.
    """

    return tl.trace(LSTMModel(), torch.randn(7, 4, 5))


@pytest.mark.smoke
def test_lstm_three_outputs_distinct_layers() -> None:
    """LSTM outputs are distinct layers, not recurrent calls."""

    trace = _lstm_trace()
    lstm = trace.modules["lstm"]
    outputs = [trace.ops[label] for label in trace.modules["lstm"].ops[0].output_ops]
    assert lstm.num_calls == 1
    assert len(outputs) == 3
    assert len({output.layer_label for output in outputs}) == 3
    assert [output.multi_output_name for output in outputs] == ["output", "h_n", "c_n"]


def test_lstm_module_call_outputs_and_structure() -> None:
    """ModuleCall exposes output OpLogs and a container spec."""

    call = _lstm_trace().module_calls["lstm:1"]
    assert len(call.output_ops) == 3
    assert call.output_structure is not None
    assert call.output_structure.kind == "tuple"
    assert call.outs[1].shape == (1, 4, 10)


def test_lstm_module_out_raises_specific_error() -> None:
    """Ambiguous single-output access raises the new specific error."""

    lstm = _lstm_trace().modules["lstm"]
    with pytest.raises(MultiOutputModuleError):
        _ = lstm.out
    with pytest.raises(TorchLensError):
        _ = lstm.out
    assert isinstance(MultiOutputModuleError("x"), ValueError)


def test_gru_two_outputs_distinct_equivalence_classes() -> None:
    """GRU parameterized outputs get distinct equivalence classes."""

    trace = tl.trace(GRUModel(), torch.randn(7, 4, 5))
    outputs = [trace.ops[label] for label in trace.modules["gru"].ops[0].output_ops]
    assert len(outputs) == 2
    assert len({output.equivalence_class for output in outputs}) == 2
    assert [output.multi_output_name for output in outputs] == ["output", "h_n"]


def test_bilstm_outputs_preserve_single_call_structure() -> None:
    """Bidirectional LSTM outputs are multi-output, not multi-pass."""

    trace = tl.trace(BiLSTMModel(), torch.randn(4, 10, 8))
    lstm = trace.modules["lstm"]
    outputs = [trace.ops[label] for label in trace.modules["lstm"].ops[0].output_ops]
    assert lstm.num_calls == 1
    assert len(outputs) == 3
    assert outputs[1].shape == (2, 4, 4)


@pytest.mark.smoke
def test_mha_attention_outputs_are_selectable() -> None:
    """MultiheadAttention output roles distinguish weights from activations."""

    x = [torch.randn(7, 4, 16), torch.randn(7, 4, 16), torch.randn(7, 4, 16)]
    trace = tl.trace(MHAModel(), x)
    outputs = [trace.ops[label] for label in trace.modules["mha"].ops[0].output_ops]
    assert [output.multi_output_name for output in outputs] == [
        "attn_output",
        "attn_output_weights",
    ]
    sites = trace.find_sites(tl.module("mha") & tl.output("attn_output_weights"))
    assert [site.layer_label for site in sites] == [outputs[1].layer_label]


def test_dict_module_outputs_keyed_by_dict_keys() -> None:
    """Dict-returning modules preserve key order and role names."""

    trace = tl.trace(DictWrapper(), torch.randn(3, 4))
    inner = trace.modules["inner"]
    outputs = [trace.ops[label] for label in trace.modules["inner"].ops[0].output_ops]
    assert [output.multi_output_name for output in outputs] == ["logits", "hidden"]
    assert inner.output_structure is not None
    assert inner.output_structure.kind == "dict"
    assert inner.output_structure.keys == ("logits", "hidden")


def test_output_selectors_match_index_and_role() -> None:
    """Output selectors match module and function outputs."""

    trace = _lstm_trace()
    h_n = trace.ops[trace.modules["lstm"].ops[0].output_ops[1]]
    assert [site.layer_label for site in trace.find_sites(tl.module("lstm") & tl.output(1))] == [
        h_n.layer_label
    ]
    assert [site.layer_label for site in trace.find_sites(tl.func("lstm", output="h_n"))] == [
        h_n.layer_label
    ]


def test_save_load_preserves_module_outputs(tmp_path: Any) -> None:
    """Portable save/load preserves module outputs, roles, and structure."""

    trace = _lstm_trace()
    path = tmp_path / "lstm.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    lstm = loaded.modules["lstm"]
    outputs = [loaded.ops[label] for label in loaded.modules["lstm"].ops[0].output_ops]
    assert len(outputs) == 3
    assert [output.multi_output_name for output in outputs] == ["output", "h_n", "c_n"]
    assert lstm.output_structure is not None
    assert [tuple(out.shape) for out in lstm.outs] == [(7, 4, 10), (1, 4, 10), (1, 4, 10)]


def test_rebuild_container_from_spec_roundtrips_supported_shapes() -> None:
    """ContainerSpec reconstruction handles tuple, namedtuple, and dataclass specs."""

    tuple_spec = ContainerSpec(kind="tuple", length=2)
    assert rebuild_container_from_spec(tuple_spec, [1, 2]) == (1, 2)

    Point = namedtuple("Point", ["x", "y"])
    named_spec = ContainerSpec(
        kind="namedtuple",
        length=2,
        fields=("x", "y"),
        type_module=Point.__module__,
        type_qualname=Point.__qualname__,
    )
    assert rebuild_container_from_spec(named_spec, [1, 2]) == Point(1, 2)

    dataclass_spec = ContainerSpec(
        kind="dataclass",
        length=2,
        fields=("first", "second"),
        type_module=PairBox.__module__,
        type_qualname=PairBox.__qualname__,
        child_specs=((DataclassField("second"), ContainerSpec(kind="tuple", length=2)),),
    )
    assert rebuild_container_from_spec(dataclass_spec, [1, 2, 3]) == PairBox(1, (2, 3))


def test_rebuild_container_from_spec_roundtrips_nested_dict_list() -> None:
    """ContainerSpec reconstruction preserves nested dict/list structure."""

    nested_spec = ContainerSpec(
        kind="dict",
        keys=("a", "nested"),
        child_specs=(
            (
                DictKey("nested"),
                ContainerSpec(
                    kind="list",
                    length=2,
                    child_specs=((TupleIndex(1), ContainerSpec(kind="tuple", length=2)),),
                ),
            ),
        ),
    )
    assert rebuild_container_from_spec(nested_spec, [1, 2, 3, 4]) == {
        "a": 1,
        "nested": [2, (3, 4)],
    }

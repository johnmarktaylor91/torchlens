"""One tensor passed at several argument positions of one call: parents and edge occurrences.

``parents`` is a deduplicated label view; the multiplicity of a repeated operand lives in
``parent_arg_positions`` and in the canonical edge table, which holds exactly one occurrence
per argument position. A repeated operand used to be listed once per use in ``parents``, so
the edge freeze emitted (uses x uses) occurrences for most ops, and for multi-output ops such
as ``torch.mkldnn_rnn_layer`` and ``torch.lstm`` the metadata invariant
``edge_use_parent_arg_consistency`` failed outright (hidden and cell state as one tensor).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._record_group_table import record_member

_HIDDEN = 4


def _state(x: torch.Tensor, shape: tuple[int, ...], parented: bool) -> torch.Tensor:
    """Return a zero state tensor, a parentless factory or one derived from ``x``.

    Parameters
    ----------
    x:
        Model input.
    shape:
        State shape.
    parented:
        Whether the state gets a traced parent (``x``) or stays a bare factory.

    Returns
    -------
    torch.Tensor
        The state tensor.
    """

    state = torch.zeros(*shape)
    return state + 0.0 * x.sum() if parented else state


class _SharedOperandModel(nn.Module):
    """Run one forward body with four recurrent weights (input and hidden size 4)."""

    def __init__(
        self, body: Callable[[_SharedOperandModel, torch.Tensor], torch.Tensor], gates: int
    ) -> None:
        """Store the body and create the weights for ``gates`` gate blocks.

        Parameters
        ----------
        body:
            Forward body taking the model and the input.
        gates:
            Gate blocks per weight (4 for LSTM, 3 for GRU, 1 for a tanh RNN).
        """

        super().__init__()
        self.body = body
        gen = torch.Generator().manual_seed(0)
        rows = gates * _HIDDEN
        self.w_ih = nn.Parameter(torch.randn(rows, _HIDDEN, generator=gen) * 0.3)
        self.w_hh = nn.Parameter(torch.randn(rows, _HIDDEN, generator=gen) * 0.3)
        self.b_ih = nn.Parameter(torch.randn(rows, generator=gen) * 0.1)
        self.b_hh = nn.Parameter(torch.randn(rows, generator=gen) * 0.1)

    def weights(self) -> list[torch.Tensor]:
        """Return the four weights in ATen RNN order.

        Returns
        -------
        list[torch.Tensor]
            ``[w_ih, w_hh, b_ih, b_hh]``.
        """

        return [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stored body.

        Parameters
        ----------
        x:
            Input tensor of shape ``(4, 4)``.

        Returns
        -------
        torch.Tensor
            Body output.
        """

        return self.body(self, x)


def _mkldnn_shared_hx_cx(
    parented: bool,
) -> Callable[[_SharedOperandModel, torch.Tensor], torch.Tensor]:
    """Build a body passing one state tensor as both ``hx_`` and ``cx_`` of mkldnn_rnn_layer.

    Parameters
    ----------
    parented:
        Whether the shared state has a traced parent.

    Returns
    -------
    Callable
        Forward body.
    """

    def body(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
        """One mkldnn LSTM layer whose hidden and cell state are one tensor.

        Parameters
        ----------
        model:
            Owning model (weights).
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output.
        """

        state = _state(x, (1, _HIDDEN), parented)
        # mode 2 is LSTM; hidden 4, one layer, biases, not bidirectional or batch-first, eval.
        out = torch.mkldnn_rnn_layer(
            torch.relu(x).unsqueeze(1),
            *model.weights(),
            state,
            state,
            False,
            [],
            2,
            _HIDDEN,
            1,
            True,
            False,
            False,
            False,
        )
        return out[0].sum() * 2 + x

    return body


def _lstm_shared_hx(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.lstm`` with one tensor as both entries of the ``hx`` list.

    Parameters
    ----------
    model:
        Owning model (weights).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    state = _state(x, (1, 1, _HIDDEN), True)
    out = torch.lstm(
        torch.relu(x).unsqueeze(1),
        [state, state],
        model.weights(),
        True,
        1,
        0.0,
        False,
        False,
        False,
    )
    return out[0].sum() + x


def _lstm_input_is_state(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.lstm`` with one tensor as the input and both hidden-state entries.

    Parameters
    ----------
    model:
        Owning model (weights).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    state = _state(x, (1, 1, _HIDDEN), False)
    out = torch.lstm(state, [state, state], model.weights(), True, 1, 0.0, False, False, False)
    return out[0].sum() + x


def _gru_input_is_state(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.gru`` with one tensor as the input and the hidden state.

    Parameters
    ----------
    model:
        Owning model (weights).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    state = _state(x, (1, 1, _HIDDEN), True)
    out = torch.gru(state, state, model.weights(), True, 1, 0.0, False, False, False)
    return out[0].sum() + x


def _rnn_tanh_input_is_state(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.rnn_tanh`` with one tensor as the input and the hidden state.

    Parameters
    ----------
    model:
        Owning model (weights).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    state = _state(x, (1, 1, _HIDDEN), False)
    out = torch.rnn_tanh(state, state, model.weights(), True, 1, 0.0, False, False, False)
    return out[0].sum() + x


def _lstm_cell_shared_hx(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.lstm_cell`` with one tensor as the input and both ``hx`` entries.

    Parameters
    ----------
    model:
        Owning model (weights).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    state = _state(x, (1, _HIDDEN), True)
    return torch.lstm_cell(state, [state, state], *model.weights())[0].sum() + x


def _cat_twice(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.cat`` with one tensor at two list positions.

    Parameters
    ----------
    model:
        Owning model (unused).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    return torch.cat([x, x]).sum() + x


def _stack_thrice(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.stack`` with one derived tensor at three list positions.

    Parameters
    ----------
    model:
        Owning model (unused).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    y = torch.relu(x)
    return torch.stack([y, y, y]).sum(0) + x


def _add_twice(model: _SharedOperandModel, x: torch.Tensor) -> torch.Tensor:
    """``torch.add`` with one tensor at both top-level positions.

    Parameters
    ----------
    model:
        Owning model (unused).
    x:
        Input tensor.

    Returns
    -------
    torch.Tensor
        Output.
    """

    return torch.add(x, x)


_CASES = {
    "mkldnn_rnn_layer-hx-cx-factory": (_mkldnn_shared_hx_cx(False), 4),
    "mkldnn_rnn_layer-hx-cx-parented": (_mkldnn_shared_hx_cx(True), 4),
    "lstm-hx-list": (_lstm_shared_hx, 4),
    "lstm-input-and-hx-list": (_lstm_input_is_state, 4),
    "gru-input-and-hx": (_gru_input_is_state, 3),
    "rnn_tanh-input-and-hx": (_rnn_tanh_input_is_state, 1),
    "lstm_cell-input-and-hx-list": (_lstm_cell_shared_hx, 4),
    "cat-list": (_cat_twice, 4),
    "stack-list": (_stack_thrice, 4),
    "add-top-level": (_add_twice, 4),
}


def _position_counts(op: object) -> Counter:
    """Count the argument positions ``parent_arg_positions`` attributes to each parent.

    Parameters
    ----------
    op:
        One per-pass op record.

    Returns
    -------
    Counter
        Parent label to number of attributed positions.
    """

    counts: Counter = Counter()
    positions = record_member(op, "parent_arg_positions", None) or {}
    for domain in ("args", "kwargs"):
        counts.update(label for label in (positions.get(domain) or {}).values())
    return counts


def _assert_one_occurrence_per_position(trace: tl.Trace) -> None:
    """Every op lists each parent once and holds one edge occurrence per argument position.

    Parameters
    ----------
    trace:
        Finished trace.
    """

    store = trace._trace_core.ops
    edges, ref_labels = store.dataflow_edges, store.ref_labels
    for layer in trace._layer_list:
        for op in getattr(layer, "ops", None) or (layer,):
            parents = list(op.parents or ())
            assert len(parents) == len(set(parents)), (op.label, parents)
            positions = _position_counts(op)
            recorded = Counter(ref_labels.get(edge.source) for edge in edges.in_edges(op._row))
            expected = {parent: positions.get(parent) or 1 for parent in parents}
            assert dict(recorded) == expected, (op.label, dict(recorded), expected)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_shared_operand_parents_and_edge_occurrences_agree(case: str) -> None:
    """A tensor at k argument positions is one parent with k edge occurrences, and validates.

    Parameters
    ----------
    case:
        Key into ``_CASES``.
    """

    body, gates = _CASES[case]
    model = _SharedOperandModel(body, gates)
    torch.manual_seed(0)
    x = torch.randn(4, _HIDDEN)
    assert tl.validate(model, x, scope="forward", validate_metadata=True) is True
    trace = tl.trace(model, x)
    try:
        _assert_one_occurrence_per_position(trace)
    finally:
        trace.cleanup()

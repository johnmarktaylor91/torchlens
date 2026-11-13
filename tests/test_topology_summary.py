"""The trace card's topology words say why a layer runs more than one pass.

``graph.is_recurrent`` keeps its meaning (some layer runs more than one pass); the card's
``graph`` row says whether that comes from a loop or a module repeated from one place
("recurrent (up to N passes)") or from a module shared by several places ("shared modules"),
and appends tied weights. A stateless module called from two lines is two layers, so a
ResNet-style block reads "feedforward".
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

import torchlens as tl


def _graph_row(model: nn.Module, x: Tensor) -> str:
    """The ``graph`` row of the trace card, and the trace's is_recurrent."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model.eval(), x)
    rows = [line.strip() for line in str(trace).splitlines() if line.strip().startswith("graph")]
    assert len(rows) == 1, str(trace)
    return rows[0].removeprefix("graph").strip()


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1, self.l2, self.relu = nn.Linear(4, 4), nn.Linear(4, 4), nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.relu(self.l1(x))
        return self.relu(self.l2(y))


class Siamese(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        a = self.enc(x)
        b = self.enc(x * 2)
        return a - b


class StepRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.RNNCell(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.zeros(x.shape[0], 4)
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
        return h


class SiameseRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = StepRNN()

    def forward(self, x: Tensor) -> Tensor:
        a = self.rnn(x)
        b = self.rnn(x * 2)
        return a - b


class Tied(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids: Tensor) -> Tensor:
        return self.head(torch.tanh(self.emb(ids)))


def test_stateless_activation_on_two_lines_reads_feedforward() -> None:
    """A ResNet-style block with one ``nn.ReLU`` called twice is feedforward."""

    assert _graph_row(Block(), torch.randn(2, 4)).startswith("feedforward, no branching")


def test_siamese_encoder_reads_shared_modules() -> None:
    """A stateful encoder called on two lines is shared, not recurrent."""

    row = _graph_row(Siamese(), torch.randn(2, 4))
    assert row.startswith("feedforward, shared modules (enc x2), ")


def test_rnn_cell_in_a_loop_reads_recurrent() -> None:
    """A cell called from one loop line is a repeat: recurrent with its pass count."""

    assert _graph_row(StepRNN(), torch.randn(2, 3, 4)).startswith("recurrent (up to 3 passes), ")


def test_loop_inside_a_shared_module_reads_both() -> None:
    """A loop inside a module shared by two lines reads recurrent and shared."""

    row = _graph_row(SiameseRNN(), torch.randn(2, 3, 4))
    assert row.startswith("recurrent (up to 6 passes), shared modules (rnn x2), ")


def test_tied_embedding_and_head_reads_tied_weights() -> None:
    """A tied embedding and output head add the tied-weights clause."""

    row = _graph_row(Tied(), torch.tensor([[1, 2, 3]]))
    assert row.startswith("feedforward, tied weights (1 group), ")

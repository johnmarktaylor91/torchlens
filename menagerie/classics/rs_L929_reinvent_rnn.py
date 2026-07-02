# SOURCE: vendored from https://github.com/MolecularAI/REINVENT4 @ main
# File: reinvent/models/reinvent/models/rnn.py (RNN)
#
# REINVENT's classical de novo molecule generator: an N-layer GRU/LSTM cell
# with an embedding layer and output linear projection back to vocabulary
# size (SMILES token vocabulary). The architecture is a specific, versioned
# configuration class living in the REINVENT4 repo (not a class exposed by
# an installed base library), so it is vendored here rather than treated as
# a base-lib recipe. Class body is unmodified from the real repo.

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as tnn
import torch.nn.functional as tnnf

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from reinvent/models/reinvent/models/rnn.py ---
class RNN(tnn.Module):
    """
    Implements an N layer GRU(M) or LSTM cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(
        self,
        voc_size: int,
        layer_size: int = 512,
        num_layers: int = 3,
        cell_type: str = "gru",
        embedding_layer_size: int = 256,
        dropout: float = 0.0,
        layer_normalization: bool = False,
        device=torch.device("cpu"),
    ) -> None:
        super(RNN, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._layer_normalization = layer_normalization
        self.device = device

        self._embedding = tnn.Embedding(voc_size, self._embedding_layer_size)

        rnn = getattr(tnn, self._cell_type.upper(), None)

        if not rnn:
            raise RuntimeError('cell type must be either "gru" or "lstm"')

        self._rnn: tnn.RNNBase = rnn(
            input_size=self._embedding_layer_size,
            hidden_size=self._layer_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
            batch_first=True,
        )

        self._linear = tnn.Linear(self._layer_size, voc_size)

        self.to(self.device)

    def forward(
        self,
        input_vector: torch.Tensor,
        hidden_state: "torch.Tensor | Sequence[torch.Tensor]" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_size = input_vector.size()

        if hidden_state is None:
            size = (self._num_layers, batch_size, self._layer_size)

            if self._cell_type == "gru":
                hidden_state = torch.zeros(*size, device=self.device)
            elif self._cell_type == "lstm":
                hidden_state = [
                    torch.zeros(*size, device=self.device),
                    torch.zeros(*size, device=self.device),
                ]
            else:
                raise ValueError(f'Invalid cell type "{self._cell_type}"')

        embedded_data = self._embedding(input_vector)  # (batch,seq,embedding)
        output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)

        if self._layer_normalization:
            output_vector = tnnf.layer_norm(output_vector, output_vector.size()[1:])

        output_vector = output_vector.reshape(-1, self._layer_size)
        output_data = self._linear(output_vector).view(batch_size, seq_size, -1)

        return output_data, hidden_state_out

    def get_params(self) -> Dict[str, Any]:
        return {
            "dropout": self._dropout,
            "layer_size": self._layer_size,
            "num_layers": self._num_layers,
            "cell_type": self._cell_type,
            "embedding_layer_size": self._embedding_layer_size,
            "layer_normalization": self._layer_normalization,
        }


# --- staging harness ---
def build_reinvent_rnn():
    return RNN(voc_size=48, layer_size=32, num_layers=2, cell_type="gru", embedding_layer_size=16)


def example_input_reinvent_rnn():
    return (torch.randint(0, 48, (3, 10), dtype=torch.long),)


MENAGERIE_ENTRIES = [
    ("REINVENT_RNN", "build_reinvent_rnn", "example_input_reinvent_rnn", 2017, "vendored-pytorch"),
]

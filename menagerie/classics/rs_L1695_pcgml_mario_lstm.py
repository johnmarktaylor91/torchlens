# FAITHFUL PORT of zhihanyang2022/super_mario_as_a_string @ master (original
# framework: Keras/TensorFlow). https://github.com/zhihanyang2022/super_mario_as_a_string
# -- community replication of "Super Mario as a String: Platformer Level Generation
# Via LSTMs" (Summerville & Mateas, arXiv 1603.00930). The repo's model (defined in
# `02_train_model.ipynb`) has no PyTorch/base-lib equivalent, so per the menagerie
# ladder this is transcribed faithfully into self-contained torch. The Keras graph
# is: 3 stacked stateful `LSTM(hidden_size, return_sequences=True, return_state=True)`
# layers, each taking an explicit incoming `(h, c)` state pair and each followed by
# `Dropout(dropout)`, then a shared `Dense(vocab_size)` + softmax applied at every
# timestep, over a one-hot tile-vocabulary input sequence `[batch, seq_length,
# vocab_size]`. Every layer/mechanism in that graph is preserved: 3 explicit
# `nn.LSTMCell`-driven recurrences unrolled over `seq_length` (Keras `LSTM(...,
# return_sequences=True)` is a per-timestep recurrence, faithfully ported as an
# explicit loop rather than approximated by `nn.LSTM`, so that (a) all 6 external
# state tensors (h/c in and h/c out for all 3 layers) are threaded through exactly
# as in the Keras functional-API graph, and (b) dropout is applied between layers
# at every timestep as in Keras' `TimeDistributed`-style `Dropout` after
# `return_sequences=True`), inter-layer dropout, and a per-timestep shared dense+
# softmax classification head.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class MarioLSTMLevelGenerator(nn.Module):
    """Port of the 02_train_model.ipynb Keras functional model: 3 stacked
    stateful LSTM layers (explicit external h/c state in and out, matching the
    Keras graph's `initial_state=[...]` / `return_state=True` wiring) with
    inter-layer dropout, followed by a shared per-timestep Dense+softmax head
    over the tile vocabulary.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 128, dropout: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm_cell_1 = nn.LSTMCell(vocab_size, hidden_size)
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_cell_3 = nn.LSTMCell(hidden_size, hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        input: torch.Tensor,
        lstm_1_state_h_in: torch.Tensor,
        lstm_1_state_c_in: torch.Tensor,
        lstm_2_state_h_in: torch.Tensor,
        lstm_2_state_c_in: torch.Tensor,
        lstm_3_state_h_in: torch.Tensor,
        lstm_3_state_c_in: torch.Tensor,
    ):
        """
        Args:
          input: [batch, seq_length, vocab_size] one-hot tile sequence.
          lstm_*_state_{h,c}_in: [batch, hidden_size] initial states for each
            of the 3 stacked LSTM layers (matches the Keras graph's explicit
            `initial_state` inputs).

        Returns:
          out_acc: [batch, seq_length, vocab_size] per-timestep softmax over
            the tile vocabulary.
          lstm_{1,2,3}_state_{h,c}_out: [batch, hidden_size] final states for
            each layer (matches the Keras graph's `return_state=True` outputs).
        """
        batch_size, seq_length, _ = input.shape

        h1, c1 = lstm_1_state_h_in, lstm_1_state_c_in
        h2, c2 = lstm_2_state_h_in, lstm_2_state_c_in
        h3, c3 = lstm_3_state_h_in, lstm_3_state_c_in

        outputs = []
        for t in range(seq_length):
            x_t = input[:, t, :]

            h1, c1 = self.lstm_cell_1(x_t, (h1, c1))
            out1 = self.dropout1(h1)

            h2, c2 = self.lstm_cell_2(out1, (h2, c2))
            out2 = self.dropout2(h2)

            h3, c3 = self.lstm_cell_3(out2, (h3, c3))
            out3 = self.dropout3(h3)

            logits_t = self.dense(out3)
            outputs.append(self.softmax(logits_t))

        out_acc = torch.stack(outputs, dim=1)

        return out_acc, h1, c1, h2, c2, h3, c3


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_pcgml_mario_lstm():
    torch.manual_seed(0)
    # Real repo uses hidden_size=128, seq_length=200, vocab_size=len(char_to_ix);
    # shrunk here for fast tracing.
    model = MarioLSTMLevelGenerator(vocab_size=12, hidden_size=8, dropout=0.5)
    model.eval()
    return model


def example_input_pcgml_mario_lstm():
    torch.manual_seed(0)
    batch_size = 2
    seq_length = 5
    vocab_size = 12
    hidden_size = 8
    input = torch.zeros(batch_size, seq_length, vocab_size)
    tile_ixs = torch.randint(0, vocab_size, (batch_size, seq_length))
    input.scatter_(-1, tile_ixs.unsqueeze(-1), 1.0)
    zeros = torch.zeros(batch_size, hidden_size)
    return (
        input,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
    )


MENAGERIE_ENTRIES = [
    (
        "PCGML-Mario-LSTM",
        build_pcgml_mario_lstm,
        example_input_pcgml_mario_lstm,
        2016,
        "ported-pytorch",
    ),
]

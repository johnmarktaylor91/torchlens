# FAITHFUL PORT of https://github.com/tsudalab/ChemTS @ master (4174c3600ebb47ed136b433b22a29c879824a6ba) (original framework:
# Keras 1.1.1 / Theano-or-TF1.x backend)
#
# ChemTS ("Molecule Design using Monte Carlo Tree Search with Neural Rollout",
# Yang et al. 2017, arXiv:1710.00616) pairs an MCTS search over SMILES-token space
# with a neural "rollout" policy: an RNN language model over SMILES tokens that
# predicts the next-token distribution used to bias/roll out tree expansion. The
# MCTS orchestration (mcts_logp.py, add_node_type.py) is search-algorithm glue, not
# a trainable neural network; the actual neural net is the pretrained RNN shipped
# as `RNN-model/model.json` + `model.h5` and loaded via `load_model.py ::
# loaded_model()` (`keras.models.model_from_json`).
#
# ChemTS's own requirements (README.md) pin Keras==2.0.5 on Python>=2.7, and the
# shipped `RNN-model/model.json` records `"keras_version": "1.1.1"` -- both
# incompatible with this environment's modern torch-only stack (no Keras/TF at
# all, and Keras 1.1.1's old-style multi-gate GRU config is not expressible via a
# `keras>=2` load path either). The model architecture is nonetheless known
# EXACTLY (not guessed from the paper) because Keras serializes the full layer
# config as JSON; this port transcribes the model.json's real layer stack
# byte-for-byte:
#
#   Embedding(input_dim=64, output_dim=64, input_length=82)
#   -> GRU(output_dim=256, activation="sigmoid", inner_activation="hard_sigmoid",
#          return_sequences=True)
#   -> GRU(output_dim=256, activation="sigmoid", inner_activation="hard_sigmoid",
#          return_sequences=True)
#   -> TimeDistributed(Dense(64, activation="softmax"))
#
# Keras 1.1.1's `GRU.step()` (`consume_less="cpu"`, as configured in model.json)
# uses non-default gate/candidate activations -- update/reset gates through
# `hard_sigmoid`, and (per this model's own config) the candidate hidden state
# through `sigmoid` rather than Keras's usual `tanh` default. `torch.nn.GRU` bakes
# in fixed sigmoid/tanh gate math and cannot reproduce this, so the GRU cell below
# is a custom hand-written recurrence transcribing Keras 1.1.1's exact step()
# equations (see `recurrent.py::GRU.step`, consume_less="cpu" branch):
#
#   x_z, x_r, x_h = per-timestep affine projections of the input (W_z/W_r/W_h + b)
#   z  = hard_sigmoid(x_z + h_{t-1} @ U_z)
#   r  = hard_sigmoid(x_r + h_{t-1} @ U_r)
#   hh = sigmoid(x_h + (r * h_{t-1}) @ U_h)
#   h_t = z * h_{t-1} + (1 - z) * hh
#
# `hard_sigmoid` itself is Keras/Theano's exact piecewise-linear approximation
# (`clip(0.2*x + 0.5, 0, 1)`), reproduced verbatim rather than substituted with
# `torch.sigmoid`.
import torch
import torch.nn as nn


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Keras/Theano's exact hard-sigmoid: clip(0.2*x + 0.5, 0, 1)."""
    return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)


class KerasGRUCell(nn.Module):
    """One timestep of Keras 1.1.1's GRU.step() (consume_less='cpu'), with the
    per-timestep input projections (`W_z x + b_z`, etc.) precomputed by the
    caller (mirroring `preprocess_input` -> `time_distributed_dense`)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.W_z = nn.Linear(input_dim, output_dim)
        self.W_r = nn.Linear(input_dim, output_dim)
        self.W_h = nn.Linear(input_dim, output_dim)
        self.U_z = nn.Linear(output_dim, output_dim, bias=False)
        self.U_r = nn.Linear(output_dim, output_dim, bias=False)
        self.U_h = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x_t: torch.Tensor, h_tm1: torch.Tensor) -> torch.Tensor:
        x_z = self.W_z(x_t)
        x_r = self.W_r(x_t)
        x_h = self.W_h(x_t)

        z = hard_sigmoid(x_z + self.U_z(h_tm1))
        r = hard_sigmoid(x_r + self.U_r(h_tm1))
        hh = torch.sigmoid(x_h + self.U_h(r * h_tm1))  # model.json activation="sigmoid"
        h = z * h_tm1 + (1 - z) * hh
        return h


class KerasGRU(nn.Module):
    """Sequence-level wrapper around KerasGRUCell, matching Keras `GRU(...,
    return_sequences=True)`: iterates the cell over the time axis and stacks
    every hidden state (Keras zero-initializes `h_0`, as here)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.cell = KerasGRUCell(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps, input_dim)
        batch, timesteps, _ = x.shape
        h = x.new_zeros(batch, self.output_dim)
        outputs = []
        for t in range(timesteps):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # (batch, timesteps, output_dim)


class ChemTSRNN(nn.Module):
    """Faithful port of ChemTS's `RNN-model/model.json`:
    Embedding(64, 64, input_length=82) -> GRU(256, return_sequences=True) x2
    -> TimeDistributed(Dense(64, softmax))."""

    VOCAB_SIZE = 64
    EMBED_DIM = 64
    SEQ_LEN = 82
    HIDDEN_DIM = 256

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.EMBED_DIM)
        self.gru_1 = KerasGRU(self.EMBED_DIM, self.HIDDEN_DIM)
        self.gru_2 = KerasGRU(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.dense = nn.Linear(self.HIDDEN_DIM, self.VOCAB_SIZE)  # TimeDistributed(Dense)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len) int64
        x = self.embedding(token_ids)
        x = self.gru_1(x)
        x = self.gru_2(x)
        logits = self.dense(x)  # TimeDistributed applies the same Dense per timestep
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------
def build_chemts():
    return ChemTSRNN()


def example_input_chemts():
    torch.manual_seed(0)
    token_ids = torch.randint(0, ChemTSRNN.VOCAB_SIZE, (2, ChemTSRNN.SEQ_LEN), dtype=torch.long)
    return (token_ids,)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("ChemTS", build_chemts, example_input_chemts, 2017, "SOURCE_AVAILABLE"),
]

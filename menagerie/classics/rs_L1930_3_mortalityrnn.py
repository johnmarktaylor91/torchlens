# SOURCE: vendored from JonathanCrabbe/Simplex @ main
# https://raw.githubusercontent.com/JonathanCrabbe/Simplex/main/src/simplexai/models/base.py
# https://raw.githubusercontent.com/JonathanCrabbe/Simplex/main/src/simplexai/models/recurrent_neural_net.py
#
# NOTE ON PROVENANCE: the queue's repo_url (YerevaNN/mimic3-benchmarks) does not contain a
# PyTorch "MortalityRNN" -- that repo's mortality models (mimic3models/in_hospital_mortality/)
# are Keras-only (mimic3models/keras_models/lstm.py etc.), confirmed by walking its full repo
# tree. No PyTorch repo literally named "MortalityRNN" exists on GitHub (repo/code search
# both empty). JonathanCrabbe/Simplex (NeurIPS 2021, "Explaining Latent Representations with
# a Corpus of Examples") ships `MortalityGRU`, a real GRU-based in-hospital-mortality
# predictor used as one of the paper's black-box case studies -- the direct real-code match
# for this family (RNN over clinical time series -> mortality risk). `BlackBox` and
# `MortalityGRU` are copied verbatim; only the `device = torch.device("cuda" if ...)` module
# import is dropped (unused at class-definition/construction time -- BlackBox/MortalityGRU
# never reference the module-level `device` name, only `.to(device)` calls made by the
# repo's separate training script would).
import torch
import torch.nn as nn
import abc

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from simplexai/models/base.py ---
class BlackBox(torch.nn.Module):
    @abc.abstractmethod
    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the latent representation for the example x
        :param x: input features
        :return:
        """
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the output for the example x
        :param x: input features
        :return:
        """
        return


# --- vendored from simplexai/models/recurrent_neural_net.py ---
class MortalityGRU(BlackBox):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,  # dropout=drop_prob
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x, h = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


def build_mortalityrnn():
    torch.manual_seed(0)
    input_dim = 16  # per-timestep clinical feature vector size
    hidden_dim = 24
    output_dim = 1  # in-hospital-mortality risk (binary)
    n_layers = 2
    return MortalityGRU(input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2)


def example_input_mortalityrnn():
    torch.manual_seed(0)
    batch_size = 4
    seq_len = 20
    input_dim = 16
    return torch.randn(batch_size, seq_len, input_dim)


MENAGERIE_ENTRIES = [
    ("MortalityRNN", "build_mortalityrnn", "example_input_mortalityrnn", 2021, "vendored"),
]

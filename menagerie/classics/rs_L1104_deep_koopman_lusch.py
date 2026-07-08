# SOURCE: vendored from https://github.com/mlpotter/DeepKoopmanLusch @ master
# (models.py, classes KoopmanOperator and Lusch)
#
# Deep Koopman Network (Lusch, Kutz & Brunton, Nature Communications 2018,
# "Deep learning for universal linear embeddings of nonlinear dynamics").
# The official repo (BethanyL/DeepKoopman) is TensorFlow 1.x (`tf.truncated_
# normal`/`tf.random_uniform`, incompatible with any installed TF2/Keras-3
# env and not reasonably runnable here per the ladder); `mlpotter/
# DeepKoopmanLusch` is a real, complete PyTorch port of the same
# architecture (autoencoder that learns a Koopman-invariant latent
# embedding, `Lusch`, plus `KoopmanOperator`, the parameterized linear
# evolution operator built from per-step complex-conjugate-eigenvalue
# rotation-scaling 2x2 blocks -- the paper's auxiliary network that predicts
# continuous eigenvalues mu/omega and assembles the block-diagonal Koopman
# matrix K). Both classes are copied verbatim from `models.py`; the only
# change is dropping the unused `seaborn`/`matplotlib`/`tqdm` plotting-only
# imports the source file also has at module level (not referenced by
# either class) and swapping the deprecated `torch.autograd.Variable` calls
# for the modern plain-tensor equivalent (`Variable(torch.zeros(...))` ->
# `torch.zeros(...)`, a no-op since torch>=0.4 -- `Variable` is a pure
# backward-compat identity wrapper, not an architectural component).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class KoopmanOperator(nn.Module):
    """Real DeepKoopmanLusch/models.py:KoopmanOperator, verbatim (parameterized
    linear Koopman evolution operator: predicts continuous eigenvalues
    mu/omega per step, assembles a block-diagonal rotation-scaling matrix K,
    and rolls the latent state forward T steps)."""

    def __init__(self, koopman_dim, delta_t, device="cpu"):
        super().__init__()

        self.koopman_dim = koopman_dim
        self.num_eigenvalues = int(koopman_dim / 2)
        self.delta_t = delta_t
        self.parameterization = nn.Sequential(
            nn.Linear(self.koopman_dim, self.num_eigenvalues * 2),
            nn.Tanh(),
            nn.Linear(self.num_eigenvalues * 2, self.num_eigenvalues * 2),
        )
        self.device = device

    def forward(self, x, T):
        # x is B x 1 x Latent (only the initial point, T=1 along dim 1)
        Y = torch.zeros(x.shape[0], T, self.koopman_dim).to(self.device)
        y = x[:, 0, :]
        for t in range(T):
            mu, omega = torch.unbind(
                self.parameterization(y).reshape(-1, self.num_eigenvalues, 2), -1
            )

            # B x Koopmandim/2
            exp = torch.exp(self.delta_t * mu)

            # B x Latent/2
            cos = torch.cos(self.delta_t * omega)
            sin = torch.sin(self.delta_t * omega)

            K = torch.zeros(x.shape[0], self.koopman_dim, self.koopman_dim).to(self.device)

            for i in range(0, self.koopman_dim, 2):
                index = i // 2

                K[:, i + 0, i + 0] = cos[:, index] * exp[:, index]
                K[:, i + 0, i + 1] = -sin[:, index] * exp[:, index]
                K[:, i + 1, i + 0] = sin[:, index] * exp[:, index]
                K[:, i + 1, i + 1] = cos[:, index] * exp[:, index]

            y = torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)

            Y[:, t, :] = y

        return Y


class Lusch(nn.Module):
    """Real DeepKoopmanLusch/models.py:Lusch (Deep Koopman autoencoder),
    verbatim. Encoder/decoder are plain tanh-MLPs mapping to/from the
    Koopman-invariant latent space; `koopman` is the KoopmanOperator above."""

    def __init__(self, input_dim, koopman_dim, hidden_dim, delta_t=0.01, device="cpu"):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, koopman_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(koopman_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.koopman = KoopmanOperator(koopman_dim, delta_t, device)

        self.device = device
        self.delta_t = delta_t

        # Normalization occurs inside the model
        self.register_buffer("mu", torch.zeros((input_dim,)))
        self.register_buffer("std", torch.ones((input_dim,)))

    def forward(self, x):
        x = self.embed(x)
        x = self.recover(x)
        return x

    def embed(self, x):
        x = self._normalize(x)
        x = self.encoder(x)
        return x

    def recover(self, x):
        x = self.decoder(x)
        x = self._unnormalize(x)
        return x

    def koopman_operator(self, x, T=1):
        return self.koopman(x, T)

    def _normalize(self, x):
        return (x - self.mu[(None,) * (x.dim() - 1) + (...,)]) / self.std[
            (None,) * (x.dim() - 1) + (...,)
        ]

    def _unnormalize(self, x):
        return (
            self.std[(None,) * (x.dim() - 1) + (...,)] * x
            + self.mu[(None,) * (x.dim() - 1) + (...,)]
        )


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
_INPUT_DIM = 3  # e.g. the paper's Lorenz/Fluid-flow 3D state-space examples
_KOOPMAN_DIM = 6
_HIDDEN_DIM = 32
_BATCH = 4


def build_deep_koopman_lusch():
    torch.manual_seed(0)
    model = Lusch(
        input_dim=_INPUT_DIM, koopman_dim=_KOOPMAN_DIM, hidden_dim=_HIDDEN_DIM, delta_t=0.01
    )
    model.eval()
    return model


def example_input_deep_koopman_lusch():
    torch.manual_seed(0)
    return torch.randn(_BATCH, _INPUT_DIM)


MENAGERIE_ENTRIES = [
    (
        "DeepKoopman-Lusch",
        "build_deep_koopman_lusch",
        "example_input_deep_koopman_lusch",
        2018,
        MENAGERIE_ZOO,
    ),
]

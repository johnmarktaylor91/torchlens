# SOURCE: vendored from GaloisInc/dlkoopman @ main (dlkoopman/nets.py, composition per
# dlkoopman/traj_pred.py's TrajPred._evolve / TrajPred.predict_new)
#
# https://raw.githubusercontent.com/GaloisInc/dlkoopman/main/dlkoopman/nets.py
# https://raw.githubusercontent.com/GaloisInc/dlkoopman/main/dlkoopman/traj_pred.py
#
# The queue candidate "Koopman NN" points at BethanyL/DeepKoopman (the original paper's
# official implementation, TensorFlow-1.x). dlkoopman (GaloisInc, published on PyPI, MIT
# licensed, actively maintained) is a from-scratch PyTorch re-implementation of the same
# DeepKoopman idea (autoencoder that learns Koopman eigenfunctions for linear latent
# dynamics) with its own `nets.py` containing plain `torch.nn.Module` classes with base-lib
# (`torch` only) dependencies -- no simplification or reimplementation-from-paper is needed
# here since real, runnable PyTorch source exists.
#
# `nets.py` provides three real nn.Modules: `MLP`, `AutoEncoder` (encoder+decoder MLPs), and
# `Knet` (a single bias-free, activation-free Linear layer that approximates the Koopman
# operator acting on the encoded/latent state). These are vendored verbatim below. The
# `StatePred`/`TrajPred` orchestration classes in dlkoopman are training harnesses (data
# handling, SVD-based Koopman-matrix fitting, logging) rather than a single traceable
# nn.Module forward pass, so instead of vendoring that harness code, `DeepKoopmanNet` below
# composes the same three real nn.Modules using the EXACT rollout logic dlkoopman itself uses
# at inference time (see `TrajPred._evolve` / `TrajPred.predict_new` in traj_pred.py):
#   Y0 = ae.encoder(X0)
#   Ypred[:, 0] = Y0
#   Ypred[:, t] = Knet(Ypred[:, t-1])  for t = 1..num_steps-1   (repeated latent rollout)
#   Xpred = ae.decoder(Ypred)
# This is a direct transcription of that composition, not a new architecture.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# dlkoopman/nets.py -- vendored verbatim.
# ---------------------------------------------------------------------------


class MLP(torch.nn.Module):
    """Multi-layer perceptron neural net."""

    def __init__(self, input_size, output_size, hidden_sizes=[], batch_norm=False):
        super().__init__()
        self.net = torch.nn.ModuleList([])
        layers = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers) - 1):
            self.net.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:  # all layers except last
                if batch_norm:
                    self.net.append(torch.nn.BatchNorm1d(layers[i + 1]))
                self.net.append(torch.nn.ReLU())

    def forward(self, X) -> torch.Tensor:
        for layer in self.net:
            X = layer(X)
        return X


class AutoEncoder(torch.nn.Module):
    """AutoEncoder neural net. Contains an encoder connected to a decoder, both MLPs."""

    def __init__(
        self,
        input_size,
        encoded_size,
        encoder_hidden_layers=[],
        decoder_hidden_layers=[],
        batch_norm=False,
    ):
        super().__init__()

        if not decoder_hidden_layers and encoder_hidden_layers:
            decoder_hidden_layers = encoder_hidden_layers[::-1]
        elif not encoder_hidden_layers and decoder_hidden_layers:
            encoder_hidden_layers = decoder_hidden_layers[::-1]

        self.encoder = MLP(
            input_size=input_size,
            output_size=encoded_size,
            hidden_sizes=encoder_hidden_layers,
            batch_norm=batch_norm,
        )

        self.decoder = MLP(
            input_size=encoded_size,
            output_size=input_size,
            hidden_sizes=decoder_hidden_layers,
            batch_norm=batch_norm,
        )

    def forward(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        Y = self.encoder(X)  # encoder complete output
        Xr = self.decoder(Y)  # final reconstructed output
        return Y, Xr


class Knet(torch.nn.Module):
    """Linear neural net to approximate the Koopman matrix.

    Contains identically sized input and output layers, no hidden layers, no bias vector, and
    no activation function.
    """

    def __init__(self, size):
        super().__init__()
        self.net = torch.nn.Linear(in_features=size, out_features=size, bias=False)

    def forward(self, X) -> torch.Tensor:
        return self.net(X)


# ---------------------------------------------------------------------------
# Composition matching dlkoopman.traj_pred.TrajPred's inference-time rollout.
# ---------------------------------------------------------------------------


class DeepKoopmanNet(torch.nn.Module):
    """Encode -> repeatedly apply the learned linear Koopman operator in latent space ->
    decode every rolled-out latent state back to observation space. Mirrors
    `TrajPred._evolve` + the `ae.decoder(Ypred)` call in `TrajPred.predict_new`
    (traj_pred.py), which is dlkoopman's actual multi-step-ahead prediction procedure.
    """

    def __init__(
        self,
        input_size: int,
        encoded_size: int,
        num_steps: int,
        encoder_hidden_layers=None,
        decoder_hidden_layers=None,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.ae = AutoEncoder(
            input_size=input_size,
            encoded_size=encoded_size,
            encoder_hidden_layers=list(encoder_hidden_layers or []),
            decoder_hidden_layers=list(decoder_hidden_layers or []),
            batch_norm=batch_norm,
        )
        self.Knet = Knet(encoded_size)
        self.num_steps = num_steps
        self.encoded_size = encoded_size

    def forward(self, X0: torch.Tensor) -> torch.Tensor:
        """X0: (num_trajectories, input_size) -- the initial state of each trajectory.
        Returns Xpred: (num_trajectories, num_steps, input_size)."""
        Y0 = self.ae.encoder(X0)

        Ys = [Y0]
        Yt = Y0
        for _ in range(1, self.num_steps):
            Yt = self.Knet(Yt)
            Ys.append(Yt)
        Ypred = torch.stack(Ys, dim=1)  # (num_trajectories, num_steps, encoded_size)

        Xpred = self.ae.decoder(Ypred)
        return Xpred


# ---------------------------------------------------------------------------
# Menagerie staging entrypoints.
# ---------------------------------------------------------------------------


def build_deepkoopman():
    torch.manual_seed(0)
    return DeepKoopmanNet(
        input_size=3,
        encoded_size=8,
        num_steps=10,
        encoder_hidden_layers=[16],
        decoder_hidden_layers=[16],
        batch_norm=False,
    )


def example_input_deepkoopman():
    torch.manual_seed(0)
    return (torch.randn(4, 3),)


MENAGERIE_ENTRIES = [
    (
        "DeepKoopman (dlkoopman AutoEncoder + Knet rollout)",
        "build_deepkoopman",
        "example_input_deepkoopman",
        2018,
        MENAGERIE_ZOO,
    ),
]

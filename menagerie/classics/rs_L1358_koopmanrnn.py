# SOURCE: vendored from https://github.com/GaloisInc/dlkoopman @ 7daacf27be712474bc4fac85cabef2a96beadb1d
#   Vendored files:
#     - dlkoopman/nets.py       -> `MLP`, `AutoEncoder`, `Knet` (the exact building-block
#       nn.Modules of the DLKoopman package: a general-purpose PyTorch implementation of deep
#       Koopman-operator learning for dynamical systems, Dey et al., GaloisInc/dlkoopman).
#     - dlkoopman/traj_pred.py  -> `TrajPred._evolve` / `TrajPred.predict_new` composition
#       logic (how `ae.encoder`, the repeated `Knet` rollout, and `ae.decoder` are wired
#       together into the trajectory-prediction forward pass).
#
# This is the "Koopman RNN" family: an autoencoder maps each observed state into a latent
# space chosen so the (unknown, nonlinear) system dynamics become approximately LINEAR;
# a single learned linear map (`Knet`, a bias-free nn.Linear -- the finite-dimensional
# approximation of the Koopman operator) is then applied repeatedly/recurrently in that
# latent space to roll a trajectory forward step by step (y_{i+1} = K y_i), i.e. exactly a
# weight-tied linear RNN cell; each rolled-forward latent state is decoded back into
# observation space. Every module (`MLP`, `AutoEncoder`, `Knet`) and the recurrent rollout
# loop (`_evolve`) are reproduced verbatim from the real repo; only the surrounding
# data-handler / training-loop / SVD-eigendecomposition bookkeeping (irrelevant to the
# architecture itself) is dropped, and the three real building blocks are composed into a
# single traceable `nn.Module.forward` matching `TrajPred.predict_new`.

import torch

MENAGERIE_ZOO = "vendored-pytorch"


class MLP(torch.nn.Module):
    """Multi-layer perceptron neural net.

    ## Parameters
    - **input_size** (*int*) - Dimension of the input layer.
    - **output_size** (*int*) - Dimension of the output layer.
    - **hidden_sizes** (*list[int], optional*) - Dimensions of hidden layers, if any.
    - **batch_norm** (*bool, optional*) - Whether to use batch normalization.
    """

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
    """AutoEncoder neural net. Contains an encoder connected to a decoder, both are
    multi-layer perceptrons.

    ## Parameters
    - **input_size** (*int*) - Number of dimensions in original data (encoder input) and
      reconstructed data (decoder output).
    - **encoded_size** (*int*) - Number of dimensions in encoded data (encoder output and
      decoder input).
    - **encoder_hidden_layers** (*list[int], optional*).
    - **decoder_hidden_layers** (*list[int], optional*).
    - **batch_norm** (*bool, optional*).
    """

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

    def forward(self, X) -> tuple:
        Y = self.encoder(X)  # encoder complete output
        Xr = self.decoder(Y)  # final reconstructed output
        return Y, Xr


class Knet(torch.nn.Module):
    """Linear neural net to approximate the Koopman matrix.

    Contains identically sized input and output layers, no hidden layers, no bias vector,
    and no activation function.

    ## Parameters
    - **size** (*int*) - Dimension of the input and output layer.
    """

    def __init__(self, size):
        super().__init__()
        self.net = torch.nn.Linear(in_features=size, out_features=size, bias=False)

    def forward(self, X) -> torch.Tensor:
        return self.net(X)


class KoopmanRNN(torch.nn.Module):
    """Composes AutoEncoder + Knet into the full trajectory-prediction forward pass, matching
    `TrajPred._evolve` + `TrajPred.predict_new` (encode initial state -> roll forward
    `num_indexes` steps in latent space via the recurrent linear `Knet` -> decode every step
    back to observation space)."""

    def __init__(self, input_size: int, encoded_size: int, num_indexes: int, hidden_sizes=[16]):
        super().__init__()
        self.ae = AutoEncoder(
            input_size=input_size,
            encoded_size=encoded_size,
            encoder_hidden_layers=hidden_sizes,
            batch_norm=False,
        )
        self.Knet = Knet(size=encoded_size)
        self.num_indexes = num_indexes
        self.encoded_size = encoded_size

    def _evolve(self, Y0: torch.Tensor) -> torch.Tensor:
        """Y0: (num_trajectories, encoded_size) -> (num_trajectories, num_indexes, encoded_size),
        reproduced verbatim from `TrajPred._evolve` (recurrent application of the linear
        Koopman-matrix layer `Knet`)."""
        steps = [Y0]
        for _index in range(1, self.num_indexes):
            steps.append(self.Knet(steps[-1]))
        return torch.stack(steps, dim=1)

    def forward(self, X0: torch.Tensor) -> torch.Tensor:
        """X0: (num_trajectories, input_size) initial states -> predicted trajectories
        (num_trajectories, num_indexes, input_size), matching `TrajPred.predict_new`."""
        Y0 = self.ae.encoder(X0)
        Ypred = self._evolve(Y0)
        Xpred = self.ae.decoder(Ypred)
        return Xpred


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_koopmanrnn():
    """Small KoopmanRNN (input_size=4, encoded_size=8, 6-step rollout) for tracing."""
    model = KoopmanRNN(input_size=4, encoded_size=8, num_indexes=6, hidden_sizes=[16])
    model.eval()
    return model


def example_input_koopmanrnn():
    """Matches KoopmanRNN.forward: (num_trajectories, input_size) initial states."""
    torch.manual_seed(0)
    return torch.randn(3, 4, dtype=torch.float32)


MENAGERIE_ENTRIES = [
    ("Koopman RNN", "build_koopmanrnn", "example_input_koopmanrnn", 2019, MENAGERIE_ZOO),
]

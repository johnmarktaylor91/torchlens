# SOURCE: vendored from RasmussenLab/vamb @ b146fb610921aef1da8aed5d739f1ce80d21b33a
# https://raw.githubusercontent.com/RasmussenLab/vamb/master/vamb/encode.py
#
# Nissen, Jakob Nybo et al. "Improved metagenome binning and assembly using deep
# variational autoencoders" (Nature Biotechnology, 2021) / VAMB project. `VAE` is a
# variational autoencoder over a concatenated (abundance, tetranucleotide-frequency,
# total-abundance) feature vector per metagenomic contig: a stack of `Linear + LeakyReLU
# + Dropout + BatchNorm1d` encoder layers (`nhiddens`) feeding a `mu` linear bottleneck
# (note: the real code has no explicit sigma head -- reparameterize() adds unit gaussian
# noise directly to `mu`, matching an upstream bugfix noted in the original docstring),
# then a symmetric decoder stack ending in a single `outputlayer` that is split back into
# (depths, tnf, abundance) reconstructions with `narrow` and a softmax on the depths
# slice. The `VAE` class body (constructor + `_encode`/`reparameterize`/`_decode`/
# `forward`) is copied verbatim from the real `vamb/encode.py`; only the `trainmodel`/
# `save`/`load`/`calc_loss`/`trainepoch` methods (which need the `dadaptation` optimizer,
# `vamb.vambtools`, and `loguru`, none of which are base libs or needed to trace a
# forward pass) were dropped from this staging copy.

import torch
from torch import Tensor
from torch import nn
from typing import Optional


class VAE(nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: list of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples: int,
        nhiddens: Optional[list] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
        seed: int = 0,
    ):
        if nlatent < 1:
            raise ValueError(f"Minimum 1 latent neuron, not {nlatent}")

        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError(f"Minimum 1 neuron per layer, not {min(nhiddens)}")

        if beta <= 0:
            raise ValueError(f"beta must be > 0, not {beta}")

        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be 0 < alpha < 1, not {alpha}")

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be 0 <= dropout < 1, not {dropout}")

        torch.manual_seed(seed)
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = 103
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout

        # Initialize lists for holding hidden layers
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip(
            # + 1 for the total abundance
            [self.nsamples + self.ntnf + 1] + self.nhiddens,
            self.nhiddens,
        ):
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = nn.Linear(self.nhiddens[-1], self.nlatent)

        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))

        # Reconstruction (output) layer. + 1 for the total abundance
        self.outputlayer = nn.Linear(self.nhiddens[0], self.nsamples + self.ntnf + 1)

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def _encode(self, tensor: Tensor) -> Tensor:
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        # Latent layers
        mu = self.mu(tensor)

        # Note: We ought to also compute logsigma here, but we had a bug in the original
        # implementation of Vamb where logsigma was fixed to zero, so we just remove it.

        return mu

    # sample with gaussian noise
    def reparameterize(self, mu: Tensor) -> Tensor:
        epsilon = torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        latent = mu + epsilon

        return latent

    def _decode(self, tensor: Tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)
        abundance_out = reconstruction.narrow(1, self.nsamples + self.ntnf, 1)

        depths_out = torch.nn.functional.softmax(depths_out, dim=1)

        return depths_out, tnf_out, abundance_out

    def forward(self, depths: Tensor, tnf: Tensor, abundance: Tensor):
        tensor = torch.cat((depths, tnf, abundance), 1)
        mu = self._encode(tensor)
        latent = self.reparameterize(mu)
        depths_out, tnf_out, abundance_out = self._decode(latent)

        return depths_out, tnf_out, abundance_out, mu


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_vamb_vae() -> nn.Module:
    # nsamples=4 (real default in the CLI is however many abundance columns the user
    # supplies); nhiddens/nlatent shrunk from real defaults ([512, 512], 32) to
    # ([32, 32], 8) for a fast trace. alpha/beta/dropout left at real defaults (0.15
    # auto for nsamples>1 in __init__, 200.0, 0.2). eval() mirrors the real
    # `evaluate=True` state VAE.load() returns, so BatchNorm1d uses running stats
    # (irrelevant here since we also warm the running stats up below, but this
    # matches the encode()-path VAE.eval() call in the real vamb/encode.py).
    model = VAE(nsamples=4, nhiddens=[32, 32], nlatent=8)
    model.eval()
    return model


def example_input_vamb_vae():
    # depths: (batch, nsamples) row-normalized abundance; tnf: (batch, 103) z-scored
    # tetranucleotide frequencies; abundance: (batch, 1) z-scored total abundance --
    # exactly the three tensors make_dataloader() produces and VAE.forward consumes,
    # per the docstring `forward(self, depths, tnf, abundance)`.
    batch = 6
    depths = torch.softmax(torch.randn(batch, 4), dim=1)
    tnf = torch.randn(batch, 103)
    abundance = torch.randn(batch, 1)
    return (depths, tnf, abundance)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("VAMB VAE", "build_vamb_vae", "example_input_vamb_vae", 2021, "vendored"),
]

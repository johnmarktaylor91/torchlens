# SOURCE: vendored from almaan/stereoscope @ 7460428d271cb36023dae2238f24f9ab169a18b9
# https://github.com/almaan/stereoscope/blob/master/stsc/models.py
#
# Stereoscope (Andersson et al. 2020, "Single-cell and spatial transcriptomics
# enables probabilistic inference of cell type topography") deconvolves
# spot-level spatial transcriptomics counts into cell-type proportions using a
# two-stage negative-binomial (NB) generative model. STModel is the spatial
# ("ST") stage: it holds per-gene reference rates R (estimated from the
# paired single-cell ScModel, frozen here), learns per-spot cell-type
# proportions (theta -> softplus -> v), a per-gene noise/unknown-type rate
# (eta -> softplus -> eps), and a per-gene multiplicative bias (beta), then
# combines them via an einsum into per-gene/per-spot NB rates and returns the
# negative log-likelihood of the observed spot counts as the forward-pass
# output (this is the genuine, unmodified training-time forward -- the model
# IS the loss function by design, per the original repo). Vendored verbatim
# (only whitespace/lint-clean; no architectural changes).

import numpy as np
import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


class STModel(nn.Module):
    def __init__(
        self,
        n_spots: int,
        R: np.ndarray,
        logits: np.ndarray,
        device: t.device,
        **kwargs,
    ) -> None:
        super().__init__()

        # Get dimensions for from data
        self.S = n_spots
        self.G, self.K = R.shape
        self.Z = self.K + 1

        # Data from single cell estimates; Rates (R) and logits (o)
        self.R = t.tensor(R.astype(np.float32)).to(device)
        self.o = t.tensor(logits.astype(np.float32).reshape(-1, 1)).to(device)

        # model specific parameters
        self.softpl = nn.functional.softplus
        self.lsig = nn.functional.logsigmoid
        self.sig = t.sigmoid

        # Learn noise from data
        self.eta = Parameter(t.tensor(np.zeros((self.G, 1)).astype(np.float32)).to(device))
        nn.init.normal_(self.eta, mean=0.0, std=1.0)

        # un-normalized proportion in log space
        self.theta = Parameter(t.tensor(np.zeros((self.Z, self.S)).astype(np.float32)).to(device))
        nn.init.normal_(self.theta, mean=0.0, std=1.0)
        # gene bias in log space
        if not kwargs.get("freeze_beta", False):
            self.beta = Parameter(t.tensor(np.zeros((self.G, 1)).astype(np.float32)).to(device))
            self.beta_trans = self.softpl
            nn.init.normal_(self.beta, mean=0.0, std=0.1)
        else:
            print("Using static beta_g")
            self.beta = t.tensor(np.ones((self.G, 1)).astype(np.float32)).to(device)
            self.beta_trans = lambda x: x
        # un-normalized proportions
        self.v = t.tensor(np.zeros((self.Z, self.S)).astype(np.float32)).to(device)

        self.loss = t.tensor(0.0)
        self.model_ll = 0.0

    def noise_loss(
        self,
    ) -> t.Tensor:
        """Regularizing term for noise"""
        return -0.5 * t.sum(t.pow(self.eta, 2))

    def _llnb(
        self,
        x: t.Tensor,
    ) -> t.Tensor:
        """Log Likelihood function for standard model"""

        log_unnormalized_prob = self.r * self.lsig(-self.o) + x * self.lsig(self.o)

        log_normalization = -t.lgamma(self.r + x) + t.lgamma(1.0 + x) + t.lgamma(self.r)

        ll = t.sum(log_unnormalized_prob - log_normalization)

        self.ll = ll.item()

        return ll

    def _lfun(
        self,
        x: t.Tensor,
    ) -> t.Tensor:
        """Loss Function

        Composed of the likelihood and prior of
        noise. Returns negative value of the above
        terms, to obtain a proper loss function.

        L(x) = -[LogLikelihood(x) + log(prior(noise))]

        Parameter
        --------
        x : t.tensor
            observed counts (n_genes x n_spots)

        """

        # log likelihood of observed count given model
        data_loss = self._llnb(x)
        # log of prior on noise elements
        noise_loss = self.noise_loss()

        return -data_loss - noise_loss

    def __str__(
        self,
    ) -> str:
        return "st_model"

    def forward(
        self,
        x: t.tensor,
        gidx: t.tensor,
        **kwargs,
    ) -> t.tensor:
        """Forward pass"""

        self.gidx = gidx
        # proportion values
        self.v = self.softpl(self.theta)
        # noise values
        self.eps = self.softpl(self.eta)
        # account for gene specific bias and add noise
        self.Rhat = t.cat((t.mul(self.beta_trans(self.beta), self.R), self.eps), dim=1)
        # combinde rates for all cell types
        self.r = t.einsum("gz,zs->gs", [self.Rhat, self.v[:, self.gidx]])
        # get loss for current parameters
        self.loss = self._lfun(x.transpose(1, 0))

        return self.loss


def build_stereoscope():
    # Tiny sizing for menagerie tracing: n_spots=6 spots, n_genes=10, n_celltypes=3
    # (paper-scale runs use thousands of genes/spots; the model's math is
    # size-invariant so this exercises the identical composition).
    n_spots, n_genes, n_celltypes = 6, 10, 3
    rng = np.random.RandomState(0)
    R = rng.rand(n_genes, n_celltypes).astype(np.float32) + 0.1
    logits = rng.randn(n_genes).astype(np.float32)
    return STModel(n_spots=n_spots, R=R, logits=logits, device=t.device("cpu"))


def example_input_stereoscope():
    n_spots, n_genes = 6, 10
    rng = np.random.RandomState(1)
    x = t.tensor(rng.poisson(5, size=(n_spots, n_genes)).astype(np.float32))
    gidx = t.arange(n_spots, dtype=t.long)
    return (x, gidx)


MENAGERIE_ENTRIES = [
    ("Stereoscope", build_stereoscope, example_input_stereoscope, 2020, "vendored-pytorch"),
]

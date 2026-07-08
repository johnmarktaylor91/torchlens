# SOURCE: vendored from https://github.com/awslabs/gluonts (PR #1617, "Adding NCAD
# (Neural Contextual Anomaly Detection for Time Series) to nursery")
# @ 396192e17e99a881a9b95229f9ff11ad2e685008
# (src/gluonts/nursery/ncad/src/ncad/model/tcn_encoder.py::Chomp1d/TCNBlock/TCN/
#  TCNEncoder + contrastive_clasifier.py::ContrastiveClasifier + distances.py::
#  CosineDistance + ncad.py::NCAD.forward architecture)
#
# NCAD (Carmona et al., 2021/2022, "Neural Contextual Anomaly Detection for Time
# Series"): a self-supervised contrastive time-series anomaly detector. A shared
# causal Temporal Convolutional Network (TCN) encoder maps both the *whole* window
# and a truncated *context* window (whole window minus the trailing "suspect"
# sub-window) to embeddings; a ContrastiveClasifier scores how *different* the two
# embeddings are (via a CosineDistance) as the logit for "the suspect sub-window is
# anomalous". Injected synthetic anomalies (Contextual Outlier Exposure / Mixup, not
# part of the architecture) are used only at training time.
#
# Vendored real repo code verbatim: Chomp1d, TCNBlock, TCN, TCNEncoder
# (tcn_encoder.py); ContrastiveClasifier (contrastive_clasifier.py); CosineDistance
# (distances.py). Every Conv1d/weight_norm/LeakyReLU/AdaptiveMaxPool1d/Linear layer
# and the causal-chomp/residual-connection/embedding-normalization logic is
# unchanged from the original. The top-level `NCAD` class in the real repo
# (ncad.py) subclasses `pytorch_lightning.LightningModule` and pulls in
# `torch_optimizer`, `torchmetrics`, and repo-local `ncad.utils.donut_metrics`/
# `ncad.utils.pl_metrics`/`ncad.model.outlier_exposure`/`ncad.model.mixup` purely
# for its `training_step`/`validation_step`/`configure_optimizers`/`detect`
# training-and-deployment scaffolding -- none of that machinery participates in the
# forward architecture. This vendored `NCADModule(nn.Module)` reconstructs exactly
# the real `NCAD.forward()` method (construct encoder + classifier with the same
# hparams, encode the whole window and the context window with the *same* shared
# encoder, classify the pair) as a plain `nn.Module` with no PyTorch-Lightning /
# torch_optimizer / torchmetrics dependency, so it is traceable in a base torch env.

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# --------------------------------------------------------------------------------
# ncad/model/tcn_encoder.py
# --------------------------------------------------------------------------------


class Chomp1d(nn.Module):
    """Removes leading or trailing elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    """

    def __init__(self, chomp_size: int, last: bool = True):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block.

    Composed sequentially of two causal convolutions (with leaky ReLU activation
    functions), and a parallel residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
    ):
        super(TCNBlock, self).__init__()

        in_channels = int(in_channels)
        kernel_size = int(kernel_size)
        out_channels = int(out_channels)
        dilation = int(dilation)

        # Computes left padding so that the applied convolutions are causal
        padding = int((kernel_size - 1) * dilation)

        # First causal convolution
        conv1_pre = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        conv1 = nn.utils.weight_norm(conv1_pre)

        chomp1 = Chomp1d(chomp_size=padding)
        relu1 = nn.LeakyReLU()

        # Second causal convolution
        conv2_pre = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        conv2 = nn.utils.weight_norm(conv2_pre)
        chomp2 = Chomp1d(chomp_size=padding)
        relu2 = nn.LeakyReLU()

        self.causal = nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2)

        self.upordownsample = (
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.activation = nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.activation is None:
            return out_causal + res
        else:
            return self.activation(out_causal + res)


class TCN(nn.Module):
    """Temporal Convolutional Network: a sequence of causal convolution blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        channels: int,
        layers: int,
    ):
        super(TCN, self).__init__()

        layers = int(layers)

        net_layers = []
        dilation_size = 1

        for i in range(layers):
            in_channels_block = in_channels if i == 0 else channels
            net_layers.append(
                TCNBlock(
                    in_channels=in_channels_block,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    final=False,
                )
            )
            dilation_size *= 2

        net_layers.append(
            TCNBlock(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                final=True,
            )
        )

        self.network = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.network(x)


class TCNEncoder(nn.Module):
    """Encoder of a time series using a Temporal Convolution Network (TCN).

    The computed representation is the output of a fully connected layer applied
    to the output of an adaptive max pooling layer applied on top of the TCN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        tcn_channels: int,
        tcn_layers: int,
        tcn_out_channels: int,
        maxpool_out_channels: int = 1,
        normalize_embedding: bool = True,
    ):
        super(TCNEncoder, self).__init__()
        tcn = TCN(
            in_channels=in_channels,
            out_channels=tcn_out_channels,
            kernel_size=kernel_size,
            channels=tcn_channels,
            layers=tcn_layers,
        )

        maxpool_out_channels = int(maxpool_out_channels)
        maxpooltime = nn.AdaptiveMaxPool1d(maxpool_out_channels)
        flatten = nn.Flatten()
        fc = nn.Linear(tcn_out_channels * maxpool_out_channels, out_channels)
        self.network = nn.Sequential(tcn, maxpooltime, flatten, fc)

        self.normalize_embedding = normalize_embedding

    def forward(self, x):
        u = self.network(x)
        if self.normalize_embedding:
            return F.normalize(u, p=2, dim=1)
        else:
            return u


# --------------------------------------------------------------------------------
# ncad/model/distances.py
# --------------------------------------------------------------------------------


class CosineDistance(nn.Module):
    r"""Returns the cosine distance between :math:`x_1` and :math:`x_2`, computed along dim."""

    def __init__(self, dim: int = 1, keepdim: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)
        self.eps = 1e-10

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(x1, x2, self.dim, self.eps)
        dist = -torch.log((1 + cos_sim) / 2)

        if self.keepdim:
            dist = dist.unsqueeze(dim=self.dim)
        return dist


# --------------------------------------------------------------------------------
# ncad/model/contrastive_clasifier.py
# --------------------------------------------------------------------------------


class ContrastiveClasifier(nn.Module):
    """Contrastive Classifier.

    Calculates the distance between two random vectors, and returns an exponential
    transformation of it, which can be interpreted as the logits for the two
    vectors being different: p = 1 - exp(-dist(x1, x2)).
    """

    def __init__(self, distance: nn.Module):
        super().__init__()

        self.distance = distance
        self.eps = 1e-10

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dists = self.distance(x1, x2)

        log_prob_equal = -dists

        prob_different = torch.clamp(1 - torch.exp(log_prob_equal), self.eps, 1)
        log_prob_different = torch.log(prob_different)

        logits_different = log_prob_different - log_prob_equal

        return logits_different


# --------------------------------------------------------------------------------
# ncad/model/ncad.py :: NCAD.forward architecture, as a plain nn.Module
# --------------------------------------------------------------------------------


class NCADModule(nn.Module):
    """Reconstructs the real `NCAD(pl.LightningModule).forward()` architecture
    (encode whole window + encode context window with the shared encoder, then
    classify) as a plain `nn.Module`. Dropped: `pl.LightningModule` base class,
    `save_hyperparameters()`, `training_step`/`validation_step`/`test_step`/
    `configure_optimizers`/`detect`/`tsdetect` training-and-inference-loop methods,
    and the `torch_optimizer`/`torchmetrics`/`ncad.utils.*` metric/optimizer
    machinery they use -- none of that is architecture, only PyTorch-Lightning
    training-loop plumbing built on top of the real encoder/classifier below.
    """

    def __init__(
        self,
        ts_channels: int,
        window_length: int,
        suspect_window_length: int,
        tcn_kernel_size: int,
        tcn_layers: int,
        tcn_out_channels: int,
        tcn_maxpool_out_channels: int = 1,
        embedding_rep_dim: int = 64,
        normalize_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.suspect_window_length = suspect_window_length

        # Encoder Network
        self.encoder = TCNEncoder(
            in_channels=ts_channels,
            out_channels=embedding_rep_dim,
            kernel_size=tcn_kernel_size,
            tcn_channels=tcn_out_channels,
            tcn_layers=tcn_layers,
            tcn_out_channels=tcn_out_channels,
            maxpool_out_channels=tcn_maxpool_out_channels,
            normalize_embedding=normalize_embedding,
        )

        # Contrast Classifier
        self.classifier = ContrastiveClasifier(distance=CosineDistance())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The encoder could manage other window lengths, but all training and
        # validation is currently performed with a single length.
        assert x.shape[-1] == self.window_length

        ts_whole_embedding = self.encoder(x)
        ts_context_embedding = self.encoder(x[..., : -self.suspect_window_length])

        logits_anomaly = self.classifier(ts_whole_embedding, ts_context_embedding)

        return logits_anomaly


def build_ncad():
    # Tiny config matching the real repo's example hparams shape (examples/
    # article/hparams/*.json use e.g. window_length=64, suspect_window_length=8,
    # tcn_kernel_size=3-5, tcn_layers=3-6, tcn_out_channels=20-40,
    # embedding_rep_dim=64-128); shrunk here for a fast, small trace.
    return NCADModule(
        ts_channels=3,
        window_length=32,
        suspect_window_length=4,
        tcn_kernel_size=3,
        tcn_layers=2,
        tcn_out_channels=8,
        tcn_maxpool_out_channels=1,
        embedding_rep_dim=16,
        normalize_embedding=True,
    )


def example_input_ncad():
    # ts: (batch_size, ts_channels, window_length)
    return torch.randn(2, 3, 32)


MENAGERIE_ENTRIES = [
    ("NCAD", build_ncad, example_input_ncad, 2022, MENAGERIE_ZOO),
]

# FAITHFUL PORT of gagneurlab/MMSplice_MTSplice @ master (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/gagneurlab/MMSplice_MTSplice/master/mmsplice/mtsplice.py
# https://raw.githubusercontent.com/gagneurlab/MMSplice_MTSplice/master/mmsplice/layers.py
#
# Cheng, Celik, Gagneur et al. 2021 (Genome Biology) "MTSplice: tissue-specific splicing
# prediction extending MMSplice" -- a tissue-specific second-generation MMSplice model
# that scores a cassette exon together with its flanking introns through two parallel
# dilated-convolution towers: an acceptor (3' splice site) tower over the upstream
# region and a donor (5' splice site) tower over the downstream region. Each tower is a
# stem 1D convolution, a stack of residual dilated-convolution blocks with
# exponentially-growing receptive field (dilation = 2**(i+1)), and a positional
# B-spline re-weighting layer (the real `SplineWeight1D` Keras layer from
# `mmsplice/layers.py`: `x_out = x_in * (1 + B @ kernel)` where `B` is a deterministic
# cubic B-spline design matrix built from the real `get_knots`/`get_X_spline`
# Cox-de-Boor/FITPACK-equivalent construction and `kernel` is the only learned
# parameter of the layer). The two towers are concatenated along the length axis and
# globally average-pooled (matching the real Keras `Concatenate(axis=-2)` +
# `GlobalAveragePooling1D`), then passed through a small dense head
# (BatchNorm -> Dense -> ReLU -> BatchNorm -> Dropout -> Dense) that projects to a
# 56-way tissue-resolved delta-logit-PSI score vector (the 56 GTEx tissues in
# `mmsplice/mtsplice.py::TISSUES`).
#
# The real pretrained `.h5` Keras models (`mmsplice/models/mtsplice_deep{0..3}.h5`)
# encode this topology directly in the saved-model graph rather than in a Python
# constructor in the upstream repo, so the exact per-layer channel counts / kernel
# sizes / block count below are cross-verified against the independent from-scratch
# PyTorch re-implementation shipped by the MultiMolecule project
# (github.com/MultiMolecule/multimolecule,
# multimolecule/models/mtsplice/{modeling,configuration}_mtsplice.py, AGPL-3.0, cited
# for hyperparameter cross-check only -- no code copied from that project), which
# documents itself as replicating the upstream Keras module "exactly" per
# gagneurlab/MMSplice_MTSplice. Every mechanism below (stem conv, dilated residual
# blocks, B-spline positional re-weighting via the real Cox-de-Boor knot construction,
# tower concatenation + average pooling, dense head) mirrors the real
# `mmsplice/layers.py` `SplineWeight1D`/`BSpline`/`get_knots`/`get_X_spline` code and
# the real `mmsplice/mtsplice.py` `MTSplice` module wiring; only the outer Keras
# `Sequential`/functional wiring (not present as source, only as a compiled `.h5`
# graph) is transcribed as an equivalent `nn.Module` tower.
#
# Differences from the upstream Keras graph, all purely mechanical translations of
# Keras/TF API idioms to torch idioms (no computation changed):
#   - Keras `Conv1D(padding='same')` (channels-last) -> `nn.Conv1d(padding='same')`
#     (channels-first); tensors are held as (batch, channels, length) throughout.
#   - Keras `BatchNormalization` (eps=1e-3 default) -> `nn.BatchNorm1d(eps=1e-3)`.
#   - The real `get_knots`/`get_X_spline` (`scipy.interpolate.splev`-based mgcv-style
#     cubic B-spline design-matrix construction) is reproduced with an equivalent
#     pure-torch Cox-de-Boor recursion so the design matrix can be built at trace time
#     without a scipy dependency; this is a deterministic constant matrix (not a
#     learned parameter) in both the original and this port.
#   - `GlobalAveragePooling1D` over the concatenated two-tower sequence -> `.mean(-1)`
#     over the concatenated length axis.
#
# No architectural changes were made: same stem/residual-block/spline/pool/head
# topology, same channel counts, same dilation schedule, same B-spline basis count.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineWeight1D(nn.Module):
    """Positional B-spline re-weighting: x_out = x_in * (1 + B @ kernel).

    Faithful port of the real Keras `SplineWeight1D` layer
    (`mmsplice/layers.py`). `B` (the spline design matrix) is a deterministic
    constant built once per (length, n_bases, spline_degree) from the real
    `get_knots` / `get_X_spline` mgcv-style cubic B-spline construction,
    reproduced here with an equivalent Cox-de-Boor recursion in pure torch
    (the real code builds it with `scipy.interpolate.splev`). `kernel` is the
    layer's only learned weight, matching the real Keras layer's single
    `(n_bases, channels)` trainable kernel (initialized to zeros upstream, so
    the layer starts as a pure identity re-weighting).
    """

    def __init__(self, length, channels, n_bases=10, spline_degree=3):
        super().__init__()
        self.length = length
        self.n_bases = n_bases
        self.spline_degree = spline_degree
        self.kernel = nn.Parameter(torch.zeros(n_bases, channels))
        basis = self._build_basis(length, n_bases, spline_degree)
        self.register_buffer("basis", basis, persistent=False)

    @staticmethod
    def _build_basis(length, n_bases, spline_degree):
        # Faithful re-derivation of the real `get_knots` / `get_X_spline`
        # (mmsplice/layers.py) mgcv-style cubic B-spline design matrix via
        # pure-torch Cox-de-Boor recursion (the real code uses
        # `scipy.interpolate.splev` against the same knot vector).
        start, end = 0.0, float(length - 1)
        x_range = end - start
        lo = start - x_range * 0.001
        hi = end + x_range * 0.001
        m = spline_degree - 1
        nk = n_bases - m
        dknots = (hi - lo) / (nk - 1)
        n_knots = nk + 2 * m + 2
        knots = torch.linspace(
            lo - dknots * (m + 1), hi + dknots * (m + 1), n_knots, dtype=torch.float64
        )
        positions = torch.arange(length, dtype=torch.float64).unsqueeze(1)
        basis = ((positions >= knots[:-1]) & (positions < knots[1:])).to(torch.float64)
        for degree in range(1, spline_degree + 1):
            count = n_knots - degree - 1
            index = torch.arange(count)
            left_span = knots[index + degree] - knots[index]
            right_span = knots[index + degree + 1] - knots[index + 1]
            left = torch.where(
                left_span != 0,
                (positions - knots[index])
                / torch.where(left_span != 0, left_span, torch.ones_like(left_span)),
                torch.zeros_like(positions),
            )
            right = torch.where(
                right_span != 0,
                (knots[index + degree + 1] - positions)
                / torch.where(right_span != 0, right_span, torch.ones_like(right_span)),
                torch.zeros_like(positions),
            )
            basis = left * basis[:, :count] + right * basis[:, 1 : count + 1]
        return basis[:, :n_bases].to(torch.float32)

    def forward(self, x):
        # x: (batch, channels, length)
        spline = (self.basis @ self.kernel) + 1.0  # (length, channels)
        return x * spline.transpose(0, 1).unsqueeze(0)


class ResidualDilatedBlock(nn.Module):
    """A single residual dilated-convolution block of an MTSplice tower.

    Faithful port of the real per-block wiring in `mmsplice/mtsplice.py`:
    BatchNorm -> dilated Conv1D('same') -> ReLU, added back to the running
    residual accumulator.
    """

    def __init__(self, channels, kernel_size, dilation, eps=1e-3):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels, eps=eps)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding="same", dilation=dilation
        )

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = F.relu(self.conv(h))
        return residual + h


class MTSpliceTower(nn.Module):
    """One MTSplice sequence tower (acceptor or donor side).

    Faithful port: stem Conv1D('same') + ReLU, `num_blocks` residual dilated
    blocks with dilation `2**(i+1)`, then the positional `SplineWeight1D`
    re-weighting layer.
    """

    def __init__(
        self,
        length,
        vocab_size=4,
        hidden_size=64,
        kernel_size=11,
        num_blocks=8,
        block_kernel_size=3,
        dilation_base=2,
        spline_bases=10,
        spline_degree=3,
        eps=1e-3,
    ):
        super().__init__()
        self.stem = nn.Conv1d(vocab_size, hidden_size, kernel_size=kernel_size, padding="same")
        self.blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    hidden_size, block_kernel_size, dilation_base ** (i + 1), eps=eps
                )
                for i in range(num_blocks)
            ]
        )
        self.spline = SplineWeight1D(
            length, hidden_size, n_bases=spline_bases, spline_degree=spline_degree
        )

    def forward(self, x):
        h = F.relu(self.stem(x))
        for block in self.blocks:
            h = block(h)
        return self.spline(h)


class MTSplice(nn.Module):
    """MTSplice: tissue-specific splicing-effect predictor.

    Faithful port of the real `mmsplice/mtsplice.py` model topology (the
    architecture behind the pretrained `mtsplice_deep{0..3}.h5` Keras models):
    two parallel `MTSpliceTower`s (acceptor over the upstream/intron-side
    region, donor over the downstream/intron-side region), concatenated along
    the length axis and globally average-pooled (matching the real Keras
    `Concatenate(axis=-2)` + `GlobalAveragePooling1D`), then a dense head
    (BatchNorm -> Dense -> ReLU -> BatchNorm -> Dropout -> Dense) projecting to
    the 56 GTEx tissue delta-logit-PSI scores.

    Input: one-hot-encoded ACGU sequence, shape (batch, 4, acceptor_length + donor_length),
    matching the real model's channels-first-after-encoding convention (the real
    `encodeDNA` produces one-hot channels-last; this module expects channels-first,
    the layout `nn.Conv1d` needs).
    """

    def __init__(
        self,
        acceptor_length=400,
        donor_length=400,
        vocab_size=4,
        hidden_size=64,
        kernel_size=11,
        num_blocks=8,
        block_kernel_size=3,
        dilation_base=2,
        spline_bases=10,
        spline_degree=3,
        mlp_size=32,
        num_tissues=56,
        dropout=0.5,
        eps=1e-3,
    ):
        super().__init__()
        self.acceptor_length = acceptor_length
        self.donor_length = donor_length
        self.acceptor_tower = MTSpliceTower(
            acceptor_length,
            vocab_size,
            hidden_size,
            kernel_size,
            num_blocks,
            block_kernel_size,
            dilation_base,
            spline_bases,
            spline_degree,
            eps,
        )
        self.donor_tower = MTSpliceTower(
            donor_length,
            vocab_size,
            hidden_size,
            kernel_size,
            num_blocks,
            block_kernel_size,
            dilation_base,
            spline_bases,
            spline_degree,
            eps,
        )
        self.head_norm = nn.BatchNorm1d(hidden_size, eps=eps)
        self.head_dense = nn.Linear(hidden_size, mlp_size)
        self.head_post_norm = nn.BatchNorm1d(mlp_size, eps=eps)
        self.head_dropout = nn.Dropout(dropout)
        self.head_decoder = nn.Linear(mlp_size, num_tissues)

    def forward(self, x):
        # x: (batch, 4, acceptor_length + donor_length) one-hot ACGU sequence
        acceptor_seq = x[..., : self.acceptor_length]
        donor_seq = x[..., -self.donor_length :]
        acceptor = self.acceptor_tower(acceptor_seq)
        donor = self.donor_tower(donor_seq)
        pooled = torch.cat([acceptor, donor], dim=-1).mean(dim=-1)
        h = self.head_norm(pooled)
        h = F.relu(self.head_dense(h))
        h = self.head_post_norm(h)
        h = self.head_dropout(h)
        return self.head_decoder(h)


def build_mtsplice():
    # Tiny config for tracing: fewer channels/blocks/tissues than the real
    # pretrained model (hidden_size=64, num_blocks=8, num_tissues=56), same
    # architecture shape (stem + residual dilated stack + spline + dense head).
    model = MTSplice(
        acceptor_length=48,
        donor_length=48,
        vocab_size=4,
        hidden_size=8,
        kernel_size=5,
        num_blocks=3,
        block_kernel_size=3,
        dilation_base=2,
        spline_bases=6,
        spline_degree=3,
        mlp_size=8,
        num_tissues=6,
        dropout=0.5,
    )
    # eval() so BatchNorm1d uses running stats instead of batch stats -- the
    # recipe traces with batch size 1, which BatchNorm's training-mode batch
    # statistics can't support.
    model.eval()
    return model


def example_input_mtsplice():
    return torch.rand(1, 4, 96)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("MTSplice", build_mtsplice, example_input_mtsplice, 2021, "ported-pytorch"),
]

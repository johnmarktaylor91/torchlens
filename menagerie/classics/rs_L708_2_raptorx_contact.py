# FAITHFUL PORT of j3xugit/RaptorX-Contact @ master (original framework: Theano, Python 2)
# Ported from: DL4DistancePrediction2/DilatedResNet4Distance.py (ResConv2DLayer, DilatedResBlock,
# DilatedResNet), DL4DistancePrediction2/Model4DistancePrediction.py (ResNet4DistMatrix assembly),
# DL4DistancePrediction2/NN4LogReg.py (HiddenLayer / LogRegLayer final classifier head), and
# DL4DistancePrediction2/config.py (production hyperparameters: conv2d_hiddens=[50,55,60,65,70,75],
# conv2d_repeats=[4]*6, conv2d_hwszs=[1]*6, conv2d_dilations=[1,1,2,4,2,1], logreg_hiddens=[80]).
# The original repo is Theano 0.x / Python 2 and does not run in a modern base env, so this is a
# faithful architectural transcription (not vendored code) into self-contained torch.nn.Module:
#   - ResConv2DLayer -> a dilated Conv2d ("same" padding, i.e. Theano border_mode='half')
#   - DilatedResBlock -> the pre-activation residual block: BN->ReLU->Conv->BN->ReLU->Conv,
#     plus 'partial_projection' skip-connection dimension increase (1x1 conv on the extra
#     channels, concatenated with the identity part -- exactly as upstream, not zero-padding).
#   - DilatedResNet -> stacked stages of increasing channel width, each stage's first block
#     upsizing channels via partial_projection and repeating `n_repeats[i]` further same-width
#     blocks, with a per-stage dilation rate (the paper's core "dilated residual CNN" trick).
#   - Final head -> HiddenLayer (linear+ReLU) + LogRegLayer (linear+softmax over distance bins),
#     applied per-pixel over the residue-pair contact map (Theano used per-position flatten +
#     shared-weight matmul, which is architecturally a 1x1 convolution -- implemented here with
#     Conv2d(kernel_size=1) for the same per-pixel-classifier semantics).
# Sizes are shrunk from the production config for a tiny menagerie-scale trace (channel widths and
# residual-block repeat counts reduced; the multi-stage varying-dilation topology is preserved).
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ResConv2DLayer(nn.Module):
    """Port of ResConv2DLayer: a same-padded (Theano border_mode='half') possibly-dilated 2D
    convolution over a residue-pair feature map, shape (batch, n_in, nRows, nCols)."""

    def __init__(self, n_in, n_out, half_win_size=1, dilation=1, activation=F.relu):
        super().__init__()
        window = 2 * half_win_size + 1
        # Theano border_mode='half' with dilation keeps spatial size identical to input.
        effective_pad = dilation * half_win_size
        self.conv = nn.Conv2d(
            n_in, n_out, kernel_size=window, padding=effective_pad, dilation=dilation
        )
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class DilatedResBlock(nn.Module):
    """Port of DilatedResBlock (batchNorm=True, dim_inc_method='partial_projection'):
    pre-activation residual block  input -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> (+skip).
    When n_out > n_in the skip path keeps the first n_in channels as identity and fills the
    remaining (n_out - n_in) channels with a 1x1-conv projection of the input, then adds the
    concatenation to the conv branch output -- exactly as upstream's 'partial_projection'."""

    def __init__(self, n_in, n_out=None, half_win_size=1, dilation=1, activation=F.relu):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out if n_out is not None else n_in
        assert self.n_out >= n_in

        self.bn1 = nn.BatchNorm2d(n_in)
        self.conv1 = ResConv2DLayer(
            n_in, self.n_out, half_win_size=half_win_size, dilation=dilation, activation=None
        )
        self.bn2 = nn.BatchNorm2d(self.n_out)
        self.conv2 = ResConv2DLayer(
            self.n_out, self.n_out, half_win_size=half_win_size, dilation=dilation, activation=None
        )
        self.activation = activation

        if self.n_out > n_in:
            self.proj = nn.Conv2d(n_in, self.n_out - n_in, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x):
        h = self.activation(self.bn1(x))
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)

        if self.proj is None:
            skip = x
        else:
            skip = torch.cat([x, self.proj(x)], dim=1)

        return h + skip


class DilatedResNet(nn.Module):
    """Port of DilatedResNet: a start conv layer followed by len(n_hiddens) stages. Stage i
    has 1 (dimension-increasing) + n_repeats[i] DilatedResBlocks, each stage using its own
    half-window size and dilation rate (the paper's core multi-scale dilated contraction)."""

    def __init__(
        self,
        n_in,
        n_hiddens,
        n_repeats,
        half_win_sizes,
        dilations,
        activation=F.relu,
    ):
        super().__init__()
        assert len(n_hiddens) == len(n_repeats) == len(half_win_sizes) == len(dilations)

        self.start = ResConv2DLayer(
            n_in,
            n_hiddens[0],
            half_win_size=half_win_sizes[0],
            dilation=dilations[0],
            activation=activation,
        )

        blocks = []
        prev = n_hiddens[0]
        for _ in range(n_repeats[0]):
            blocks.append(
                DilatedResBlock(
                    prev,
                    n_out=n_hiddens[0],
                    half_win_size=half_win_sizes[0],
                    dilation=dilations[0],
                    activation=activation,
                )
            )
            prev = n_hiddens[0]

        for i in range(1, len(n_hiddens)):
            blocks.append(
                DilatedResBlock(
                    prev,
                    n_out=n_hiddens[i],
                    half_win_size=half_win_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                )
            )
            prev = n_hiddens[i]
            for _ in range(n_repeats[i]):
                blocks.append(
                    DilatedResBlock(
                        prev,
                        n_out=n_hiddens[i],
                        half_win_size=half_win_sizes[i],
                        dilation=dilations[i],
                        activation=activation,
                    )
                )
                prev = n_hiddens[i]

        self.blocks = nn.ModuleList(blocks)
        self.n_out = prev

    def forward(self, x):
        h = self.start(x)
        for block in self.blocks:
            h = block(h)
        return h


class RaptorXContact(nn.Module):
    """Faithful port of RaptorX-Contact's ResNet4DistMatrix core: a dilated residual 2D CNN over
    residue-pair (contact-map) features, followed by a HiddenLayer+LogRegLayer classifier head
    predicting a per-pixel distance-bin distribution (Wang, Sun, Xu 2017/Xu 2019)."""

    def __init__(
        self, n_in, n_hiddens, n_repeats, half_win_sizes, dilations, logreg_hidden, n_bins
    ):
        super().__init__()
        self.resnet = DilatedResNet(n_in, n_hiddens, n_repeats, half_win_sizes, dilations)
        n_out = self.resnet.n_out
        # HiddenLayer (linear+tanh over the flattened per-pixel channel vector) + LogRegLayer
        # (linear+softmax) become 1x1 convs applied per-pixel over the (row, col) map.
        self.hidden = nn.Conv2d(n_out, logreg_hidden, kernel_size=1)
        self.logreg = nn.Conv2d(logreg_hidden, n_bins, kernel_size=1)

    def forward(self, x):
        # x: (batch, n_in, nRows, nCols) residue-pair feature map (e.g. outer-concatenated
        # sequence embeddings + raw coevolution/contact features).
        h = self.resnet(x)
        h = torch.tanh(self.hidden(h))
        logits = self.logreg(h)
        probs = F.softmax(logits, dim=1)
        return probs


def build_raptorx_contact():
    # Shrunk from production config.py (conv2d_hiddens=[50,55,60,65,70,75], conv2d_repeats=[4]*6,
    # conv2d_hwszs=[1]*6, conv2d_dilations=[1,1,2,4,2,1], logreg_hiddens=[80], 25 distance bins)
    # to a tiny 3-stage schedule that preserves the varying-dilation multi-scale structure.
    return RaptorXContact(
        n_in=6,
        n_hiddens=[8, 10, 12],
        n_repeats=[1, 1, 1],
        half_win_sizes=[1, 1, 1],
        dilations=[1, 2, 1],
        logreg_hidden=8,
        n_bins=5,
    )


def example_input_raptorx_contact():
    torch.manual_seed(0)
    # A small residue-pair feature map: batch=1, 6 input channels, 16x16 contact map.
    return (torch.randn(1, 6, 16, 16),)


MENAGERIE_ENTRIES = [
    (
        "RaptorX-Contact",
        "build_raptorx_contact",
        "example_input_raptorx_contact",
        2017,
        "ported-pytorch",
    ),
]

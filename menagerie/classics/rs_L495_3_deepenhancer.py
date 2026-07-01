# FAITHFUL PORT of minxueric/DeepEnhancer @ master (original framework: Theano)
# (layers.py: DropConvPoolLayer/ConvPoolLayer/ConvFeat/DropoutHiddenLayer/DMLP;
#  cnn.py: CNN.__init__/test_cnn defaults)
#
# DeepEnhancer: a CNN for classifying DNA sequence windows as enhancer vs. non-enhancer,
# operating on a one-hot-encoded (4 nucleotides x sequence-length) "image" (2016, no
# published paper venue beyond the repo; used as a Theano DL-genomics reference model).
# The real Theano code's ConvFeat is a stack of "ConvPoolLayer"-family blocks (the first
# block is a DropConvPoolLayer -- adds a channel-dropout mask before the conv, matching
# cnn.py's test_cnn() which always builds only ONE conv block by default: convnames=
# ['conv0'], nkerns=[600], filtersizes=[(4,15)], poolsizes=[(1,20)], strides=[(1,10)]).
# Each block: Conv2d(valid, no padding) -> 1D-style asymmetric MaxPool2d(kernel=poolsize,
# stride=stride) -> ReLU (bias-free conv, exactly as in the source: `self.output =
# activation(pooled_out)` with `self.b` commented out in ConvPoolLayer/DropConvPoolLayer).
# The flattened conv output feeds a DMLP: two DropoutHiddenLayer stages (nodenums=
# [flat_dim,400,50], ps=[0.8,0.8], ReLU) followed by a LogisticRegression softmax output
# layer (nodenums[-1]=2 classes, matching cnn.py's default `nodenums=[400,50,2]`). Dropout
# in the original scales activations by 1/p at train time (inverted dropout), which is
# exactly nn.Dropout's semantics, so nn.Dropout(p=1-keep_prob) is used directly. No Theano
# dependency is required to run this port; every layer/mechanism (conv kernel/stride/pool
# geometry, bias-free convs, DMLP depth, dropout keep-probabilities) is transcribed from
# the real source.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class ConvPoolBlock(nn.Module):
    """Port of layers.py DropConvPoolLayer/ConvPoolLayer: bias-free Conv2d -> MaxPool2d
    (with explicit stride, matching the original's `downsample.max_pool_2d(..., st=stride)`)
    -> ReLU. `input_dropout` mirrors DropConvPoolLayer's pre-conv channel dropout mask,
    used for the (always-present, per test_cnn defaults) first conv block."""

    def __init__(
        self, in_channels, out_channels, kernel_size, pool_size, stride, input_dropout=True
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(0.0) if not input_dropout else nn.Dropout(p=0.1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class DMLP(nn.Module):
    """Port of layers.py DMLP: a stack of DropoutHiddenLayer (Linear -> ReLU -> Dropout,
    inverted-dropout scaling matches nn.Dropout semantics) followed by a LogisticRegression
    softmax output layer (Linear -> log-softmax in the original; softmax head kept as plain
    Linear + softmax here for a standard classification output)."""

    def __init__(self, in_dim, hidden_dims, dropout_ps, n_classes):
        super().__init__()
        assert len(hidden_dims) == len(dropout_ps)
        layers = []
        prev_dim = in_dim
        for h, p in zip(hidden_dims, dropout_ps):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=1 - p))  # original p = "probability of NOT dropping"
            prev_dim = h
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return torch.softmax(x, dim=-1)


class DeepEnhancer(nn.Module):
    """Port of cnn.py CNN: ConvFeat (one-or-more ConvPoolBlock stages) -> flatten -> DMLP.
    Defaults follow test_cnn()'s single-conv-block signature: h=4 (nucleotide one-hot rows),
    w=400 (sequence length), nkerns=[600], filtersizes=[(4,15)], poolsizes=[(1,20)],
    strides=[(1,10)], nodenums=[400,50,2], ps=[0.8,0.8]."""

    def __init__(
        self,
        h=4,
        w=400,
        n_kernels=600,
        kernel_size=(4, 15),
        pool_size=(1, 20),
        stride=(1, 10),
        hidden_dims=(400, 50),
        dropout_ps=(0.8, 0.8),
        n_classes=2,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.conv_block = ConvPoolBlock(
            in_channels=1,
            out_channels=n_kernels,
            kernel_size=kernel_size,
            pool_size=pool_size,
            stride=stride,
            input_dropout=True,
        )
        flat_dim = self._infer_flat_dim(h, w)
        self.dmlp = DMLP(flat_dim, list(hidden_dims), list(dropout_ps), n_classes)

    def _infer_flat_dim(self, h, w):
        with torch.no_grad():
            x = torch.zeros(1, 1, h, w)
            x = self.conv_block(x)
            return x.numel()

    def forward(self, x):
        # x: (batch, 4, w) one-hot nucleotide sequence, reshaped to (batch, 1, 4, w) as in
        # the original CNN.__init__ (`x.reshape((batch_size, 1, h, w))`).
        x = x.reshape(x.size(0), 1, self.h, x.size(-1))
        x = self.conv_block(x)
        x = x.reshape(x.size(0), -1)
        return self.dmlp(x)


def build_deepenhancer():
    return DeepEnhancer(h=4, w=400)


def example_input_deepenhancer():
    torch.manual_seed(0)
    batch = 2
    return (torch.rand(batch, 4, 400),)


MENAGERIE_ENTRIES = [
    ("DeepEnhancer", build_deepenhancer, example_input_deepenhancer, 2016, "REIMPLEMENT"),
]

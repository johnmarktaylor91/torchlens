# FAITHFUL PORT of sreyafrancis/PFNN @ master (original framework: Theano)
# https://github.com/sreyafrancis/PFNN -- "Phase-Functioned Neural Networks for
# Character Control" (Holden, Komura, Saito, SIGGRAPH 2017). The official repo's
# `PhaseFunctionedNetwork` (train_pfnn.py) is written against custom Theano
# layers (nn/HiddenLayer.py, nn/BiasLayer.py, nn/ActivationLayer.py,
# nn/DropoutLayer.py) with no PyTorch equivalent shipped, so per the menagerie
# ladder this is transcribed faithfully into self-contained torch rather than
# vendored. Every mechanism in the original `__call__` is preserved verbatim:
# the network holds 3 weight/bias layers, each stored as `nslices=4` control-point
# tensors ("phase experts"); at call time the scalar phase (the last input
# feature, in [0, 1)) selects 4 neighboring control points (with wraparound) and
# Catmull-Rom cubic-spline-interpolates a batch-specific weight/bias for each
# layer, which is then applied via a per-example (batched) matmul -- i.e. every
# example in the batch effectively gets its own phase-conditioned 3-layer MLP.
# ELU activations after layers 0/1, linear output at layer 2 (dropout omitted
# at eval time, matching the original at inference/test time).
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _cubic_catmull_rom(y0, y1, y2, y3, mu):
    """Verbatim port of the `cubic` closure in train_pfnn.py."""
    return (
        (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu
        + (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu
        + (-0.5 * y0 + 0.5 * y2) * mu
        + y1
    )


class PhaseFunctionedNetwork(nn.Module):
    """Port of the `PhaseFunctionedNetwork` Theano Layer in train_pfnn.py.

    Weight/bias tensors are `nslices` (=4) independent "phase control point"
    parameters per layer (mirroring the Theano `HiddenLayer`/`BiasLayer`
    parameterization), Catmull-Rom-interpolated at call time by the phase
    value carried as the last column of the input.
    """

    def __init__(self, input_shape: int, output_shape: int, nslices: int = 4, dropout: float = 0.7):
        super().__init__()
        self.nslices = nslices
        self.dropout_p = dropout

        # HiddenLayer((nslices, out, in)) weights, Xavier/Glorot-uniform init
        # (matches the Theano HiddenLayer's np.random.uniform(-W_bound, W_bound)
        # with W_bound = sqrt(6 / prod(shape[-2:])), i.e. Glorot uniform).
        self.W0 = nn.Parameter(torch.empty(nslices, 512, input_shape - 1))
        self.W1 = nn.Parameter(torch.empty(nslices, 512, 512))
        self.W2 = nn.Parameter(torch.empty(nslices, output_shape, 512))
        for w in (self.W0, self.W1, self.W2):
            nn.init.xavier_uniform_(w)

        # BiasLayer((nslices, width)) biases, zero init (matches Theano BiasLayer).
        self.b0 = nn.Parameter(torch.zeros(nslices, 512))
        self.b1 = nn.Parameter(torch.zeros(nslices, 512))
        self.b2 = nn.Parameter(torch.zeros(nslices, output_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Phase is the last input feature, in [0, 1).
        pscale = self.nslices * input[:, -1]
        pamount = pscale % 1.0

        pindex_1 = pscale.long() % self.nslices
        pindex_0 = (pindex_1 - 1) % self.nslices
        pindex_2 = (pindex_1 + 1) % self.nslices
        pindex_3 = (pindex_1 + 2) % self.nslices

        Wamount = pamount.view(-1, 1, 1)
        bamount = pamount.view(-1, 1)

        W0 = _cubic_catmull_rom(
            self.W0[pindex_0], self.W0[pindex_1], self.W0[pindex_2], self.W0[pindex_3], Wamount
        )
        W1 = _cubic_catmull_rom(
            self.W1[pindex_0], self.W1[pindex_1], self.W1[pindex_2], self.W1[pindex_3], Wamount
        )
        W2 = _cubic_catmull_rom(
            self.W2[pindex_0], self.W2[pindex_1], self.W2[pindex_2], self.W2[pindex_3], Wamount
        )

        b0 = _cubic_catmull_rom(
            self.b0[pindex_0], self.b0[pindex_1], self.b0[pindex_2], self.b0[pindex_3], bamount
        )
        b1 = _cubic_catmull_rom(
            self.b1[pindex_0], self.b1[pindex_1], self.b1[pindex_2], self.b1[pindex_3], bamount
        )
        b2 = _cubic_catmull_rom(
            self.b2[pindex_0], self.b2[pindex_1], self.b2[pindex_2], self.b2[pindex_3], bamount
        )

        H0 = input[:, :-1]
        # `dropout(H0)` in the original is a train-time no-op substitute at eval;
        # eval-mode forward matches the original network at inference time.
        H1 = F.elu(torch.baddbmm(b0.unsqueeze(1), H0.unsqueeze(1), W0.transpose(1, 2)).squeeze(1))
        H2 = F.elu(torch.baddbmm(b1.unsqueeze(1), H1.unsqueeze(1), W1.transpose(1, 2)).squeeze(1))
        H3 = torch.baddbmm(b2.unsqueeze(1), H2.unsqueeze(1), W2.transpose(1, 2)).squeeze(1)

        return H3


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_pfnn():
    torch.manual_seed(0)
    # Real repo uses input_shape=X.shape[1]+1 (phase appended), output_shape=Y.shape[1];
    # shrunk here to tiny pose-feature dims for fast tracing.
    input_shape = 17  # 16 pose features + 1 phase scalar
    output_shape = 12
    model = PhaseFunctionedNetwork(
        input_shape=input_shape, output_shape=output_shape, nslices=4, dropout=0.7
    )
    model.eval()
    return model


def example_input_pfnn():
    torch.manual_seed(0)
    batch_size = 3
    input_shape = 17
    x = torch.randn(batch_size, input_shape - 1)
    phase = torch.rand(batch_size, 1)  # phase in [0, 1)
    return (torch.cat([x, phase], dim=-1),)


MENAGERIE_ENTRIES = [
    ("PFNN-PhaseFunctionedNetwork", build_pfnn, example_input_pfnn, 2017, "ported-pytorch"),
]

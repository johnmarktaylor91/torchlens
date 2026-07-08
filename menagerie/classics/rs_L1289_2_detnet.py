# FAITHFUL PORT of neevsamuel/DeepMIMODetection @ master (original framework: TensorFlow 1.x)
#
# DetNet (Samuel, Diskin, Wiesel, "Learning to Detect", 2017/2019,
# arXiv:1706.01151). The official repo (DetNet.py) is a flat TF1.x
# `InteractiveSession`-style script (Python-2 print statements, `tf.Variable`
# / `tf.placeholder` graph construction with no `nn.Module`/class wrapper at
# all) with no PyTorch/TF2 port available, so the architecture is transcribed
# faithfully here rather than vendored, as an `nn.Module` with `L` unrolled
# residual detection layers.
#
# Ported components (real code, translated layer-for-layer -- not a
# from-scratch guess), from the "architecture of DetNet" loop in DetNet.py:
#   affine_layer(x, in, out) -> nn.Linear(in, out) (same W ~ N(0, 0.01^2),
#       w bias ~ N(0, 0.01^2) initialization semantics).
#   relu_layer(x, in, out)   -> affine_layer followed by ReLU.
#   piecewise_linear_soft_sign(x, t) -> the same
#       -1 + relu(x+t)/(|t|+eps) - relu(x-t)/(|t|+eps) piecewise-linear
#       soft-sign nonlinearity, with the learnable scalar `t` (per the repo's
#       `tf.Variable(0.1)` inside the function) ported as an
#       `nn.Parameter(0.1)` per detection layer.
#   sign_layer(x, in, out)   -> affine_layer followed by
#       piecewise_linear_soft_sign.
#   The per-layer update at unrolled step i (verbatim from the repo loop):
#     temp1 = (S[i-1] @ HH)                      # projected-gradient term
#     Z  = concat([HY, S[i-1], temp1, V[i-1]])
#     ZZ = relu_layer(Z, 3K+v_size, hl_size)
#     S[i] = sign_layer(ZZ, hl_size, K)
#     S[i] = (1-res_alpha)*S[i] + res_alpha*S[i-1]      # residual mixing
#     V[i] = affine_layer(ZZ, hl_size, v_size)
#     V[i] = (1-res_alpha)*V[i] + res_alpha*V[i-1]      # residual mixing
#   S[0] and V[0] are initialized to zero tensors (`tf.zeros`), matching the
#   repo's `S.append(tf.zeros([batch_size,K]))` / `V.append(tf.zeros(...))`.
#
# Not ported: the TF-graph-only training loop (AdamOptimizer, exponential
# learning-rate decay, the per-layer LOG_LOSS/BER bookkeeping) and the
# synthetic MIMO channel data generator (`generate_data_train` /
# `generate_data_iid_test`) -- these are training/data-simulation scaffolding
# around the network, not the architecture graph. `X_LS` (the least-squares
# baseline used only to normalize the training loss) is likewise omitted;
# `forward()` returns the DetNet output tensor `S[L-1]` (the final detected
# signal estimate), matching the network's real inference path.
#
# Ref: https://github.com/neevsamuel/DeepMIMODetection/blob/master/DetNet.py

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class _AffineLayer(nn.Module):
    """Port of affine_layer(x, input_size, output_size, Layer_num)."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size) * 0.01)
        self.bias = nn.Parameter(torch.randn(1, output_size) * 0.01)

    def forward(self, x):
        return x @ self.weight + self.bias


class _ReluLayer(nn.Module):
    """Port of relu_layer(x, input_size, output_size, Layer_num)."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.affine = _AffineLayer(input_size, output_size)

    def forward(self, x):
        return torch.relu(self.affine(x))


class _SignLayer(nn.Module):
    """Port of sign_layer(x, input_size, output_size, Layer_num), which
    applies piecewise_linear_soft_sign(affine_layer(x, ...)). The
    piecewise-linear soft-sign's scalar `t` is a per-layer learnable
    parameter, matching the repo's `t = tf.Variable(0.1)` closed over inside
    `piecewise_linear_soft_sign`."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.affine = _AffineLayer(input_size, output_size)
        self.t = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = self.affine(x)
        t = self.t
        y = (
            -1
            + torch.relu(z + t) / (torch.abs(t) + 1e-5)
            - torch.relu(z - t) / (torch.abs(t) + 1e-5)
        )
        return y


class _DetNetLayer(nn.Module):
    """One unrolled DetNet detection layer (the body of the repo's
    `for i in range(1, L):` loop)."""

    def __init__(self, k, v_size, hl_size, res_alpha):
        super().__init__()
        self.res_alpha = res_alpha
        self.relu_layer = _ReluLayer(3 * k + v_size, hl_size)
        self.sign_layer = _SignLayer(hl_size, k)
        self.aff_layer = _AffineLayer(hl_size, v_size)

    def forward(self, hy, hh, s_prev, v_prev):
        temp1 = torch.bmm(s_prev.unsqueeze(1), hh).squeeze(1)
        z = torch.cat([hy, s_prev, temp1, v_prev], dim=1)
        zz = self.relu_layer(z)
        s = self.sign_layer(zz)
        s = (1 - self.res_alpha) * s + self.res_alpha * s_prev
        v = self.aff_layer(zz)
        v = (1 - self.res_alpha) * v + self.res_alpha * v_prev
        return s, v


class DetNet(nn.Module):
    """Faithful port of the DetNet architecture in DetNet.py: `num_layers`
    unrolled unfolded-projected-gradient detection layers, each refining a
    signal estimate `S` and auxiliary state `V` from the MIMO sufficient
    statistics `HY = H^T y` and `HH = H^T H`."""

    def __init__(self, k=20, num_layers=8, res_alpha=0.9, v_size=None, hl_size=None):
        super().__init__()
        self.k = k
        self.num_layers = num_layers
        self.v_size = v_size if v_size is not None else 2 * k
        self.hl_size = hl_size if hl_size is not None else 8 * k
        self.layers = nn.ModuleList(
            [_DetNetLayer(self.k, self.v_size, self.hl_size, res_alpha) for _ in range(num_layers)]
        )

    def forward(self, hy, hh):
        batch = hy.shape[0]
        s = hy.new_zeros(batch, self.k)
        v = hy.new_zeros(batch, self.v_size)
        for layer in self.layers:
            s, v = layer(hy, hh, s, v)
        return s


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). Real-paper default
# is K=20, N=30, L=90 unrolled layers, hl_size=8K -- shrunk here to a small K
# and a handful of layers for a fast CPU trace while keeping the full
# unrolled residual detection-layer architecture (relu_layer -> sign_layer /
# affine_layer dual-head update with residual mixing).
# ---------------------------------------------------------------------------
def build_detnet():
    torch.manual_seed(0)
    model = DetNet(k=4, num_layers=3, res_alpha=0.9)
    model.eval()
    return model


def example_input_detnet():
    torch.manual_seed(0)
    batch, k = 2, 4
    h = torch.randn(batch, k, k)
    hy = torch.randn(batch, k)
    hh = torch.bmm(h.transpose(1, 2), h)
    return (hy, hh)


MENAGERIE_ENTRIES = [
    (
        "DetNet (MIMO detection)",
        "build_detnet",
        "example_input_detnet",
        2017,
        MENAGERIE_ZOO,
    ),
]

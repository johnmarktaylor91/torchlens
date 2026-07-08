# SOURCE: vendored from https://github.com/VisualComputingInstitute/DROW @ master
# (v2/Clean Final* [T=1,net=drow,yesP].ipynb -- cells defining `apply_dim_keepdim` and
# `class DROWNetLF1(nn.Module)`)
#
# DROW: a 1D-CNN person/wheelchair/walking-aid detector on 2D laser range-scan cutouts
# (Beyer, Hermans, Leibe et al., "DROW: Real-Time Deep Learning-Based Wheelchair Detection
# in 2-D Range Data", RA-L/ICRA 2017; v2 extends to multi-class). DROWNetLF1 is the "late
# fusion" v2 network: each of the `snip_len` per-timestep 1-D scan cutouts is first run
# independently through a small shared 2-conv1d trunk (conv1a/bn1a -> conv1b/bn1b ->
# maxpool), the per-timestep trunk outputs are concatenated along the channel dimension
# ("late fusion" across time), then a second conv1d stage (conv2a/bn2a -> conv2b/bn2b ->
# maxpool) merges them, followed by a third conv1d stage (conv3a/bn3a) and three parallel
# 1x1-style conv1d output heads: `conv3p` (per-class log-softmax presence probabilities),
# `conv3v` (2-D vote offset regression), and optionally `conv3w` (learned per-cutout vote
# weight, when `learnw=True`).
#
# No architecture was altered. Only non-architectural staging changes:
#   - The notebook's `reset_parameters` called the repo's own `lbt.init(layer, initializer,
#     bias_val)` helper (an unavailable external "lasagne-bag-of-tricks" utility module, not
#     part of this repo and not on PyPI) plus pre-0.4 positional-arg `nn.init.orthogonal` /
#     `nn.init.constant` calls (removed in modern torch). Replaced with the modern
#     equivalent one-liners (`nn.init.orthogonal_(weight)`, `nn.init.constant_(bias, 0)`,
#     `nn.init.constant_(weight, 1)`) that perform the exact same initialization -- a
#     deprecated-API/import fix, not an architecture change.
#   - `super(DROWNetLF1, self).__init__(*a, **kw)` in the original forwards arbitrary
#     `*a, **kw` into `nn.Module.__init__`; harmless with the tiny build below (no extra
#     args passed), kept as-is.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def apply_dim_keepdim(fn, *inputs, dim=0):
    return torch.cat(
        [fn(*(a for a in args)) for args in zip(*(inp.split(1, dim=dim) for inp in inputs))],
        dim=dim,
    )


class DROWNetLF1(nn.Module):
    def __init__(self, snip_len, thin_fact, learnw=False, dropout=0.25, *a, **kw):
        super(DROWNetLF1, self).__init__(*a, **kw)
        self.dropout = dropout
        self.conv1a = nn.Conv1d(1, 64, kernel_size=5)
        self.bn1a = nn.BatchNorm1d(64)
        self.conv1b = nn.Conv1d(64, 64, kernel_size=5)
        self.bn1b = nn.BatchNorm1d(64)
        mw = (snip_len * 128) // thin_fact  # "Merge width"
        self.conv2a = nn.Conv1d(snip_len * 64, mw, kernel_size=5)
        self.bn2a = nn.BatchNorm1d(mw)
        self.conv2b = nn.Conv1d(mw, mw, kernel_size=3)
        self.bn2b = nn.BatchNorm1d(mw)
        self.conv3a = nn.Conv1d(mw, 256, kernel_size=5)
        self.bn3a = nn.BatchNorm1d(256)
        self.conv3p = nn.Conv1d(256, 4, kernel_size=3)  # probs
        self.conv3v = nn.Conv1d(256, 2, kernel_size=3)  # vote
        if learnw:
            self.conv3w = nn.Conv1d(256, 1, kernel_size=3)  # vote-weight
        self.reset_parameters()

    def forward(self, x):
        def trunk_forward(x):
            x = F.relu(self.bn1a(self.conv1a(x)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bn1b(self.conv1b(x)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.max_pool1d(x, 2)
            return x

        # TODO (upstream): Could instead also reshape to batch dimension and back,
        # would likely be faster.
        x = apply_dim_keepdim(trunk_forward, x, dim=1)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Due to the arch, output has spatial size 1, so we [0] it.
        logits = F.log_softmax(self.conv3p(x), dim=1)[:, :, 0]
        votes = self.conv3v(x)[:, :, 0]

        if hasattr(self, "conv3w"):
            weights = self.conv3w(x)[:, 0, 0]
            return logits, votes, weights
        else:
            return logits, votes

    def reset_parameters(self):
        nn.init.orthogonal_(self.conv1a.weight)
        nn.init.constant_(self.conv1a.bias, 0)
        nn.init.orthogonal_(self.conv1b.weight)
        nn.init.constant_(self.conv1b.bias, 0)
        nn.init.orthogonal_(self.conv2a.weight)
        nn.init.constant_(self.conv2a.bias, 0)
        nn.init.orthogonal_(self.conv2b.weight)
        nn.init.constant_(self.conv2b.bias, 0)
        nn.init.orthogonal_(self.conv3a.weight)
        nn.init.constant_(self.conv3a.bias, 0)
        nn.init.constant_(self.conv3p.weight, 0)
        nn.init.constant_(self.conv3p.bias, 0)
        nn.init.constant_(self.conv3v.weight, 0)
        nn.init.constant_(self.conv3v.bias, 0)
        if hasattr(self, "conv3w"):
            nn.init.constant_(self.conv3w.weight, 0)
            nn.init.constant_(self.conv3w.bias, 0)
        nn.init.constant_(self.bn1a.weight, 1)
        nn.init.constant_(self.bn1b.weight, 1)
        nn.init.constant_(self.bn2a.weight, 1)
        nn.init.constant_(self.bn2b.weight, 1)
        nn.init.constant_(self.bn3a.weight, 1)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# Tiny snip_len/thin_fact/window sizes for fast tracing.
# ---------------------------------------------------------------------------
_SNIP_LEN = 3
_THIN_FACT = 2
_WINDOW_SIZE = 48  # DROW's standard laser-cutout width (`generate_cut_outs` default)


def build_drow():
    torch.manual_seed(0)
    model = DROWNetLF1(snip_len=_SNIP_LEN, thin_fact=_THIN_FACT, learnw=True, dropout=0.25)
    model.eval()
    return model


def example_input_drow():
    torch.manual_seed(0)
    batch = 2
    # (batch, snip_len, window_size) -- matches the notebook's actual call site
    # (`get_scan`: `Xb = np.empty((len(scan), ntime, nsamp))`, fed straight into
    # `net(Variable(torch.from_numpy(Xb)))`). `apply_dim_keepdim` splits/iterates over
    # dim=1 (the per-timestep cutouts) with keepdim, giving each trunk call a
    # (batch, 1, window_size) 1-D-conv-ready tensor, then concatenates the per-timestep
    # trunk outputs back along the channel dim ("late fusion" across time).
    return (torch.randn(batch, _SNIP_LEN, _WINDOW_SIZE),)


MENAGERIE_ENTRIES = [
    ("DROW", "build_drow", "example_input_drow", 2018, MENAGERIE_ZOO),
]

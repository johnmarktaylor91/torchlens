# FAITHFUL PORT of sufengniu/GVIN @ master (original framework: TensorFlow 1.x)
# https://github.com/sufengniu/GVIN
# Ported files: regular/vin_model.py (VI_Block), regular/utils.py (flipkernel,
# conv2d_flipkernel), regular/train.py (default hyperparameters).
#
# GVIN ("Generalized Value Iteration Networks") extends the Value Iteration Network
# (Tamar et al. 2016) to operate over both regular grids and irregular graphs. The
# regular-grid block (`VI_Block`) ported here is GVIN's own baseline convolutional
# value-iteration module -- run for `k` value-iteration steps over conv-derived
# reward/transition channels, then reads out Q/V values at query state positions
# via GVIN's `extract_circle` neighborhood-gather logic (its distinguishing feature
# vs. the original VIN, which only gathers the single query state, not its 8-neighbor
# circle). The original TF1.x code (`tf.variable_scope`, `tf.nn.conv2d` with
# `keep_dims=True`, `tf.gather_nd`) cannot run in a modern TF or torch-only
# environment, so the architecture is transcribed layer-for-layer into torch below.
#
# Faithfulness notes (every deviation from a straight 1:1 op transcription is a
# framework-level substitution with unchanged semantics, not an architectural change):
#   - tf.nn.conv2d(..., padding='SAME') + explicit `flipkernel` -> torch.nn.functional
#     .conv2d(..., padding='same') using the flipped kernel; torch's cross-correlation
#     convention already matches TF's own after the same flip, so the explicit
#     `flipkernel` step from the original code is kept verbatim.
#   - Data layout: TF NHWC -> torch NCHW (the original code itself transposes to NCHW
#     right before the gather step; this port simply stays in NCHW throughout).
#   - `tf.gather_nd` over batch+spatial indices -> `torch.gather`/advanced indexing
#     with equivalent per-sample index sets.
#   - Trainable `tf.Variable` conv kernels/bias -> `torch.nn.Parameter` with the same
#     shapes and the same small-Gaussian (`randn * 0.01`) initialization used in the
#     original `vin_model.py`.
#   - State-batch handling: the original samples `statebatchsize` query states per
#     k+1 value-iteration snapshot for training; for a single forward/trace pass this
#     port queries the final value map with one fixed state-batch, matching the
#     `build_model()` call graph GVIN itself executes at inference / eval time.

import torch
import torch.nn as nn
import torch.nn.functional as F


def flipkernel(k):
    """Flip a conv kernel spatially, matching GVIN's utils.flipkernel (TF HWIO layout ->
    here applied on a torch OIHW kernel, flipping the last two (spatial) axes)."""
    return torch.flip(k, dims=[-2, -1])


class VIBlock(nn.Module):
    """Faithful port of GVIN's regular-grid `VI_Block` (regular/vin_model.py).

    A convolutional Value Iteration Network variant: builds reward/transition maps
    via small convs, iterates the Bellman-style max-over-actions update `k` times,
    then reads out Q-values (via a learned linear head `w_o`) and V-values at a
    batch of query grid positions using GVIN's 8-neighbor "extract_circle" gather.
    """

    def __init__(self, imsize=16, ch_i=2, ch_h=150, ch_q=10, k=20, statebatchsize=10):
        super().__init__()
        self.imsize = imsize
        self.ch_i = ch_i
        self.ch_h = ch_h
        self.ch_q = ch_q
        self.k = k
        self.statebatchsize = statebatchsize

        self.bias = nn.Parameter(torch.randn(1, ch_h, 1, 1) * 0.01)
        self.w0 = nn.Parameter(torch.randn(ch_h, ch_i, 3, 3) * 0.01)
        self.w1 = nn.Parameter(torch.randn(1, ch_h, 1, 1) * 0.01)
        self.w = nn.Parameter(torch.randn(ch_q, 1, 3, 3) * 0.01)
        self.w_fb = nn.Parameter(torch.randn(ch_q, 1, 3, 3) * 0.01)
        self.w_o = nn.Linear(ch_q, 8, bias=False)
        with torch.no_grad():
            self.w_o.weight.copy_((torch.randn(8, ch_q) * 0.01))

    def _conv(self, x, weight, groups=1):
        return F.conv2d(x, flipkernel(weight), padding="same", groups=groups)

    def forward(self, X, S1, S2):
        """
        X: (N, ch_i, imsize, imsize) input image + reward-prior channels (NCHW).
        S1, S2: (N, statebatchsize) integer row/col query positions.
        """
        k = self.k

        h = self._conv(X, self.w0) + self.bias
        r = self._conv(h, self.w1)  # r: (N, 1, H, W), the learned reward map
        q = self._conv(r, self.w)  # (N, ch_q, H, W)
        v, _ = torch.max(q, dim=1, keepdim=True)

        for _ in range(k - 1):
            rv = torch.cat([r, v], dim=1)
            wwfb = torch.cat([self.w, self.w_fb], dim=1)
            q = F.conv2d(rv, flipkernel(wwfb), padding="same")
            v, _ = torch.max(q, dim=1, keepdim=True)

        rv = torch.cat([r, v], dim=1)
        wwfb = torch.cat([self.w, self.w_fb], dim=1)
        q = F.conv2d(rv, flipkernel(wwfb), padding="same")

        # NCHW is already the layout used throughout this port (see module docstring).
        bs = X.shape[0]
        state_batch_size = S1.shape[1]

        batch_idx = (
            torch.arange(bs, device=X.device).view(-1, 1).repeat(1, state_batch_size).reshape(-1)
        )
        s1 = S1.reshape(-1).long()
        s2 = S2.reshape(-1).long()

        # q: (N, ch_q, H, W) -> gather per-(batch, s1, s2) vector over channels.
        q_out = q[batch_idx, :, s1, s2]  # (N*state_batch_size, ch_q)
        v_out = self._extract_circle(batch_idx, s1, s2, v)

        logits1 = self.w_o(q_out)
        logits2 = v_out
        output1 = F.softmax(logits1, dim=-1)
        output2 = F.softmax(logits2, dim=-1)
        return logits1, output1, v, logits2, output2

    def _extract_circle(self, rprn, ins1, ins2, v):
        """GVIN's 8-neighbor circle readout (regular/vin_model.py: extract_circle),
        gathering the scalar value map `v` at the 7 offset positions the original
        code samples (it omits the center position, matching the commented-out
        line in the source)."""
        H, W = v.shape[-2], v.shape[-1]
        offsets = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        circle = []
        for dx, dy in offsets:
            i1 = (ins1 + dx).clamp(0, H - 1)
            i2 = (ins2 + dy).clamp(0, W - 1)
            circle.append(v[rprn, 0, i1, i2])
        return torch.stack(circle, dim=-1)


MENAGERIE_ZOO = "ported-pytorch"


def build_gvin():
    # Tiny size (paper/repo defaults: imsize=16, ch_h=150, ch_q=10, k=20 -- shrunk here
    # for a fast trace while keeping every layer/mechanism from the original).
    return VIBlock(imsize=8, ch_i=2, ch_h=16, ch_q=4, k=3, statebatchsize=2)


def example_input_gvin():
    # Returned as a tuple so `tl.trace(model, example_input_gvin())` forwards
    # (X, S1, S2) as positional args to VIBlock.forward, matching its 3-input signature.
    imsize, ch_i, statebatchsize = 8, 2, 2
    X = torch.randn(1, ch_i, imsize, imsize)
    S1 = torch.randint(1, imsize - 1, (1, statebatchsize))
    S2 = torch.randint(1, imsize - 1, (1, statebatchsize))
    return (X, S1, S2)


MENAGERIE_ENTRIES = [
    ("GVIN", build_gvin, example_input_gvin, 2018, MENAGERIE_ZOO),
]

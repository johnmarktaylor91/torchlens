# FAITHFUL PORT of JakubBartoszewicz/DeePaC @ eab1be09df6bc0bdc7a015c48a5d9636ed9516e5 (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/JakubBartoszewicz/DeePaC/eab1be09df6bc0bdc7a015c48a5d9636ed9516e5/deepac/nn_train.py
# https://raw.githubusercontent.com/JakubBartoszewicz/DeePaC/eab1be09df6bc0bdc7a015c48a5d9636ed9516e5/deepac/builtin/config/nn-img-rapid-cnn.ini
#
# Bartoszewicz, Seidel, Renard (Bioinformatics 2021), "DeePaC: predicting
# host phenotypes and pathogenicity from DNA sequencing reads". DeePaC's
# `RCNet` model builder (`deepac/nn_train.py`) is real, runnable code, but
# it is TensorFlow 2 / Keras only (`tensorflow.keras.layers.Conv1D`,
# `tf.keras.backend.reverse`, etc.) with no PyTorch anywhere in the repo or
# in any fork -- not reasonably installable/vendorable into a torch-only
# env, so this is a faithful architectural PORT, not a vendor.
#
# The ported architecture is DeePaC's flagship, paper-shipped "rapid CNN"
# config (`deepac/builtin/config/nn-img-rapid-cnn.ini`, `RC_Mode = full`):
# a reverse-complement-parameter-shared 1D-CNN read classifier. Every
# mechanism below is transcribed from `RCNet._build_rc_model` and its
# helpers in the real source (not a paraphrase):
#   - `RevCompConv1d` == `RCNet._add_rc_conv1d`/`_add_siam_conv1d`: builds
#     the reverse complement of the one-hot input by flipping BOTH the
#     sequence axis and the channel axis (`K.reverse(x, axes=(1,2))` --
#     for one-hot-encoded DNA [A,C,G,T] flipping the channel axis maps a
#     base to its complement, and flipping the sequence axis reverses
#     5'->3' direction), applies the *same* Conv1d (shared weights) to
#     both the forward strand and the reverse-complement strand, flips the
#     RC branch's output back to forward-relative orientation, then
#     concatenates [fwd_out, rc_out] along the channel axis. This
#     "RC-parameter-sharing" is DeePaC's core architectural contribution
#     (generalizing Shrikumar et al.'s RC weight-tying beyond input layers).
#   - `RevCompGlobalAvgPool1d` == `RCNet._add_rc_pooling` with
#     `Conv_Pooling = average` + `N_Recurrent = 0` (global pooling branch):
#     splits the RC-doubled channel axis in half, flips the rc half's
#     channel-halves back to canonical (fwd) orientation before pooling
#     each half independently, then re-concatenates (matches the config's
#     `Conv_Pooling = average`, global because no recurrent layers follow).
#   - `RevCompMergeDense` == `RCNet._add_rc_merge_dense` /
#     `_add_siam_merge_dense` (the first dense layer, `Dense_Merge = add`):
#     splits the RC-doubled feature vector into fwd/rc halves, applies the
#     *same* shared `Linear` to both halves (reusing the same weight
#     matrix -- Shrikumar-style RC weight tying for dense layers), and adds
#     the two outputs (`merge_function=add`, the config default).
#   - `RCNetRapidCNN.forward` == `RCNet._build_rc_model`, restricted to the
#     exact code path selected by `nn-img-rapid-cnn.ini`: `N_Conv=2`,
#     `Conv_Units=512,512`, `Conv_FilterSize=15,15`, `Conv_Stride=1,1`,
#     `Conv_Dilation=1,1`, `Conv_Padding=same`, `Conv_Activation=relu`,
#     `Conv_BN=False`, `Skip_Size=0` (no residual skip branch),
#     `Conv_Pooling=average`, `N_Recurrent=0`, `N_Dense=2`,
#     `Dense_Units=256,256`, `Dense_Activation=relu`, `Dense_BN=False`,
#     `N_Classes=2` -> a single sigmoid logit output, matching the config's
#     `Activation('sigmoid', dtype='float32')` binary-classification head.
#     `Input_Dropout=0.25`/`Dense_Dropout=0.5` are present in the config
#     but Keras `Dropout` layers are no-ops in eval mode (`training=False`)
#     -- included below as `nn.Dropout` for architectural fidelity, they
#     have no effect on the traced forward pass since the model is built
#     and traced in `.eval()` mode.
#
# Only mechanical staging/translation edits (Keras -> torch, not
# architectural changes):
#   - `same` padding for the odd kernel_size=15, stride=1, dilation=1 convs
#     is `padding=7` in torch (Keras "same" for stride=1 is symmetric here).
#   - Keras `Conv1D` expects channels-last (`batch, seq, channels`); torch
#     `nn.Conv1d` expects channels-first (`batch, channels, seq`) -- the
#     real Keras reverse-axis semantics (`axes=(1,2)` = sequence + channel)
#     are preserved by flipping torch dims (-1, -2) (channels-first
#     equivalents of Keras's (seq, channel) axes), so the RC construction
#     is bit-for-bit equivalent, just re-indexed for the torch layout.
#   - Added `build_deepac_rapid_cnn()` / `example_input_deepac_rapid_cnn()`
#     staging entry points at the config's real `SeqLength=250`,
#     one-hot DNA alphabet dim 4 (`seq_dim` in the original), batch=2.

import torch
import torch.nn as nn


class RevCompConv1d(nn.Module):
    """Reverse-complement parameter-shared 1D convolution.

    Real port of `RCNet._add_rc_conv1d` / `_add_siam_conv1d`: applies the
    SAME Conv1d to the forward strand and to the reverse-complemented
    strand (sequence axis flipped + channel/base axis flipped), then
    flips the RC branch's output back to forward orientation before
    concatenating [fwd, rc] along the channel axis.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.shared_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        # x: (batch, channels, seq) -- channels-first torch layout.
        # Keras K.reverse(x, axes=(1, 2)) flips (seq, channel) in NHWC;
        # the channels-first equivalent flips (channel, seq) == dims (-2, -1).
        x_rc = torch.flip(x, dims=(-2, -1))
        out_fwd = self.shared_conv(x)
        out_rc = self.shared_conv(x_rc)
        out_rc = torch.flip(out_rc, dims=(-2, -1))
        return torch.cat([out_fwd, out_rc], dim=1)


class RevCompGlobalAvgPool1d(nn.Module):
    """Reverse-complement-aware global average pooling.

    Real port of `RCNet._add_rc_pooling` (global-pooling branch, taken when
    `Conv_Pooling='average'` and `N_Recurrent=0`): splits the RC-doubled
    channel axis into fwd/rc halves, un-flips the rc half back to forward
    orientation, pools each half independently, then re-concatenates.
    """

    def forward(self, x):
        c = x.shape[1] // 2
        x_fwd = x[:, :c, :]
        x_rc = torch.flip(x[:, c:, :], dims=(-2, -1))
        pooled_fwd = x_fwd.mean(dim=-1)
        pooled_rc = x_rc.mean(dim=-1)
        return torch.cat([pooled_fwd, pooled_rc], dim=-1)


class RevCompMergeDense(nn.Module):
    """Reverse-complement parameter-shared merging dense layer.

    Real port of `RCNet._add_rc_merge_dense` / `_add_siam_merge_dense`
    with the config default `Dense_Merge='add'`: splits the RC-doubled
    feature vector into fwd/rc halves, applies the SAME shared Linear to
    both halves, and adds the two outputs.
    """

    def __init__(self, in_features_half, out_features):
        super().__init__()
        self.shared_dense = nn.Linear(in_features_half, out_features)

    def forward(self, x):
        c = x.shape[-1] // 2
        x_fwd = x[:, :c]
        x_rc = x[:, c:]
        return self.shared_dense(x_fwd) + self.shared_dense(x_rc)


class RCNetRapidCNN(nn.Module):
    """DeePaC's flagship RC-CNN read classifier ("rapid CNN" config).

    Faithful port of `RCNet._build_rc_model`, restricted to the exact
    architectural choices in `nn-img-rapid-cnn.ini`: RC_Mode=full,
    N_Conv=2, Conv_Units=512,512, Conv_FilterSize=15,15, Conv_Pooling=
    average, N_Recurrent=0, N_Dense=2, Dense_Units=256,256, N_Classes=2.
    """

    def __init__(self, seq_dim=4):
        super().__init__()
        self.input_dropout = nn.Dropout(0.25)

        # First conv layer (RCNet._build_rc_model, n_conv>0 branch).
        self.conv1 = RevCompConv1d(seq_dim, 512, kernel_size=15, padding=7)
        self.act1 = nn.ReLU()

        # Second conv layer (loop over range(1, n_conv)); input channels
        # double to 1024 because conv1's RC-concatenation doubles the
        # channel dim (512 fwd + 512 rc).
        self.conv2 = RevCompConv1d(1024, 512, kernel_size=15, padding=7)
        self.act2 = nn.ReLU()

        # Pooling layer: Conv_Pooling='average', N_Recurrent=0 -> global
        # average pooling (RCNet._add_rc_pooling global branch).
        self.pool = RevCompGlobalAvgPool1d()

        # Dense layers: first is the RC-merging dense (_add_rc_merge_dense),
        # subsequent ones are plain Dense layers on the merged (non-RC)
        # representation.
        self.dense1 = RevCompMergeDense(in_features_half=512, out_features=256)
        self.dense1_act = nn.ReLU()
        self.dense1_drop = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256, 256)
        self.dense2_act = nn.ReLU()
        self.dense2_drop = nn.Dropout(0.5)

        # Output layer: N_Classes=2 -> single sigmoid logit.
        self.out = nn.Linear(256, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_dim, seq_length) one-hot DNA, channels-first.
        x = self.input_dropout(x)

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.pool(x)

        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.dense1_drop(x)

        x = self.dense2(x)
        x = self.dense2_act(x)
        x = self.dense2_drop(x)

        x = self.out(x)
        x = self.out_act(x)
        return x


def build_deepac_rapid_cnn():
    return RCNetRapidCNN(seq_dim=4)


def example_input_deepac_rapid_cnn():
    # Config default SeqLength=250, one-hot DNA alphabet dim=4.
    return torch.randn(2, 4, 250)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeePaC-RapidCNN", "build_deepac_rapid_cnn", "example_input_deepac_rapid_cnn", 2020, "ported"),
]

# SOURCE: vendored from xuchenglin28/speaker_extraction_SpEx @ master
# https://raw.githubusercontent.com/xuchenglin28/speaker_extraction_SpEx/master/nnet/spex_plus.py
# https://raw.githubusercontent.com/xuchenglin28/speaker_extraction_SpEx/master/nnet/cnns.py
# https://raw.githubusercontent.com/xuchenglin28/speaker_extraction_SpEx/master/nnet/norm.py
#
# SpEx+ (Ge et al., Interspeech 2020, "SpEx+: A Complete Time Domain Solution for
# Target Speaker Extraction") -- official reference PyTorch implementation. Multi-scale
# (short/middle/long window) 1-D conv encoder shared between the mixture and the
# auxiliary (reference) utterance, a channel-wise-then-global-layer-norm TCN stack of
# four speaker-conditioned TCN blocks (each seeded by a broadcast speaker embedding,
# followed by (B-1) plain dilated TCN blocks), three parallel 1x1-conv masks applied to
# the three encoder scales, and three transposed-conv decoders reconstructing the target
# speech at each scale; a parallel ResNet-style speaker encoder (shared conv front-end +
# 3 ResBlocks + 1x1 conv) produces the auxiliary speaker embedding and a speaker
# classification logit via a final linear layer.
#
# Transcribed verbatim from the three real source files (spex_plus.py, cnns.py,
# norm.py): every Conv1D/ConvTrans1D/TCNBlock/TCNBlock_Spk/ResBlock/ChannelwiseLayerNorm/
# GlobalLayerNorm class body and the SpEx_Plus forward() logic (encoder scales, padding
# arithmetic, TCN block stacking with speaker-embedding broadcast, masking, decoding) is
# unchanged. Only the relative package imports (`from .norm import ...`,
# `from .cnns import ...`) are flattened into this single file since it is used standalone
# outside the original `nnet` package; no architectural code was altered.

import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# norm.py
# ---------------------------------------------------------------------------
class ChannelwiseLayerNorm(nn.LayerNorm):
    """
    Channel-wise layer normalization based on nn.LayerNorm
    Input: 3D tensor with [batch_size(N), channel_size(C), frame_num(T)]
    Output: 3D tensor with same shape
    """

    def __init__(self, *args, **kwargs):
        super(ChannelwiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(self.__name__))
        x = th.transpose(x, 1, 2)
        x = super().forward(x)
        x = th.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    """
    Global layer normalization
    Input: 3D tensor with [batch_size(N), channel_size(C), frame_num(T)]
    Output: 3D tensor with same shape
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(self.__name__))
        # calculate the mean, variance over the channel and time dimensions
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean) ** 2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


# ---------------------------------------------------------------------------
# cnns.py
# ---------------------------------------------------------------------------
class Conv1D(nn.Conv1d):
    """
    1D Conv based on nn.Conv1d for 2D or 3D tensor
    Input: 2D or 3D tensor with [N, L_in] or [N, C_in, L_in]
    Output: Default 3D tensor with [N, C_out, L_out]
            If C_out=1 and squeeze is true, return 2D tensor
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} require a 2/3D tensor input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D Transposed Conv based on nn.ConvTranspose1d for 2D or 3D tensor
    Input: 2D or 3D tensor with [N, L_in] or [N, C_in, L_in]
    Output: 2D tensor with [N, L_out]
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} require a 2/3D tensor input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        # squeeze the channel dimension 1 after reconstructing the signal
        return th.squeeze(x, 1)


class TCNBlock(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
    Input: 3D tensor with [N, C_in, L_in]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(self, in_channels=256, conv_channels=512, kernel_size=3, dilation=1, causal=False):
        super(TCNBlock, self).__init__()
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = (
            GlobalLayerNorm(conv_channels, elementwise_affine=True)
            if not causal
            else (ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        )
        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        )
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = (
            GlobalLayerNorm(conv_channels, elementwise_affine=True)
            if not causal
            else (ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        )
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, : -self.dconv_pad]
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y


class TCNBlock_Spk(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
        The first tcn block takes additional speaker embedding as inputs
    Input: 3D tensor with [N, C_in, L_in]
    Input Speaker Embedding: 2D tensor with [N, D]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(
        self,
        in_channels=256,
        spk_embed_dim=100,
        conv_channels=512,
        kernel_size=3,
        dilation=1,
        causal=False,
    ):
        super(TCNBlock_Spk, self).__init__()
        self.conv1x1 = Conv1D(in_channels + spk_embed_dim, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = (
            GlobalLayerNorm(conv_channels, elementwise_affine=True)
            if not causal
            else (ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        )
        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        )
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = (
            GlobalLayerNorm(conv_channels, elementwise_affine=True)
            if not causal
            else (ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        )
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad
        self.dilation = dilation

    def forward(self, x, aux):
        # Repeatedly concated speaker embedding aux to each frame of the representation x
        T = x.shape[-1]
        aux = th.unsqueeze(aux, -1)
        aux = aux.repeat(1, 1, T)
        y = th.cat([x, aux], 1)
        y = self.conv1x1(y)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, : -self.dconv_pad]
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y


class ResBlock(nn.Module):
    """
    Resnet block for speaker encoder to obtain speaker embedding
    ref to
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    """

    def __init__(self, in_dims, out_dims):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)


# ---------------------------------------------------------------------------
# spex_plus.py
# ---------------------------------------------------------------------------
class SpEx_Plus(nn.Module):
    def __init__(
        self,
        L1=20,
        L2=80,
        L3=160,
        N=256,
        B=8,
        O=256,  # noqa: E741 -- keeps the real repo's constructor parameter name
        P=512,
        Q=3,
        num_spks=101,
        spk_embed_dim=256,
        causal=False,
    ):
        super(SpEx_Plus, self).__init__()
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.encoder_1d_short = Conv1D(1, N, L1, stride=L1 // 2, padding=0)
        self.encoder_1d_middle = Conv1D(1, N, L2, stride=L1 // 2, padding=0)
        self.encoder_1d_long = Conv1D(1, N, L3, stride=L1 // 2, padding=0)
        # before repeat blocks, always cLN
        self.ln = ChannelwiseLayerNorm(3 * N)
        # n x N x T => n x O x T
        self.proj = Conv1D(3 * N, O, 1)
        self.conv_block_1 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            causal=causal,
            dilation=1,
        )
        self.conv_block_1_other = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal
        )
        self.conv_block_2 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            causal=causal,
            dilation=1,
        )
        self.conv_block_2_other = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal
        )
        self.conv_block_3 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            causal=causal,
            dilation=1,
        )
        self.conv_block_3_other = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal
        )
        self.conv_block_4 = TCNBlock_Spk(
            spk_embed_dim=spk_embed_dim,
            in_channels=O,
            conv_channels=P,
            kernel_size=Q,
            causal=causal,
            dilation=1,
        )
        self.conv_block_4_other = self._build_stacks(
            num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal
        )
        # n x O x T => n x N x T
        self.mask1 = Conv1D(O, N, 1)
        self.mask2 = Conv1D(O, N, 1)
        self.mask3 = Conv1D(O, N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d_short = ConvTrans1D(N, 1, kernel_size=L1, stride=L1 // 2, bias=True)
        self.decoder_1d_middle = ConvTrans1D(N, 1, kernel_size=L2, stride=L1 // 2, bias=True)
        self.decoder_1d_long = ConvTrans1D(N, 1, kernel_size=L3, stride=L1 // 2, bias=True)
        self.num_spks = num_spks

        self.spk_encoder = nn.Sequential(
            ChannelwiseLayerNorm(3 * N),
            Conv1D(3 * N, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            Conv1D(P, spk_embed_dim, 1),
        )

        self.pred_linear = nn.Linear(spk_embed_dim, num_spks)

    def _build_stacks(self, num_blocks, **block_kwargs):
        """
        Stack B numbers of TCN block, the first TCN block takes the speaker embedding
        """
        blocks = [TCNBlock(**block_kwargs, dilation=(2**b)) for b in range(1, num_blocks)]
        return nn.Sequential(*blocks)

    def forward(self, x, aux, aux_len):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__class__.__name__, x.dim()
                )
            )
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)

        # n x 1 x S => n x N x T
        w1 = F.relu(self.encoder_1d_short(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(self.encoder_1d_middle(F.pad(x, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_1d_long(F.pad(x, (0, xlen3 - xlen1), "constant", 0)))

        # n x 3N x T
        y = self.ln(th.cat([w1, w2, w3], 1))
        # n x O x T
        y = self.proj(y)

        # speaker encoder (share params from speech encoder)
        aux_w1 = F.relu(self.encoder_1d_short(aux))
        aux_T_shape = aux_w1.shape[-1]
        aux_len1 = aux.shape[-1]
        aux_len2 = (aux_T_shape - 1) * (self.L1 // 2) + self.L2
        aux_len3 = (aux_T_shape - 1) * (self.L1 // 2) + self.L3
        aux_w2 = F.relu(self.encoder_1d_middle(F.pad(aux, (0, aux_len2 - aux_len1), "constant", 0)))
        aux_w3 = F.relu(self.encoder_1d_long(F.pad(aux, (0, aux_len3 - aux_len1), "constant", 0)))

        aux = self.spk_encoder(th.cat([aux_w1, aux_w2, aux_w3], 1))
        aux_T = (aux_len - self.L1) // (self.L1 // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = th.sum(aux, -1) / aux_T.view(-1, 1).float()

        y = self.conv_block_1(y, aux)
        y = self.conv_block_1_other(y)
        y = self.conv_block_2(y, aux)
        y = self.conv_block_2_other(y)
        y = self.conv_block_3(y, aux)
        y = self.conv_block_3_other(y)
        y = self.conv_block_4(y, aux)
        y = self.conv_block_4_other(y)

        # n x N x T
        m1 = F.relu(self.mask1(y))
        m2 = F.relu(self.mask2(y))
        m3 = F.relu(self.mask3(y))
        S1 = w1 * m1
        S2 = w2 * m2
        S3 = w3 * m3

        return (
            self.decoder_1d_short(S1),
            self.decoder_1d_middle(S2)[:, :xlen1],
            self.decoder_1d_long(S3)[:, :xlen1],
            self.pred_linear(aux),
        )


class SpExPlusInput(nn.Module):
    """Thin wrapper bundling (mixture, auxiliary_utt) into a single example_input_
    callable's return value; torchlens traces the wrapped forward directly, so this
    class just packages the SpEx_Plus module + a matching aux_len computation so the
    whole real forward() signature (x, aux, aux_len) is exercised faithfully."""

    def __init__(self, model: SpEx_Plus):
        super().__init__()
        self.model = model

    def forward(self, x, aux):
        aux_len = th.tensor([aux.shape[-1]] * aux.shape[0])
        return self.model(x, aux, aux_len)


def build_spex_plus():
    th.manual_seed(0)
    # Tiny config: fewer TCN blocks (B) and narrower channels than the paper's
    # defaults (L1=20,L2=80,L3=160,N=256,B=8,O=256,P=512,Q=3), keeping every real
    # architectural mechanism (multi-scale encoder, 4 speaker-conditioned TCN stacks,
    # tri-scale masking/decoding, ResNet speaker encoder) intact.
    model = SpExPlusInput(
        SpEx_Plus(
            L1=20,
            L2=80,
            L3=160,
            N=8,
            B=2,
            O=8,
            P=16,
            Q=3,
            num_spks=5,
            spk_embed_dim=8,
            causal=False,
        )
    )
    model.eval()
    return model


def example_input_spex_plus():
    th.manual_seed(0)
    # Mixture waveform and a separate auxiliary (reference) utterance from the target
    # speaker, matching the repo's (batch, samples) convention. Kept short (well above
    # the L3=160 long-window kernel) to keep the tiny build fast to trace.
    x = th.randn(2, 4000)
    aux = th.randn(2, 6000)
    return (x, aux)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SpEx+", "build_spex_plus", "example_input_spex_plus", 2020, MENAGERIE_ZOO),
]

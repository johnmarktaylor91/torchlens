# SOURCE: vendored from mrjunjieli/speaker_extraction @ c646de515c0ae3bc1f8c71f83f6aadf8a2b6bd2d
# https://raw.githubusercontent.com/mrjunjieli/speaker_extraction/main/AVModel.py
# https://raw.githubusercontent.com/mrjunjieli/speaker_extraction/main/Modelutils.py
# https://raw.githubusercontent.com/mrjunjieli/speaker_extraction/main/Normolization.py
#
# SpeakerFilter: Conv-TasNet-based target-speaker extraction. Despite the repo/file
# name "AVModel", the traced forward() path is audio-only target-speaker extraction:
# a shared TasNet Encoder produces latents for the mixture and for a reference
# (enrollment) clip of the target speaker; the reference latent is passed through its
# own TCN stack + BatchNorm + a strided Conv1d "speaker embedding" head to produce a
# fixed-length speaker embedding, which is broadcast-concatenated with the mixture's
# TCN output and fed through a further TCN ("concatNet") to produce a mask; the mask
# multiplies the mixture latent and a ConvTranspose1d Decoder reconstructs the
# extracted-speaker waveform. `AVModel`/`concatNet`/`Encoder`/`Decoder`/`TCN` are
# transcribed verbatim from `AVModel.py`; `Conv1D`/`Conv1D_Block`/`select_norm` are
# transcribed verbatim from `Modelutils.py`; `GlobalLayerNorm` is transcribed verbatim
# from `Normolization.py` (the repo's own `Modelutils.select_norm` imports
# `GlobalLayerNorm` from `Normolization`, so that definition is used here, shadowing
# the incomplete `Modelutils.GlobalLayerNorm`/`CumulativeLayerNorm` duplicates the
# original repo also defines but never calls from this forward path). The unused
# vision branch of the original repo (`ResNet`/`BasicBlock`/`Conv1D_Block_in_Visual`,
# gated behind `cv2`/`torchsummary`, never invoked by `AVModel.forward`) is dropped;
# no change to any layer/mechanism actually exercised by `AVModel.forward`.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- from Normolization.py (GlobalLayerNorm actually used by Modelutils.select_norm) ---


class GlobalLayerNorm(nn.Module):
    """

    normalize over both the channel and the time dimensions

    gLN(F) = (F-E[F])/(Var[F]+eps)**0.5 element-wise y+beta
    E[F] = 1/(NT)*sum_NT(F)[add elements in F along N and T dimensions]
    Var[F] = 1/(NT)*sum_NT((F-E[F])**2)

    N:channle dimension
    T:time dimension
    y and beta are trainable parameters ->R^{N*1}

    where F ->R^{N*T}
    dim:(int or list or torch.Size) - input shape from an expected input of size
    elementwise_affine: a boolean value that when set to True, then this module has
    learneable parameter initialized to ones(for weights) and zeros (for bias)
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


# --- from Modelutils.py ---


def select_norm(norm, dim):
    """
    select normolization method
    norm: the one in ['gln','cln','bn']
    """
    if norm not in ["gln", "cln", "bn"]:
        raise RuntimeError("only accept['gln','cln','bn']")
    if norm == "gln":
        return GlobalLayerNorm(dim, elementwise_affine=True)
    elif norm == "cln":
        raise RuntimeError(
            "cln norm not usable: original repo's CumulativeLayerNorm.__init__ references an undefined name"
        )
    elif norm == "bn":
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Conv1d):
    """
    Applies a 1D convolution over an input signal composed of several input planes.
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    """
    sub-block with the exponential growth dilation factors 2**d
    """

    def __init__(
        self, in_channels=256, out_channels=512, kernel_size=3, dilation=1, norm="gln", causal=False
    ):
        super(Conv1D_Block, self).__init__()
        # this conv1d determines the number of channels
        self.linear = Conv1D(in_channels, out_channels, 1)  # set kernel_size=1
        self.ReLu = nn.ReLU(True)
        self.norm = select_norm(norm, out_channels)
        # keep time length unchanged
        self.pad = (
            (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        )

        self.DepthwiseConv = Conv1D(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=self.pad,
            dilation=dilation,
        )
        self.SeparableConv = Conv1D(out_channels, in_channels, 1)
        self.causal = causal

    def forward(self, x):
        c = self.linear(x)
        c = self.ReLu(c)
        c = self.norm(c)
        c = self.DepthwiseConv(c)
        if self.causal:
            c = c[:, :, : -self.pad]
        c = self.SeparableConv(c)
        return x + c


# --- from AVModel.py ---


class concatNet(nn.Module):
    def __init__(
        self,
        num_repeats,
        num_blocks,
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        norm="gln",
        causal=False,
        concat_in_dim=556,
    ):
        super(concatNet, self).__init__()
        self.liner = Conv1D(concat_in_dim, in_channels, kernel_size=1)
        self.TCN = self._Sequential_repeat(
            num_repeats,
            num_blocks,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm=norm,
            causal=causal,
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.liner(x)
        out = self.TCN(out)
        out = self.relu(out)

        return out

    def _Sequential_repeat(self, num_repeats, num_blocks, **kwargs):
        repeat_lists = [self._Sequential_block(num_blocks, **kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeat_lists)

    def _Sequential_block(self, num_blocks, **kwargs):
        """
        Sequential 1-D Conv Block
        input:
            num_blocks:times the block appears
            **block_kwargs
        """
        Conv1D_Block_lists = [Conv1D_Block(**kwargs, dilation=(2**i)) for i in range(num_blocks)]
        return nn.Sequential(*Conv1D_Block_lists)


class Encoder(nn.Module):
    """
    Encoder of the TasNet
    """

    def __init__(self, kernel_size, stride, outputDim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv1d(1, outputDim, kernel_size, stride=stride)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.encoder(x)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """
    Decoder of the TasNet
    """

    def __init__(self, kernel_size, stride, inputDim=256):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose1d(inputDim, 1, kernel_size, stride)

    def forward(self, x):
        out = self.decoder(x)
        return out


class TCN(nn.Module):
    """
    in_channels:the encoder out_channels

    """

    def __init__(
        self,
        out_channels,
        num_repeats,
        num_blocks,
        kernel_size,
        norm="gln",
        causal=False,
        feat_dim=256,
    ):
        super(TCN, self).__init__()

        self.TCN = self._Sequential_repeat(
            num_repeats,
            num_blocks,
            in_channels=feat_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm=norm,
            causal=causal,
        )

    def forward(self, x):
        c = self.TCN(x)
        return c  # shape [-1,1,256]

    def _Sequential_repeat(self, num_repeats, num_blocks, **kwargs):
        repeat_lists = [self._Sequential_block(num_blocks, **kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeat_lists)

    def _Sequential_block(self, num_blocks, **kwargs):
        """
        Sequential 1-D Conv Block
        input:
            num_blocks:times the block appears
            **block_kwargs
        """
        Conv1D_Block_lists = [Conv1D_Block(**kwargs, dilation=(2**i)) for i in range(num_blocks)]
        return nn.Sequential(*Conv1D_Block_lists)


class AVModel(nn.Module):
    def __init__(
        self,
        num_repeats=1,
        num_blocks=2,
        concat_repeats=1,
        concat_blocks=2,
        feat_dim=256,
        tcn_bottleneck_channels=256,
        speaker_emb_out_len=300,
        speaker_emb_target_len=5033,
    ):
        super(AVModel, self).__init__()

        # feat_dim is the shared TasNet feature width (Encoder outputDim == TCN's
        # internal in_channels/out_channels == Decoder inputDim, all 256 in the
        # original hardcoded repo); tcn_bottleneck_channels is the Conv1D_Block
        # bottleneck width inside each TCN (the repo's `out_channels` TCN arg).
        self.audio_model_encoder = Encoder(kernel_size=40, stride=20, outputDim=feat_dim)
        self.audio_model_TCN1 = TCN(
            out_channels=tcn_bottleneck_channels,
            num_repeats=num_repeats,
            num_blocks=num_blocks,
            kernel_size=3,
            feat_dim=feat_dim,
        )
        self.audio_model_TCN2 = TCN(
            out_channels=tcn_bottleneck_channels,
            num_repeats=num_repeats,
            num_blocks=num_blocks,
            kernel_size=3,
            feat_dim=feat_dim,
        )

        self.concat_model = concatNet(
            in_channels=feat_dim,
            out_channels=feat_dim,
            num_repeats=concat_repeats,
            num_blocks=concat_blocks,
            concat_in_dim=feat_dim + speaker_emb_out_len,
        )
        self.decoder = Decoder(kernel_size=40, stride=20, inputDim=feat_dim)

        self.speakerembedding = nn.Conv1d(
            in_channels=feat_dim, out_channels=1, kernel_size=3, stride=8
        )
        self.transToSpeakerEmb = nn.Linear(speaker_emb_out_len, speaker_emb_target_len)
        self.batch = nn.BatchNorm1d(feat_dim)

    def forward(self, data):
        audio_mix, audio_ref = data

        encoder_output = self.audio_model_encoder(audio_mix)

        TCN_output = self.audio_model_TCN1(encoder_output)
        length = TCN_output.shape[2]

        encoder_output_aux = self.audio_model_encoder(audio_ref)
        TCN_output_aux = self.audio_model_TCN2(encoder_output_aux)
        TCN_output_aux = self.batch(TCN_output_aux)
        speakeremb = self.speakerembedding(TCN_output_aux)  # shape[batch,1,300]

        speakeremb_new = self.transToSpeakerEmb(speakeremb)
        speakeremb_new = torch.squeeze(speakeremb_new, dim=1)

        speakerembs = speakeremb.repeat(1, length, 1)
        speakerembs = speakerembs.permute(0, 2, 1)

        concat_input = torch.cat((TCN_output, speakerembs), dim=1)
        concat_output = self.concat_model(concat_input)

        decoder_input = encoder_output * concat_output
        output = self.decoder(decoder_input)  # output [B,C,lenght]

        return output, speakeremb_new


def build_speakerfilter():
    # tiny config: 1 repeat x 2 blocks per TCN stack (vs. paper-scale 1x8/3x8) and a
    # shrunk bottleneck width, but feat_dim stays at the architecture's real 256 (the
    # Conv1D_Block residual-add path only balances when in_channels==out_channels of
    # the block's own kernel_size=1 SeparableConv projection, so it cannot be freely
    # shrunk without changing the block's own hardcoded shapes). Short waveforms keep
    # the trace cheap; speaker_emb_out_len is computed from example_input_speakerfilter's
    # sample count so the strided speakerembedding Conv1d + Linear head line up exactly.
    model = AVModel(
        num_repeats=1,
        num_blocks=2,
        concat_repeats=1,
        concat_blocks=2,
        feat_dim=256,
        tcn_bottleneck_channels=32,
        speaker_emb_out_len=25,
        speaker_emb_target_len=8,
    )
    model.eval()
    return model


def example_input_speakerfilter():
    # (audio_mix, audio_ref): both [Batch, 1, num_samples]; kernel=40/stride=20
    # TasNet encoder needs num_samples >= ~40. speaker_emb_out_len=25 above matches
    # a 4000-sample reference clip: Encoder(k=40,s=20) -> 199 frames, then the
    # speakerembedding Conv1d(k=3,s=8) -> floor((199-3)/8)+1 = 25 frames.
    audio_mix = torch.randn(1, 1, 4000)
    audio_ref = torch.randn(1, 1, 4000)
    return (audio_mix, audio_ref)


MENAGERIE_ENTRIES = [
    ("SpeakerFilter", "build_speakerfilter", "example_input_speakerfilter", 2021, MENAGERIE_ZOO),
]

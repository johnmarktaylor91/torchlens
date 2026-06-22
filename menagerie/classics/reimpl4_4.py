"""Compact faithful reconstructions for dependency-gated reimpl4_4 targets.

Sources researched for the distinctive primitives:

* NVIDIA NeMo ASR/TTS docs: Jasper/QuartzNet/Citrinet separable time-channel
  convolutions, Conformer/FastConformer/Squeezeformer encoders, Parakeet
  FastConformer plus CTC/RNN-T/TDT heads, FastPitch/MixerTTS/RadTTS/TalkNet,
  HiFi-GAN, WaveGlow, and speaker encoders.
* Instant-NGP and nerfstudio docs: multiresolution hash encodings, NeRF fields,
  proposal/occupancy sampling, mip/Zip anti-aliasing, Nerfacto field heads, and
  Splatfacto Gaussian parameters.
* Monodepth2 (Godard et al. ICCV 2019): ResNet encoder plus multiscale sigmoid
  disparity decoder for self-supervised monocular depth.
* Latent Neural Processes (Garnelo et al. 2018): deterministic set aggregation
  plus a global latent Gaussian sampled/reparameterized into target decoding.
* ACMix (Pan et al. CVPR 2022): shared 1x1 Q/K/V projection feeding attention
  aggregation and lightweight convolutional aggregation branches.
* Agent57 (Badia et al. ICML 2020): R2D2/NGU-style recurrent Q-network with
  separate extrinsic/intrinsic heads and a bandit meta-controller over policies.
* AIFS/Anemoi (ECMWF): graph encoder-processor-decoder weather forecaster.
* Hedgehog (Zhang et al. ICLR 2024): learnable MLP feature map for linear
  attention with softmax-mimicry.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDSeparableBlock(nn.Module):
    """Depthwise-separable temporal convolution block used by Jasper descendants."""

    def __init__(self, channels: int, kernel: int, dilation: int = 1, se: bool = False) -> None:
        """Initialize a time-channel separable convolution block.

        Parameters
        ----------
        channels:
            Number of temporal channels.
        kernel:
            Temporal convolution kernel size.
        dilation:
            Temporal dilation.
        se:
            Whether to include squeeze-excitation gating.
        """

        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv1d(
            channels, channels, kernel, padding=pad, dilation=dilation, groups=channels
        )
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.se = (
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, max(4, channels // 4), 1),
                nn.ReLU(),
                nn.Conv1d(max(4, channels // 4), channels, 1),
                nn.Sigmoid(),
            )
            if se
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual separable temporal convolution.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Updated temporal features.
        """

        y = F.relu(self.norm(self.pw(self.dw(x))))
        if self.se is not None:
            y = y * self.se(y)
        return x + y


class ConvASR(nn.Module):
    """Compact NeMo-style convolutional CTC acoustic model."""

    def __init__(
        self,
        feat_dim: int = 32,
        channels: int = 40,
        vocab: int = 64,
        kernels: tuple[int, ...] = (11, 13, 17),
        se: bool = False,
    ) -> None:
        """Initialize a compact separable-convolution ASR network.

        Parameters
        ----------
        feat_dim:
            Acoustic feature dimension.
        channels:
            Hidden channel count.
        vocab:
            Number of CTC output classes.
        kernels:
            Temporal kernels for Jasper/QuartzNet/Citrinet-style blocks.
        se:
            Whether to include squeeze-excitation gates.
        """

        super().__init__()
        self.in_proj = nn.Conv1d(feat_dim, channels, 1)
        self.blocks = nn.ModuleList(
            [
                TDSeparableBlock(channels, k, dilation=1 + (i % 3), se=se)
                for i, k in enumerate(kernels)
            ]
        )
        self.head = nn.Conv1d(channels, vocab, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute frame-level CTC logits.

        Parameters
        ----------
        feats:
            Acoustic features of shape ``(batch, time, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, time, vocab)``.
        """

        x = self.in_proj(feats.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.head(x).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Macaron feed-forward, MHSA, depthwise-conv Conformer block."""

    def __init__(self, dim: int, heads: int = 4, kernel: int = 7, squeeze: bool = False) -> None:
        """Initialize a compact Conformer/Squeezeformer block.

        Parameters
        ----------
        dim:
            Model width.
        heads:
            Attention heads.
        kernel:
            Depthwise convolution kernel.
        squeeze:
            Whether to add a Squeezeformer-style temporal pooling branch.
        """

        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv_in = nn.Linear(dim, 2 * dim)
        self.conv_dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.conv_bn = nn.BatchNorm1d(dim)
        self.conv_out = nn.Conv1d(dim, dim, 1)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )
        self.squeeze = nn.AvgPool1d(2, stride=1, padding=1) if squeeze else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Conformer block.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Updated sequence tensor.
        """

        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        c = F.glu(self.conv_in(self.conv_norm(x)), dim=-1).transpose(1, 2)
        c = self.conv_out(F.silu(self.conv_bn(self.conv_dw(c))))
        if self.squeeze is not None:
            c = self.squeeze(c)[..., : x.shape[1]]
        x = x + c.transpose(1, 2)
        return x + 0.5 * self.ff2(x)


class TransducerJoint(nn.Module):
    """RNN-T/TDT joint network over acoustic and prediction states."""

    def __init__(self, dim: int, vocab: int, durations: int = 0) -> None:
        """Initialize joint projections.

        Parameters
        ----------
        dim:
            Hidden dimension.
        vocab:
            Token vocabulary.
        durations:
            Optional number of TDT duration classes.
        """

        super().__init__()
        self.pred = nn.GRU(vocab, dim, batch_first=True)
        self.joint = nn.Linear(dim, vocab + durations)

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """Compute compact transducer logits.

        Parameters
        ----------
        enc:
            Encoder states.

        Returns
        -------
        torch.Tensor
            Token or token-duration logits.
        """

        seed = F.one_hot(
            torch.zeros(enc.shape[0], 4, dtype=torch.long, device=enc.device),
            self.joint.out_features,
        )
        pred, _ = self.pred(seed[..., : self.pred.input_size].float())
        fused = enc[:, :4].unsqueeze(2) + pred.unsqueeze(1)
        return self.joint(torch.tanh(fused)).mean(dim=2)


class ConformerASR(nn.Module):
    """Compact Conformer/FastConformer/Squeezeformer ASR family."""

    def __init__(
        self,
        feat_dim: int = 32,
        dim: int = 48,
        vocab: int = 64,
        fast: bool = False,
        squeeze: bool = False,
        head: str = "ctc",
    ) -> None:
        """Initialize compact ASR encoder and requested output head.

        Parameters
        ----------
        feat_dim:
            Acoustic feature dimension.
        dim:
            Encoder width.
        vocab:
            Output vocabulary size.
        fast:
            Whether to use FastConformer-style 8x depthwise subsampling.
        squeeze:
            Whether to include Squeezeformer pooling.
        head:
            One of ``"ctc"``, ``"rnnt"``, ``"tdt"``, or ``"hybrid"``.
        """

        super().__init__()
        stride = 2 if fast else 1
        self.subsample = nn.Sequential(
            nn.Conv1d(feat_dim, dim, 3, stride=stride, padding=1, groups=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, stride=stride if fast else 1, padding=1, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, stride=2 if fast else 1, padding=1, groups=dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ConformerBlock(dim, kernel=9 if fast else 7, squeeze=squeeze) for _ in range(2)]
        )
        self.ctc = nn.Linear(dim, vocab)
        self.transducer = TransducerJoint(
            dim, vocab, durations=4 if head in {"tdt", "hybrid"} else 0
        )
        self.head = head

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Run ASR encoder and selected decoding head.

        Parameters
        ----------
        feats:
            Acoustic features of shape ``(batch, time, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Decoder logits.
        """

        x = self.subsample(feats.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        if self.head == "ctc":
            return self.ctc(x)
        if self.head == "hybrid":
            return torch.cat([self.ctc(x[:, :4]), self.transducer(x)], dim=-1)
        return self.transducer(x)


class CacheAwareFastConformerASR(nn.Module):
    """FastConformer encoder with chunk lookahead and activation cache."""

    def __init__(self, feat_dim: int = 32, dim: int = 48, vocab: int = 64) -> None:
        """Initialize cache-aware streaming FastConformer.

        Parameters
        ----------
        feat_dim:
            Acoustic feature dimension.
        dim:
            Encoder width.
        vocab:
            Vocabulary size.
        """

        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv1d(feat_dim, dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, stride=2, padding=1, groups=dim),
            nn.SiLU(),
        )
        self.block = ConformerBlock(dim, heads=4, kernel=9)
        self.cache_proj = nn.Linear(dim, dim)
        self.ctc = nn.Linear(dim, vocab)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Run chunked FastConformer inference with cache reuse.

        Parameters
        ----------
        feats:
            Acoustic features.

        Returns
        -------
        torch.Tensor
            Streaming CTC logits.
        """

        x = self.subsample(feats.transpose(1, 2)).transpose(1, 2)
        cache = torch.zeros(x.shape[0], 2, x.shape[-1], device=x.device, dtype=x.dtype)
        outs = []
        for start in range(0, 8, 2):
            chunk = x[:, start : start + 2]
            if chunk.shape[1] == 0:
                continue
            lookahead = x[:, start + 2 : start + 3]
            context = torch.cat([self.cache_proj(cache), chunk, lookahead], dim=1)
            encoded = self.block(context)
            outs.append(encoded[:, cache.shape[1] : cache.shape[1] + chunk.shape[1]])
            cache = context[:, -2:].detach() + 0.0 * context[:, -2:]
        return self.ctc(torch.cat(outs, dim=1))


class ContextNetRNNT(nn.Module):
    """ContextNet fully convolutional SE encoder with RNN-T joint network."""

    def __init__(self, feat_dim: int = 32, dim: int = 48, vocab: int = 64) -> None:
        """Initialize compact ContextNet RNN-T.

        Parameters
        ----------
        feat_dim:
            Acoustic feature dimension.
        dim:
            Hidden channel count.
        vocab:
            Token vocabulary.
        """

        super().__init__()
        self.in_proj = nn.Conv1d(feat_dim, dim, 1)
        self.blocks = nn.ModuleList(
            [
                TDSeparableBlock(dim, kernel, dilation=idx + 1, se=True)
                for idx, kernel in enumerate((9, 15, 23))
            ]
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(dim, dim, 1), nn.Sigmoid()
        )
        self.enc_proj = nn.Linear(dim, dim)
        self.joint = TransducerJoint(dim, vocab)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute ContextNet transducer logits.

        Parameters
        ----------
        feats:
            Acoustic feature sequence.

        Returns
        -------
        torch.Tensor
            RNN-T joint logits.
        """

        x = self.in_proj(feats.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
            x = x * self.global_context(x)
        return self.joint(self.enc_proj(x.transpose(1, 2)))


class LSTMTransducer(nn.Module):
    """LSTM encoder, prediction network, and RNN-T joint mechanism."""

    def __init__(self, feat_dim: int = 32, dim: int = 48, vocab: int = 64) -> None:
        """Initialize compact LSTM transducer.

        Parameters
        ----------
        feat_dim:
            Acoustic feature dimension.
        dim:
            Hidden dimension.
        vocab:
            Token vocabulary.
        """

        super().__init__()
        self.encoder = nn.LSTM(feat_dim, dim, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(dim * 2, dim)
        self.pred_embed = nn.Embedding(vocab, dim)
        self.pred = nn.LSTM(dim, dim, batch_first=True)
        self.joint = nn.Linear(dim, vocab)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute compact RNN-T logits.

        Parameters
        ----------
        feats:
            Acoustic feature sequence.

        Returns
        -------
        torch.Tensor
            Joint network logits.
        """

        enc, _ = self.encoder(feats)
        enc = self.enc_proj(enc[:, :4])
        seed = torch.zeros(feats.shape[0], 4, dtype=torch.long, device=feats.device)
        pred, _ = self.pred(self.pred_embed(seed))
        return self.joint(torch.tanh(enc.unsqueeze(2) + pred.unsqueeze(1))).mean(dim=2)


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN speaker encoder with Res2Net, SE, and attentive statistics."""

    def __init__(self, feat_dim: int = 32, channels: int = 48, emb: int = 32) -> None:
        """Initialize ECAPA-style speaker encoder.

        Parameters
        ----------
        feat_dim:
            Acoustic feature dimension.
        channels:
            TDNN channel count.
        emb:
            Speaker embedding dimension.
        """

        super().__init__()
        self.stem = nn.Conv1d(feat_dim, channels, 5, padding=2)
        self.branches = nn.ModuleList(
            [TDSeparableBlock(channels, 3, dilation=i + 1, se=True) for i in range(3)]
        )
        self.attn = nn.Conv1d(channels * 3, 1, 1)
        self.proj = nn.Linear(channels * 6, emb)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute a speaker embedding.

        Parameters
        ----------
        feats:
            Acoustic features.

        Returns
        -------
        torch.Tensor
            Speaker embedding.
        """

        x = F.relu(self.stem(feats.transpose(1, 2)))
        outs = []
        for block in self.branches:
            x = block(x)
            outs.append(x)
        h = torch.cat(outs, dim=1)
        w = torch.softmax(self.attn(h), dim=-1)
        mean = (h * w).sum(dim=-1)
        std = torch.sqrt(((h - mean.unsqueeze(-1)).pow(2) * w).sum(dim=-1).clamp_min(1e-6))
        return self.proj(torch.cat([mean, std], dim=-1))


class FastPitchTTS(nn.Module):
    """FastPitch-style parallel duration, pitch, and mel predictor."""

    def __init__(self, vocab: int = 80, dim: int = 48, mel: int = 32) -> None:
        """Initialize FastPitch compact model.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        dim:
            Hidden width.
        mel:
            Mel channel count.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.encoder = nn.ModuleList([ConformerBlock(dim, heads=4, kernel=5) for _ in range(2)])
        self.duration = nn.Linear(dim, 1)
        self.pitch = nn.Linear(dim, 1)
        self.pitch_embed = nn.Linear(1, dim)
        self.decoder = nn.ModuleList([ConformerBlock(dim, heads=4, kernel=5) for _ in range(2)])
        self.mel = nn.Linear(dim, mel)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate parallel mel frames from token ids.

        Parameters
        ----------
        tokens:
            Token ids of shape ``(batch, text_time)``.

        Returns
        -------
        torch.Tensor
            Mel frames.
        """

        x = self.embed(tokens)
        for block in self.encoder:
            x = block(x)
        pitch = self.pitch(x)
        x = x + self.pitch_embed(pitch)
        durations = F.softplus(self.duration(x))
        x = x * (1.0 + durations)
        for block in self.decoder:
            x = block(x)
        return self.mel(x)


class MixerTTS(nn.Module):
    """Mixer-TTS with token/channel MLP mixing and FastPitch predictors."""

    def __init__(self, vocab: int = 80, dim: int = 48, mel: int = 32) -> None:
        """Initialize Mixer-TTS.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        dim:
            Hidden width.
        mel:
            Mel channel count.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.token_mix = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.duration = nn.Linear(dim, 1)
        self.pitch = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, mel)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate compact Mixer-TTS mel frames.

        Parameters
        ----------
        tokens:
            Token ids.

        Returns
        -------
        torch.Tensor
            Mel frames.
        """

        x = self.embed(tokens)
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(x)
        x = x * (1.0 + F.softplus(self.duration(x))) + self.pitch(x)
        return self.head(x)


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator with transposed upsampling and MRF residual branches."""

    def __init__(self, mel: int = 32, channels: int = 48) -> None:
        """Initialize compact HiFi-GAN generator.

        Parameters
        ----------
        mel:
            Mel channel count.
        channels:
            Hidden channels.
        """

        super().__init__()
        self.pre = nn.Conv1d(mel, channels, 7, padding=3)
        self.up = nn.ModuleList(
            [nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1) for _ in range(2)]
        )
        self.res = nn.ModuleList(
            [nn.Conv1d(channels, channels, 3, padding=d, dilation=d) for d in (1, 3, 5)]
        )
        self.post = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform from mel features.

        Parameters
        ----------
        mel:
            Mel tensor of shape ``(batch, mel, time)``.

        Returns
        -------
        torch.Tensor
            Waveform tensor.
        """

        x = F.leaky_relu(self.pre(mel), 0.1)
        for up in self.up:
            x = F.leaky_relu(up(x), 0.1)
            x = sum(F.leaky_relu(branch(x), 0.1) for branch in self.res) / len(self.res)
        return torch.tanh(self.post(x))


class FastPitchHiFiGANE2E(nn.Module):
    """End-to-end FastPitch text-to-mel plus HiFi-GAN waveform generator."""

    def __init__(self) -> None:
        """Initialize FastPitch and HiFi-GAN stages."""

        super().__init__()
        self.fastpitch = FastPitchTTS()
        self.vocoder = HiFiGANGenerator()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate waveform directly from text tokens.

        Parameters
        ----------
        tokens:
            Text token ids.

        Returns
        -------
        torch.Tensor
            Synthesized waveform.
        """

        mel = self.fastpitch(tokens).transpose(1, 2)
        return self.vocoder(mel)


class TalkNetTTS(nn.Module):
    """TalkNet duration, pitch, and spectrogram convolutional networks."""

    def __init__(self, vocab: int = 80, dim: int = 48, mel: int = 32) -> None:
        """Initialize compact TalkNet.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        dim:
            Hidden width.
        mel:
            Mel channel count.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.duration_net = nn.Sequential(
            TDSeparableBlock(dim, 5), TDSeparableBlock(dim, 7), nn.Conv1d(dim, 1, 1)
        )
        self.pitch_net = nn.Sequential(TDSeparableBlock(dim, 5), nn.Conv1d(dim, 1, 1))
        self.spect_net = nn.Sequential(
            TDSeparableBlock(dim, 7), TDSeparableBlock(dim, 9), nn.Conv1d(dim, mel, 1)
        )
        self.pitch_embed = nn.Conv1d(1, dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate mel frames with explicit duration and pitch predictors.

        Parameters
        ----------
        tokens:
            Text token ids.

        Returns
        -------
        torch.Tensor
            Mel spectrogram frames.
        """

        x = self.embed(tokens).transpose(1, 2)
        durations = F.softplus(self.duration_net(x))
        expanded = x * (1.0 + durations)
        pitch = self.pitch_net(expanded)
        return self.spect_net(expanded + self.pitch_embed(pitch)).transpose(1, 2)


class LocationVariableConv1d(nn.Module):
    """Location-variable convolution with mel-predicted kernels."""

    def __init__(self, channels: int, cond_channels: int, kernel: int = 3) -> None:
        """Initialize kernel predictor and base projection.

        Parameters
        ----------
        channels:
            Noise/audio channels.
        cond_channels:
            Conditioning mel channels.
        kernel:
            Dynamic kernel size.
        """

        super().__init__()
        self.channels = channels
        self.kernel = kernel
        self.kernel_predictor = nn.Conv1d(cond_channels, channels * kernel, 3, padding=1)
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, audio: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply location-variable convolution to an audio feature sequence.

        Parameters
        ----------
        audio:
            Audio/noise feature sequence.
        cond:
            Conditioning mel sequence.

        Returns
        -------
        torch.Tensor
            Dynamically filtered sequence.
        """

        kernels = self.kernel_predictor(cond).view(
            audio.shape[0], self.channels, self.kernel, audio.shape[-1]
        )
        padded = F.pad(audio, (self.kernel // 2, self.kernel // 2))
        windows = padded.unfold(dimension=-1, size=self.kernel, step=1).transpose(-1, -2)
        return self.pointwise((windows * kernels).sum(dim=2))


class UnivNetGenerator(nn.Module):
    """UnivNet generator with noise input and LVC residual stack."""

    def __init__(self, mel: int = 32, channels: int = 40) -> None:
        """Initialize compact UnivNet generator.

        Parameters
        ----------
        mel:
            Mel channel count.
        channels:
            Hidden channel count.
        """

        super().__init__()
        self.noise = nn.Conv1d(mel, channels, 1)
        self.cond = nn.Conv1d(mel, mel, 3, padding=1)
        self.lvc = nn.ModuleList([LocationVariableConv1d(channels, mel, 3) for _ in range(3)])
        self.up = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
        self.post = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform with noise-conditioned LVC blocks.

        Parameters
        ----------
        mel:
            Mel spectrogram.

        Returns
        -------
        torch.Tensor
            Waveform tensor.
        """

        cond = self.cond(mel)
        x = torch.tanh(self.noise(mel))
        for layer in self.lvc:
            x = x + F.leaky_relu(layer(x, cond), 0.2)
        return torch.tanh(self.post(F.leaky_relu(self.up(x), 0.2)))


class StyleSpectrogramEnhancer(nn.Module):
    """StyleGAN2-like spectrogram enhancer with modulated convolution."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize compact spectrogram enhancer.

        Parameters
        ----------
        channels:
            Hidden channel count.
        """

        super().__init__()
        self.enc = nn.Conv2d(1, channels, 3, padding=1)
        self.style = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels * 2)
        )
        self.weight = nn.Parameter(torch.randn(channels, channels, 3, 3) * 0.02)
        self.noise_strength = nn.Parameter(torch.ones(1) * 0.05)
        self.out = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Enhance a noisy spectrogram image.

        Parameters
        ----------
        spec:
            Spectrogram image.

        Returns
        -------
        torch.Tensor
            Enhanced spectrogram image.
        """

        x = F.leaky_relu(self.enc(spec), 0.2)
        scale, bias = self.style(x).chunk(2, dim=-1)
        demod = torch.rsqrt((self.weight.pow(2).sum(dim=(1, 2, 3)) + 1e-6)).view(-1, 1, 1, 1)
        y = F.conv2d(x, self.weight * demod, padding=1)
        y = y * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        y = y + self.noise_strength * torch.tanh(spec.mean(dim=1, keepdim=True))
        return self.out(F.leaky_relu(y, 0.2))


class WaveGlow(nn.Module):
    """WaveGlow flow with invertible 1x1 convolution and affine coupling."""

    def __init__(self, channels: int = 8, mel: int = 32) -> None:
        """Initialize compact WaveGlow.

        Parameters
        ----------
        channels:
            Audio channel grouping width.
        mel:
            Mel feature count.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.eye(channels))
        self.cond = nn.Conv1d(mel, channels, 1)
        self.coupling = nn.Sequential(
            nn.Conv1d(channels, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, channels, 3, padding=1)
        )

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Apply one conditional normalizing-flow step.

        Parameters
        ----------
        packed:
            Tensor of shape ``(batch, channels + mel, time)``.

        Returns
        -------
        torch.Tensor
            Flow-transformed audio channels.
        """

        audio, mel = packed[:, : self.weight.shape[0]], packed[:, self.weight.shape[0] :]
        z = torch.einsum("ij,bjt->bit", self.weight, audio)
        h = z + self.cond(mel)
        a, b = self.coupling(h).chunk(2, dim=1)
        z1, z2 = z.chunk(2, dim=1)
        return torch.cat(
            [z1, z2 * torch.exp(torch.tanh(a[:, : z2.shape[1]])) + b[:, : z2.shape[1]]], dim=1
        )


class RadTTS(nn.Module):
    """RadTTS-like text encoder with duration prior and affine flow decoder."""

    def __init__(self, vocab: int = 80, dim: int = 48, mel: int = 32) -> None:
        """Initialize compact RadTTS.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        dim:
            Hidden width.
        mel:
            Mel channel count.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.flow = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(3)])
        self.duration = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, mel)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Generate mel features through affine coupling-style transforms.

        Parameters
        ----------
        tokens:
            Text token ids.

        Returns
        -------
        torch.Tensor
            Mel features.
        """

        x = self.embed(tokens)
        x = x * (1.0 + F.softplus(self.duration(x)))
        for layer in self.flow:
            shift, log_scale = layer(x).chunk(2, dim=-1)
            x = x * torch.exp(torch.tanh(log_scale)) + shift
        return self.head(x)


class Tacotron2Tiny(nn.Module):
    """Tacotron2-style encoder, location-sensitive attention, autoregressive decoder."""

    def __init__(self, vocab: int = 80, dim: int = 48, mel: int = 32) -> None:
        """Initialize compact Tacotron2.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        dim:
            Hidden width.
        mel:
            Mel channel count.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.encoder = nn.GRU(dim, dim // 2, bidirectional=True, batch_first=True)
        self.attn_loc = nn.Conv1d(1, 1, 3, padding=1)
        self.decoder = nn.GRUCell(dim + mel, dim)
        self.prenet = nn.Linear(mel, mel)
        self.out = nn.Linear(dim, mel)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode a short teacher-forced mel sequence.

        Parameters
        ----------
        tokens:
            Text token ids.

        Returns
        -------
        torch.Tensor
            Mel frames.
        """

        enc, _ = self.encoder(self.embed(tokens))
        weights = torch.zeros(tokens.shape[0], tokens.shape[1], device=tokens.device)
        frame = torch.zeros(tokens.shape[0], self.prenet.in_features, device=tokens.device)
        hidden = torch.zeros(tokens.shape[0], enc.shape[-1], device=tokens.device)
        outs = []
        for _ in range(4):
            loc = self.attn_loc(weights.unsqueeze(1)).squeeze(1)
            score = (enc * hidden.unsqueeze(1)).sum(-1) + loc
            weights = torch.softmax(score, dim=-1)
            ctx = (enc * weights.unsqueeze(-1)).sum(1)
            hidden = self.decoder(torch.cat([ctx, F.relu(self.prenet(frame))], dim=-1), hidden)
            frame = self.out(hidden)
            outs.append(frame)
        return torch.stack(outs, dim=1)


class HashNeRF(nn.Module):
    """Instant-NGP-style multiresolution hash-grid NeRF."""

    def __init__(
        self, levels: int = 4, table_size: int = 32, feat: int = 2, hidden: int = 32
    ) -> None:
        """Initialize hash tables and radiance MLPs.

        Parameters
        ----------
        levels:
            Number of hash-grid resolutions.
        table_size:
            Entries per level.
        feat:
            Features per entry.
        hidden:
            MLP hidden width.
        """

        super().__init__()
        self.levels = levels
        self.table_size = table_size
        self.tables = nn.Parameter(torch.randn(levels, table_size, feat) * 0.01)
        self.density = nn.Sequential(
            nn.Linear(levels * feat + 3, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.sigma = nn.Linear(hidden, 1)
        self.rgb = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 3), nn.Sigmoid()
        )

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """Hash coordinates at several resolutions.

        Parameters
        ----------
        xyz:
            Coordinates in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Concatenated hash features.
        """

        feats = []
        primes = torch.tensor([1, 2654435761, 805459861], device=xyz.device, dtype=torch.long)
        for level in range(self.levels):
            res = 4 * (2**level)
            grid = torch.floor(xyz * res).long()
            idx = ((grid * primes).sum(dim=-1) % self.table_size).clamp_min(0)
            feats.append(self.tables[level, idx])
        return torch.cat(feats, dim=-1)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Render compact radiance samples from packed positions and directions.

        Parameters
        ----------
        rays:
            Tensor ``(..., 6)`` with xyz and view direction.

        Returns
        -------
        torch.Tensor
            RGB and density.
        """

        xyz = torch.sigmoid(rays[..., :3])
        dirs = F.normalize(rays[..., 3:6], dim=-1)
        h = self.density(torch.cat([xyz, self.encode(xyz)], dim=-1))
        return torch.cat(
            [self.rgb(torch.cat([h, dirs], dim=-1)), F.softplus(self.sigma(h))], dim=-1
        )


class EventNeRFRenderer(nn.Module):
    """Event-camera NeRF renderer using brightness-difference events."""

    def __init__(self, hidden: int = 32) -> None:
        """Initialize compact event radiance field.

        Parameters
        ----------
        hidden:
            Hidden feature width.
        """

        super().__init__()
        self.field = HashNeRF(hidden=hidden)
        self.threshold = nn.Parameter(torch.tensor(0.2))

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Render event polarities from two timestamped ray samples.

        Parameters
        ----------
        rays:
            Packed ray origins/directions.

        Returns
        -------
        torch.Tensor
            Positive and negative event responses with RGB brightness.
        """

        early = self.field(rays)
        shifted = rays.clone()
        shifted[..., :3] = shifted[..., :3] + 0.05 * torch.tanh(rays[..., 3:6])
        late = self.field(shifted)
        log_delta = torch.log(late[..., :3].clamp_min(1e-4)) - torch.log(
            early[..., :3].clamp_min(1e-4)
        )
        pos = F.relu(log_delta - self.threshold.abs())
        neg = F.relu(-log_delta - self.threshold.abs())
        return torch.cat([pos, neg, late[..., 3:4]], dim=-1)


class PropNetEstimatorTiny(nn.Module):
    """nerfacc-style proposal-network PDF sampler."""

    def __init__(self, samples: int = 16) -> None:
        """Initialize proposal estimator.

        Parameters
        ----------
        samples:
            Number of ray samples.
        """

        super().__init__()
        self.samples = samples
        self.proposal = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Estimate proposal weights and resampled intervals.

        Parameters
        ----------
        rays:
            Packed ray origins/directions.

        Returns
        -------
        torch.Tensor
            Weighted sample depths and transmittance estimates.
        """

        t = torch.linspace(0.0, 1.0, self.samples, device=rays.device)
        pts = rays[:, None, :3] + rays[:, None, 3:6] * t[None, :, None]
        sigma = F.softplus(
            self.proposal(torch.cat([pts, t.view(1, -1, 1).expand(rays.shape[0], -1, -1)], dim=-1))
        ).squeeze(-1)
        pdf = sigma * torch.exp(-torch.cumsum(sigma, dim=1) / self.samples)
        weights = pdf / pdf.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.stack([t.expand_as(weights), weights, torch.cumsum(weights, dim=1)], dim=-1)


class OccGridEstimatorTiny(nn.Module):
    """nerfacc-style occupancy-grid ray sampler."""

    def __init__(self, resolution: int = 8, samples: int = 16) -> None:
        """Initialize occupancy grid estimator.

        Parameters
        ----------
        resolution:
            Grid resolution per axis.
        samples:
            Number of ray samples.
        """

        super().__init__()
        self.resolution = resolution
        self.samples = samples
        self.occupancy = nn.Parameter(torch.randn(resolution, resolution, resolution) * 0.1)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Sample along rays while skipping low-occupancy cells.

        Parameters
        ----------
        rays:
            Packed ray origins/directions.

        Returns
        -------
        torch.Tensor
            Occupancy weights and cumulative transmittance proxy.
        """

        t = torch.linspace(0.0, 1.0, self.samples, device=rays.device)
        pts = torch.sigmoid(rays[:, None, :3] + rays[:, None, 3:6] * t[None, :, None])
        idx = (pts * (self.resolution - 1)).long().clamp(0, self.resolution - 1)
        occ = torch.sigmoid(self.occupancy[idx[..., 0], idx[..., 1], idx[..., 2]])
        mask = (occ > occ.mean(dim=1, keepdim=True)).float()
        trans = 1.0 - torch.cumsum(mask, dim=1) / self.samples
        return torch.stack([t.expand_as(occ), occ * mask, trans.clamp_min(0.0)], dim=-1)


class MipNeRFRenderer(nn.Module):
    """Mip-NeRF renderer with integrated positional encoding."""

    def __init__(self, samples: int = 12, bands: int = 4, hidden: int = 32) -> None:
        """Initialize compact mip-NeRF renderer.

        Parameters
        ----------
        samples:
            Number of conical frustum samples.
        bands:
            Fourier frequency bands.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.samples = samples
        self.register_buffer("freq", 2.0 ** torch.arange(bands).float())
        encoded = 3 + 3 * bands * 2
        self.mlp = nn.Sequential(nn.Linear(encoded + 3, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def _ipe(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Compute integrated positional encoding.

        Parameters
        ----------
        mean:
            Gaussian mean positions.
        variance:
            Gaussian positional variances.

        Returns
        -------
        torch.Tensor
            Anti-aliased Fourier features.
        """

        scaled = mean[..., None, :] * self.freq.to(mean.device).view(1, 1, -1, 1)
        attenuation = torch.exp(
            -0.5 * variance[..., None, :] * self.freq.to(mean.device).view(1, 1, -1, 1).pow(2)
        )
        return torch.cat(
            [
                mean,
                (attenuation * torch.sin(scaled)).flatten(-2),
                (attenuation * torch.cos(scaled)).flatten(-2),
            ],
            dim=-1,
        )

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Render ray colors from conical frustum samples.

        Parameters
        ----------
        rays:
            Packed ray origins/directions.

        Returns
        -------
        torch.Tensor
            RGB and density summary.
        """

        t = torch.linspace(0.05, 1.0, self.samples, device=rays.device)
        mean = rays[:, None, :3] + rays[:, None, 3:6] * t[None, :, None]
        variance = (0.01 * t[None, :, None]).pow(2).expand_as(mean)
        dirs = F.normalize(rays[:, 3:6], dim=-1).unsqueeze(1).expand(-1, self.samples, -1)
        raw = self.mlp(torch.cat([self._ipe(mean, variance), dirs], dim=-1))
        sigma = F.softplus(raw[..., 3])
        weights = torch.softmax(sigma, dim=1)
        rgb = torch.sigmoid(raw[..., :3])
        return torch.cat(
            [(weights[..., None] * rgb).sum(dim=1), sigma.mean(dim=1, keepdim=True)], dim=-1
        )


class ZipNeRFTiny(nn.Module):
    """Zip-NeRF anti-aliased grid renderer with conical multisampling."""

    def __init__(self, samples: int = 10, features: int = 8) -> None:
        """Initialize compact Zip-NeRF renderer.

        Parameters
        ----------
        samples:
            Number of ray samples.
        features:
            Grid feature width.
        """

        super().__init__()
        self.samples = samples
        self.grid = nn.Parameter(torch.randn(1, features, 8, 8) * 0.02)
        self.proj = nn.Linear(features + 3, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Render with multisampled, radius-downweighted grid features.

        Parameters
        ----------
        rays:
            Packed ray origins/directions.

        Returns
        -------
        torch.Tensor
            RGB and density summary.
        """

        t = torch.linspace(0.05, 1.0, self.samples, device=rays.device)
        pts = rays[:, None, :3] + rays[:, None, 3:6] * t[None, :, None]
        offsets = torch.tensor([[0.0, 0.0], [0.03, -0.02], [-0.02, 0.03]], device=rays.device)
        xy = torch.tanh(pts[..., :2]).unsqueeze(2) + offsets.view(1, 1, 3, 2) * t.view(1, -1, 1, 1)
        flat_grid = xy.reshape(rays.shape[0], self.samples * 3, 1, 2)
        feat = F.grid_sample(
            self.grid.expand(rays.shape[0], -1, -1, -1), flat_grid, align_corners=False
        )
        feat = feat.squeeze(-1).transpose(1, 2).reshape(rays.shape[0], self.samples, 3, -1)
        radius_weight = torch.exp(
            -t.view(1, -1, 1, 1)
            * torch.arange(1, feat.shape[-1] + 1, device=rays.device).view(1, 1, 1, -1)
        )
        feat = (feat * radius_weight).mean(dim=2)
        dirs = F.normalize(rays[:, 3:6], dim=-1).unsqueeze(1).expand(-1, self.samples, -1)
        raw = self.head(F.relu(self.proj(torch.cat([feat, dirs], dim=-1))))
        sigma = F.softplus(raw[..., 3])
        weights = torch.softmax(sigma, dim=1)
        rgb = torch.sigmoid(raw[..., :3])
        return torch.cat(
            [(weights[..., None] * rgb).sum(dim=1), sigma.mean(dim=1, keepdim=True)], dim=-1
        )


class SplatfactoTiny(nn.Module):
    """3D Gaussian Splatting/Splatfacto parameter renderer."""

    def __init__(self, gaussians: int = 24) -> None:
        """Initialize trainable Gaussian cloud parameters.

        Parameters
        ----------
        gaussians:
            Number of compact splats.
        """

        super().__init__()
        self.means = nn.Parameter(torch.randn(gaussians, 3) * 0.1)
        self.scales = nn.Parameter(torch.zeros(gaussians, 3) - 3.0)
        self.quats = nn.Parameter(torch.randn(gaussians, 4))
        self.colors = nn.Parameter(torch.zeros(gaussians, 3))
        self.opacity = nn.Parameter(torch.zeros(gaussians, 1))

    def forward(self, camera: torch.Tensor) -> torch.Tensor:
        """Composite a tiny Gaussian cloud for camera samples.

        Parameters
        ----------
        camera:
            Camera/query tensor used as view direction.

        Returns
        -------
        torch.Tensor
            RGB-like composited color.
        """

        direction = F.normalize(camera[..., :3], dim=-1)
        view = torch.matmul(direction, self.means.t())
        weights = torch.softmax(view * torch.exp(self.scales.mean(-1)).unsqueeze(0), dim=-1)
        colors = torch.sigmoid(self.colors) * torch.sigmoid(self.opacity)
        return torch.matmul(weights, colors)


class ResidualDepthBlock(nn.Module):
    """ResNet-style residual block for Monodepth2 encoders."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, bottleneck: bool = False) -> None:
        """Initialize a residual encoder block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        stride:
            Spatial stride.
        bottleneck:
            Whether to use a ResNet-50-style bottleneck block.
        """

        super().__init__()
        if bottleneck:
            mid = max(out_ch // 4, 4)
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual depth-encoder block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual output feature map.
        """

        return F.relu(self.skip(x) + self.body(x))


class Monodepth2Tiny(nn.Module):
    """Monodepth2 ResNet encoder and multiscale sigmoid disparity decoder."""

    def __init__(self, base: int = 16, bottleneck: bool = False) -> None:
        """Initialize compact depth network.

        Parameters
        ----------
        base:
            Base channel width.
        bottleneck:
            Whether to use bottleneck residual blocks.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ResidualDepthBlock(base, base, bottleneck=bottleneck)
        self.layer2 = ResidualDepthBlock(base, base * 2, stride=2, bottleneck=bottleneck)
        self.layer3 = ResidualDepthBlock(base * 2, base * 4, stride=2, bottleneck=bottleneck)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(base * 4, base, 4, stride=2, padding=1)
        self.disp2 = nn.Conv2d(base * 2, 1, 3, padding=1)
        self.disp1 = nn.Conv2d(base * 2, 1, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict two-scale inverse-depth maps.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Full-resolution-ish disparity map.
        """

        e0 = self.stem(image)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        u2 = F.relu(self.up2(e3))
        d2 = torch.sigmoid(self.disp2(u2))
        u1 = F.relu(self.up1(torch.cat([u2, e2], dim=1)))
        d1 = torch.sigmoid(self.disp1(torch.cat([u1, e1], dim=1)))
        return F.interpolate(
            d1, size=image.shape[-2:], mode="bilinear", align_corners=False
        ) + F.interpolate(d2, size=image.shape[-2:], mode="bilinear", align_corners=False)


class LatentNPTiny(nn.Module):
    """Latent Neural Process with global Gaussian latent variable."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize latent NP encoders and decoder.

        Parameters
        ----------
        dim:
            Representation width.
        """

        super().__init__()
        self.xy = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.mu = nn.Linear(dim, dim)
        self.log_sigma = nn.Linear(dim, dim)
        self.dec = nn.Sequential(nn.Linear(dim + 1, dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Decode target distribution from context and global latent state.

        Parameters
        ----------
        inputs:
            ``(context_x, context_y, target_x)``.

        Returns
        -------
        torch.Tensor
            Target mean and positive scale.
        """

        cx, cy, tx = inputs
        r = self.xy(torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1)], dim=-1)).mean(dim=1)
        mu = self.mu(r)
        sigma = F.softplus(self.log_sigma(r))
        z = mu + 0.1 * sigma
        zt = z.unsqueeze(1).expand(-1, tx.shape[1], -1)
        out = self.dec(torch.cat([tx.unsqueeze(-1), zt], dim=-1))
        return torch.stack([out[..., 0], F.softplus(out[..., 1])], dim=-1)


class ACMixTiny(nn.Module):
    """ACMix block combining self-attention and convolutional aggregation."""

    def __init__(self, channels: int = 32, heads: int = 4) -> None:
        """Initialize ACMix projections.

        Parameters
        ----------
        channels:
            Feature channels.
        heads:
            Attention heads.
        """

        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.conv_kernel = nn.Conv2d(channels * 3, channels, 3, padding=1, groups=channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply shared-projection attention and convolution mixing.

        Parameters
        ----------
        image:
            Feature map.

        Returns
        -------
        torch.Tensor
            Mixed feature map.
        """

        q, k, v = self.qkv(image).chunk(3, dim=1)
        b, c, h, w = q.shape
        dim = c // self.heads
        qf = q.reshape(b, self.heads, dim, h * w).transpose(-1, -2)
        kf = k.reshape(b, self.heads, dim, h * w)
        vf = v.reshape(b, self.heads, dim, h * w).transpose(-1, -2)
        attn = torch.softmax(torch.matmul(qf, kf) / math.sqrt(dim), dim=-1)
        attn_out = torch.matmul(attn, vf).transpose(-1, -2).reshape(b, c, h, w)
        conv_out = self.conv_kernel(torch.cat([q, k, v], dim=1))
        return self.proj(attn_out + conv_out)


class Agent57Tiny(nn.Module):
    """Agent57-style recurrent Q-network with intrinsic/extrinsic heads."""

    def __init__(self, actions: int = 8, policies: int = 4) -> None:
        """Initialize compact Agent57 network.

        Parameters
        ----------
        actions:
            Number of Atari actions.
        policies:
            Number of beta/gamma policy arms.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(), nn.Conv2d(16, 32, 4, stride=2), nn.ReLU()
        )
        self.rnn = nn.GRU(32 * 9 * 9 + policies, 64, batch_first=True)
        self.ext = nn.Linear(64, actions)
        self.intr = nn.Linear(64, actions)
        self.bandit = nn.Linear(64, policies)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute mixed Q-values selected by the policy-arm controller.

        Parameters
        ----------
        frames:
            Atari frame stack ``(batch, 4, 84, 84)``.

        Returns
        -------
        torch.Tensor
            Action values.
        """

        f = self.encoder(frames).flatten(1)
        arm_seed = torch.zeros(frames.shape[0], 1, self.bandit.out_features, device=frames.device)
        h, _ = self.rnn(torch.cat([f.unsqueeze(1), arm_seed], dim=-1))
        beta = torch.softmax(self.bandit(h[:, -1]), dim=-1).mean(dim=-1, keepdim=True)
        return self.ext(h[:, -1]) + beta * self.intr(h[:, -1])


class AIFSTiny(nn.Module):
    """AIFS attention-GNN encoder/decoder with sliding-window transformer."""

    def __init__(self, nodes: int = 12, dim: int = 32) -> None:
        """Initialize graph weather forecaster.

        Parameters
        ----------
        nodes:
            Number of compact grid nodes.
        dim:
            Hidden width.
        """

        super().__init__()
        self.adj = nn.Parameter(torch.randn(nodes, nodes) * 0.02)
        self.enc = nn.Linear(6, dim)
        self.edge_q = nn.Linear(dim, dim)
        self.edge_k = nn.Linear(dim, dim)
        self.edge_v = nn.Linear(dim, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True)
        self.processor = nn.TransformerEncoder(layer, 2)
        self.window_bias = nn.Parameter(torch.randn(nodes, nodes) * 0.01)
        self.dec_q = nn.Linear(dim, dim)
        self.dec_k = nn.Linear(dim, dim)
        self.dec_v = nn.Linear(dim, dim)
        self.dec = nn.Linear(dim, 6)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forecast node variables after graph message passing.

        Parameters
        ----------
        state:
            Weather node state ``(batch, nodes, variables)``.

        Returns
        -------
        torch.Tensor
            Forecast node variables.
        """

        x = self.enc(state)
        graph_prior = torch.softmax(self.adj, dim=-1)
        q = self.edge_q(x)
        k = self.edge_k(x)
        v = self.edge_v(x)
        graph_attn = torch.softmax(
            torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]) + graph_prior, dim=-1
        )
        encoded = torch.matmul(graph_attn, v)
        nodes = encoded.shape[1]
        distance = torch.arange(nodes, device=state.device)
        window = (distance[:, None] - distance[None, :]).abs() <= 2
        mask = torch.where(window, self.window_bias, torch.full_like(self.window_bias, -1e4))
        processed = self.processor(encoded, mask=mask)
        dq = self.dec_q(processed)
        dk = self.dec_k(processed)
        dv = self.dec_v(processed)
        dec_attn = torch.softmax(
            torch.matmul(dq, dk.transpose(1, 2)) / math.sqrt(dq.shape[-1]) + graph_prior, dim=-1
        )
        return self.dec(torch.matmul(dec_attn, dv))


class HedgehogTiny(nn.Module):
    """Hedgehog linear attention with trainable softmax-mimic feature map."""

    def __init__(self, dim: int = 48, heads: int = 4, feat: int = 12) -> None:
        """Initialize Hedgehog attention.

        Parameters
        ----------
        dim:
            Model width.
        heads:
            Attention heads.
        feat:
            Feature-map dimension per head.
        """

        super().__init__()
        self.heads = heads
        self.feat = feat
        self.qkv = nn.Linear(dim, 3 * dim)
        self.map = nn.Sequential(
            nn.Linear(dim // heads, feat), nn.GELU(), nn.Linear(feat, feat), nn.Softplus()
        )
        self.out = nn.Linear(heads * feat, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal linear attention.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Updated sequence tensor.
        """

        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, d // self.heads)
        k = k.view(b, t, self.heads, d // self.heads)
        v = v.view(b, t, self.heads, d // self.heads)[..., : self.feat]
        phi_q = self.map(q)
        phi_k = self.map(k)
        kv = torch.einsum("bthf,bthg->bthfg", phi_k, v).cumsum(dim=1)
        norm = (phi_q * phi_k.cumsum(dim=1)).sum(-1, keepdim=True).clamp_min(1e-5)
        y = torch.einsum("bthf,bthfg->bthg", phi_q, kv) / norm
        return self.out(y.reshape(b, t, self.heads * self.feat))


class HiFiCTiny(nn.Module):
    """HiFiC image compression autoencoder with hyperprior and GAN decoder core."""

    def __init__(self, latent: int = 32) -> None:
        """Initialize compact HiFiC.

        Parameters
        ----------
        latent:
            Latent channel count.
        """

        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, latent, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(latent, latent, 5, stride=2, padding=2),
        )
        self.hyper = nn.Sequential(
            nn.Conv2d(latent, latent, 3, padding=1), nn.GELU(), nn.Conv2d(latent, latent, 1)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent, latent, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(latent, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compress and reconstruct an image.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Reconstructed image.
        """

        y = self.enc(image)
        scale = F.softplus(self.hyper(y))
        quantized = y + 0.01 * torch.tanh(scale)
        return self.dec(quantized)


class TimeLLMTiny(nn.Module):
    """Time-LLM reprogramming forecaster with frozen-token prototype prompts."""

    def __init__(self, dim: int = 48, prototypes: int = 16) -> None:
        """Initialize compact Time-LLM.

        Parameters
        ----------
        dim:
            Hidden width.
        prototypes:
            Number of text prototype tokens.
        """

        super().__init__()
        self.patch = nn.Conv1d(1, dim, 4, stride=2, padding=1)
        self.prototypes = nn.Parameter(torch.randn(prototypes, dim) * 0.02)
        self.reprogram = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.llm_block = ConformerBlock(dim, heads=4, kernel=3)
        self.head = nn.Linear(dim, 4)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """Forecast a univariate time series.

        Parameters
        ----------
        series:
            Tensor of shape ``(batch, time, 1)``.

        Returns
        -------
        torch.Tensor
            Forecast values.
        """

        x = self.patch(series.transpose(1, 2)).transpose(1, 2)
        p = self.prototypes.unsqueeze(0).expand(series.shape[0], -1, -1)
        x, _ = self.reprogram(x, p, p, need_weights=False)
        return self.head(self.llm_block(x).mean(dim=1))


def build_nemo_citrinet_ctc_bpe() -> nn.Module:
    """Build compact Citrinet CTC with SE separable convolutions."""

    return ConvASR(kernels=(11, 13, 15, 17), se=True)


def build_nemo_jasper_ctc() -> nn.Module:
    """Build compact Jasper CTC with large residual temporal convolutions."""

    return ConvASR(kernels=(11, 13, 17), se=False)


def build_nemo_quartznet_ctc() -> nn.Module:
    """Build compact QuartzNet CTC with separable temporal convolutions."""

    return ConvASR(kernels=(33, 39, 51), se=False)


def build_nemo_contextnet_rnnt() -> nn.Module:
    """Build compact ContextNet RNN-T with SE convolution encoder."""

    return ContextNetRNNT()


def build_nemo_conformer_ctc() -> nn.Module:
    """Build compact Conformer CTC."""

    return ConformerASR(head="ctc")


def build_nemo_conformer_transducer() -> nn.Module:
    """Build compact Conformer RNN-T."""

    return ConformerASR(head="rnnt")


def build_nemo_fastconformer_ctc() -> nn.Module:
    """Build compact FastConformer CTC with depthwise subsampling."""

    return ConformerASR(fast=True, head="ctc")


def build_nemo_fastconformer_transducer() -> nn.Module:
    """Build compact FastConformer RNN-T."""

    return ConformerASR(fast=True, head="rnnt")


def build_nemo_fastconformer_hybrid_ctc_rnnt() -> nn.Module:
    """Build compact FastConformer hybrid CTC/RNN-T."""

    return ConformerASR(fast=True, head="hybrid")


def build_nemo_fastconformer_cache_aware_streaming() -> nn.Module:
    """Build compact cache-aware streaming FastConformer."""

    return CacheAwareFastConformerASR()


def build_nemo_lstm_ctc() -> nn.Module:
    """Build compact LSTM-CTC acoustic model."""

    return nn.Sequential(
        nn.LSTM(32, 48, batch_first=True, bidirectional=True), _TupleFirst(), nn.Linear(96, 64)
    )


def build_nemo_lstm_transducer() -> nn.Module:
    """Build compact LSTM transducer acoustic model."""

    return LSTMTransducer()


class _TupleFirst(nn.Module):
    """Select the first element from recurrent module outputs."""

    def forward(self, pair: tuple[torch.Tensor, object]) -> torch.Tensor:
        """Return the sequence output.

        Parameters
        ----------
        pair:
            Recurrent output tuple.

        Returns
        -------
        torch.Tensor
            Sequence tensor.
        """

        return pair[0]


def build_nemo_squeezeformer_ctc() -> nn.Module:
    """Build compact Squeezeformer CTC."""

    return ConformerASR(squeeze=True, head="ctc")


def build_nemo_parakeet_rnnt_0_6b() -> nn.Module:
    """Build compact Parakeet FastConformer RNN-T."""

    return ConformerASR(fast=True, head="rnnt")


def build_nemo_parakeet_rnnt_1_1b() -> nn.Module:
    """Build compact larger Parakeet FastConformer RNN-T."""

    return ConformerASR(fast=True, dim=56, head="rnnt")


def build_nemo_parakeet_tdt_0_6b_v2() -> nn.Module:
    """Build compact Parakeet FastConformer TDT."""

    return ConformerASR(fast=True, head="tdt")


def build_nemo_parakeet_tdt_1_1b() -> nn.Module:
    """Build compact larger Parakeet FastConformer TDT."""

    return ConformerASR(fast=True, dim=56, head="tdt")


def build_nemo_ecapa_tdnn() -> nn.Module:
    """Build compact ECAPA-TDNN speaker model."""

    return ECAPATDNN()


def build_nemo_speakernet() -> nn.Module:
    """Build compact SpeakerNet x-vector/TDNN model."""

    return ECAPATDNN(channels=40)


def build_nemo_speaker_decoder() -> nn.Module:
    """Build compact speaker-classification decoder."""

    return nn.Sequential(ECAPATDNN(), nn.Linear(32, 8))


def build_nemo_titanet_large() -> nn.Module:
    """Build compact TitaNet speaker encoder."""

    return ECAPATDNN(channels=56, emb=40)


def build_nemo_marblenet_vad() -> nn.Module:
    """Build compact MarbleNet VAD model."""

    return ConvASR(vocab=2, kernels=(11, 13, 15), se=True)


def build_nemo_fastpitch() -> nn.Module:
    """Build compact FastPitch."""

    return FastPitchTTS()


def build_nemo_fastpitch_hifigan_e2e() -> nn.Module:
    """Build compact end-to-end FastPitch plus HiFi-GAN."""

    return FastPitchHiFiGANE2E()


def build_nemo_hifigan() -> nn.Module:
    """Build compact HiFi-GAN generator."""

    return HiFiGANGenerator()


def build_nemo_waveglow() -> nn.Module:
    """Build compact WaveGlow."""

    return WaveGlow()


def build_nemo_mixer_tts() -> nn.Module:
    """Build compact Mixer-TTS."""

    return MixerTTS()


def build_nemo_radtts() -> nn.Module:
    """Build compact RadTTS."""

    return RadTTS()


def build_nemo_tacotron2_encoder() -> nn.Module:
    """Build compact Tacotron2 encoder+attention+decoder."""

    return Tacotron2Tiny()


def build_nemo_tacotron2_decoder() -> nn.Module:
    """Build compact Tacotron2 decoder."""

    return Tacotron2Tiny()


def build_nemo_talknet() -> nn.Module:
    """Build compact TalkNet non-autoregressive TTS."""

    return TalkNetTTS()


def build_nemo_univnet() -> nn.Module:
    """Build compact UnivNet neural vocoder."""

    return UnivNetGenerator()


def build_nemo_spectrogram_enhancer() -> nn.Module:
    """Build compact StyleGAN-like spectrogram enhancer."""

    return StyleSpectrogramEnhancer()


def build_hashnerf() -> nn.Module:
    """Build compact hash-grid NeRF."""

    return HashNeRF()


def build_eventnerf_renderer() -> nn.Module:
    """Build compact EventNeRF renderer."""

    return EventNeRFRenderer()


def build_nerfacc_propnet_estimator() -> nn.Module:
    """Build compact nerfacc proposal-network estimator."""

    return PropNetEstimatorTiny()


def build_nerfacc_occgrid_estimator() -> nn.Module:
    """Build compact nerfacc occupancy-grid estimator."""

    return OccGridEstimatorTiny()


def build_nerfstudio_mipnerf_model() -> nn.Module:
    """Build compact mip-NeRF renderer."""

    return MipNeRFRenderer()


def build_nerfstudio_zipnerf() -> nn.Module:
    """Build compact Zip-NeRF renderer."""

    return ZipNeRFTiny()


def build_splatfacto() -> nn.Module:
    """Build compact Splatfacto Gaussian model."""

    return SplatfactoTiny()


def build_monodepth2_resnet18() -> nn.Module:
    """Build compact Monodepth2 ResNet-18-style model."""

    return Monodepth2Tiny(base=12, bottleneck=False)


def build_monodepth2_resnet50() -> nn.Module:
    """Build compact Monodepth2 ResNet-50-style model."""

    return Monodepth2Tiny(base=16, bottleneck=True)


def build_latent_np() -> nn.Module:
    """Build compact Latent Neural Process."""

    return LatentNPTiny()


def build_neuralcompression_hific() -> nn.Module:
    """Build compact HiFiC image compression model."""

    return HiFiCTiny()


def build_timellm() -> nn.Module:
    """Build compact Time-LLM."""

    return TimeLLMTiny()


def build_acmix() -> nn.Module:
    """Build compact ACMix block."""

    return ACMixTiny()


def build_agent57() -> nn.Module:
    """Build compact Agent57 network."""

    return Agent57Tiny()


def build_aifs() -> nn.Module:
    """Build compact AIFS graph weather forecaster."""

    return AIFSTiny()


def build_hedgehog() -> nn.Module:
    """Build compact Hedgehog attention."""

    return HedgehogTiny()


def example_audio() -> torch.Tensor:
    """Create compact acoustic features."""

    return torch.randn(1, 32, 32)


def example_tokens() -> torch.Tensor:
    """Create compact text tokens."""

    return torch.randint(0, 80, (1, 10))


def example_mel() -> torch.Tensor:
    """Create compact mel spectrogram."""

    return torch.randn(1, 32, 12)


def example_waveglow() -> torch.Tensor:
    """Create packed WaveGlow audio and mel features."""

    return torch.randn(1, 40, 12)


def example_spec2d() -> torch.Tensor:
    """Create compact spectrogram image."""

    return torch.randn(1, 1, 24, 24)


def example_rays() -> torch.Tensor:
    """Create compact NeRF ray samples."""

    return torch.randn(32, 6)


def example_camera() -> torch.Tensor:
    """Create compact camera query samples."""

    return torch.randn(8, 3)


def example_image() -> torch.Tensor:
    """Create compact RGB image."""

    return torch.randn(1, 3, 32, 32)


def example_latent_np() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact Neural Process context and target sets."""

    cx = torch.linspace(-1.0, 1.0, 8).unsqueeze(0)
    cy = torch.sin(3.0 * cx)
    tx = torch.linspace(-1.2, 1.2, 10).unsqueeze(0)
    return cx, cy, tx


def example_feature_map() -> torch.Tensor:
    """Create compact ACMix feature map."""

    return torch.randn(1, 32, 8, 8)


def example_atari() -> torch.Tensor:
    """Create compact Atari frame stack."""

    return torch.randn(1, 4, 84, 84)


def example_weather() -> torch.Tensor:
    """Create compact graph weather state."""

    return torch.randn(1, 12, 6)


def example_sequence() -> torch.Tensor:
    """Create compact sequence features."""

    return torch.randn(1, 12, 48)


def example_series() -> torch.Tensor:
    """Create compact time-series window."""

    return torch.randn(1, 24, 1)


def _entry(
    name: str, build: str, example: str, year: str, code: str
) -> tuple[str, str, str, str, str]:
    """Create a MENAGERIE_ENTRIES tuple.

    Parameters
    ----------
    name:
        Catalog name.
    build:
        Build function name.
    example:
        Example-input function name.
    year:
        Publication year.
    code:
        Menagerie code.

    Returns
    -------
    tuple[str, str, str, str, str]
        Registry tuple.
    """

    return (name, build, example, year, code)


MENAGERIE_ENTRIES = [
    _entry("nemo_citrinet_ctc_bpe", "build_nemo_citrinet_ctc_bpe", "example_audio", "2021", "ASR"),
    _entry("nemo_conformer_ctc", "build_nemo_conformer_ctc", "example_audio", "2020", "ASR"),
    _entry(
        "nemo_conformer_transducer",
        "build_nemo_conformer_transducer",
        "example_audio",
        "2020",
        "ASR",
    ),
    _entry(
        "nemo_fastconformer_cache_aware_streaming",
        "build_nemo_fastconformer_cache_aware_streaming",
        "example_audio",
        "2023",
        "ASR",
    ),
    _entry(
        "nemo_fastconformer_ctc", "build_nemo_fastconformer_ctc", "example_audio", "2023", "ASR"
    ),
    _entry(
        "nemo_fastconformer_hybrid_ctc_rnnt",
        "build_nemo_fastconformer_hybrid_ctc_rnnt",
        "example_audio",
        "2023",
        "ASR",
    ),
    _entry(
        "nemo_fastconformer_transducer",
        "build_nemo_fastconformer_transducer",
        "example_audio",
        "2023",
        "ASR",
    ),
    _entry("nemo_contextnet_rnnt", "build_nemo_contextnet_rnnt", "example_audio", "2020", "ASR"),
    _entry("nemo_ecapa_tdnn", "build_nemo_ecapa_tdnn", "example_audio", "2020", "SPK"),
    _entry("nemo_fastpitch", "build_nemo_fastpitch", "example_tokens", "2020", "TTS"),
    _entry("nemo_fastpitch_model", "build_nemo_fastpitch", "example_tokens", "2020", "TTS"),
    _entry("nemo_fastpitch_module", "build_nemo_fastpitch", "example_tokens", "2020", "TTS"),
    _entry(
        "nemo_fastpitch_hifigan_e2e",
        "build_nemo_fastpitch_hifigan_e2e",
        "example_tokens",
        "2020",
        "TTS",
    ),
    _entry("nemo_hifigan", "build_nemo_hifigan", "example_mel", "2020", "VOC"),
    _entry("nemo_hifigan_generator", "build_nemo_hifigan", "example_mel", "2020", "VOC"),
    _entry("nemo_hifigan_model", "build_nemo_hifigan", "example_mel", "2020", "VOC"),
    _entry("nemo_jasper_ctc", "build_nemo_jasper_ctc", "example_audio", "2019", "ASR"),
    _entry("nemo_lstm_ctc", "build_nemo_lstm_ctc", "example_audio", "2017", "ASR"),
    _entry("nemo_lstm_transducer", "build_nemo_lstm_transducer", "example_audio", "2017", "ASR"),
    _entry("nemo_marblenet_vad", "build_nemo_marblenet_vad", "example_audio", "2020", "ASR"),
    _entry("nemo_mixer_tts", "build_nemo_mixer_tts", "example_tokens", "2021", "TTS"),
    _entry("nemo_mixertts_model", "build_nemo_mixer_tts", "example_tokens", "2021", "TTS"),
    _entry("nemo_mixertts_module", "build_nemo_mixer_tts", "example_tokens", "2021", "TTS"),
    _entry(
        "nemo_parakeet_rnnt_0_6b", "build_nemo_parakeet_rnnt_0_6b", "example_audio", "2024", "ASR"
    ),
    _entry(
        "nemo_parakeet_rnnt_1_1b", "build_nemo_parakeet_rnnt_1_1b", "example_audio", "2024", "ASR"
    ),
    _entry(
        "nemo_parakeet_tdt_0_6b_v2",
        "build_nemo_parakeet_tdt_0_6b_v2",
        "example_audio",
        "2025",
        "ASR",
    ),
    _entry(
        "nemo_parakeet_tdt_1_1b", "build_nemo_parakeet_tdt_1_1b", "example_audio", "2024", "ASR"
    ),
    _entry("nemo_quartznet_ctc", "build_nemo_quartznet_ctc", "example_audio", "2019", "ASR"),
    _entry("nemo_radtts", "build_nemo_radtts", "example_tokens", "2021", "TTS"),
    _entry("nemo_radtts_module", "build_nemo_radtts", "example_tokens", "2021", "TTS"),
    _entry("nemo_speakernet", "build_nemo_speakernet", "example_audio", "2020", "SPK"),
    _entry(
        "nemo_squeezeformer_ctc", "build_nemo_squeezeformer_ctc", "example_audio", "2022", "ASR"
    ),
    _entry(
        "nemo_spectrogram_enhancer",
        "build_nemo_spectrogram_enhancer",
        "example_spec2d",
        "2021",
        "TTS",
    ),
    _entry(
        "nemo_tacotron2_decoder", "build_nemo_tacotron2_decoder", "example_tokens", "2018", "TTS"
    ),
    _entry(
        "nemo_tacotron2_encoder", "build_nemo_tacotron2_encoder", "example_tokens", "2018", "TTS"
    ),
    _entry("nemo_tacotron2_model", "build_nemo_tacotron2_encoder", "example_tokens", "2018", "TTS"),
    _entry("nemo_talknet", "build_nemo_talknet", "example_tokens", "2020", "TTS"),
    _entry("nemo_speaker_decoder", "build_nemo_speaker_decoder", "example_audio", "2020", "SPK"),
    _entry("nemo_titanet_large", "build_nemo_titanet_large", "example_audio", "2022", "SPK"),
    _entry("nemo_univnet", "build_nemo_univnet", "example_mel", "2021", "VOC"),
    _entry("nemo_univnet_generator", "build_nemo_univnet", "example_mel", "2021", "VOC"),
    _entry("nemo_waveglow", "build_nemo_waveglow", "example_waveglow", "2018", "VOC"),
    _entry("nemo_waveglow_module", "build_nemo_waveglow", "example_waveglow", "2018", "VOC"),
    _entry("eventnerf_renderer", "build_eventnerf_renderer", "example_rays", "2022", "3D"),
    _entry("torch_ngp_nerfnetwork", "build_hashnerf", "example_rays", "2022", "3D"),
    _entry("HashNeRF_NeRFSmall", "build_hashnerf", "example_rays", "2022", "3D"),
    _entry(
        "nerfacc_PropNetEstimator", "build_nerfacc_propnet_estimator", "example_rays", "2022", "3D"
    ),
    _entry(
        "nerfacc_OccGridEstimator", "build_nerfacc_occgrid_estimator", "example_rays", "2022", "3D"
    ),
    _entry("nerfstudio_instant_ngp_model", "build_hashnerf", "example_rays", "2022", "3D"),
    _entry(
        "nerfstudio_mipnerf_model", "build_nerfstudio_mipnerf_model", "example_rays", "2021", "3D"
    ),
    _entry("NerfactoModel", "build_hashnerf", "example_rays", "2023", "3D"),
    _entry("SplatfactoModel", "build_splatfacto", "example_camera", "2023", "3D"),
    _entry("nerfstudio_zipnerf", "build_nerfstudio_zipnerf", "example_rays", "2023", "3D"),
    _entry("NerfactoField", "build_hashnerf", "example_rays", "2023", "3D"),
    _entry("monodepth2_resnet18", "build_monodepth2_resnet18", "example_image", "2019", "DEP"),
    _entry("monodepth2_resnet50", "build_monodepth2_resnet50", "example_image", "2019", "DEP"),
    _entry("NP (Latent NP)", "build_latent_np", "example_latent_np", "2018", "NP"),
    _entry(
        "neuralcompression_hific", "build_neuralcompression_hific", "example_image", "2020", "IMG"
    ),
    _entry("NeuralForecast TimeLLM", "build_timellm", "example_series", "2024", "TS"),
    _entry("ACMix", "build_acmix", "example_feature_map", "2022", "CV"),
    _entry("Agent57", "build_agent57", "example_atari", "2020", "RL"),
    _entry("AIFS", "build_aifs", "example_weather", "2024", "WX"),
    _entry("Hedgehog", "build_hedgehog", "example_sequence", "2024", "LM"),
]

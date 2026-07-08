"""Wave-C compact reimplementations for dependency-gated audio and misc models.

The entries in this module replace recipes whose original packages are not
base-environment buildable.  They keep the load-bearing architecture primitives:
ESPnet/NeMo speech encoders and decoders, NCSN++ speech enhancement, tabular
attention MLPs, SpikingTorch-style spike backprop, and Allegro's strictly local
equivariant potential.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DepthwiseConvModule(nn.Module):
    """Conformer-style depthwise temporal convolution module."""

    def __init__(self, dim: int) -> None:
        """Initialize pointwise, gated depthwise, and output projections.

        Parameters
        ----------
        dim:
            Sequence feature width.
        """

        super().__init__()
        self.pw_in = nn.Conv1d(dim, dim * 2, 1)
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pw_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a gated depthwise temporal convolution.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, time, dim)``.

        Returns
        -------
        Tensor
            Updated sequence tensor.
        """

        y = self.pw_in(x.transpose(1, 2))
        gate, value = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * value
        y = F.silu(self.norm(self.dw(y)))
        return self.pw_out(y).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Macaron Conformer block with attention and convolution."""

    def __init__(self, dim: int = 64, heads: int = 4) -> None:
        """Initialize feed-forward, self-attention, and convolution paths."""

        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = DepthwiseConvModule(dim)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Run one Conformer block.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Updated sequence tensor.
        """

        x = x + 0.5 * self.ff1(x)
        attn, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))
        x = x + attn
        x = x + self.conv(self.conv_norm(x))
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class SpeechEncoder(nn.Module):
    """Configurable ESPnet/NeMo speech encoder."""

    def __init__(self, kind: str = "conformer", dim: int = 64, blocks: int = 2) -> None:
        """Initialize subsampling and the requested speech block family.

        Parameters
        ----------
        kind:
            Encoder family name.
        dim:
            Hidden width.
        blocks:
            Number of compact blocks.
        """

        super().__init__()
        self.kind = kind
        self.front = nn.Sequential(
            nn.Conv1d(80, dim, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        if kind in {"conformer", "fastconformer", "squeezeformer"}:
            self.blocks = nn.ModuleList([ConformerBlock(dim) for _ in range(blocks)])
            self.squeeze = (
                nn.Conv1d(dim, dim, 3, stride=2, padding=1) if kind == "squeezeformer" else None
            )
        elif kind == "ebranchformer":
            self.blocks = nn.ModuleList([EBranchformerBlock(dim) for _ in range(blocks)])
            self.squeeze = None
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, blocks)
            self.blocks = nn.ModuleList()
            self.squeeze = None
        else:
            self.blocks = nn.ModuleList(
                [JasperBlock(dim, separable=kind != "jasper") for _ in range(blocks + 1)]
            )
            self.squeeze = None
        self.out = nn.Linear(dim, dim)

    def forward(self, mel: Tensor) -> Tensor:
        """Encode log-mel speech features.

        Parameters
        ----------
        mel:
            Mel features as ``(batch, 80, time)``.

        Returns
        -------
        Tensor
            Encoded sequence.
        """

        x = self.front(mel).transpose(1, 2)
        if self.kind == "transformer":
            return self.out(self.transformer(x))
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 0 and self.squeeze is not None:
                x = self.squeeze(x.transpose(1, 2)).transpose(1, 2)
        return self.out(x)


class EBranchformerBlock(nn.Module):
    """E-Branchformer block with attention and cgMLP branches."""

    def __init__(self, dim: int = 64) -> None:
        """Initialize attention, convolutional gating MLP, and merge projection."""

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.gate = nn.Conv1d(dim, dim * 2, 5, padding=2, groups=1)
        self.merge = nn.Linear(dim * 2, dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply merged attention and convolutional-gating branches."""

        q = self.norm(x)
        attn, _ = self.attn(q, q, q)
        conv_a, conv_b = self.gate(q.transpose(1, 2)).chunk(2, dim=1)
        conv = (conv_a * torch.sigmoid(conv_b)).transpose(1, 2)
        x = x + self.merge(torch.cat([attn, conv], dim=-1))
        return x + self.ff(x)


class JasperBlock(nn.Module):
    """Jasper/Citrinet/ContextNet separable convolution block with SE."""

    def __init__(self, dim: int = 64, *, separable: bool = True) -> None:
        """Initialize a residual temporal convolution block."""

        super().__init__()
        groups = dim if separable else 1
        self.conv = nn.Conv1d(dim, dim, 9, padding=4, groups=groups)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.norm = nn.BatchNorm1d(dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 4, 1),
            nn.SiLU(),
            nn.Conv1d(dim // 4, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual convolution and squeeze-excitation."""

        y = self.pointwise(self.conv(x.transpose(1, 2)))
        y = F.silu(self.norm(y))
        y = y * self.se(y)
        return x + y.transpose(1, 2)


class SpeechHead(nn.Module):
    """Encoder wrapper with CTC/classification style projection."""

    def __init__(self, encoder: SpeechEncoder, classes: int = 32) -> None:
        """Initialize the encoder and output projection."""

        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(64, classes)

    def forward(self, mel: Tensor) -> Tensor:
        """Return frame logits from mel features."""

        return self.head(self.encoder(mel))


class RNNAttentionModel(nn.Module):
    """ESPnet-style RNN encoder/decoder with location attention."""

    def __init__(self, mode: str = "encoder") -> None:
        """Initialize an RNN attention model.

        Parameters
        ----------
        mode:
            ``"encoder"`` or ``"decoder"``.
        """

        super().__init__()
        self.mode = mode
        self.encoder = nn.LSTM(
            80, 64, num_layers=2, bidirectional=mode == "encoder", batch_first=True
        )
        enc_out = 128 if mode == "encoder" else 64
        self.query = nn.Linear(enc_out, 64)
        self.loc = nn.Conv1d(1, 8, 3, padding=1)
        self.energy = nn.Linear(72, 1)
        self.out = nn.Linear(enc_out, 32)

    def forward(self, mel: Tensor) -> Tensor:
        """Run RNN encoder or decoder attention over mel-like inputs."""

        seq = mel.transpose(1, 2) if mel.shape[1] == 80 else mel
        h, _ = self.encoder(seq)
        previous = torch.zeros(h.shape[0], 1, h.shape[2], device=h.device, dtype=h.dtype)
        loc = self.loc(
            torch.zeros(h.shape[0], 1, h.shape[1], device=h.device, dtype=h.dtype)
        ).transpose(1, 2)
        score = self.energy(torch.cat([torch.tanh(self.query(h)), loc], dim=-1)).softmax(dim=1)
        context = (score * h).sum(dim=1, keepdim=True) + previous
        return self.out(context.squeeze(1))


class DPRNNSeparator(nn.Module):
    """Dual-path RNN source separator."""

    def __init__(self, sources: int = 2) -> None:
        """Initialize encoder, intra/inter chunk RNNs, and mask head."""

        super().__init__()
        self.sources = sources
        self.enc = nn.Conv1d(1, 32, 8, stride=4, padding=2)
        self.intra = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.inter = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
        self.mask = nn.Conv1d(64, sources * 32, 1)
        self.dec = nn.ConvTranspose1d(32, 1, 8, stride=4, padding=2)

    def forward(self, audio: Tensor) -> Tensor:
        """Separate a waveform into source estimates."""

        z = self.enc(audio)
        chunks = z.unfold(-1, 8, 4).transpose(1, 2)
        bsz, chunks_n, channels, chunk = chunks.shape
        intra, _ = self.intra(chunks.permute(0, 1, 3, 2).reshape(bsz * chunks_n, chunk, channels))
        intra = intra.mean(dim=1).reshape(bsz, chunks_n, -1)
        inter, _ = self.inter(intra)
        masks = torch.sigmoid(self.mask(inter.transpose(1, 2))).view(
            bsz, self.sources, 32, chunks_n
        )
        base = F.interpolate(z, size=chunks_n, mode="nearest").unsqueeze(1)
        base = base.expand(-1, self.sources, -1, -1)
        return torch.stack(
            [self.dec(base[:, i] * masks[:, i]) for i in range(self.sources)], dim=1
        ).squeeze(2)


class TFGridNetSeparator(nn.Module):
    """TF-GridNet-style time-frequency separator."""

    def __init__(self) -> None:
        """Initialize grid embedding, BLSTM, attention, and mask projection."""

        super().__init__()
        self.embed = nn.Conv2d(1, 32, 3, padding=1)
        self.freq_rnn = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.mask = nn.Conv2d(64, 2, 1)

    def forward(self, audio: Tensor) -> Tensor:
        """Estimate two time-frequency masks from waveform samples."""

        grid = audio.squeeze(1).unfold(-1, 32, 16).abs().unsqueeze(1)
        x = F.gelu(self.embed(grid)).permute(0, 2, 3, 1)
        bsz, time, freq, channels = x.shape
        x, _ = self.freq_rnn(x.reshape(bsz * time, freq, channels))
        x = x.reshape(bsz, time * freq, -1)
        x, _ = self.attn(x, x, x)
        masks = torch.sigmoid(self.mask(x.reshape(bsz, time, freq, -1).permute(0, 3, 1, 2)))
        return masks.mean(dim=(-2, -1))


class Tacotron2Tiny(nn.Module):
    """Tacotron2-style encoder, attention decoder, and postnet."""

    def __init__(self, *, gst: bool = False) -> None:
        """Initialize compact Tacotron2.

        Parameters
        ----------
        gst:
            Whether to include global style token attention.
        """

        super().__init__()
        self.gst = gst
        self.embed = nn.Embedding(100, 64)
        self.enc = nn.Sequential(
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 5, padding=2), nn.ReLU()
        )
        self.style_tokens = nn.Parameter(torch.randn(8, 64) * 0.02)
        self.decoder = nn.GRUCell(80 + 64, 64)
        self.prenet = nn.Linear(80, 80)
        self.mel = nn.Linear(64, 80)
        self.postnet = nn.Conv1d(80, 80, 5, padding=2)

    def forward(self, tokens: Tensor) -> Tensor:
        """Generate teacher-forced mel frames from token ids."""

        x = self.embed(tokens.long()).transpose(1, 2)
        memory = self.enc(x).transpose(1, 2)
        context = memory.mean(dim=1)
        if self.gst:
            weights = torch.softmax(context @ self.style_tokens.t(), dim=-1)
            context = context + weights @ self.style_tokens
        frame = torch.zeros(tokens.shape[0], 80, device=tokens.device)
        outputs = []
        hidden = context
        for _ in range(6):
            hidden = self.decoder(torch.cat([self.prenet(frame), context], dim=-1), hidden)
            frame = self.mel(hidden)
            outputs.append(frame)
        mel = torch.stack(outputs, dim=2)
        return mel + self.postnet(mel)


class TransformerTTS(nn.Module):
    """ESPnet Transformer ASR/TTS style encoder-decoder."""

    def __init__(self, *, text_input: bool = True) -> None:
        """Initialize token/mel projections and transformer core."""

        super().__init__()
        self.text_input = text_input
        self.embed = nn.Embedding(100, 64)
        self.mel_in = nn.Linear(80, 64)
        self.pos = nn.Parameter(torch.randn(1, 64, 64) * 0.02)
        self.transformer = nn.Transformer(64, 4, 2, 2, 128, batch_first=True)
        self.out = nn.Linear(64, 80 if text_input else 32)

    def forward(self, x: Tensor) -> Tensor:
        """Run transformer encoder-decoder on token or mel inputs."""

        if self.text_input:
            src = self.embed(x.long())
            tgt = torch.zeros(x.shape[0], 8, 64, device=x.device)
        else:
            src = self.mel_in(x.transpose(1, 2))
            tgt = torch.zeros(x.shape[0], 8, 64, device=x.device)
        src = src + self.pos[:, : src.shape[1]]
        tgt = tgt + self.pos[:, : tgt.shape[1]]
        return self.out(self.transformer(src, tgt))


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN/MelGAN-style neural vocoder generator."""

    def __init__(self, bands: int = 1, *, residual: bool = True) -> None:
        """Initialize upsampling stack and residual blocks."""

        super().__init__()
        self.pre = nn.Conv1d(80, 64, 7, padding=3)
        self.up1 = nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2)
        self.up2 = nn.ConvTranspose1d(32, 16, 8, stride=4, padding=2)
        self.residual = residual
        self.resblocks = nn.ModuleList(
            [nn.Conv1d(16, 16, k, padding=k // 2, dilation=1) for k in (3, 7, 11)]
        )
        self.out = nn.Conv1d(16, bands, 7, padding=3)

    def forward(self, mel: Tensor) -> Tensor:
        """Synthesize waveform samples from mel features."""

        x = F.leaky_relu(self.pre(mel), 0.2)
        x = F.leaky_relu(self.up1(x), 0.2)
        x = F.leaky_relu(self.up2(x), 0.2)
        if self.residual:
            x = sum(F.leaky_relu(block(x), 0.2) for block in self.resblocks) / len(self.resblocks)
        return torch.tanh(self.out(x))


class NCSNppSpeechEnhancer(nn.Module):
    """NCSN++ score model for speech enhancement spectrograms."""

    def __init__(self) -> None:
        """Initialize Fourier noise embedding, U-Net blocks, and attention."""

        super().__init__()
        self.noise = nn.Linear(16, 32)
        self.down1 = nn.Conv2d(2, 32, 3, padding=1)
        self.down2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.up = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, spec: Tensor) -> Tensor:
        """Predict a complex spectrogram score."""

        sigma = spec.new_tensor([0.1])
        freqs = torch.arange(8, device=spec.device, dtype=spec.dtype)
        emb = torch.cat([torch.sin(freqs * sigma), torch.cos(freqs * sigma)]).unsqueeze(0)
        temb = self.noise(emb).view(1, 32, 1, 1)
        h1 = F.silu(self.down1(spec) + temb)
        h2 = F.silu(self.down2(h1))
        tokens = h2.flatten(2).transpose(1, 2)
        tokens, _ = self.attn(tokens, tokens, tokens)
        h2 = tokens.transpose(1, 2).reshape_as(h2)
        up = F.silu(self.up(h2))
        return self.out(torch.cat([up[..., : h1.shape[-2], : h1.shape[-1]], h1], dim=1))


class WideDeepAttentionMLP(nn.Module):
    """pytorch-widedeep context/self-attention tabular MLP."""

    def __init__(self, *, context: bool) -> None:
        """Initialize categorical embeddings, attention, and MLP head."""

        super().__init__()
        self.context = context
        self.cat0 = nn.Embedding(10, 8)
        self.cat1 = nn.Embedding(20, 8)
        self.cont = nn.Linear(11, 16)
        self.query = nn.Parameter(torch.randn(1, 1, 16) * 0.02)
        self.attn = nn.MultiheadAttention(16, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Score a mixed categorical/continuous tabular row."""

        cat0 = torch.remainder(torch.round(torch.abs(x[:, 0])), 10).long()
        cat1 = torch.remainder(torch.round(torch.abs(x[:, 1])), 20).long()
        tokens = torch.stack([self.cat0(cat0), self.cat1(cat1)], dim=1)
        tokens = F.pad(tokens, (0, 8))
        cont = self.cont(x[:, 2:]).unsqueeze(1)
        seq = torch.cat([tokens, cont], dim=1)
        query = self.query.expand(x.shape[0], -1, -1) if self.context else seq
        attended, _ = self.attn(query, seq, seq)
        pooled = attended.mean(dim=1)
        return self.mlp(torch.cat([pooled, cont.squeeze(1)], dim=-1))


class SpikingTorchMLP(nn.Module):
    """SpikingTorch-style MLP with surrogate spike backprop."""

    def __init__(self) -> None:
        """Initialize dense layers and learnable membrane decay."""

        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.beta = nn.Parameter(torch.tensor(0.85))

    def forward(self, x: Tensor) -> Tensor:
        """Unroll leaky integrate-and-fire dynamics over four timesteps."""

        mem = torch.zeros(x.shape[0], 128, device=x.device, dtype=x.dtype)
        out = torch.zeros(x.shape[0], 10, device=x.device, dtype=x.dtype)
        current = self.fc1(x)
        for step in range(4):
            mem = self.beta.sigmoid() * mem + current / float(step + 1)
            spike = torch.sigmoid(8.0 * (mem - 1.0))
            mem = mem * (1.0 - spike.detach())
            out = out + self.fc2(spike)
        return out / 4.0


class AllegroLocalPotential(nn.Module):
    """Allegro-style strictly local equivariant interatomic potential."""

    def __init__(self, types: int = 4, hidden: int = 32) -> None:
        """Initialize radial, angular, pair, and energy networks."""

        super().__init__()
        self.type_emb = nn.Embedding(types, hidden)
        self.radial = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.pair = nn.Sequential(
            nn.Linear(hidden * 2 + 9, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.energy = nn.Linear(hidden, 1)

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Compute local edge energies without message passing."""

        positions, atom_types, edge_index = inputs
        src, dst = edge_index
        vec = positions[dst] - positions[src]
        dist = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radial = torch.sin(
            torch.arange(1, 9, device=positions.device, dtype=positions.dtype) * dist
        )
        direction = vec / dist
        angular = _angular_features(direction)
        pair = torch.cat(
            [self.type_emb(atom_types[src]), self.type_emb(atom_types[dst]), angular], dim=-1
        )
        edge_feat = self.pair(pair) * self.radial(radial)
        atom_feat = torch.zeros(
            positions.shape[0], edge_feat.shape[-1], device=positions.device, dtype=positions.dtype
        )
        atom_feat = atom_feat.index_add(0, dst, edge_feat)
        return self.energy(atom_feat).sum(dim=0, keepdim=True)


def _angular_features(direction: Tensor) -> Tensor:
    """Return compact l=0/1/2 spherical-harmonic-like edge features."""

    x, y, z = direction.unbind(dim=-1)
    return torch.stack(
        [
            torch.ones_like(x),
            x,
            y,
            z,
            x * y,
            y * z,
            z * x,
            x.square() - y.square(),
            3 * z.square() - 1,
        ],
        dim=-1,
    )


def _mel() -> Tensor:
    """Return compact log-mel speech features."""

    return torch.randn(1, 80, 96)


def _wave() -> Tensor:
    """Return a compact mono waveform."""

    return torch.randn(1, 1, 1024)


def _tokens() -> Tensor:
    """Return compact token ids."""

    return torch.randint(0, 100, (1, 24))


def _spec() -> Tensor:
    """Return compact real/imag speech spectrogram."""

    return torch.randn(1, 2, 32, 32)


def _atoms() -> tuple[Tensor, Tensor, Tensor]:
    """Return a small molecular graph."""

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.0, 0.1], [1.0, 1.0, -0.2]])
    atom_types = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]], dtype=torch.long
    )
    return positions, atom_types, edge_index


def build_conformer() -> nn.Module:
    """Build a compact Conformer encoder."""

    return SpeechHead(SpeechEncoder("conformer")).eval()


def build_fastconformer() -> nn.Module:
    """Build a compact FastConformer encoder."""

    return SpeechHead(SpeechEncoder("fastconformer", blocks=1)).eval()


def build_squeezeformer() -> nn.Module:
    """Build a compact Squeezeformer encoder."""

    return SpeechHead(SpeechEncoder("squeezeformer")).eval()


def build_ebranchformer() -> nn.Module:
    """Build a compact E-Branchformer encoder."""

    return SpeechHead(SpeechEncoder("ebranchformer")).eval()


def build_transformer_asr() -> nn.Module:
    """Build a compact Transformer ASR encoder-decoder."""

    return SpeechHead(SpeechEncoder("transformer")).eval()


def build_jasper() -> nn.Module:
    """Build a compact Jasper/ConvASR encoder."""

    return SpeechHead(SpeechEncoder("jasper")).eval()


def build_citrinet() -> nn.Module:
    """Build a compact Citrinet encoder with separable SE blocks."""

    return SpeechHead(SpeechEncoder("citrinet")).eval()


def build_contextnet() -> nn.Module:
    """Build a compact ContextNet encoder with separable swish blocks."""

    return SpeechHead(SpeechEncoder("contextnet")).eval()


def build_rnn_encoder() -> nn.Module:
    """Build an ESPnet RNN encoder."""

    return RNNAttentionModel("encoder").eval()


def build_rnn_decoder() -> nn.Module:
    """Build an ESPnet RNN decoder."""

    return RNNAttentionModel("decoder").eval()


def build_dprnn() -> nn.Module:
    """Build a compact DPRNN separator."""

    return DPRNNSeparator().eval()


def build_tfgridnet() -> nn.Module:
    """Build a compact TF-GridNet separator."""

    return TFGridNetSeparator().eval()


def build_tacotron2() -> nn.Module:
    """Build compact Tacotron2."""

    return Tacotron2Tiny().eval()


def build_tacotron2_gst() -> nn.Module:
    """Build compact Tacotron2 with global style tokens."""

    return Tacotron2Tiny(gst=True).eval()


def build_transformer_tts() -> nn.Module:
    """Build compact Transformer-TTS."""

    return TransformerTTS(text_input=True).eval()


def build_hifigan() -> nn.Module:
    """Build compact HiFi-GAN vocoder."""

    return HiFiGANGenerator(residual=True).eval()


def build_melgan() -> nn.Module:
    """Build compact MelGAN vocoder."""

    return HiFiGANGenerator(residual=False).eval()


def build_multiband_melgan() -> nn.Module:
    """Build compact multiband MelGAN vocoder."""

    return HiFiGANGenerator(bands=4, residual=False).eval()


def build_sgmse() -> nn.Module:
    """Build compact NCSN++ speech enhancement score model."""

    return NCSNppSpeechEnhancer().eval()


def build_widedeep_context() -> nn.Module:
    """Build compact ContextAttentionMLP."""

    return WideDeepAttentionMLP(context=True).eval()


def build_widedeep_self() -> nn.Module:
    """Build compact SelfAttentionMLP."""

    return WideDeepAttentionMLP(context=False).eval()


def build_spikingtorch_bp() -> nn.Module:
    """Build compact SpikingTorch backprop-through-spikes MLP."""

    return SpikingTorchMLP().eval()


def build_allegro_potential() -> nn.Module:
    """Build compact Allegro local equivariant potential."""

    return AllegroLocalPotential().eval()


def build_darts_alias() -> nn.Module:
    """Build the existing DARTS CIFAR classic under the source target name."""

    from menagerie.classics.darts_cifar_derived_network import build

    return build()


def build_attngan_alias() -> nn.Module:
    """Build the existing AttnGAN generator under the source target name."""

    from menagerie.classics.text_conditioned_gans import build_attngan_generator

    return build_attngan_generator()


def build_controlgan_alias() -> nn.Module:
    """Build the existing ControlGAN generator under the source target name."""

    from menagerie.classics.text_conditioned_gans import build_controlgan_generator

    return build_controlgan_generator()


def build_dmgan_alias() -> nn.Module:
    """Build the existing DM-GAN generator under the source target name."""

    from menagerie.classics.text_conditioned_gans import build_dmgan_generator

    return build_dmgan_generator()


def build_lerobot_act_alias() -> nn.Module:
    """Build the existing compact ACT policy under the source target name."""

    from menagerie.classics.lerobot_vla_policies import build_act

    return build_act()


def example_mel() -> Tensor:
    """Return mel features for speech encoders."""

    return _mel()


def example_wave() -> Tensor:
    """Return waveform input for separators."""

    return _wave()


def example_tokens() -> Tensor:
    """Return token ids for TTS models."""

    return _tokens()


def example_spec() -> Tensor:
    """Return complex spectrogram channels for NCSN++."""

    return _spec()


def example_tabular() -> Tensor:
    """Return tabular mixed feature input."""

    return torch.randn(1, 13)


def example_spiking() -> Tensor:
    """Return flattened MNIST-like input."""

    return torch.randn(1, 784)


def example_atoms() -> tuple[Tensor, Tensor, Tensor]:
    """Return Allegro molecular graph input."""

    return _atoms()


def example_text_gan() -> tuple[Tensor, Tensor, Tensor]:
    """Return text-to-image GAN conditioning inputs."""

    from menagerie.classics.text_conditioned_gans import example_text_gan_input

    return example_text_gan_input()


def example_act() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ACT policy inputs."""

    from menagerie.classics.lerobot_vla_policies import example_act as _example_act

    return _example_act()


def example_darts() -> Tensor:
    """Return CIFAR image input for DARTS."""

    from menagerie.classics.darts_cifar_derived_network import example_input

    return example_input()


def _build_registered_classic(name: str) -> nn.Module:
    """Build an already-registered classic by canonical name."""

    from menagerie.classics import CLASSICS

    return CLASSICS[name]["build"]().eval()


def _example_registered_classic(name: str) -> object:
    """Return an example input from an already-registered classic."""

    from menagerie.classics import CLASSICS

    return CLASSICS[name]["example_input"]()


def build_mmagic_edsr() -> nn.Module:
    """Build the existing EDSR super-resolution classic."""

    return _build_registered_classic("EDSR x4 (batch-norm-free residual scaling SR)")


def build_mmagic_esrgan() -> nn.Module:
    """Build the existing ESRGAN RRDBNet classic."""

    return _build_registered_classic("ESRGAN RRDBNet x4 (residual-in-residual dense SR generator)")


def build_mmagic_rdn() -> nn.Module:
    """Build the existing RDN super-resolution classic."""

    return _build_registered_classic("RDN x2 (residual dense network super-resolution)")


def build_mmagic_real_esrgan() -> nn.Module:
    """Build the existing Real-ESRGAN RRDBNet classic."""

    return _build_registered_classic(
        "Real-ESRGAN RRDBNet x4plus (synthetic-degradation RRDBNet generator)"
    )


def build_mmagic_restormer() -> nn.Module:
    """Build the existing Restormer restoration classic."""

    return _build_registered_classic("mmagic_restormer")


def build_mmagic_srcnn() -> nn.Module:
    """Build the existing SRCNN super-resolution classic."""

    return _build_registered_classic("SRCNN x4 (9-5-5 convolutional super-resolution)")


def build_mmagic_srgan() -> nn.Module:
    """Build the existing SRGAN/SRResNet classic."""

    return _build_registered_classic("SRGAN (SRResNet generator, PixelShuffle upsampling)")


def build_mmagic_swinir() -> nn.Module:
    """Build the existing SwinIR x4 classic."""

    return _build_registered_classic(
        "SwinIR Classical SR x4 (window/shifted-window MSA, RSTB, pixel-shuffle x4)"
    )


def example_mmagic_edsr() -> object:
    """Return the existing EDSR example input."""

    return _example_registered_classic("EDSR x4 (batch-norm-free residual scaling SR)")


def example_mmagic_esrgan() -> object:
    """Return the existing ESRGAN example input."""

    return _example_registered_classic(
        "ESRGAN RRDBNet x4 (residual-in-residual dense SR generator)"
    )


def example_mmagic_rdn() -> object:
    """Return the existing RDN example input."""

    return _example_registered_classic("RDN x2 (residual dense network super-resolution)")


def example_mmagic_real_esrgan() -> object:
    """Return the existing Real-ESRGAN example input."""

    return _example_registered_classic(
        "Real-ESRGAN RRDBNet x4plus (synthetic-degradation RRDBNet generator)"
    )


def example_mmagic_restormer() -> object:
    """Return the existing Restormer example input."""

    return _example_registered_classic("mmagic_restormer")


def example_mmagic_srcnn() -> object:
    """Return the existing SRCNN example input."""

    return _example_registered_classic("SRCNN x4 (9-5-5 convolutional super-resolution)")


def example_mmagic_srgan() -> object:
    """Return the existing SRGAN/SRResNet example input."""

    return _example_registered_classic("SRGAN (SRResNet generator, PixelShuffle upsampling)")


def example_mmagic_swinir() -> object:
    """Return the existing SwinIR x4 example input."""

    return _example_registered_classic(
        "SwinIR Classical SR x4 (window/shifted-window MSA, RSTB, pixel-shuffle x4)"
    )


MENAGERIE_ENTRIES = [
    ("speechbrain_sgmse_voicebank", "build_sgmse", "example_spec", "2023", "audio/speech"),
    (
        "WideDeep-ContextAttentionMLP",
        "build_widedeep_context",
        "example_tabular",
        "2021",
        "tabular",
    ),
    ("WideDeep-SelfAttentionMLP", "build_widedeep_self", "example_tabular", "2021", "tabular"),
    ("spikingtorch_bp", "build_spikingtorch_bp", "example_spiking", "2021", "spiking"),
    ("Allegro", "build_allegro_potential", "example_atoms", "2022", "atomistic/equivariant"),
    ("DARTS-CIFAR-NormalCell", "build_darts_alias", "example_darts", "2019", "vision/nas"),
    ("attngan_g_net", "build_attngan_alias", "example_text_gan", "2018", "text-to-image"),
    ("controlgan_g_net", "build_controlgan_alias", "example_text_gan", "2019", "text-to-image"),
    ("dm_gan_g_net", "build_dmgan_alias", "example_text_gan", "2019", "text-to-image"),
    ("lerobot_act", "build_lerobot_act_alias", "example_act", "2023", "robotics"),
    ("espnet_asr_conformer_ctc", "build_conformer", "example_mel", "2020", "audio/speech"),
    ("espnet_asr_conformer_transducer", "build_conformer", "example_mel", "2020", "audio/speech"),
    ("espnet_conformer_encoder", "build_conformer", "example_mel", "2020", "audio/speech"),
    ("espnet_conformer_separator", "build_conformer", "example_mel", "2020", "audio/speech"),
    (
        "espnet_contextual_block_conformer_encoder",
        "build_conformer",
        "example_mel",
        "2020",
        "audio/speech",
    ),
    ("espnet_transducer_encoder", "build_conformer", "example_mel", "2020", "audio/speech"),
    ("espnet_asr_e_branchformer_ctc", "build_ebranchformer", "example_mel", "2022", "audio/speech"),
    ("espnet_ebranchformer_encoder", "build_ebranchformer", "example_mel", "2022", "audio/speech"),
    (
        "espnet_asr_transformer_ctc_attention",
        "build_transformer_asr",
        "example_mel",
        "2018",
        "audio/speech",
    ),
    (
        "espnet_contextual_block_transformer_encoder",
        "build_transformer_asr",
        "example_mel",
        "2018",
        "audio/speech",
    ),
    ("espnet_transformer_decoder", "build_transformer_asr", "example_mel", "2018", "audio/speech"),
    ("espnet_transformer_encoder", "build_transformer_asr", "example_mel", "2018", "audio/speech"),
    ("espnet_transformer_tts", "build_transformer_tts", "example_tokens", "2018", "audio/tts"),
    ("espnet_tts_transformer_tts", "build_transformer_tts", "example_tokens", "2018", "audio/tts"),
    ("espnet_ctc", "build_conformer", "example_mel", "2006", "audio/speech"),
    ("espnet_dprnn_separator", "build_dprnn", "example_wave", "2020", "audio/separation"),
    ("espnet_enh_dprnn", "build_dprnn", "example_wave", "2020", "audio/separation"),
    ("espnet_rnn_decoder", "build_rnn_decoder", "example_mel", "2018", "audio/speech"),
    ("espnet_rnn_encoder", "build_rnn_encoder", "example_mel", "2018", "audio/speech"),
    ("espnet_tacotron2", "build_tacotron2", "example_tokens", "2018", "audio/tts"),
    ("espnet_tts_tacotron2_gst", "build_tacotron2_gst", "example_tokens", "2018", "audio/tts"),
    ("espnet_tfgridnet_separator", "build_tfgridnet", "example_wave", "2022", "audio/separation"),
    ("espnet_transducer_joint_network", "build_conformer", "example_mel", "2012", "audio/speech"),
    ("espnet_transducer_rnn_decoder", "build_rnn_decoder", "example_mel", "2012", "audio/speech"),
    ("espnet_vocoder_hifigan", "build_hifigan", "example_mel", "2020", "audio/vocoder"),
    ("espnet_vocoder_melgan", "build_melgan", "example_mel", "2019", "audio/vocoder"),
    (
        "espnet_vocoder_multiband_melgan",
        "build_multiband_melgan",
        "example_mel",
        "2020",
        "audio/vocoder",
    ),
    ("nemo_citrinet_encoder", "build_citrinet", "example_mel", "2021", "audio/speech"),
    ("nemo_conformer_encoder", "build_conformer", "example_mel", "2020", "audio/speech"),
    ("nemo_contextnet_encoder", "build_contextnet", "example_mel", "2020", "audio/speech"),
    ("nemo_convasr_encoder", "build_jasper", "example_mel", "2019", "audio/speech"),
    ("nemo_fastconformer_encoder", "build_fastconformer", "example_mel", "2023", "audio/speech"),
    ("nemo_squeezeformer_encoder", "build_squeezeformer", "example_mel", "2022", "audio/speech"),
    ("mmagic:edsr", "build_mmagic_edsr", "example_mmagic_edsr", "2017", "super-resolution"),
    ("mmagic_edsr", "build_mmagic_edsr", "example_mmagic_edsr", "2017", "super-resolution"),
    ("mmagic:esrgan", "build_mmagic_esrgan", "example_mmagic_esrgan", "2018", "super-resolution"),
    ("mmagic_esrgan", "build_mmagic_esrgan", "example_mmagic_esrgan", "2018", "super-resolution"),
    ("mmagic:rdn", "build_mmagic_rdn", "example_mmagic_rdn", "2018", "super-resolution"),
    ("mmagic_rdn", "build_mmagic_rdn", "example_mmagic_rdn", "2018", "super-resolution"),
    (
        "mmagic:real_esrgan",
        "build_mmagic_real_esrgan",
        "example_mmagic_real_esrgan",
        "2021",
        "super-resolution",
    ),
    (
        "mmagic_real_esrgan",
        "build_mmagic_real_esrgan",
        "example_mmagic_real_esrgan",
        "2021",
        "super-resolution",
    ),
    (
        "mmagic:restormer",
        "build_mmagic_restormer",
        "example_mmagic_restormer",
        "2022",
        "restoration",
    ),
    ("mmagic:srcnn", "build_mmagic_srcnn", "example_mmagic_srcnn", "2014", "super-resolution"),
    ("mmagic_srcnn", "build_mmagic_srcnn", "example_mmagic_srcnn", "2014", "super-resolution"),
    (
        "mmagic:srgan_resnet",
        "build_mmagic_srgan",
        "example_mmagic_srgan",
        "2017",
        "super-resolution",
    ),
    (
        "mmagic_srgan_resnet",
        "build_mmagic_srgan",
        "example_mmagic_srgan",
        "2017",
        "super-resolution",
    ),
    ("mmagic:swinir", "build_mmagic_swinir", "example_mmagic_swinir", "2021", "super-resolution"),
]

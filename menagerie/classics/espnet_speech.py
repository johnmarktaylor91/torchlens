"""Faithful pure-torch reimplementations of ESPnet speech models.

Mirrors github.com/espnet/espnet (espnet2/): Mask-CTC ConformerEncoder + MaskCTC
TransformerDecoder; hybrid RNN CTC/attention ASR (VGG subsampling + BiLSTM +
location-aware attention decoder); TransformerSeparator for separation; VITS-style
Text2Wav generator; StyleMelGAN vocoder (TADEResBlock + noise upsampling).
Random-init; used by the model menagerie catalog.
"""

import torch
import torch.nn as nn
import math


# ============================================================
# Model 1: espnet_asr_mask_ctc
# ConformerEncoder + MaskCTC TransformerDecoder
# ============================================================


class RelPosEnc(nn.Module):
    __module__ = "__main__"

    def __init__(self, d, drop=0.1):
        super().__init__()
        self.d = d
        self.dropout = nn.Dropout(drop)
        self.pe = None

    def _make_pe(self, x):
        L = x.size(1)
        pe = torch.zeros(L, self.d)
        pos = torch.arange(0, L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.d, 2).float() * (-math.log(10000.0) / self.d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0).to(x.device)

    def forward(self, x):
        self._make_pe(x)
        return self.dropout(x + self.pe[:, : x.size(1)])


class ConvSubsampling(nn.Module):
    __module__ = "__main__"

    def __init__(self, idim, odim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2), nn.ReLU(), nn.Conv2d(odim, odim, 3, 2), nn.ReLU()
        )
        # compute exact freq dim with a dummy forward
        import torch as _t

        with _t.no_grad():
            dummy = _t.zeros(1, 1, 10, idim)
            dummy_out = self.conv(dummy)
            fdim = odim * dummy_out.shape[-1]
        self.out = nn.Linear(fdim, odim)

    def forward(self, x, lens):
        x = x.unsqueeze(1)  # (B,1,T,F)
        x = self.conv(x)  # (B,C,T',F')
        B, C, T, F = x.size()
        x = self.out(x.transpose(1, 2).reshape(B, T, C * F))
        # approximate output lengths after 2x strided conv each
        lens = (lens - 1) // 2 - 1
        return x, lens


class ConformerBlock(nn.Module):
    __module__ = "__main__"

    def __init__(self, d, h, ffn, kconv=31):
        super().__init__()
        # macaron-style: two half-step FF
        self.ff1_norm = nn.LayerNorm(d)
        self.ff1 = nn.Sequential(nn.Linear(d, ffn), nn.SiLU(), nn.Linear(ffn, d))
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.conv_norm = nn.LayerNorm(d)
        # depthwise conv module
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kconv, padding=kconv // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.ff2_norm = nn.LayerNorm(d)
        self.ff2 = nn.Sequential(nn.Linear(d, ffn), nn.SiLU(), nn.Linear(ffn, d))
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        # half-step FF1 (macaron)
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        # self-attention
        xn = self.attn_norm(x)
        xn, _ = self.attn(xn, xn, xn, key_padding_mask=mask)
        x = x + xn
        # conv module
        xc = self.conv_norm(x).transpose(1, 2)  # (B, d, T)
        xc = self.pw1(xc)
        xc = torch.nn.functional.glu(xc, dim=1)  # -> (B, d, T)
        xc = nn.functional.silu(self.bn(self.dw(xc)))
        xc = self.pw2(xc).transpose(1, 2)
        x = x + xc
        # half-step FF2 (macaron)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    __module__ = "__main__"

    def __init__(self, idim, odim=256, heads=4, ffn=2048, nblk=4):
        super().__init__()
        self.subsampling = ConvSubsampling(idim, odim)
        self.pos_enc = RelPosEnc(odim)
        self.blocks = nn.ModuleList([ConformerBlock(odim, heads, ffn) for _ in range(nblk)])

    def forward(self, x, lens):
        x, lens = self.subsampling(x, lens)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        return x, lens


class MaskCTCDecoder(nn.Module):
    __module__ = "__main__"

    def __init__(self, vocab, enc_dim, heads=4, ffn=2048, nblk=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, enc_dim)
        self.pos_enc = RelPosEnc(enc_dim)
        self.layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(enc_dim, heads, ffn, batch_first=True) for _ in range(nblk)]
        )
        self.out = nn.Linear(enc_dim, vocab)

    def forward(self, enc, tgt):
        tgt = self.pos_enc(self.embed(tgt))
        for layer in self.layers:
            tgt = layer(tgt, enc)
        return self.out(tgt)


class MaskCTCModel(nn.Module):
    __module__ = "__main__"

    def __init__(self, idim=80, odim=256, heads=4, ffn=2048, nblk=4, vocab=500):
        super().__init__()
        self.encoder = ConformerEncoder(idim, odim, heads, ffn, nblk)
        self.ctc_out = nn.Linear(odim, vocab)
        self.decoder = MaskCTCDecoder(vocab, odim, heads, ffn, nblk)

    def forward(self, feats, feats_len, ys):
        enc, lens = self.encoder(feats, feats_len)
        ctc_out = self.ctc_out(enc)
        dec_out = self.decoder(enc, ys)
        return ctc_out, dec_out


# ============================================================
# Model 2: espnet_asr_rnn_ctc_attention
# VGG subsampling + BiLSTM encoder + attention decoder + CTC
# ============================================================


class VGGSubsampling(nn.Module):
    __module__ = "__main__"

    def __init__(self, idim, odim):
        super().__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        fdim = 128 * (idim // 4)
        self.out = nn.Linear(fdim, odim)

    def forward(self, x, lens):
        x = x.unsqueeze(1)  # (B,1,T,F)
        x = self.vgg(x)  # (B,128,T//4,F//4)
        B, C, T, F = x.size()
        x = self.out(x.transpose(1, 2).contiguous().reshape(B, T, C * F))
        lens = torch.clamp(lens // 4, min=1)
        return x, lens


class LocationAwareAttention(nn.Module):
    __module__ = "__main__"

    def __init__(self, enc_dim, dec_dim, att_dim=256, conv_channels=10, conv_size=100):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, att_dim, bias=False)
        self.dec_proj = nn.Linear(dec_dim, att_dim, bias=False)
        # use (conv_size-1)//2 padding to keep T unchanged
        self.loc_conv = nn.Conv1d(
            1, conv_channels, conv_size, padding=(conv_size - 1) // 2, bias=False
        )
        self.loc_proj = nn.Linear(conv_channels, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, enc, dec_state, prev_att):
        # enc: (B, T, enc_dim), dec_state: (B, dec_dim), prev_att: (B, T)
        B, T, _ = enc.size()
        e = self.enc_proj(enc)  # (B, T, att_dim)
        d = self.dec_proj(dec_state).unsqueeze(1)  # (B, 1, att_dim)
        loc = self.loc_conv(prev_att.unsqueeze(1))  # (B, conv_ch, T')
        # match T exactly: trim or pad
        if loc.size(-1) > T:
            loc = loc[:, :, :T]
        elif loc.size(-1) < T:
            loc = torch.nn.functional.pad(loc, (0, T - loc.size(-1)))
        loc = self.loc_proj(loc.transpose(1, 2))  # (B, T, att_dim)
        energy = self.v(torch.tanh(e + d + loc)).squeeze(-1)  # (B, T)
        alpha = torch.softmax(energy, dim=-1)  # (B, T)
        context = (alpha.unsqueeze(-1) * enc).sum(1)  # (B, enc_dim)
        return context, alpha


class RNNAttentionDecoder(nn.Module):
    __module__ = "__main__"

    def __init__(self, vocab, enc_dim, dec_dim=256, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab, enc_dim)
        self.attention = LocationAwareAttention(enc_dim, dec_dim)
        self.rnn = nn.LSTM(enc_dim + enc_dim, dec_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(dec_dim, vocab)

    def forward(self, enc, ys):
        B, T, E = enc.size()
        L = ys.size(1)
        h = torch.zeros(1, B, 256, device=enc.device)
        c = torch.zeros(1, B, 256, device=enc.device)
        prev_att = torch.zeros(B, T, device=enc.device)
        outs = []
        for i in range(L):
            dec_in = self.embed(ys[:, i])  # (B, E)
            context, prev_att = self.attention(enc, h[-1], prev_att)
            rnn_in = torch.cat([dec_in, context], dim=-1).unsqueeze(1)
            out, (h, c) = self.rnn(rnn_in, (h, c))
            outs.append(self.out(out.squeeze(1)))
        return torch.stack(outs, dim=1)


class RNNCTCAttentionModel(nn.Module):
    __module__ = "__main__"

    def __init__(self, idim=80, enc_dim=512, vocab=500, rnn_layers=3):
        super().__init__()
        self.subsampling = VGGSubsampling(idim, enc_dim)
        self.encoder = nn.LSTM(
            enc_dim,
            enc_dim // 2,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.ctc_out = nn.Linear(enc_dim, vocab)
        self.decoder = RNNAttentionDecoder(vocab, enc_dim)

    def forward(self, feats, feats_len, ys):
        enc, lens = self.subsampling(feats, feats_len)
        enc, _ = self.encoder(enc)
        ctc_out = self.ctc_out(enc)
        dec_out = self.decoder(enc, ys)
        return ctc_out, dec_out


# ============================================================
# Model 3: espnet_enh_transformer_separator
# TransformerSeparator: input (B,T,F), num_spk=2
# ============================================================


class TransformerSeparator(nn.Module):
    __module__ = "__main__"

    def __init__(self, input_dim=64, num_spk=2, adim=256, aheads=4, layers=4, linear_units=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, adim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=adim, nhead=aheads, dim_feedforward=linear_units, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.mask_heads = nn.ModuleList([nn.Linear(adim, input_dim) for _ in range(num_spk)])
        self.num_spk = num_spk

    def forward(self, mixture):
        # mixture: (B, T, F)
        x = self.input_proj(mixture)
        x = self.transformer(x)
        masks = [torch.relu(head(x)) for head in self.mask_heads]
        separated = [m * mixture for m in masks]
        return separated


# ============================================================
# Model 4: espnet_tts_joint_text2wav (VITS-style)
# Text encoder (transformer) + flow + HiFiGAN decoder
# ============================================================


class VITSTextEncoder(nn.Module):
    __module__ = "__main__"

    def __init__(self, vocab, hidden=192, heads=2, ffn=768, nblk=6, kernel=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.pre = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nblk)
        # project to mean and logvar for posterior
        self.proj = nn.Conv1d(hidden, hidden * 2, 1)

    def forward(self, x):
        x = self.embed(x)  # (B, L, H)
        x = self.pre(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        stats = self.proj(x.transpose(1, 2))  # (B, 2H, L)
        m, logs = stats.chunk(2, dim=1)
        return x, m, logs


class WaveNetResBlock(nn.Module):
    __module__ = "__main__"

    def __init__(self, channels=192, kernel=5, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2
        )
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x, c=None):
        residual = x
        x = self.conv(x)
        x_tanh, x_sig = x.chunk(2, dim=1)
        x = torch.tanh(x_tanh) * torch.sigmoid(x_sig)
        return self.res_conv(x) + residual, self.skip_conv(x)


class PosteriorEncoder(nn.Module):
    __module__ = "__main__"

    def __init__(self, in_channels=1, hidden=192, out_channels=192, n_layers=16):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden, 1)
        self.resblocks = nn.ModuleList([WaveNetResBlock(hidden) for _ in range(n_layers)])
        self.proj = nn.Conv1d(hidden, out_channels * 2, 1)

    def forward(self, x):
        x = self.pre(x)
        skip = None
        for blk in self.resblocks:
            x, s = blk(x)
            skip = s if skip is None else skip + s
        stats = self.proj(skip)
        m, logs = stats.chunk(2, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs


class AffineFlow(nn.Module):
    __module__ = "__main__"

    def __init__(self, channels=192, hidden=192, n_layers=4, kernel=5):
        super().__init__()
        self.pre = nn.Conv1d(channels // 2, hidden, 1)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden, hidden, kernel, dilation=2**i, padding=2**i * (kernel - 1) // 2)
                for i in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.post = nn.Conv1d(hidden, channels // 2, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, reverse=False):
        x0, x1 = x.chunk(2, dim=1)
        h = self.pre(x0)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(norm(conv(h).transpose(1, 2)).transpose(1, 2))
        m = self.post(h)
        if not reverse:
            x1 = m + x1
        else:
            x1 = x1 - m
        return torch.cat([x0, x1], dim=1)


class ResidualCouplingBlock(nn.Module):
    __module__ = "__main__"

    def __init__(self, channels=192, n_flows=4):
        super().__init__()
        self.flows = nn.ModuleList([AffineFlow(channels) for _ in range(n_flows)])

    def forward(self, x, reverse=False):
        flows = self.flows if not reverse else reversed(self.flows)
        for f in flows:
            x = f(x, reverse=reverse)
            x = x.flip(1)
        return x


class HiFiGANResBlock(nn.Module):
    __module__ = "__main__"

    def __init__(self, channels, kernel=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel, dilation=d, padding=d * (kernel - 1) // 2)
                for d in dilations
            ]
        )

    def forward(self, x):
        for c in self.convs:
            x = x + c(torch.nn.functional.leaky_relu(x, 0.1))
        return x


class HiFiGANGenerator(nn.Module):
    __module__ = "__main__"

    def __init__(
        self,
        in_channels=192,
        upsample_rates=(8, 8, 2, 2),
        upsample_initial=512,
        resblock_kernels=(3, 7, 11),
    ):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, upsample_initial, 7, padding=3)
        ch = upsample_initial
        self.ups = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        for r in upsample_rates:
            self.ups.append(nn.ConvTranspose1d(ch, ch // 2, r * 2, stride=r, padding=r // 2))
            ch = ch // 2
            self.res_blocks.append(
                nn.ModuleList([HiFiGANResBlock(ch, k) for k in resblock_kernels])
            )
        self.post = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, x):
        x = self.pre(x)
        for up, rbs in zip(self.ups, self.res_blocks):
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = up(x)
            for rb in rbs:
                x = rb(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        return torch.tanh(self.post(x))


class StochasticDurationPredictor(nn.Module):
    __module__ = "__main__"

    def __init__(self, in_channels=192, hidden=256, kernel=3, n_flows=4):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden, 1)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2) for _ in range(n_flows)]
        )
        self.post = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, C, L)
        x = self.pre(x)
        for c in self.convs:
            x = x + torch.relu(c(x))
        # compute log-duration
        dur = self.post(x.mean(-1))
        return dur


class VITSGenerator(nn.Module):
    __module__ = "__main__"

    def __init__(self, vocab=100, hidden=192, heads=2, ffn=768, nblk=6):
        super().__init__()
        self.text_encoder = VITSTextEncoder(vocab, hidden, heads, ffn, nblk)
        self.duration_predictor = StochasticDurationPredictor(hidden)
        self.flow = ResidualCouplingBlock(hidden, n_flows=4)
        self.decoder = HiFiGANGenerator(in_channels=hidden)

    def forward(self, tokens):
        # tokens: (B, L)
        enc_out, m_p, logs_p = self.text_encoder(tokens)
        # sample latent
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z = self.flow(z_p, reverse=True)
        wav = self.decoder(z)
        return wav


# ============================================================
# Model 5: espnet_vocoder_style_melgan
# StyleMelGANGenerator: noise upsampling + TADEResBlocks
# ============================================================


class TADELayer(nn.Module):
    __module__ = "__main__"

    def __init__(self, channels=64, aux_channels=80, upsample_factor=1, upsample_mode="nearest"):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels)
        self.upsample = (
            nn.Upsample(scale_factor=upsample_factor, mode=upsample_mode)
            if upsample_factor > 1
            else nn.Identity()
        )
        self.aux_conv = nn.Conv1d(aux_channels, channels * 2, 1)

    def forward(self, x, c):
        x = self.norm(x)
        x = self.upsample(x)
        c = nn.functional.interpolate(c, size=x.size(-1), mode="nearest")
        params = self.aux_conv(c)
        gamma, beta = params.chunk(2, dim=1)
        return (1 + gamma) * x + beta


class TADEResBlock(nn.Module):
    __module__ = "__main__"

    def __init__(
        self,
        channels=64,
        aux_channels=80,
        kernel=9,
        dilation=2,
        upsample_factor=2,
        upsample_mode="nearest",
        gated_function="softmax",
    ):
        super().__init__()
        self.tade1 = TADELayer(channels, aux_channels, 1, upsample_mode)
        self.gated_conv1 = nn.Conv1d(channels, channels * 2, kernel, padding=kernel // 2)
        self.tade2 = TADELayer(channels, aux_channels, upsample_factor, upsample_mode)
        self.gated_conv2 = nn.Conv1d(
            channels, channels * 2, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2
        )
        self.upsample = (
            nn.Upsample(scale_factor=upsample_factor, mode=upsample_mode)
            if upsample_factor > 1
            else nn.Identity()
        )
        self.gated_fn = gated_function

    def _gate(self, x):
        x1, x2 = x.chunk(2, dim=1)
        if self.gated_fn == "softmax":
            w = torch.softmax(torch.stack([x1, x2], dim=-1), dim=-1)
            return w[..., 0] * x1 + w[..., 1] * x2
        return torch.sigmoid(x2) * torch.tanh(x1)

    def forward(self, x, c):
        residual = self.upsample(x)
        x = self.tade1(x, c)
        x = self._gate(self.gated_conv1(x))
        x = self.tade2(x, c)
        x = self._gate(self.gated_conv2(x))
        return x + residual


class StyleMelGANGenerator(nn.Module):
    __module__ = "__main__"

    def __init__(
        self,
        in_channels=128,
        aux_channels=80,
        channels=64,
        out_channels=1,
        kernel_size=9,
        dilation=2,
        noise_upsample_scales=(11, 2, 2, 2),
        upsample_scales=(2, 2, 2, 2, 2, 2, 2, 2, 1),
        upsample_mode="nearest",
    ):
        super().__init__()
        # noise upsampling path: expand in_channels noise
        noise_ups = []
        for scale in noise_upsample_scales:
            noise_ups += [
                nn.ConvTranspose1d(
                    in_channels, in_channels, scale * 2, stride=scale, padding=scale // 2
                ),
                nn.LeakyReLU(0.2),
            ]
        self.noise_upsample = nn.Sequential(*noise_ups)
        # project noise to channel size
        self.noise_proj = nn.Conv1d(in_channels, channels, 1)
        # TADE residual blocks
        self.tade_blocks = nn.ModuleList()
        for scale in upsample_scales:
            self.tade_blocks.append(
                TADEResBlock(
                    channels,
                    aux_channels,
                    kernel_size,
                    dilation,
                    upsample_factor=scale,
                    upsample_mode=upsample_mode,
                )
            )
        self.out_conv = nn.Sequential(
            nn.Conv1d(channels, out_channels, kernel_size, padding=kernel_size // 2), nn.Tanh()
        )
        self.in_channels = in_channels

    def forward(self, c, z=None):
        # c: (B, aux_channels, T_mel)
        B = c.size(0)
        if z is None:
            z = torch.randn(B, self.in_channels, 1, device=c.device)
        x = self.noise_upsample(z)
        x = self.noise_proj(x)
        for blk in self.tade_blocks:
            x = blk(x, c)
        return self.out_conv(x)


# ============================================================
# Menagerie wiring: single-input wrappers + builders + example inputs.
#
# Several ESPnet speech models are natively multi-input (acoustic features +
# feature lengths + target tokens for the autoregressive/teacher-forced paths).
# TorchLens menagerie capture traces a single example tensor, so each such model
# is wrapped to carry its auxiliary inputs as fixed buffers and expose a single
# tensor argument. The wrapped forward is otherwise the faithful model forward.
# ============================================================


class _MaskCTCForMenagerie(nn.Module):
    """MaskCTC (Conformer encoder + MaskCTC transformer decoder), single-input."""

    __module__ = "__main__"

    def __init__(self, idim=80, odim=256, vocab=500, ys_len=8):
        super().__init__()
        self.model = MaskCTCModel(idim, odim, 4, 2048, 4, vocab)
        self.register_buffer("feats_len", torch.tensor([128]))
        self.register_buffer("ys", torch.randint(0, vocab, (1, ys_len)))

    def forward(self, feats):
        return self.model(feats, self.feats_len, self.ys)


class _RNNCTCAttForMenagerie(nn.Module):
    """Hybrid RNN CTC/attention ASR (VGG + BiLSTM + location-aware att), single-input."""

    __module__ = "__main__"

    def __init__(self, idim=80, enc_dim=512, vocab=500, rnn_layers=3, ys_len=4):
        super().__init__()
        self.model = RNNCTCAttentionModel(idim, enc_dim, vocab, rnn_layers)
        self.register_buffer("feats_len", torch.tensor([128]))
        self.register_buffer("ys", torch.randint(0, vocab, (1, ys_len)))

    def forward(self, feats):
        return self.model(feats, self.feats_len, self.ys)


class _SeparatorForMenagerie(nn.Module):
    """TransformerSeparator (masking-based source separation), single-input."""

    __module__ = "__main__"

    def __init__(self, input_dim=64, num_spk=2):
        super().__init__()
        self.model = TransformerSeparator(input_dim, num_spk, 256, 4, 4, 256)

    def forward(self, mixture):
        return self.model(mixture)


class _VITSForMenagerie(nn.Module):
    """VITS-style joint text-to-wav generator (text enc + flow + HiFiGAN), single-input."""

    __module__ = "__main__"

    def __init__(self, vocab=100, hidden=192):
        super().__init__()
        self.model = VITSGenerator(vocab, hidden, 2, 768, 6)

    def forward(self, tokens):
        return self.model(tokens)


class _StyleMelGANForMenagerie(nn.Module):
    """StyleMelGAN vocoder (TADEResBlock + noise upsampling), single-input."""

    __module__ = "__main__"

    def __init__(self, in_channels=128, aux_channels=80, channels=64):
        super().__init__()
        self.model = StyleMelGANGenerator(in_channels, aux_channels, channels)

    def forward(self, mel):
        return self.model(mel)


def build_mask_ctc() -> nn.Module:
    """Build the ESPnet Mask-CTC ASR model (Conformer encoder + MaskCTC decoder)."""
    return _MaskCTCForMenagerie()


def example_input_mask_ctc() -> "torch.Tensor":
    """Example log-mel acoustic features ``(1, 128, 80)`` for Mask-CTC ASR."""
    return torch.randn(1, 128, 80)


def build_rnn_ctc_attention() -> nn.Module:
    """Build the hybrid RNN CTC/attention ASR model (VGG + BiLSTM + location attention)."""
    return _RNNCTCAttForMenagerie()


def example_input_rnn_ctc_attention() -> "torch.Tensor":
    """Example log-mel acoustic features ``(1, 128, 80)`` for the RNN CTC/attention ASR."""
    return torch.randn(1, 128, 80)


def build_separator() -> nn.Module:
    """Build the ESPnet TransformerSeparator (masking-based source separation)."""
    return _SeparatorForMenagerie()


def example_input_separator() -> "torch.Tensor":
    """Example mixture spectrogram ``(1, 100, 64)`` for the TransformerSeparator."""
    return torch.randn(1, 100, 64)


def build_vits() -> nn.Module:
    """Build the ESPnet VITS-style joint text-to-wav generator."""
    return _VITSForMenagerie()


def example_input_vits() -> "torch.Tensor":
    """Example token ids ``(1, 32)`` for the VITS text-to-wav generator."""
    return torch.randint(0, 100, (1, 32))


def build_style_melgan() -> nn.Module:
    """Build the ESPnet StyleMelGAN neural vocoder."""
    return _StyleMelGANForMenagerie()


def example_input_style_melgan() -> "torch.Tensor":
    """Example mel-spectrogram conditioning ``(1, 80, 32)`` for the StyleMelGAN vocoder."""
    return torch.randn(1, 80, 32)


MENAGERIE_ENTRIES = [
    (
        "ESPnet Mask-CTC ASR (Conformer + MaskCTC decoder)",
        "build_mask_ctc",
        "example_input_mask_ctc",
        "2020",
        "DE",
    ),
    (
        "ESPnet hybrid RNN CTC/attention ASR (VGG + BiLSTM + location attention)",
        "build_rnn_ctc_attention",
        "example_input_rnn_ctc_attention",
        "2018",
        "DE",
    ),
    (
        "ESPnet TransformerSeparator (masking-based source separation)",
        "build_separator",
        "example_input_separator",
        "2020",
        "DE",
    ),
    ("ESPnet VITS joint text-to-wav generator", "build_vits", "example_input_vits", "2021", "DE"),
    (
        "ESPnet StyleMelGAN neural vocoder",
        "build_style_melgan",
        "example_input_style_melgan",
        "2021",
        "DE",
    ),
]


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if model_name in ("1", "mask_ctc", "all"):
        print("Testing MaskCTC...")
        m = MaskCTCModel(80, 256, 4, 2048, 4, 500)
        feats = torch.randn(1, 128, 80)
        feats_len = torch.tensor([128])
        ys = torch.randint(0, 500, (1, 32))
        out = m(feats, feats_len, ys)
        print(f"  MaskCTC CTC out: {out[0].shape}, Dec out: {out[1].shape}")

    if model_name in ("2", "rnn", "all"):
        print("Testing RNN CTC Attention...")
        m = RNNCTCAttentionModel(80, 512, 500, 3)
        feats = torch.randn(1, 128, 80)
        feats_len = torch.tensor([128])
        ys = torch.randint(0, 500, (1, 32))
        out = m(feats, feats_len, ys)
        print(f"  RNN CTC out: {out[0].shape}, Att out: {out[1].shape}")

    if model_name in ("3", "sep", "all"):
        print("Testing TransformerSeparator...")
        m = TransformerSeparator(64, 2, 256, 4, 4, 256)
        x = torch.randn(1, 100, 64)
        out = m(x)
        print(f"  Separator outputs: {len(out)} speakers, shape: {out[0].shape}")

    if model_name in ("4", "vits", "all"):
        print("Testing VITS Generator...")
        m = VITSGenerator(100, 192, 2, 768, 6)
        tokens = torch.randint(0, 100, (1, 32))
        out = m(tokens)
        print(f"  VITS wav out: {out.shape}")

    if model_name in ("5", "stylemelgan", "all"):
        print("Testing StyleMelGAN Generator...")
        m = StyleMelGANGenerator(128, 80, 64)
        c = torch.randn(1, 80, 32)
        out = m(c)
        print(f"  StyleMelGAN wav out: {out.shape}")

    print("All tests passed!")

# FAITHFUL REIMPLEMENTATION from Chang, Zhang, Qian, Le Roux, Watanabe,
# "MIMO-SPEECH: End-to-End Multi-Channel Multi-Speaker Speech Recognition"
# (ASRU 2019, arXiv:1910.06522) -- no public code.
#
# The queue candidate's repo, github.com/simpleoier/MIMO-Speech, is a demo-
# only GitHub Pages companion site: it contains nothing but `index.html` /
# `index_dereverb.html` audio-sample players and `.wav` files -- no MATLAB,
# Python, or any model-definition source of any kind (confirmed via
# `gh api repos/simpleoier/MIMO-Speech/git/trees/master?recursive=1`). The
# real reference implementation is built into the ESPnet framework (per the
# paper's Section 3.1: "Our end-to-end multi-channel multi-speaker model is
# completely built based on the ESPnet framework"), which is not installed
# here and is not the queue candidate's own repo, so rungs 2/3 (vendor/port
# from the candidate repo) do not apply. The paper itself (Sections 2.1-2.1.3)
# gives a fully detailed, equation-level architecture spec, so this is a
# RUNG 4 faithful reimplementation transcribed directly from those equations
# and Section 3.1's stated configuration (not a loose paraphrase):
#
#   1) Monaural masking network (Sec 2.1.1): per input channel c, a shared
#      mask-estimation network (paper: 3-layer BLSTMP, 512 cells/direction,
#      512-dim projection) maps the channel's STFT magnitude+phase features
#      to (J+1) time-frequency masks in [0,1] -- J per-speaker masks plus one
#      noise mask (Eq. 1: M_c = MaskNet(X_c)).
#   2) Multi-source neural beamformer (Sec 2.1.2, Eq. 2-4): for each source
#      i, the per-channel masks and STFT are used to estimate a spatial PSD
#      (covariance) matrix Phi^i(f) = (1/sum_t m^i_t,f) * sum_t m^i_t,f
#      x_t,f x_t,f^H (Eq. 2); an MVDR beamforming filter g^i(f) is derived
#      from the target PSD and the summed interference+noise PSD via the
#      MVDR formalization used in the paper (Eq. 3, with the paper's SSN-
#      style interference PSD = sum over all OTHER sources+noise); the
#      filter is applied to produce a single-channel enhanced STFT estimate
#      per source (Eq. 4: s_hat^i_t,f = (g^i(f))^H x_t,f).
#   3) End-to-end multi-speaker ASR (Sec 2.1.3, Eq. 5-10): a log-mel filter-
#      bank (Eq. 5-6) is computed from each source's enhanced STFT magnitude
#      and globally mean-variance normalized, then fed through a joint CTC/
#      attention encoder-decoder (paper Sec 3.1.2: two VGG-motivated CNN
#      blocks -- 3x3 kernels, 64/128 feature maps -- followed by three
#      BLSTMP layers with 1024 cells/direction and 1024-dim projection as
#      the encoder; a single unidirectional LSTM decoder with 300 cells and
#      additive/content-based attention, matching Eq. 7-10). PIT-based CTC
#      permutation search (Eq. 11) and the combined CTC/attention loss
#      (Eq. 12-14) are training-time-only objectives and are not part of the
#      forward inference architecture ported here; the module exposes the
#      full forward path (masking -> MVDR beamforming -> log-mel -> encoder
#      -> one greedy/teacher-forced decoder step per output source), which
#      is the genuine multi-stage neural architecture the paper describes.
#
# Because the paper's own encoder/decoder sizes (512/1024-cell BLSTMPs) are
# meant for full WSJ-scale ASR training and would make a CPU trace far too
# slow/heavy, the tiny build below keeps every ARCHITECTURAL mechanism and
# stage from the paper (mask net -> MVDR PSD/beamforming -> VGG+BLSTM
# encoder -> attention decoder) but uses drastically smaller channel/cell
# counts and a short synthetic multi-channel STFT input, consistent with
# the menagerie's tiny-random-init tracing convention.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


class MonauralMaskingNet(nn.Module):
    """Sec 2.1.1 / Eq. 1: per-channel BLSTMP mask estimator.

    Applied independently to each channel's STFT magnitude+phase (here:
    concatenated real/imag parts, a standard STFT-feature front end for
    mask-estimation BLSTMs); outputs (n_speakers + 1) masks in [0,1] per
    time-frequency bin (the "+1" is the noise mask, i=0 in the paper).
    """

    def __init__(self, n_freq: int, hidden: int, n_speakers: int):
        super().__init__()
        self.n_freq = n_freq
        self.n_out = n_speakers + 1  # + noise mask
        self.blstm = nn.LSTM(
            input_size=2 * n_freq,
            hidden_size=hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * hidden, n_freq * self.n_out)

    def forward(self, x_complex):
        """x_complex: [B, T, F] complex STFT of one channel -> masks [B, n_out, T, F] in [0,1]."""
        feat = torch.cat([x_complex.real, x_complex.imag], dim=-1)  # [B,T,2F]
        h, _ = self.blstm(feat)  # [B,T,2*hidden]
        out = self.proj(h)  # [B,T,F*n_out]
        b, t, _ = out.shape
        out = out.view(b, t, self.n_out, self.n_freq).permute(0, 2, 1, 3)  # [B,n_out,T,F]
        return torch.sigmoid(out)


class MultiSourceMVDRBeamformer(nn.Module):
    """Sec 2.1.2 / Eq. 2-4: mask-based PSD estimation + MVDR filtering.

    Given per-channel STFTs x [B,C,T,F] (complex) and per-channel masks
    (from MonauralMaskingNet, stacked over channels) [B,C,n_out,T,F], for
    each source i estimates the spatial covariance (PSD) matrix Phi^i(f)
    (Eq. 2), derives the MVDR filter g^i(f) using the SSN-style
    "everything else is interference" PSD (Eq. 3), and applies it to
    produce a single-channel enhanced STFT per source (Eq. 4).

    The reference-microphone vector u (Eq. 3) is a real learned parameter
    here in place of the paper's attention-derived reference vector -- a
    standard MVDR simplification -- while every other operator (PSD
    accumulation, matrix inverse, MVDR ratio) is the paper's actual formula.
    """

    def __init__(self, n_channels: int, n_speakers: int, eps: float = 1e-6):
        super().__init__()
        self.n_channels = n_channels
        self.n_speakers = n_speakers
        self.eps = eps
        # reference-microphone selection vector u in Eq. 3 (paper derives it
        # via an attention mechanism; a learned softmax-weighted vector is
        # used here as the reference-vector mechanism)
        self.ref_logits = nn.Parameter(torch.zeros(n_channels))

    def _psd(self, masks_c, x_c):
        """Eq. 2: Phi^i(f) = (1/sum_t m_tf) * sum_t m_tf * x_tf x_tf^H, per source i.

        masks_c: [B,C,n_out,T,F] real masks; x_c: [B,C,T,F] complex STFT.
        Returns Phi: [B,n_out,F,C,C] complex PSD matrices.
        """
        # x_t,f vector across channels: [B,T,F,C]
        x_vec = x_c.permute(0, 2, 3, 1)  # [B,T,F,C]
        outer = x_vec.unsqueeze(-1) * x_vec.unsqueeze(-2).conj()  # [B,T,F,C,C]
        # masks per source, averaged over channels (paper computes per
        # channel c independently; we use the cross-channel mean mask as
        # the weighting, a standard simplification for the PSD estimate)
        m = masks_c.mean(dim=1)  # [B,n_out,T,F]
        m_w = m.permute(0, 2, 3, 1).to(outer.dtype)  # [B,T,F,n_out], complex-cast for einsum
        num = torch.einsum("btfi,btfcd->bifcd", m_w, outer)
        den = m.sum(dim=2).clamp_min(self.eps)  # sum over t -> [B,n_out,F]
        phi = num / den.to(num.dtype).unsqueeze(-1).unsqueeze(-1)
        return phi  # [B,n_out,F,C,C]

    def forward(self, x_c, masks_c):
        """x_c: [B,C,T,F] complex STFT. masks_c: [B,C,n_out,T,F] real masks in [0,1].

        Returns: list of J enhanced single-channel STFTs [B,T,F] complex,
        one per speaker (source indices 1..J; index 0 is the noise source
        and is used only as interference, matching the paper).
        """
        b, c, t, f = x_c.shape
        phi = self._psd(masks_c, x_c)  # [B,n_out,F,C,C]
        eye = torch.eye(c, dtype=phi.dtype, device=phi.device) * self.eps

        u = F.softmax(self.ref_logits, dim=0).to(phi.dtype)  # [C]

        enhanced = []
        for i in range(1, self.n_speakers + 1):
            phi_i = phi[:, i]  # [B,F,C,C] target PSD
            # SSN interference PSD: sum over all j != i (other speakers + noise)
            other_idx = [j for j in range(self.n_speakers + 1) if j != i]
            phi_interf = phi[:, other_idx].sum(dim=1)  # [B,F,C,C]
            phi_interf = phi_interf + eye  # regularize for a stable inverse

            phi_interf_inv = torch.linalg.inv(phi_interf)  # [B,F,C,C]
            num = torch.matmul(phi_interf_inv, phi_i)  # [B,F,C,C]
            denom = torch.diagonal(num, dim1=-2, dim2=-1).sum(-1).real  # trace, [B,F]
            denom = denom.clamp_min(self.eps).to(num.dtype).unsqueeze(-1).unsqueeze(-1)

            g = torch.matmul(num, u.view(1, 1, c, 1).expand(b, f, c, 1)) / denom  # [B,F,C,1]
            g = g.squeeze(-1)  # [B,F,C]

            # Eq. 4: s_hat^i_t,f = (g^i(f))^H x_t,f
            x_vec = x_c.permute(0, 2, 3, 1)  # [B,T,F,C]
            s_hat = torch.einsum("bfc,btfc->btf", g.conj(), x_vec)
            enhanced.append(s_hat)

        return enhanced  # list of J tensors, each [B,T,F] complex


class VGGBLSTMEncoder(nn.Module):
    """Sec 3.1.2: two VGG-motivated CNN blocks + three BLSTMP layers.

    Two conv blocks (3x3 kernels, {64,128} feature maps, each followed by
    max-pool) act on the log-mel-filterbank feature "image" (time x freq),
    then three BLSTM layers (with a linear projection after each, i.e.
    "BLSTMP") produce the encoder representation H^i (Eq. 7).
    """

    def __init__(
        self,
        n_mels: int,
        cnn_ch1: int,
        cnn_ch2: int,
        blstm_hidden: int,
        proj_dim: int,
        n_blstm_layers: int = 3,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, cnn_ch1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_ch1, cnn_ch1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(cnn_ch1, cnn_ch2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_ch2, cnn_ch2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        freq_after = n_mels // 4
        self.blstm_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        in_dim = cnn_ch2 * freq_after
        for _ in range(n_blstm_layers):
            self.blstm_layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=blstm_hidden,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.proj_layers.append(nn.Linear(2 * blstm_hidden, proj_dim))
            in_dim = proj_dim
        self.out_dim = proj_dim

    def forward(self, fbank):
        """fbank: [B,T,n_mels] log-mel features -> H: [B,T',out_dim]."""
        x = fbank.unsqueeze(1)  # [B,1,T,n_mels]
        x = self.block1(x)
        x = self.block2(x)  # [B,C,T',F']
        b, ch, t, fr = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, ch * fr)  # [B,T',C*F']
        for blstm, proj in zip(self.blstm_layers, self.proj_layers):
            x, _ = blstm(x)
            x = proj(x)
        return x


class AttentionDecoder(nn.Module):
    """Sec 2.1.3 / Eq. 7-10: single-layer LSTM decoder with additive attention.

    c^i_n, alpha^i_n = Attention(alpha^i_{n-1}, e^i_{n-1}, H^i)   (Eq. 8)
    e^i_n = Update(e^i_{n-1}, c^i_n, y^i_{n-1})                    (Eq. 9)
    y^i_n ~ Decoder(c^i_n, y^i_n)                                  (Eq. 10)
    """

    def __init__(self, enc_dim: int, dec_hidden: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn_w_enc = nn.Linear(enc_dim, dec_hidden, bias=False)
        self.attn_w_dec = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.attn_v = nn.Linear(dec_hidden, 1, bias=False)
        self.lstm_cell = nn.LSTMCell(embed_dim + enc_dim, dec_hidden)
        self.out_proj = nn.Linear(dec_hidden, vocab_size)
        self.dec_hidden = dec_hidden

    def forward(self, enc_h, target_tokens):
        """enc_h: [B,T,enc_dim] encoder output for one source.
        target_tokens: [B,N] teacher-forcing token ids (int64).
        Returns logits [B,N,vocab_size] (Eq. 10's y^i_n distribution)."""
        b, t_len, _ = enc_h.shape
        n_steps = target_tokens.shape[1]

        h = enc_h.new_zeros(b, self.dec_hidden)
        c = enc_h.new_zeros(b, self.dec_hidden)
        ctx = enc_h.new_zeros(b, enc_h.shape[-1])

        enc_proj = self.attn_w_enc(enc_h)  # [B,T,dec_hidden]

        logits = []
        for n in range(n_steps):
            y_prev = target_tokens[:, n]
            emb = self.embed(y_prev)  # [B,embed_dim]

            # Eq. 8: additive attention over encoder states
            dec_proj = self.attn_w_dec(h).unsqueeze(1)  # [B,1,dec_hidden]
            score = self.attn_v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # [B,T]
            alpha = F.softmax(score, dim=-1)  # [B,T]
            ctx = torch.einsum("bt,btd->bd", alpha, enc_h)  # [B,enc_dim]

            # Eq. 9: LSTM state update
            lstm_in = torch.cat([emb, ctx], dim=-1)
            h, c = self.lstm_cell(lstm_in, (h, c))

            # Eq. 10: output distribution
            logits.append(self.out_proj(h))

        return torch.stack(logits, dim=1)  # [B,N,vocab_size]


class MIMOSpeech(nn.Module):
    """
    Faithful reimplementation of the MIMO-Speech end-to-end multi-channel
    multi-speaker ASR model (Chang et al., ASRU 2019): monaural masking
    network -> multi-source MVDR neural beamformer -> per-speaker log-mel
    filterbank front end -> shared VGG+BLSTM encoder + attention decoder
    (Sections 2.1.1-2.1.3). CTC/PIT permutation search and the combined
    CTC/attention training loss (Eq. 11-14) are training-only and are not
    part of this forward-inference module.
    """

    def __init__(
        self,
        n_channels: int,
        n_speakers: int,
        n_freq: int,
        n_mels: int,
        mask_hidden: int,
        cnn_ch1: int,
        cnn_ch2: int,
        enc_hidden: int,
        enc_proj_dim: int,
        dec_hidden: int,
        vocab_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_speakers = n_speakers
        self.n_freq = n_freq

        self.mask_net = MonauralMaskingNet(n_freq, mask_hidden, n_speakers)
        self.beamformer = MultiSourceMVDRBeamformer(n_channels, n_speakers)

        mel_fb = torch.zeros(n_mels, n_freq)
        # simple triangular-ish mel-like filterbank (Eq. 5's MelFilterBank);
        # exact mel-scale placement is not architecturally load-bearing for
        # tracing purposes, only the linear-projection *mechanism* is.
        step = n_freq / (n_mels + 1)
        for m in range(n_mels):
            center = int((m + 1) * step)
            lo = max(0, center - int(step))
            hi = min(n_freq, center + int(step))
            if hi > lo:
                mel_fb[m, lo:hi] = 1.0 / (hi - lo)
        self.register_buffer("mel_fb", mel_fb)  # [n_mels, n_freq]

        self.encoder = VGGBLSTMEncoder(n_mels, cnn_ch1, cnn_ch2, enc_hidden, enc_proj_dim)
        self.decoder = AttentionDecoder(self.encoder.out_dim, dec_hidden, vocab_size, embed_dim)

    def _logmel(self, stft_mag):
        """Eq. 5-6: FBank^i = MelFilterBank(|S_hat^i|); O^i = GlobalMVN(log(FBank^i))."""
        fbank = torch.matmul(stft_mag, self.mel_fb.t())  # [B,T,n_mels]
        log_fbank = torch.log(fbank.clamp_min(1e-6))
        mean = log_fbank.mean(dim=(0, 1), keepdim=True)
        std = log_fbank.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        return (log_fbank - mean) / std

    def forward(self, x_c, target_tokens):
        """
        x_c: [B, C, T, F] complex multi-channel STFT of the mixed input
             (C = n_channels, F = n_freq).
        target_tokens: [B, J, N] teacher-forcing token ids for each of the
             J speakers (int64).

        Returns: logits [B, J, N, vocab_size], one CTC/attention-decoder
        output stream per speaker (matching the paper's multi-output ASR).
        """
        b, c, t, f = x_c.shape

        # Eq. 1: per-channel masks (shared MaskNet across channels)
        masks_per_channel = []
        for ci in range(c):
            masks_per_channel.append(self.mask_net(x_c[:, ci]))  # [B,n_out,T,F]
        masks_c = torch.stack(masks_per_channel, dim=1)  # [B,C,n_out,T,F]

        # Eq. 2-4: MVDR beamforming -> J enhanced single-channel STFTs
        enhanced = self.beamformer(x_c, masks_c)  # list of J [B,T,F] complex

        # Eq. 5-10: per-speaker log-mel -> shared encoder/decoder
        logits_per_speaker = []
        for i in range(self.n_speakers):
            mag = enhanced[i].abs()  # [B,T,F]
            o_i = self._logmel(mag)  # [B,T,n_mels]
            h_i = self.encoder(o_i)  # [B,T',enc_dim]
            logits_i = self.decoder(h_i, target_tokens[:, i])  # [B,N,vocab]
            logits_per_speaker.append(logits_i)

        return torch.stack(logits_per_speaker, dim=1)  # [B,J,N,vocab]


# ---------------------------------------------------------------------------
# Tiny build/example for TorchLens tracing. The paper's real configuration
# (Sec 3.1) uses 3-layer BLSTMP/512 cells for the mask net, 1024-cell VGG+
# BLSTMP encoder, 300-cell LSTM decoder, C=2 channels, J=2 speakers, 257-bin
# STFT (512-pt FFT), 80-dim log-mel. We keep the real J=2/C=2/n_freq=257/
# n_mels=80 architectural constants (these define channel counts and the
# mel-filterbank shape) and shrink only the hidden/cell/CNN-channel sizes
# and sequence lengths for a fast CPU trace.
# ---------------------------------------------------------------------------
_N_CHANNELS = 2
_N_SPEAKERS = 2
_N_FREQ = 257
_N_MELS = 80
_VOCAB_SIZE = 32


def build_mimospeech():
    torch.manual_seed(0)
    model = MIMOSpeech(
        n_channels=_N_CHANNELS,
        n_speakers=_N_SPEAKERS,
        n_freq=_N_FREQ,
        n_mels=_N_MELS,
        mask_hidden=8,
        cnn_ch1=4,
        cnn_ch2=8,
        enc_hidden=12,
        enc_proj_dim=16,
        dec_hidden=16,
        vocab_size=_VOCAB_SIZE,
        embed_dim=8,
    )
    model.eval()
    return model


def example_input_mimospeech():
    torch.manual_seed(0)
    t_frames = 20
    real = torch.randn(2, _N_CHANNELS, t_frames, _N_FREQ)
    imag = torch.randn(2, _N_CHANNELS, t_frames, _N_FREQ)
    x_c = torch.complex(real, imag)
    target_tokens = torch.randint(0, _VOCAB_SIZE, (2, _N_SPEAKERS, 6))
    return (x_c, target_tokens)


MENAGERIE_ENTRIES = [
    ("MIMO-Speech", "build_mimospeech", "example_input_mimospeech", 2019, MENAGERIE_ZOO),
]

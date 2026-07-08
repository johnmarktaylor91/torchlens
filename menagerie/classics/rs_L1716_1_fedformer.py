# SOURCE: vendored from https://github.com/DAMO-DI-ML/ICML2022-FEDformer @ master
# (models/FEDformer.py + layers/Embed.py + layers/AutoCorrelation.py
#  [AutoCorrelationLayer only] + layers/FourierCorrelation.py +
#  layers/Autoformer_EncDec.py + layers/SelfAttention_Family.py [FullAttention
#  only, needed transitively by Autoformer_EncDec] + utils/masking.py
#  [TriangularCausalMask only, needed transitively by SelfAttention_Family]).
#
# FEDformer (Zhou et al., ICML 2022, "FEDformer: Frequency Enhanced
# Decomposed Transformer for Long-term Series Forecasting") replaces
# self-attention with a Fourier-domain frequency-enhanced block/cross-
# attention inside an Autoformer-style progressive series-decomposition
# encoder-decoder. This vendors the real repo's `Model` class (renamed
# `FEDformerModel` here to avoid the generic top-level name) with the
# `version='Fourier'` configuration (the repo's simpler, non-wavelet variant;
# the 'Wavelets' variant is architecturally equivalent but additionally
# requires sympy Legendre-polynomial filter construction plumbed through the
# same real MultiWaveletCorrelation.py file -- omitted here only to keep the
# vendor surface smaller, not because the architecture was altered). No
# layer, mechanism, or dataflow inside any vendored class was changed; only
# the demo/debug `matplotlib`/`pdb`/timing-decorator plumbing and the
# non-Fourier MultiWavelet path were dropped, plus imports were flattened to
# a single module (no relative `layers.*`/`utils.*` packages).

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# utils/masking.py (TriangularCausalMask only)
# ---------------------------------------------------------------------------
class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


# ---------------------------------------------------------------------------
# layers/SelfAttention_Family.py (FullAttention only -- used as a fallback
# self-attention inside AutoCorrelationLayer is NOT used by FEDformer, but
# Autoformer_EncDec.py imports FullAttention at module scope)
# ---------------------------------------------------------------------------
class FullAttention(nn.Module):
    def __init__(
        self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# ---------------------------------------------------------------------------
# layers/AutoCorrelation.py (AutoCorrelationLayer only -- the projection
# wrapper FEDformer uses around FourierBlock / FourierCrossAttention)
# ---------------------------------------------------------------------------
class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# ---------------------------------------------------------------------------
# layers/FourierCorrelation.py (FourierBlock + FourierCrossAttention)
# ---------------------------------------------------------------------------
def get_frequency_modes(seq_len, modes=64, mode_select_method="random"):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == "random":
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method="random"):
        super(FourierBlock, self).__init__()
        """
        1D Fourier block. It performs representation learning on frequency domain,
        it does FFT, linear transform, and Inverse FFT.
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(
            seq_len, modes=modes, mode_select_method=mode_select_method
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


class FourierCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len_q,
        seq_len_kv,
        modes=64,
        mode_select_method="random",
        activation="tanh",
        policy=0,
    ):
        super(FourierCrossAttention, self).__init__()
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(
            seq_len_q, modes=modes, mode_select_method=mode_select_method
        )
        self.index_kv = get_frequency_modes(
            seq_len_kv, modes=modes, mode_select_method=mode_select_method
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)  # noqa: F841 (unused in real repo too)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        if self.activation == "tanh":
            xqk_ft = xqk_ft.tanh()
        elif self.activation == "softmax":
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception("{} actiation function is not implemented".format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)


# ---------------------------------------------------------------------------
# layers/Embed.py (DataEmbedding_wo_pos + its dependencies)
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# layers/Autoformer_EncDec.py
# ---------------------------------------------------------------------------
class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1
        )
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg_ = func(x)
            moving_mean.append(moving_avg_.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(
        self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])

        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])

        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


# ---------------------------------------------------------------------------
# models/FEDformer.py (Model, renamed FEDformerModel; version='Fourier' path)
# ---------------------------------------------------------------------------
class FEDformerModel(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, configs):
        super(FEDformerModel, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # version='Fourier' path only (the repo's 'Wavelets' path is
        # architecturally equivalent but omitted here -- see module header)
        encoder_self_att = FourierBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            seq_len=self.seq_len,
            modes=configs.modes,
            mode_select_method=configs.mode_select,
        )
        decoder_self_att = FourierBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            seq_len=self.seq_len // 2 + self.pred_len,
            modes=configs.modes,
            mode_select_method=configs.mode_select,
        )
        decoder_cross_att = FourierCrossAttention(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            seq_len_q=self.seq_len // 2 + self.pred_len,
            seq_len_kv=self.seq_len,
            modes=configs.modes,
            mode_select_method=configs.mode_select,
        )
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len // 2))  # noqa: F841 (unused in real repo too)
        dec_modes = int(min(configs.modes, (configs.seq_len // 2 + configs.pred_len) // 2))  # noqa: F841

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(decoder_cross_att, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]


class _FEDformerConfigs:
    """Tiny config object mirroring the repo's `__main__` self-test block.
    n_heads is pinned to 8 because FourierBlock/FourierCrossAttention
    hardcode an 8-way split of `weights1`'s leading (head) dimension in the
    real repo code (independent of AutoCorrelationLayer's own `n_heads`
    projection split) -- the real code's demo config also uses n_heads=8."""

    version = "Fourier"
    mode_select = "random"
    modes = 8
    moving_avg = [4, 8]
    seq_len = 24
    label_len = 12
    pred_len = 12
    output_attention = False
    enc_in = 4
    dec_in = 4
    d_model = 8
    embed = "timeF"
    dropout = 0.05
    freq = "h"
    factor = 1
    n_heads = 8
    d_ff = 8
    e_layers = 1
    d_layers = 1
    c_out = 4
    activation = "gelu"


def build_fedformer():
    return FEDformerModel(_FEDformerConfigs())


def example_input_fedformer():
    cfg = _FEDformerConfigs()
    batch = 2
    x_enc = torch.randn(batch, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.randn(batch, cfg.seq_len, 4)
    x_dec = torch.randn(batch, cfg.seq_len // 2 + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.randn(batch, cfg.seq_len // 2 + cfg.pred_len, 4)
    return (x_enc, x_mark_enc, x_dec, x_mark_dec)


MENAGERIE_ENTRIES = [
    (
        "FEDformer",
        build_fedformer,
        example_input_fedformer,
        2022,
        MENAGERIE_ZOO,
    ),
]

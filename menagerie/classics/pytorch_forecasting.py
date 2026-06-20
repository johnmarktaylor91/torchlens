"""Faithful core reimplementations of PyTorch-Forecasting time-series models.

Source: https://github.com/sktime/pytorch-forecasting (maintained fork of
jdb78/pytorch-forecasting).  The shipped models are *dataset-bound*: the only
public constructor is ``Model.from_dataset(TimeSeriesDataSet, ...)`` and the
forward consumes a ``dict[str, Tensor]`` batch with keys ``encoder_cont`` /
``decoder_cont`` / ``encoder_lengths`` / ``decoder_lengths`` / ``target_scale``.
That is why these "ceiling" for the catalog -- there is no tractable standalone
constructor and the input is a structured dict, not a plain tensor.

These reimplementations reproduce the *core forward network* of each model with
a small config and a plain fixed-shape tuple input ``(encoder_cont,
decoder_cont)`` -- dropping only the dataset machinery (``TimeSeriesDataSet``,
``from_dataset``, length packing, ``target_scale`` rescaling, the loss/metric
layer).  Random init, CPU, forward-only.

Models:
  - **DeepAR** (Salinas et al. 2017/2020): autoregressive LSTM with the
    target-roll input trick + a Normal distribution head (loc, softplus(scale)).
  - **N-HiTS** (Challu et al. 2022): doubly-residual stacks of
    MaxPool1d -> MLP -> theta-split -> hierarchical interpolation blocks
    (N-BEATS-style backcast subtraction / forecast accumulation).
  - **TFT** (Lim et al. 2019/2021): GRN + Variable-Selection-Network +
    seq2seq LSTM + interpretable multi-head attention (shared value
    projection, head averaging) + gated-skip fusion -> quantile head.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. DeepAR -- autoregressive LSTM + Normal distribution head
# ============================================================


class DeepAR(nn.Module):
    """DeepAR core: target-roll input + LSTM encoder/decoder + Normal head.

    The distinctive op is the autoregressive *target roll*: at time ``t`` the
    network is fed the target value at ``t-1`` (``torch.roll(..., shifts=1)``),
    and the decoder's first step is seeded with the last true encoder target
    (``one_off_target``).  Teacher-forced forward (the static-architecture path;
    the sampling AR loop reuses the same modules one step at a time).
    """

    def __init__(
        self,
        n_features: int = 4,
        hidden_size: int = 32,
        rnn_layers: int = 2,
        target_position: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.target_position = target_position
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        # NormalDistributionLoss.distribution_arguments = ["loc", "scale"] -> 2 outputs
        self.distribution_projector = nn.Linear(hidden_size, 2)

    def _roll_target(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        tp = self.target_position
        x[..., tp] = torch.roll(x[..., tp], shifts=1, dims=1)
        return x

    def forward(self, encoder_cont: torch.Tensor, decoder_cont: torch.Tensor) -> torch.Tensor:
        # ENCODE: roll target, drop the (garbage) first rolled position.
        enc_in = self._roll_target(encoder_cont)[:, 1:]
        _, (h, c) = self.rnn(enc_in)

        # DECODE (teacher forcing): seed decoder first target with last true encoder target.
        one_off = encoder_cont[:, -1, self.target_position]
        dec_in = self._roll_target(decoder_cont)
        dec_in[:, 0, self.target_position] = one_off
        dec_out, _ = self.rnn(dec_in, (h, c))

        params = self.distribution_projector(dec_out)  # (B, T_dec, 2)
        loc = params[..., 0]
        scale = F.softplus(params[..., 1])
        return torch.stack([loc, scale], dim=-1)


def build_deepar() -> nn.Module:
    return DeepAR(n_features=4, hidden_size=32, rnn_layers=2)


def example_input_deepar() -> List[torch.Tensor]:
    """``[encoder_cont (1,12,4), decoder_cont (1,6,4)]`` covariate windows."""
    return [torch.randn(1, 12, 4), torch.randn(1, 6, 4)]


# ============================================================
# 2. N-HiTS -- doubly-residual pooling/MLP/interpolation stacks
# ============================================================


class _NHiTSMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_features, hidden[0]), nn.ReLU()]
        for i in range(len(hidden) - 1):
            layers += [nn.Linear(hidden[i], hidden[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden[-1], out_features)]  # no final activation
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _NHiTSBlock(nn.Module):
    """MaxPool1d -> MLP -> split theta -> hierarchical interpolation.

    Backcast thetas are taken at full resolution; forecast thetas are a small
    set of ``n_theta`` knots linearly upsampled to the prediction length (the
    N-HiTS hierarchical-interpolation trick).
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_theta: int,
        pooling_size: int,
        hidden: List[int],
        interpolation_mode: str = "linear",
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_theta = n_theta
        self.interpolation_mode = interpolation_mode
        self.pool = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size, ceil_mode=True)
        pooled_len = math.ceil(context_length / pooling_size)
        self.mlp = _NHiTSMLP(pooled_len, context_length + n_theta, hidden)

    def forward(self, encoder_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_y: (B, T_enc, 1)
        x = encoder_y.transpose(1, 2)  # (B, 1, T_enc)
        x = self.pool(x)  # (B, 1, pooled)
        x = x.transpose(1, 2).reshape(x.shape[0], -1)  # (B, pooled)
        theta = self.mlp(x)  # (B, T_enc + n_theta)
        backcast = theta[:, : self.context_length]  # (B, T_enc)
        knots = theta[:, self.context_length :]  # (B, n_theta)
        forecast = F.interpolate(
            knots[:, None, :], size=self.prediction_length, mode=self.interpolation_mode
        )[:, 0, :]  # (B, T_dec)
        return backcast.unsqueeze(-1), forecast.unsqueeze(-1)


class NHiTS(nn.Module):
    """N-HiTS stack: per-stack pooling rate + doubly-residual backcast/forecast."""

    def __init__(
        self,
        context_length: int = 24,
        prediction_length: int = 8,
        pooling_sizes: List[int] = None,
        downsample_frequencies: List[int] = None,
        hidden: List[int] = None,
        naive_level: bool = True,
    ) -> None:
        super().__init__()
        if pooling_sizes is None:
            pooling_sizes = [4, 2, 1]
        if downsample_frequencies is None:
            downsample_frequencies = [8, 4, 2]
        if hidden is None:
            hidden = [64, 64]
        self.naive_level = naive_level
        self.prediction_length = prediction_length
        blocks = []
        for pool, df in zip(pooling_sizes, downsample_frequencies):
            n_theta = max(prediction_length // df, 1)
            blocks.append(
                _NHiTSBlock(context_length, prediction_length, n_theta, pool, list(hidden))
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, encoder_y: torch.Tensor) -> torch.Tensor:
        # encoder_y: (B, T_enc, 1)
        residuals = encoder_y
        if self.naive_level:
            forecast = encoder_y[:, -1:].repeat(1, self.prediction_length, 1)  # Naive-1 seed
        else:
            forecast = torch.zeros(
                encoder_y.shape[0], self.prediction_length, 1, device=encoder_y.device
            )
        for block in self.blocks:
            block_backcast, block_forecast = block(residuals)
            residuals = residuals - block_backcast  # backcast subtraction
            forecast = forecast + block_forecast  # forecast accumulation
        return forecast


def build_nhits() -> nn.Module:
    return NHiTS(context_length=24, prediction_length=8)


def example_input_nhits() -> torch.Tensor:
    """Target backcast series ``(1, 24, 1)`` (N-HiTS core works on the target)."""
    return torch.randn(1, 24, 1)


# ============================================================
# 3. Temporal Fusion Transformer (TFT)
# ============================================================


class _GLU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(self.fc(self.dropout(x)), dim=-1)


class _AddNorm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(x + skip)


class _GateAddNorm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.glu = _GLU(input_size, hidden_size, dropout)
        self.add_norm = _AddNorm(hidden_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.add_norm(self.glu(x), skip)


class _GRN(nn.Module):
    """Gated Residual Network: Linear -> (+ctx) -> ELU -> Linear -> GLU -> +res -> LayerNorm."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.resample = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.context = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate_norm = _GateAddNorm(hidden_size, output_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if residual is None:
            residual = x
        if self.resample is not None:
            residual = self.resample(residual)
        h = self.fc1(x)
        if self.context is not None and context is not None:
            h = h + self.context(context)
        h = self.elu(h)
        h = self.fc2(h)
        return self.gate_norm(h, residual)


class _VariableSelectionNetwork(nn.Module):
    """Per-variable GRN transforms + softmax-weighted sum (selection)."""

    def __init__(
        self,
        n_vars: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        # each continuous variable: scalar -> hidden via a per-variable GRN
        self.prescalers = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(n_vars)])
        self.var_grns = nn.ModuleList(
            [_GRN(hidden_size, hidden_size, hidden_size, dropout) for _ in range(n_vars)]
        )
        self.flattened_grn = _GRN(
            hidden_size * n_vars, hidden_size, n_vars, dropout, context_size=context_size
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, n_vars) continuous reals
        var_embeds = [
            self.prescalers[i](x[..., i : i + 1]) for i in range(self.n_vars)
        ]  # each (B, T, H)
        var_outputs = torch.stack(
            [grn(e) for grn, e in zip(self.var_grns, var_embeds)], dim=-1
        )  # (B, T, H, n_vars)
        flat = torch.cat(var_embeds, dim=-1)  # (B, T, H*n_vars)
        weights = self.softmax(self.flattened_grn(flat, context)).unsqueeze(-2)  # (B,T,1,n_vars)
        outputs = (var_outputs * weights).sum(dim=-1)  # (B, T, H)
        return outputs, weights


class _InterpretableMultiHeadAttention(nn.Module):
    """Shared value projection across heads + head averaging (interpretable)."""

    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.v_layer = nn.Linear(d_model, self.d_k)  # ONE shared value projection
        self.q_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(n_head)])
        self.dropout = nn.Dropout(dropout)
        self.w_h = nn.Linear(self.d_k, d_model, bias=False)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        return torch.matmul(self.dropout(torch.softmax(scores, dim=-1)), v)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        vs = self.v_layer(v)  # shared across heads
        heads = [
            self.dropout(self._attn(self.q_layers[i](q), self.k_layers[i](k), vs))
            for i in range(self.n_head)
        ]
        head = torch.stack(heads, dim=2).mean(dim=2)  # average heads -> (B, Tq, d_k)
        return self.dropout(self.w_h(head))


class TemporalFusionTransformer(nn.Module):
    """TFT core: VSN -> seq2seq LSTM -> static enrichment -> interpretable attn -> quantiles."""

    def __init__(
        self,
        n_cont: int = 4,
        hidden_size: int = 16,
        attn_heads: int = 4,
        lstm_layers: int = 1,
        n_quantiles: int = 7,
        dropout: float = 0.1,
        encoder_length: int = 12,
    ) -> None:
        super().__init__()
        H = hidden_size
        self.encoder_length = encoder_length
        self.encoder_vsn = _VariableSelectionNetwork(n_cont, H, dropout, context_size=H)
        self.decoder_vsn = _VariableSelectionNetwork(n_cont, H, dropout, context_size=H)
        self.lstm_encoder = nn.LSTM(H, H, lstm_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(H, H, lstm_layers, batch_first=True)
        self.post_lstm_gate = _GateAddNorm(H, H, dropout)
        self.static_enrichment = _GRN(H, H, H, dropout, context_size=H)
        self.attn = _InterpretableMultiHeadAttention(attn_heads, H, dropout)
        self.post_attn_gate = _GateAddNorm(H, H, dropout)
        self.pos_wise_ff = _GRN(H, H, H, dropout)
        self.pre_output_gate = _GateAddNorm(H, H, dropout)
        self.output_layer = nn.Linear(H, n_quantiles)
        self.hidden_size = H

    def forward(self, encoder_cont: torch.Tensor, decoder_cont: torch.Tensor) -> torch.Tensor:
        B = encoder_cont.shape[0]
        H = self.hidden_size
        # No static covariates: zero static context.
        static_ctx = torch.zeros(B, H, device=encoder_cont.device)
        enc_ctx = static_ctx.unsqueeze(1).expand(-1, encoder_cont.shape[1], -1)
        dec_ctx = static_ctx.unsqueeze(1).expand(-1, decoder_cont.shape[1], -1)

        emb_enc, _ = self.encoder_vsn(encoder_cont, enc_ctx)
        emb_dec, _ = self.decoder_vsn(decoder_cont, dec_ctx)

        enc_out, (h, c) = self.lstm_encoder(emb_enc)
        dec_out, _ = self.lstm_decoder(emb_dec, (h, c))

        lstm_enc = self.post_lstm_gate(enc_out, emb_enc)
        lstm_dec = self.post_lstm_gate(dec_out, emb_dec)
        lstm_output = torch.cat([lstm_enc, lstm_dec], dim=1)  # (B, T, H)

        enrich_ctx = static_ctx.unsqueeze(1).expand(-1, lstm_output.shape[1], -1)
        attn_input = self.static_enrichment(lstm_output, enrich_ctx)

        T_enc = encoder_cont.shape[1]
        q = attn_input[:, T_enc:]  # decoder steps query
        attn_out = self.attn(q, attn_input, attn_input)
        attn_out = self.post_attn_gate(attn_out, q)

        out = self.pos_wise_ff(attn_out)
        out = self.pre_output_gate(out, lstm_output[:, T_enc:])
        return self.output_layer(out)  # (B, T_dec, n_quantiles)


def build_tft() -> nn.Module:
    return TemporalFusionTransformer(
        n_cont=4, hidden_size=16, attn_heads=4, lstm_layers=1, n_quantiles=7, encoder_length=12
    )


def example_input_tft() -> List[torch.Tensor]:
    """``[encoder_cont (1,12,4), decoder_cont (1,6,4)]`` continuous covariates."""
    return [torch.randn(1, 12, 4), torch.randn(1, 6, 4)]


MENAGERIE_ENTRIES = [
    (
        "PyTorch-Forecasting DeepAR (autoregressive LSTM + Normal head)",
        "build_deepar",
        "example_input_deepar",
        "2017",
        "DE",
    ),
    (
        "PyTorch-Forecasting N-HiTS (hierarchical interpolation MLP stacks)",
        "build_nhits",
        "example_input_nhits",
        "2022",
        "DE",
    ),
    (
        "PyTorch-Forecasting TFT (variable selection + interpretable attention)",
        "build_tft",
        "example_input_tft",
        "2019",
        "DE",
    ),
]

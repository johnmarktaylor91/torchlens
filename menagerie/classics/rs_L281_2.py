# SOURCE: vendored from Gutianpei/MID @ 3b61674bd4beb13dbd662ec3b95542911176189d
# Files: models/diffusion.py, models/common.py (verbatim real architecture; only
# import paths flattened and the Trajectron++ trajectory encoder replaced at the
# INPUT boundary with a random `context` tensor of the same shape it produces --
# the paper's denoising network (DiffusionTraj + TransformerConcatLinear, the
# `diffnet` selected by the repo's own configs/baseline.yaml) is captured verbatim).
"""MID: Motion Indeterminacy Diffusion (CVPR 2022) -- denoising diffusion network
for stochastic pedestrian trajectory prediction.

Real repo: https://github.com/Gutianpei/MID

`models/autoencoder.py::AutoEncoder` wraps a full Trajectron++ encoder (RNN/graph
pipeline that consumes pickled `Environment`/`Scene` objects, not a plain tensor)
around this diffusion module. That encoder cannot be constructed from a tiny
random-init tensor input, so this staging module traces the diffusion network --
the paper's actual contribution -- directly, with `context` (the encoder's latent
output, shape `[B, encoder_dim]` per the real `AutoEncoder.encode`) supplied as a
random tensor input instead of coming from the Trajectron++ encoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/common.py (verbatim)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


# ---------------------------------------------------------------------------
# models/diffusion.py (verbatim)
# ---------------------------------------------------------------------------
class VarianceSchedule(Module):
    def __init__(self, num_steps, mode="linear", beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ("linear", "cosine")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "cosine":
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        import numpy as np

        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffusionTraj(Module):
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        batch_size, _, point_dim = x_0.size()
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)

        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction="mean")
        return loss

    def forward(self, x_0, context, t=None):
        # NOTE (staging addition, not in the original repo): the real class only
        # exposes get_loss()/sample() as its usage entry points. This forward()
        # replicates exactly what get_loss() does to reach self.net (its one
        # noise-prediction call) so the module is directly callable/traceable;
        # no new architecture/behavior is introduced.
        batch_size = context.size(0)
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)
        beta = self.var_sched.betas[t]
        return self.net(x_0, beta=beta, context=context)


class TrajNet(Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList(
            [
                ConcatSquashLinear(2, 128, context_dim + 3),
                ConcatSquashLinear(128, 256, context_dim + 3),
                ConcatSquashLinear(256, 512, context_dim + 3),
                ConcatSquashLinear(512, 256, context_dim + 3),
                ConcatSquashLinear(256, 128, context_dim + 3),
                ConcatSquashLinear(128, 2, context_dim + 3),
            ]
        )

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(2, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim, nhead=4, dim_feedforward=4 * context_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
        self.linear = ConcatSquashLinear(context_dim // 2, 2, context_dim + 3)
        # self.linear = nn.Linear(128,2)

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


class TransformerLinear(Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim + 3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(1, 0, 2)  # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)


class LinearDecoder(Module):
    def __init__(self):
        super().__init__()
        self.act = F.leaky_relu
        self.layers = ModuleList(
            [
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12),
            ]
        )

    def forward(self, code):
        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out


# ---------------------------------------------------------------------------
# staging harness -- matches the repo's own configs/baseline.yaml default
# hyperparameters (diffnet=TransformerConcatLinear, encoder_dim=256, tf_layer=3),
# scaled down (fewer diffusion steps) purely for fast tracing.
# ---------------------------------------------------------------------------
_ENCODER_DIM = 256
_TF_LAYER = 3
_NUM_STEPS = 20  # real config uses 100; reduced only to keep the VarianceSchedule
# buffers small for a fast trace -- does not change the architecture.


def build_mid_diffusion():
    net = TransformerConcatLinear(
        point_dim=2, context_dim=_ENCODER_DIM, tf_layer=_TF_LAYER, residual=False
    )
    var_sched = VarianceSchedule(num_steps=_NUM_STEPS, beta_T=5e-2, mode="linear")
    return DiffusionTraj(net=net, var_sched=var_sched)


def example_input_mid_diffusion():
    batch_size = 3
    num_future_points = 12  # matches the repo's `num_points=12` prediction horizon
    x_0 = torch.randn(batch_size, num_future_points, 2)
    context = torch.randn(batch_size, _ENCODER_DIM)  # Trajectron++ encoder latent (real shape)
    t = torch.randint(1, _NUM_STEPS + 1, (batch_size,)).tolist()
    return (x_0, context, t)


MENAGERIE_ENTRIES = [
    ("MID", build_mid_diffusion, example_input_mid_diffusion, 2022, "vendored-pytorch"),
]

# SOURCE: vendored from facebookresearch/svg @ main
# https://github.com/facebookresearch/svg/blob/main/svg/dx.py
# https://github.com/facebookresearch/svg/blob/main/svg/utils.py (mlp, weight_init only)
#
# Amos, Stanton, Yarats & Wilson, "On the model-based stochastic value gradient
# for continuous reinforcement learning" (L4DC 2021). `SeqDx` is the paper's
# differentiable, recurrent STOCHASTIC VALUE GRADIENT dynamics model: an MLP
# state+action encoder feeds an LSTM/GRU recurrent core, whose latent state is
# decoded by a second MLP into a residual next-state delta
# (`xtp1 = xt + self.x_dec(...)`), with optional frozen "goal" observation
# dimensions carried through unchanged. This class -- the differentiable world
# model that gives SVG its name -- is transcribed VERBATIM from `svg/dx.py`
# (constructor, `init_hidden_state`, `unroll_policy`, `unroll`, `forward`;
# `update_step` is a training-loop optimizer method, dropped here since it is
# not exercised by tracing `forward()`).
#
# Import-isolation fixes only (no architectural edit):
#   - Upstream `dx.py` does `from . import utils`, and `svg/utils.py`
#     unconditionally imports `gym` and `from .env import dmc` (which in turn
#     requires the non-base `dmc2gym` package purely for RL-environment
#     construction, unrelated to the model itself). Only the two helper
#     functions `dx.py` actually calls -- `utils.mlp` and `utils.weight_init`
#     -- are inlined here verbatim from `utils.py`, avoiding the unusable
#     environment-registration import chain.
#   - `self.opt = torch.optim.Adam(...)` (used only by `update_step`, which is
#     not vendored) is dropped from `__init__` to avoid depending on
#     `utils.get_params`; nothing on the traced forward path depends on it.

import torch
from torch import nn


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    """Verbatim from svg/utils.py."""
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def weight_init(m):
    """Verbatim from svg/utils.py. Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class SeqDx(nn.Module):
    def __init__(
        self,
        env_name,
        obs_dim,
        action_dim,
        action_range,
        horizon,
        device,
        detach_xt,
        clip_grad_norm,
        xu_enc_hidden_dim,
        xu_enc_hidden_depth,
        x_dec_hidden_dim,
        x_dec_hidden_depth,
        rec_type,
        rec_latent_dim,
        rec_num_layers,
        lr,
    ):
        super().__init__()

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm

        # Manually freeze the goal locations
        if env_name == "gym_petsReacher":
            self.freeze_dims = torch.LongTensor([7, 8, 9])
        elif env_name == "gym_petsPusher":
            self.freeze_dims = torch.LongTensor([20, 21, 22])
        else:
            self.freeze_dims = None

        self.rec_type = rec_type
        self.rec_num_layers = rec_num_layers
        self.rec_latent_dim = rec_latent_dim

        self.xu_enc = mlp(
            obs_dim + action_dim, xu_enc_hidden_dim, rec_latent_dim, xu_enc_hidden_depth
        )
        self.x_dec = mlp(rec_latent_dim, x_dec_hidden_dim, obs_dim, x_dec_hidden_depth)

        self.apply(weight_init)  # Don't apply this to the recurrent unit.

        if rec_num_layers > 0:
            if rec_type == "LSTM":
                self.rec = nn.LSTM(rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            elif rec_type == "GRU":
                self.rec = nn.GRU(rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            else:
                assert False

    def init_hidden_state(self, init_x):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.rec_type == "LSTM":
            h = torch.zeros(self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == "GRU":
            h = torch.zeros(self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
        else:
            assert False

        return h

    def unroll_policy(self, init_x, policy, sample=True, last_u=True, detach_xt=False):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = init_x[:, self.freeze_dims]

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(init_x)

        pred_xs = []
        us = []
        log_p_us = []
        xt = init_x
        for t in range(self.horizon - 1):
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen

            pred_xs.append(xtp1)
            xt = xtp1

        if last_u:
            policy_kwargs = {}
            if sample:
                _, ut, log_p_ut = policy(xt, **policy_kwargs)
            else:
                ut, _, log_p_ut = policy(xt, **policy_kwargs)
            us.append(ut)
            log_p_us.append(log_p_ut)

        us = torch.stack(us)
        log_p_us = torch.stack(log_p_us).squeeze(2)
        if self.horizon <= 1:
            pred_xs = torch.empty(0, n_batch, self.obs_dim).to(init_x.device)
        else:
            pred_xs = torch.stack(pred_xs)

        return us, log_p_us, pred_xs

    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(x)

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            if self.freeze_dims is not None:
                xtp1[:, self.freeze_dims] = obs_frozen
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        return pred_xs

    def forward(self, x, us):
        return self.unroll(x, us)


def build_seqdx():
    """Tiny-config SVG recurrent dynamics model (CPU, LSTM core)."""
    return SeqDx(
        env_name="dmc_cheetah_run",
        obs_dim=6,
        action_dim=3,
        action_range=(-1.0, 1.0),
        horizon=4,
        device="cpu",
        detach_xt=False,
        clip_grad_norm=None,
        xu_enc_hidden_dim=32,
        xu_enc_hidden_depth=1,
        x_dec_hidden_dim=32,
        x_dec_hidden_depth=1,
        rec_type="LSTM",
        rec_latent_dim=16,
        rec_num_layers=1,
        lr=1e-3,
    )


def example_input_seqdx():
    seq_len = 3
    batch_size = 4
    x = torch.randn(batch_size, 6)
    us = torch.randn(seq_len, batch_size, 3)
    return (x, us)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SVG Recurrent Dynamics Model (SeqDx)", "build_seqdx", "example_input_seqdx", 2021, "RL"),
]

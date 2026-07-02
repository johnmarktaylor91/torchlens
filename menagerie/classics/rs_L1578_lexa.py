# FAITHFUL PORT of https://github.com/orybkin/lexa @ master (original framework: TensorFlow 2 / Keras)
# (lexa/networks.py: RSSM, GRUCell, ConvEncoder, ConvDecoder, GC_Encoder,
#  GC_Actor, GC_Critic, lines 11-527)
#
# LEXA / "Latent Explorer-Achiever" (Mendonca & Rybkin et al. 2021, NeurIPS
# 2021, "Discovering and Achieving Goals via World Models"). Official
# orybkin/lexa repo, built on a TF2/Keras DreamerV2-style world model
# (`tensorflow`, `tensorflow_probability`); not installable in a base torch
# env, so the real graph structure is transcribed faithfully into
# self-contained torch. Ported here (forward-pass architecture only, dropping
# the tfd.Distribution / imagination-rollout / loss-computation machinery that
# is orthogonal to the module graph):
#   - `GRUCell`: the project's layer-norm GRU recurrent cell (single
#     `Dense(3*size)` producing reset/candidate/update gates, LayerNorm on the
#     pre-activation, matching `norm=True` used by the RSSM's `cell='gru_layer_norm'`).
#   - `RSSM`: the Recurrent State-Space Model -- `img_step` (deterministic GRU
#     transition conditioned on previous stochastic state + action, followed
#     by an MLP producing Gaussian (mean, std) "prior" stats) and `obs_step`
#     (the same GRU state additionally conditioned on the image embedding to
#     produce the "posterior" stats) -- ported as `RSSM.forward` running one
#     observe step for a traceable single-step forward pass.
#   - `ConvEncoder` / `ConvDecoder`: the image encoder/decoder conv towers.
#   - `GC_Encoder` / `GC_Actor` / `GC_Critic`: LEXA's goal-conditioned heads --
#     the "achiever" policy/critic that LEXA's explorer-achiever design adds on
#     top of the base world model, each conditioned on a (state, goal) pair.
# All layer widths/depths/activations match the real `__init__` defaults in
# networks.py; only image resolution is reduced for a fast trace.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class GRUCell(nn.Module):
    """networks.py GRUCell(size, norm=True): single Dense(3*size) -> split
    into reset/candidate/update gates, with LayerNorm on the pre-activation
    (the `norm` branch used by RSSM's default `cell='gru_layer_norm'`)."""

    def __init__(self, size, act=torch.tanh, update_bias=-1):
        super().__init__()
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self._layer = nn.Linear(2 * size, 3 * size, bias=False)
        self._norm = nn.LayerNorm(3 * size)

    def forward(self, inputs, state):
        parts = self._layer(torch.cat([inputs, state], -1))
        parts = self._norm(parts)
        reset, cand, update = torch.chunk(parts, 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output


class RSSM(nn.Module):
    """networks.py RSSM: layer-norm-GRU recurrent state-space model. Ported
    as a single `img_step` + `obs_step` pair exercised once per forward call
    (the real `observe`/`imagine` are just `tf.scan` over this same step)."""

    def __init__(self, stoch=30, deter=200, hidden=200, embed_size=1024, action_size=6):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden

        # img_step MLPs: [prev_stoch, prev_action] -> hidden -> GRU -> hidden -> stats
        self._img_in = nn.Sequential(nn.Linear(stoch + action_size, hidden), nn.ELU())
        self._cell = GRUCell(deter)
        self._img_out = nn.Sequential(nn.Linear(deter, hidden), nn.ELU())
        self._img_stats = nn.Linear(hidden, 2 * stoch)

        # obs_step MLP: [deter, embed] -> hidden -> stats
        self._obs_out = nn.Sequential(nn.Linear(deter + embed_size, hidden), nn.ELU())
        self._obs_stats = nn.Linear(hidden, 2 * stoch)

    def img_step(self, prev_stoch, prev_action, prev_deter):
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_in(x)
        deter = self._cell(x, prev_deter)
        x = self._img_out(deter)
        mean, std = torch.chunk(self._img_stats(x), 2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch = mean + std * torch.randn_like(std)
        return stoch, deter

    def obs_step(self, prev_stoch, prev_action, prev_deter, embed):
        prior_stoch, deter = self.img_step(prev_stoch, prev_action, prev_deter)
        x = torch.cat([deter, embed], -1)
        x = self._obs_out(x)
        mean, std = torch.chunk(self._obs_stats(x), 2, dim=-1)
        std = F.softplus(std) + 0.1
        post_stoch = mean + std * torch.randn_like(std)
        return post_stoch, deter, prior_stoch

    def get_feat(self, stoch, deter):
        return torch.cat([stoch, deter], -1)

    def forward(self, prev_stoch, prev_action, prev_deter, embed):
        post_stoch, deter, prior_stoch = self.obs_step(prev_stoch, prev_action, prev_deter, embed)
        return self.get_feat(post_stoch, deter)


class ConvEncoder(nn.Module):
    """networks.py ConvEncoder(depth=32, kernels=(4,4,4,4)): 4-layer strided
    conv tower over the image observation."""

    def __init__(self, in_channels=3, depth=32):
        super().__init__()
        self.h1 = nn.Conv2d(in_channels, depth, 4, stride=2)
        self.h2 = nn.Conv2d(depth, 2 * depth, 4, stride=2)
        self.h3 = nn.Conv2d(2 * depth, 4 * depth, 4, stride=2)
        self.h4 = nn.Conv2d(4 * depth, 8 * depth, 4, stride=2)

    def forward(self, image):
        x = F.relu(self.h1(image))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.relu(self.h4(x))
        return x.reshape(x.size(0), -1)


class ConvDecoder(nn.Module):
    """networks.py ConvDecoder(depth=32, kernels=(5,5,6,6), thin=True):
    Dense -> reshape -> 4-layer transposed-conv tower reconstructing the
    image."""

    def __init__(self, feat_size, depth=32, out_channels=3):
        super().__init__()
        self.h1 = nn.Linear(feat_size, 32 * depth)
        self.h2 = nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride=2)
        self.h3 = nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride=2)
        self.h4 = nn.ConvTranspose2d(2 * depth, depth, 6, stride=2)
        self.h5 = nn.ConvTranspose2d(depth, out_channels, 6, stride=2)

    def forward(self, features):
        x = self.h1(features)
        x = x.view(-1, x.size(-1), 1, 1)
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.relu(self.h4(x))
        x = self.h5(x)
        return x


class GC_Encoder(nn.Module):
    """networks.py GC_Encoder(depth=8, kernels=(4,4,4,4)): the goal-image
    encoder used by the achiever's critic/actor (leaky-relu conv tower with
    two interleaved BatchNorm layers, matching the real layer order)."""

    def __init__(self, in_channels=3, depth=8):
        super().__init__()
        self.h1 = nn.Conv2d(in_channels, depth, 4, stride=2)
        self.h2 = nn.Conv2d(depth, 2 * depth, 4, stride=2)
        self.bn1 = nn.BatchNorm2d(2 * depth)
        self.h3 = nn.Conv2d(2 * depth, 4 * depth, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(4 * depth)
        self.h4 = nn.Conv2d(4 * depth, 8 * depth, 4, stride=2)

    def forward(self, gc_obs):
        x = F.leaky_relu(self.h1(gc_obs))
        x = F.leaky_relu(self.h2(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.h3(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.h4(x))
        return x.reshape(x.size(0), -1)


class GC_Critic(nn.Module):
    """networks.py GC_Critic(layers=4, units=128): the achiever's
    goal-conditioned critic -- encode(goal-obs) concatenated with action, fed
    through 4 Dense+BatchNorm blocks to a scalar Q-value."""

    def __init__(self, in_channels=3, action_size=6, layers=4, units=128):
        super().__init__()
        self.encoder = GC_Encoder(in_channels=in_channels)
        self._layers = layers
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        # in_dim resolved lazily on first forward (mirrors TF's shape-inferred `self.get`)
        self._action_size = action_size
        self._units = units
        self._built = False

    def _build(self, in_dim):
        for _ in range(self._layers):
            self.fc.append(nn.Linear(in_dim, self._units))
            self.bn.append(nn.BatchNorm1d(self._units))
            in_dim = self._units
        self.hout = nn.Linear(in_dim, 1)
        self._built = True

    def forward(self, gc_obs, action):
        enc = self.encoder(gc_obs)
        x = torch.cat([enc, action], dim=-1)
        if not self._built:
            self._build(x.size(-1))
        for fc, bn in zip(self.fc, self.bn):
            x = F.relu(fc(x))
            x = bn(x)
        return self.hout(x).squeeze(-1)


class GC_Actor(nn.Module):
    """networks.py GC_Actor(layers=10, units=128, from_images=True): the
    achiever's goal-conditioned policy -- encode(goal-obs) fed through 10
    Dense+BatchNorm blocks to a tanh-bounded action."""

    def __init__(self, action_size, in_channels=3, layers=10, units=128):
        super().__init__()
        self.encoder = GC_Encoder(in_channels=in_channels)
        self._layers = layers
        self._units = units
        self._action_size = action_size
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self._built = False

    def _build(self, in_dim):
        for _ in range(self._layers):
            self.fc.append(nn.Linear(in_dim, self._units))
            self.bn.append(nn.BatchNorm1d(self._units))
            in_dim = self._units
        self.hout = nn.Linear(in_dim, self._action_size)
        self._built = True

    def forward(self, gc_obs):
        x = self.encoder(gc_obs)
        if not self._built:
            self._build(x.size(-1))
        for fc, bn in zip(self.fc, self.bn):
            x = F.relu(fc(x))
            x = bn(x)
        return torch.tanh(self.hout(x))


class LexaAgent(nn.Module):
    """Staging wrapper exercising the world-model RSSM + image codec together
    with LEXA's explorer/achiever goal-conditioned actor+critic heads in a
    single traceable forward pass."""

    def __init__(self, image_size=16, stoch=8, deter=16, hidden=16, action_size=4):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=3, depth=4)
        embed_size = self._probe_embed_size(image_size)
        self.rssm = RSSM(
            stoch=stoch, deter=deter, hidden=hidden, embed_size=embed_size, action_size=action_size
        )
        self.decoder = ConvDecoder(feat_size=stoch + deter, depth=4, out_channels=3)
        self.gc_actor = GC_Actor(action_size=action_size, in_channels=3, layers=2, units=16)
        self.gc_critic = GC_Critic(in_channels=3, action_size=action_size, layers=2, units=16)
        self.stoch = stoch
        self.deter = deter
        self.action_size = action_size

    def _probe_embed_size(self, image_size):
        probe = ConvEncoder(in_channels=3, depth=4)
        with torch.no_grad():
            out = probe(torch.zeros(1, 3, image_size, image_size))
        return out.size(-1)

    def forward(self, image, goal_image, prev_action):
        batch = image.size(0)
        embed = self.encoder(image)
        prev_stoch = torch.zeros(batch, self.stoch)
        prev_deter = torch.zeros(batch, self.deter)
        feat = self.rssm(prev_stoch, prev_action, prev_deter, embed)
        recon = self.decoder(feat)
        goal_action = self.gc_actor(goal_image)
        value = self.gc_critic(goal_image, goal_action)
        return recon, goal_action, value


def build_lexa():
    return LexaAgent(image_size=64, stoch=8, deter=16, hidden=16, action_size=4)


def example_input_lexa():
    image = torch.randn(2, 3, 64, 64)
    goal_image = torch.randn(2, 3, 64, 64)
    prev_action = torch.zeros(2, 4)
    return image, goal_image, prev_action


MENAGERIE_ENTRIES = [
    ("LEXA", "build_lexa", "example_input_lexa", 2021, "ported-pytorch"),
]

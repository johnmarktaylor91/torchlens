# FAITHFUL PORT of https://github.com/google-deepmind/deepmind-research @ master (original framework: TensorFlow-v1 / dm-sonnet<2)
# (visr/VISR_ICLR2020.ipynb, "Define Computational Graph" cell)
#
# VISR: Variational Intrinsic Successor Features (Hansen, Dabney, Barreto, Warde-Farley,
# Van de Wiele, Mnih 2020, ICLR, "Fast Task Inference with Variational Intrinsic Successor
# Features", arXiv:1906.05030). DeepMind's official release ships only as a TF1 + Sonnet<2
# Colab notebook driving a `tf.placeholder`/`SingularMonitoredSession` graph on a toy
# GridWorld -- no standalone .py module and no TF2/torch port exists, and TF1+legacy-Sonnet
# cannot reasonably be installed alongside this repo's torch stack, so the "Define
# Computational Graph" cell's real network code is transcribed faithfully into
# self-contained torch. Architecture: a `phi_net` state-embedding MLP
# (`snt.Sequential([snt.nets.MLP([hid_dim, phi_dim]), l2_normalize])` -- the "successor
# feature" encoder phi(s), L2-normalized as VISR's variational bound requires) and a
# `psi_net` successor-feature-predictor MLP (`snt.nets.MLP([hid_dim, hid_dim,
# phi_dim*num_a])`, taking the concatenation of the one-hot observation and the sampled
# policy-conditioning vector `w` and producing per-action successor features psi(s, w) --
# VISR's defining architectural move of conditioning the value/successor network directly
# on a sampled latent task/option vector `w`). `psi` is reshaped to `(phi_dim, num_a)` and
# contracted with `w` via `einsum('tbpa,tbp->tba', psi, w)` to produce Q-values, exactly as
# the original graph computes `q`. The GPI (generalized policy improvement) max-over-samples
# branch and the training-loss/session-feed-dict machinery are notebook-level driver code,
# not part of the traceable network, so only the two MLPs plus the psi-reshape/einsum
# forward computation are ported here.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class MLP(nn.Module):
    """Faithful port of snt.nets.MLP(output_sizes): stack of Linear layers with ReLU
    between hidden layers (Sonnet's MLP default activation), no activation after the
    final layer -- matching `snt.nets.MLP([hid_dim, hid_dim, phi_dim*num_a])` /
    `snt.nets.MLP([hid_dim, phi_dim])` usage in the original notebook."""

    def __init__(self, in_dim, output_sizes):
        super().__init__()
        layers = []
        prev = in_dim
        for i, size in enumerate(output_sizes):
            layers.append(nn.Linear(prev, size))
            if i < len(output_sizes) - 1:
                layers.append(nn.ReLU())
            prev = size
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VISRNet(nn.Module):
    """Faithful port of the VISR_ICLR2020.ipynb "Define Computational Graph" cell:
    phi_net (state-embedding, L2-normalized) + psi_net (successor-feature predictor
    conditioned on sampled task vector w), combined via the notebook's
    `einsum('tbpa,tbp->tba', psi, w)` to produce per-action Q-values."""

    def __init__(self, num_states, num_actions, phi_dim, hid_dim):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.phi_dim = phi_dim

        # phi_net = snt.Sequential([snt.nets.MLP([hid_dim, phi_dim]), l2_normalize])
        self.phi_net = MLP(num_states, [hid_dim, phi_dim])
        # psi_net = snt.nets.MLP([hid_dim, hid_dim, phi_dim*num_a])
        self.psi_net = MLP(num_states + phi_dim, [hid_dim, hid_dim, phi_dim * num_actions])

    def forward(self, obs, w):
        """
        :param obs: (T, B, num_states) one-hot state observations (tf.one_hot(s_ph, num_states))
        :param w: (T, B, phi_dim) sampled policy-conditioning vectors (options)
        :return: (phi, psi, q) matching the original graph's `phi`, `psi`, `q` tensors
        """
        t_len, batch = obs.shape[0], obs.shape[1]

        # phi = snt.BatchApply(phi_net)(obs); l2_normalize applied by phi_net's Sequential tail
        phi = self.phi_net(obs)
        phi = F.normalize(phi, p=2, dim=-1)

        # psi = reshape(snt.BatchApply(psi_net)(concat([obs, w_ph], -1)), [T, B, phi_dim, num_a])
        psi_in = torch.cat([obs, w], dim=-1)
        psi = self.psi_net(psi_in).view(t_len, batch, self.phi_dim, self.num_actions)

        # q = einsum('tbpa,tbp->tba', psi, w_ph)
        q = torch.einsum("tbpa,tbp->tba", psi, w)

        return phi, psi, q


def build_visr():
    return VISRNet(num_states=16, num_actions=4, phi_dim=2, hid_dim=32)


def example_input_visr():
    torch.manual_seed(0)
    t_len, batch, num_states, phi_dim = 3, 2, 16, 2
    obs = F.one_hot(torch.randint(0, num_states, (t_len, batch)), num_states).float()
    w = torch.randn(t_len, batch, phi_dim)
    w = F.normalize(w, p=2, dim=-1)
    return (obs, w)


MENAGERIE_ENTRIES = [
    ("VISR", "build_visr", "example_input_visr", 2020, "ported-pytorch"),
]

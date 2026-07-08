# SOURCE: vendored from https://github.com/Shaswat2001/Multi_Agent_Path_Finding @ main
# (marl_planner/marl_planner/network/vdn_net.py: VDNMixer, VDNCritic, lines 1-38)
#
# VDN / Value Decomposition Networks (Sunehag et al. 2017, "Value-Decomposition Networks
# For Cooperative Multi-Agent Learning", arXiv:1706.05296). The original Oxford PYMARL
# implementation's `VDNMixer` (oxwhirl/pymarl, src/modules/mixers/vdn.py) is a single
# parameter-free `torch.sum` over stacked per-agent Q-values -- correct, but degenerate as
# a standalone traceable module (no learnable weights, one op). Shaswat2001's
# Multi_Agent_Path_Finding repo (a multi-agent RL planner library) pairs that same
# summation mixer with a real per-agent `VDNCritic` (dueling-style Q-network: shared MLP
# trunk -> separate Value and Advantage heads recombined as Q = V + (A - mean(A))), which
# is the actual per-agent network VDN's decomposition assumption is applied to in practice.
# Vendored verbatim from vdn_net.py; only the config-namespace access
# (`args.input_shape[agent]` / `args.n_actions[agent]` / `args.critic_hidden`) is
# unpacked into plain constructor args in the staging wrapper below.

import torch
from torch import nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- marl_planner/network/vdn_net.py (vendored verbatim, config-namespace unpacked) ----
class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, q_values):
        return sum(q_values)


class VDNCritic(nn.Module):
    def __init__(self, input_dim, n_action, critic_hidden):
        super(VDNCritic, self).__init__()

        self.input_dim = input_dim
        self.n_action = n_action

        self.criticNet = nn.Sequential(nn.Linear(self.input_dim, critic_hidden), nn.ReLU())

        self.VNet = nn.Sequential(
            nn.Linear(critic_hidden, critic_hidden), nn.ReLU(), nn.Linear(critic_hidden, 1)
        )
        self.AdvNet = nn.Sequential(
            nn.Linear(critic_hidden, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, self.n_action),
        )

    def forward(self, obs):
        out = self.criticNet(obs)

        V = self.VNet(out)
        Adv = self.AdvNet(out)
        Adv = Adv - Adv.mean(dim=-1, keepdim=True)
        Qval = Adv + V

        return Qval


# ---- end vendored vdn_net.py ----


class VDNNet(nn.Module):
    """Staging wrapper exercising the real per-agent construction VDN.reset() performs
    (`self.PolicyNetwork = {agent: VDNCritic(...) for agent in env_agents}`) plus the
    `VDNMixer` sum-decomposition used in VDN.learn() (`self.VDNMixer(torch.hstack(q_values))`)
    as a single traceable module: N per-agent VDNCritic dueling Q-networks whose scalar
    outputs (gathered/argmaxed per-agent in the real algorithm; here summed directly per
    VDNMixer's contract) get combined by the real VDNMixer sum."""

    def __init__(self, n_agents=3, obs_dim=8, n_action=4, critic_hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.critics = nn.ModuleList(
            [VDNCritic(obs_dim, n_action, critic_hidden) for _ in range(n_agents)]
        )
        self.mixer = VDNMixer()

    def forward(self, obs_n):
        # obs_n: (batch, n_agents, obs_dim) -- per-agent local observations
        q_values = [self.critics[i](obs_n[:, i, :]) for i in range(self.n_agents)]
        q_tot = self.mixer(q_values)
        return q_tot


def build_vdn():
    return VDNNet(n_agents=3, obs_dim=8, n_action=4, critic_hidden=64)


def example_input_vdn():
    torch.manual_seed(0)
    return (torch.randn(2, 3, 8),)


MENAGERIE_ENTRIES = [
    ("VDN", "build_vdn", "example_input_vdn", 2017, "vendored-pytorch"),
]

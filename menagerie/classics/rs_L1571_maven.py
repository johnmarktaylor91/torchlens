# SOURCE: vendored from https://github.com/AnujMahajanOxf/MAVEN @ master
# MAVEN (Multi-Agent Variational Exploration): the noise-conditioned RNN agent
# network from "MAVEN: Multi-Agent Variational Exploration" (Mahajan et al.,
# NeurIPS 2019). This is the per-agent Q-network that consumes a shared
# latent exploration noise `z` (produced by the hierarchical policy, not part
# of this per-agent forward pass) via a hypernetwork that maps
# `(noise, agent_id)` to a per-action weighting of the RNN's Q-values -- the
# architectural novelty over vanilla QMIX/VDN-style independent RNN agents.
#
# Vendored real repo code from maven_code/src/modules/agents/noise_rnn_agent.py
# verbatim (RNNAgent class). Only non-architectural portability fix applied:
#   - the original `args` parameter is a SimpleNamespace/argparse.Namespace
#     built from a sacred config; replaced with a plain SimpleNamespace
#     carrying the same fields (values only, no logic changed).
# No layer, head, or dataflow was changed from the real implementation.

from types import SimpleNamespace

import torch as th
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- maven_code/src/modules/agents/noise_rnn_agent.py (verbatim) ----
class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.noise_embedding_dim)
        self.noise_fc2 = nn.Linear(args.noise_embedding_dim, args.noise_embedding_dim)
        self.noise_fc3 = nn.Linear(args.noise_embedding_dim, args.n_actions)

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(
            args.noise_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, noise):
        agent_ids = th.eye(self.args.n_agents, device=inputs.device).repeat(noise.shape[0], 1)
        noise_repeated = noise.repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        noise_input = th.cat([noise_repeated, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_noise_fc1(noise_input).reshape(
                -1, self.args.n_actions, self.args.rnn_hidden_dim
            )
            wq = th.bmm(W, h.unsqueeze(2))
        else:
            z = F.tanh(self.noise_fc1(noise_input))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)

            wq = q * wz

        return wq, h


def build_maven_rnn_agent():
    args = SimpleNamespace(
        rnn_hidden_dim=16,
        n_actions=5,
        noise_dim=4,
        n_agents=3,
        noise_embedding_dim=8,
    )
    input_shape = 10  # flattened per-agent observation feature size
    return RNNAgent(input_shape, args)


def example_input_maven_rnn_agent():
    n_agents = 3
    batch = n_agents  # one row per agent, as the real training loop feeds it
    inputs = th.randn(batch, 10)
    hidden_state = th.zeros(batch, 16)
    noise = th.randn(1, 4)  # shared latent exploration noise, batch-of-1 (broadcast via repeat)
    return (inputs, hidden_state, noise)


MENAGERIE_ENTRIES = [
    (
        "MAVEN (Multi-Agent Variational Exploration)",
        build_maven_rnn_agent,
        example_input_maven_rnn_agent,
        2019,
        MENAGERIE_ZOO,
    ),
]

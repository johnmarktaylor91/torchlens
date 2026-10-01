# SOURCE: vendored from oxwhirl/wqmix @ master (files: src/modules/mixers/qmix.py,
# src/modules/mixers/qmix_central_no_hyper.py, src/modules/agents/rnn_agent.py,
# src/modules/agents/central_rnn_agent.py -- fetched 2026-07-02).
#
# This vendors the REAL WQMIX (Weighted QMIX) architecture: the per-agent recurrent Q-network
# (`RNNAgent`), the monotonic hypernetwork mixer (`QMixer`, plain QMIX's mixing network, which
# WQMIX reuses as its "greedy" branch), and the unrestricted central feed-forward mixer
# (`QMixerCentralFF`, from `qmix_central_no_hyper.py`) whose weighted regression against the
# monotonic mixer's output is exactly the OW/CW weighting scheme that gives WQMIX its name
# (Rashid et al., NeurIPS 2020, "Weighted QMIX: Expanding Monotonic Value Function
# Factorisation for Deep Multi-Agent Reinforcement Learning").
#
# Code below is the upstream source with only mechanical edits: cross-file imports flattened
# into this single module, the `args` namespace argument replaced by explicit constructor
# kwargs (upstream builds `args` from a Sacred config dict at training-script level, not part
# of the model architecture), everything else untouched.

import torch as th
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# src/modules/agents/rnn_agent.py
# ---------------------------------------------------------------------------
class RNNAgent(nn.Module):
    """Per-agent recurrent Q-network shared across agents (GRU-based DRQN)."""

    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNNAgent, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# ---------------------------------------------------------------------------
# src/modules/mixers/qmix.py
# ---------------------------------------------------------------------------
class QMixer(nn.Module):
    """Monotonic hypernetwork mixing network (plain QMIX; WQMIX's "greedy" branch)."""

    def __init__(self, n_agents, state_dim, mixing_embed_dim, hypernet_layers=1, hypernet_embed=64):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


# ---------------------------------------------------------------------------
# src/modules/mixers/qmix_central_no_hyper.py
# ---------------------------------------------------------------------------
class QMixerCentralFF(nn.Module):
    """Unrestricted central feed-forward mixer. WQMIX regresses this (unrestricted) joint-action
    value estimator against the monotonic QMixer output, weighted by the OW/CW schemes -- the
    architectural contribution the "Weighted" in WQMIX refers to."""

    def __init__(self, n_agents, state_dim, central_action_embed, central_mixing_embed_dim):
        super(QMixerCentralFF, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.central_action_embed = central_action_embed

        self.input_dim = self.n_agents * self.central_action_embed + self.state_dim
        self.embed_dim = central_mixing_embed_dim

        non_lin = nn.ReLU

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            non_lin(),
            nn.Linear(self.embed_dim, self.embed_dim),
            non_lin(),
            nn.Linear(self.embed_dim, self.embed_dim),
            non_lin(),
            nn.Linear(self.embed_dim, 1),
        )

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim), non_lin(), nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents * self.central_action_embed)

        inputs = th.cat([states, agent_qs], dim=1)

        advs = self.net(inputs)
        vs = self.V(states)

        y = advs + vs

        q_tot = y.view(bs, -1, 1)
        return q_tot


# ---------------------------------------------------------------------------
# Staging module: full WQMIX forward (agents -> monotonic mixer + central mixer)
# ---------------------------------------------------------------------------
class WQMIXTraceWrapper(nn.Module):
    """Wraps the real WQMIX components (RNNAgent x n_agents (shared weights, as upstream),
    QMixer monotonic mixer, QMixerCentralFF unrestricted central mixer) into a single
    forward pass so the whole real architecture is traceable end to end."""

    N_AGENTS = 3
    RNN_HIDDEN = 16
    N_ACTIONS = 4
    STATE_DIM = 20
    OBS_DIM = 8
    MIXING_EMBED = 12
    CENTRAL_ACTION_EMBED = 1
    CENTRAL_MIXING_EMBED = 16

    def __init__(self):
        super().__init__()
        self.agent = RNNAgent(
            input_shape=self.OBS_DIM, rnn_hidden_dim=self.RNN_HIDDEN, n_actions=self.N_ACTIONS
        )
        self.mixer = QMixer(
            n_agents=self.N_AGENTS, state_dim=self.STATE_DIM, mixing_embed_dim=self.MIXING_EMBED
        )
        self.central_mixer = QMixerCentralFF(
            n_agents=self.N_AGENTS,
            state_dim=self.STATE_DIM,
            central_action_embed=self.CENTRAL_ACTION_EMBED,
            central_mixing_embed_dim=self.CENTRAL_MIXING_EMBED,
        )

    def forward(self, obs, state):
        # obs: (n_agents, obs_dim); state: (1, state_dim)
        hidden = self.agent.init_hidden().expand(self.N_AGENTS, -1)
        q_vals, _hidden_out = self.agent(obs, hidden)  # (n_agents, n_actions)
        chosen_qs = q_vals.max(dim=-1).values.view(1, self.N_AGENTS)  # greedy action per agent

        q_tot_mono = self.mixer(chosen_qs, state)

        central_qs = chosen_qs.view(1, self.N_AGENTS, self.CENTRAL_ACTION_EMBED)
        q_tot_central = self.central_mixer(central_qs, state)

        return q_tot_mono, q_tot_central


def build_wqmix():
    return WQMIXTraceWrapper()


def example_input_wqmix():
    obs = th.zeros(WQMIXTraceWrapper.N_AGENTS, WQMIXTraceWrapper.OBS_DIM)
    state = th.zeros(1, WQMIXTraceWrapper.STATE_DIM)
    return (obs, state)


MENAGERIE_ENTRIES = [
    (
        "WQMIX (Weighted QMIX)",
        build_wqmix,
        example_input_wqmix,
        2020,
        MENAGERIE_ZOO,
    ),
]

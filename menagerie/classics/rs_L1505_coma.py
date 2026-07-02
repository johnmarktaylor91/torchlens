# SOURCE: vendored from https://github.com/oxwhirl/pymarl @ master
# (src/modules/agents/rnn_agent.py + src/modules/critics/coma.py, unmodified
# architecture; only added lightweight config/batch shims so the real modules
# can be constructed and called outside the full pymarl runner.)
"""COMA (Counterfactual Multi-Agent Policy Gradients, Foerster et al. 2018):
decentralized per-agent recurrent actor (RNNAgent) + centralized critic
(COMACritic) as shipped in the reference pymarl implementation."""

from types import SimpleNamespace

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class COMACritic(nn.Module):
    def __init__(self, scheme, args):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = (
            batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        )
        agent_mask = 1 - th.eye(self.n_agents, device=batch.device)
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if t == 0:
            inputs.append(
                th.zeros_like(batch["actions_onehot"][:, 0:1])
                .view(bs, max_t, 1, -1)
                .repeat(1, 1, self.n_agents, 1)
            )
        elif isinstance(t, int):
            inputs.append(
                batch["actions_onehot"][:, slice(t - 1, t)]
                .view(bs, max_t, 1, -1)
                .repeat(1, 1, self.n_agents, 1)
            )
        else:
            last_actions = th.cat(
                [th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                dim=1,
            )
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(
            th.eye(self.n_agents, device=batch.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, max_t, -1, -1)
        )

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape


# ---- staging scaffolding (config/batch shims; no architecture changes) ----

_N_AGENTS = 3
_N_ACTIONS = 5
_OBS_DIM = 8
_STATE_DIM = 12
_RNN_HIDDEN = 16


class _FakeBatch:
    """Minimal stand-in for pymarl's EpisodeBatch, exposing exactly the
    fields/attrs COMACritic._build_inputs reads from a real batch."""

    def __init__(self, bs=2, seq_len=1):
        self.batch_size = bs
        self.max_seq_length = seq_len
        self.device = th.device("cpu")
        self.data = {
            "state": th.randn(bs, seq_len, _STATE_DIM),
            "obs": th.randn(bs, seq_len, _N_AGENTS, _OBS_DIM),
            "actions_onehot": F.one_hot(
                th.randint(0, _N_ACTIONS, (bs, seq_len, _N_AGENTS)), num_classes=_N_ACTIONS
            ).float(),
        }

    def __getitem__(self, key):
        return self.data[key]


def build_coma_actor():
    args = SimpleNamespace(rnn_hidden_dim=_RNN_HIDDEN, n_actions=_N_ACTIONS)
    return RNNAgent(input_shape=_OBS_DIM, args=args)


def example_input_coma_actor():
    obs = th.randn(4, _OBS_DIM)
    hidden = th.zeros(4, _RNN_HIDDEN)
    return (obs, hidden)


def build_coma_critic():
    scheme = {
        "state": {"vshape": _STATE_DIM},
        "obs": {"vshape": _OBS_DIM},
        "actions_onehot": {"vshape": (_N_ACTIONS,)},
    }
    args = SimpleNamespace(n_actions=_N_ACTIONS, n_agents=_N_AGENTS)
    return COMACritic(scheme=scheme, args=args)


def example_input_coma_critic():
    return (_FakeBatch(),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("COMA Actor (RNNAgent)", build_coma_actor, example_input_coma_actor, 2018, MENAGERIE_ZOO),
    ("COMA Critic", build_coma_critic, example_input_coma_critic, 2018, MENAGERIE_ZOO),
]

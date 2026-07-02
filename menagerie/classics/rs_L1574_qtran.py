# SOURCE: vendored from starry-sky6688/MARL-Algorithms @ master
#
# Files vendored (paths in the original repo):
#   network/base_net.py -> RNN (per-agent recurrent Q-network)
#   network/qtran_net.py -> QtranQBase, QtranV (QTRAN-base joint-action-value and state-value nets)
#
# QTRAN ("Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement
# Learning", Son et al., ICML 2019) factorizes the joint action-value function via a joint network
# (QtranQBase) and a state-value network (QtranV), built on top of a per-agent GRU recurrent
# Q-network (RNN). All code below is the REAL model definitions from the repo (only the relative
# `from network.base_net import RNN` / `from network.qtran_net import ...` imports were flattened
# into this single file, and the original Chinese-language inline comments are preserved verbatim
# since they document the exact tensor semantics of the authors' implementation). Only base
# packages (torch) are required -- no vendored-only third-party deps.

import torch
import torch.nn as nn
import torch.nn.functional as f
from types import SimpleNamespace


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# Joint action-value network, 输入state,所有agent的hidden_state，所有agent的动作，输出对应的联合Q值
class QtranQBase(nn.Module):
    def __init__(self, args):
        super(QtranQBase, self).__init__()
        self.args = args
        # action_encoding对输入的每个agent的hidden_state和动作进行编码，从而将所有agents的hidden_state和动作相加得到近似的联合hidden_state和动作
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(
            nn.Linear(ae_input, ae_input), nn.ReLU(), nn.Linear(ae_input, ae_input)
        )

        # 编码求和之后输入state、所有agent的hidden_state和动作之和
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
        self.q = nn.Sequential(
            nn.Linear(q_input, self.args.qtran_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.qtran_hidden_dim, 1),
        )

    # 因为所有时刻所有agent的hidden_states在之前已经计算好了，所以联合Q值可以一次计算所有transition的，不需要一条一条计算。
    def forward(
        self, state, hidden_states, actions
    ):  # (episode_num, max_episode_len, n_agents, n_actions)
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(
            episode_num * max_episode_len, n_agents, -1
        )  # 变回n_agents维度用于求和
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)

        inputs = torch.cat(
            [state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1
        )
        q = self.q(inputs)
        return q


# 输入当前的state与所有agent的hidden_state, 输出V值
class QtranV(nn.Module):
    def __init__(self, args):
        super(QtranV, self).__init__()
        self.args = args

        # hidden_encoding对输入的每个agent的hidden_state编码，从而将所有agents的hidden_state相加得到近似的联合hidden_state
        hidden_input = self.args.rnn_hidden_dim
        self.hidden_encoding = nn.Sequential(
            nn.Linear(hidden_input, hidden_input), nn.ReLU(), nn.Linear(hidden_input, hidden_input)
        )

        # 编码求和之后输入state、所有agent的hidden_state之和
        v_input = self.args.state_shape + self.args.rnn_hidden_dim
        self.v = nn.Sequential(
            nn.Linear(v_input, self.args.qtran_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.qtran_hidden_dim, 1),
        )

    def forward(self, state, hidden):
        episode_num, max_episode_len, n_agents, _ = hidden.shape
        state = state.reshape(episode_num * max_episode_len, -1)
        hidden_encoding = self.hidden_encoding(hidden.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num * max_episode_len, n_agents, -1).sum(
            dim=-2
        )
        inputs = torch.cat([state, hidden_encoding], dim=-1)
        v = self.v(inputs)
        return v


# --- Menagerie staging: QtranQBase is the headline QTRAN joint-action-value network (the paper's
# novel factorization contribution); state/hidden_states/actions are concrete random tensors sized
# for a tiny episode_num=2, max_episode_len=3, n_agents=3 setup.

MENAGERIE_ZOO = "vendored-pytorch"


def _qtran_args():
    return SimpleNamespace(
        n_actions=4,
        n_agents=3,
        rnn_hidden_dim=16,
        state_shape=20,
        qtran_hidden_dim=32,
    )


def build_qtran_joint_q():
    return QtranQBase(_qtran_args())


def example_input_qtran_joint_q():
    args = _qtran_args()
    episode_num, max_episode_len, n_agents = 2, 3, args.n_agents
    state = torch.randn(episode_num, max_episode_len, args.state_shape)
    hidden_states = torch.randn(episode_num, max_episode_len, n_agents, args.rnn_hidden_dim)
    actions = torch.randn(episode_num, max_episode_len, n_agents, args.n_actions)
    return state, hidden_states, actions


def build_qtran_rnn_agent():
    args = SimpleNamespace(rnn_hidden_dim=16, n_actions=4)
    return RNN(input_shape=15, args=args)


def example_input_qtran_rnn_agent():
    obs = torch.randn(2, 15)
    hidden = torch.zeros(2, 16)
    return obs, hidden


MENAGERIE_ENTRIES = [
    (
        "qtran_joint_q_network",
        build_qtran_joint_q,
        example_input_qtran_joint_q,
        2019,
        MENAGERIE_ZOO,
    ),
    ("qtran_rnn_agent", build_qtran_rnn_agent, example_input_qtran_rnn_agent, 2019, MENAGERIE_ZOO),
]

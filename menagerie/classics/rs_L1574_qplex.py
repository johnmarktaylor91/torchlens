# SOURCE: vendored from wjh720/QPLEX @ master
#
# Files vendored (paths in the original repo, under pymarl-master/src/modules/):
#   agents/rnn_agent.py       -> RNNAgent
#   mixers/dmaq_si_weight.py  -> DMAQ_SI_Weight
#   mixers/dmaq_qatten_weight.py -> Qatten_Weight
#   mixers/dmaq_qatten.py     -> DMAQ_QattenMixer
#   mixers/dmaq_general.py    -> DMAQer
#
# QPLEX ("Duplex Dueling Multi-Agent Q-Learning", Wang et al., ICLR 2021) implements the duplex
# dueling mixing network as `DMAQer`/`DMAQ_QattenMixer` (advantage-value decomposition with a
# multi-head attention hypernetwork, `Qatten_Weight` + `DMAQ_SI_Weight`), stacked on top of a
# per-agent GRU recurrent Q-network (`RNNAgent`). All code below is the REAL model definitions
# from the repo (only imports/relative-module-paths and a `SimpleNamespace`-based `args` staging
# shim were adjusted; the architecture is byte-for-byte the original nn.Module classes). Only
# base packages (torch, numpy) are required -- no vendored-only third-party deps.

from types import SimpleNamespace

import numpy as np
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


class DMAQ_SI_Weight(nn.Module):
    def __init__(self, args):
        super(DMAQ_SI_Weight, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.num_kernel = args.num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        adv_hypernet_embed = self.args.adv_hypernet_embed
        for i in range(self.num_kernel):  # multi-head attention
            if getattr(args, "adv_hypernet_layers", 1) == 1:
                self.key_extractors.append(nn.Linear(self.state_dim, 1))  # key
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  # agent
                self.action_extractors.append(
                    nn.Linear(self.state_action_dim, self.n_agents)
                )  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 2:
                self.key_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, 1),
                    )
                )  # key
                self.agents_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, self.n_agents),
                    )
                )  # agent
                self.action_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_action_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, self.n_agents),
                    )
                )  # action
            elif getattr(args, "adv_hypernet_layers", 1) == 3:
                self.key_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, 1),
                    )
                )  # key
                self.agents_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, self.n_agents),
                    )
                )  # agent
                self.action_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_action_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, self.n_agents),
                    )
                )  # action
            else:
                raise Exception("Error setting number of adv hypernet layers.")

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = th.cat([states, actions], dim=1)

        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(
            all_head_key, all_head_agents, all_head_action
        ):
            x_key = th.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = th.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = th.sum(head_attend, dim=1)

        return head_attend


class Qatten_Weight(nn.Module):
    def __init__(self, args):
        super(Qatten_Weight, self).__init__()

        self.name = "qatten_weight"
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.n_actions = args.n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.n_head  # attention head num

        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):  # multi-head attention
            selector_nn = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim, bias=False),
            )
            self.selector_extractors.append(selector_nn)  # query
            if self.args.nonlinear:  # add qs
                self.key_extractors.append(
                    nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False)
                )  # key
            else:
                self.key_extractors.append(
                    nn.Linear(self.unit_dim, self.embed_dim, bias=False)
                )  # key
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.n_head),
            )

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states, actions):
        states = states.reshape(-1, self.state_dim)
        unit_states = states[
            :, : self.unit_dim * self.n_agents
        ]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(
                curr_head_selector.view(-1, 1, self.embed_dim),
                th.stack(curr_head_keys).permute(1, 2, 0),
            )
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            if self.args.mask_dead:
                # actions: (episode_batch, episode_length - 1, agent_num, 1)
                actions = actions.reshape(-1, 1, self.n_agents)
                # actions: (batch_size, 1, agent_num)
                scaled_attend_logits[actions == 0] = -99999999  # action == 0 means the unit is dead
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = th.stack(
            head_attend_weights, dim=1
        )  # (batch_size, self.n_head, self.n_agents)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        v = self.V(states).view(-1, 1)  # v: (bs, 1)
        # head_qs: [head_num, bs, 1]
        if self.args.weighted_head:
            w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
            w_head = w_head.view(-1, self.n_head, 1).repeat(
                1, 1, self.n_agents
            )  # w_head: (bs, head_num, self.n_agents)
            head_attend *= w_head

        head_attend = th.sum(head_attend, dim=1)

        if not self.args.state_bias:
            v *= 0.0

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum(
            (logit**2).mean() for logit in head_attend_logits
        )
        head_entropies = [
            (-((probs + 1e-8).log() * probs).squeeze().sum(1).mean())
            for probs in head_attend_weights
        ]

        return head_attend, v, attend_mag_regs, head_entropies


class DMAQ_QattenMixer(nn.Module):
    def __init__(self, args):
        super(DMAQ_QattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)
        self.si_weight = DMAQ_SI_Weight(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.0), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(
            agent_qs, states, actions
        )
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies


class DMAQer(nn.Module):
    def __init__(self, args):
        super(DMAQer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.embed_dim = args.mixing_embed_dim

        hypernet_embed = self.args.hypernet_embed
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents),
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents),
        )

        self.si_weight = DMAQ_SI_Weight(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.0), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        w_final = self.hyper_w_final(states)
        w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = self.V(states)
        v = v.view(-1, self.n_agents)

        if self.args.weighted_head:
            agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            if self.args.weighted_head:
                max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot


# --- Menagerie staging: DMAQer mixer (the QPLEX "duplex dueling" mixing network) is the novel
# contribution being catalogued; agent_qs/states/actions/max_q_i are provided as concrete random
# tensors sized for a tiny n_agents=3, n_actions=4 StarCraft-style SMAC-like setup.

MENAGERIE_ZOO = "vendored-pytorch"


def _qplex_args():
    return SimpleNamespace(
        n_agents=3,
        n_actions=4,
        state_shape=(24,),
        mixing_embed_dim=16,
        hypernet_embed=32,
        adv_hypernet_embed=32,
        adv_hypernet_layers=2,
        num_kernel=2,
        is_minus_one=True,
        weighted_head=True,
    )


class QPLEXMixerWrapper(nn.Module):
    """Thin wrapper fixing is_v=False, actions/max_q_i supplied, matching the repo's
    non-V-branch forward call used during TD-error computation."""

    def __init__(self, args):
        super().__init__()
        self.mixer = DMAQer(args)

    def forward(self, agent_qs, states, actions, max_q_i):
        return self.mixer(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=False)


def build_qplex_dmaq_mixer():
    return QPLEXMixerWrapper(_qplex_args())


def example_input_qplex_dmaq_mixer():
    args = _qplex_args()
    bs = 2
    agent_qs = th.randn(bs, args.n_agents)
    states = th.randn(bs, *args.state_shape)
    actions = th.randn(bs, args.n_agents * args.n_actions)
    max_q_i = th.randn(bs, args.n_agents)
    return agent_qs, states, actions, max_q_i


def build_qplex_rnn_agent():
    args = SimpleNamespace(rnn_hidden_dim=32, n_actions=4)
    return RNNAgent(input_shape=20, args=args)


def example_input_qplex_rnn_agent():
    inputs = th.randn(2, 20)
    hidden = th.zeros(2, 32)
    return inputs, hidden


MENAGERIE_ENTRIES = [
    (
        "qplex_dmaq_mixer",
        build_qplex_dmaq_mixer,
        example_input_qplex_dmaq_mixer,
        2021,
        MENAGERIE_ZOO,
    ),
    ("qplex_rnn_agent", build_qplex_rnn_agent, example_input_qplex_rnn_agent, 2021, MENAGERIE_ZOO),
]

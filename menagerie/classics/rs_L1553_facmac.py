# SOURCE: vendored from oxwhirl/facmac @ main
# https://github.com/oxwhirl/facmac
# Files: src/modules/agents/mlp_agent.py (MLPAgent), src/modules/critics/facmac.py
# (FACMACCritic)
#
# FACMAC (Factored Multi-Agent Centralised policy gradients, Peng et al. 2021)
# is a multi-agent actor-critic method built on top of the PyMARL codebase.
# Its architectural contribution is the pair of small per-agent networks that
# the FACMAC learner (src/learners/facmac_learner.py) trains: an MLP actor
# ("MLPAgent", mapping local per-agent observations to continuous actions via
# a 3-layer MLP with tanh-squashed outputs) and a centralised MLP critic
# ("FACMACCritic", mapping the concatenation of an agent's observation and
# action to a scalar Q-value via a 3-layer MLP). This module vendors both
# real network classes verbatim (only the `args`/`scheme` plumbing was
# replaced with a plain namespace + dict literal at construction time; no
# architecture was altered) and composes them into a single forward-passable
# module that runs the real actor forward pass followed by the real critic
# forward pass on the resulting action, mirroring how facmac_learner.py calls
# `self.mac.forward(...)` and then `self.critic(inputs, actions)`.

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class MLPAgent(nn.Module):
    """Per-agent MLP actor. (src/modules/agents/mlp_agent.py MLPAgent)"""

    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        if self.agent_return_logits:
            actions = self.fc3(x)
        else:
            actions = torch.tanh(self.fc3(x))
        return {"actions": actions, "hidden_state": hidden_state}


class FACMACCritic(nn.Module):
    """Centralised MLP critic. (src/modules/critics/facmac.py FACMACCritic)"""

    def __init__(self, scheme, args):
        super(FACMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = torch.cat(
                [
                    inputs.view(-1, self.input_shape - self.n_actions),
                    actions.contiguous().view(-1, self.n_actions),
                ],
                dim=-1,
            )
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


class FACMACActorCritic(nn.Module):
    """Composition of the real MLPAgent actor with the real FACMACCritic,
    matching how facmac_learner.py chains the actor's action output into the
    critic's forward pass (`self.critic(agent_outs, actions)`)."""

    def __init__(self, obs_dim, n_actions, rnn_hidden_dim):
        super().__init__()
        args = SimpleNamespace(
            rnn_hidden_dim=rnn_hidden_dim,
            n_actions=n_actions,
            n_agents=1,
            agent_return_logits=False,
        )
        scheme = {"obs": {"vshape": obs_dim}}
        self.actor = MLPAgent(obs_dim, args)
        self.critic = FACMACCritic(scheme, args)

    def forward(self, obs):
        hidden = self.actor.init_hidden()
        actor_out = self.actor(obs, hidden)
        actions = actor_out["actions"]
        q, _ = self.critic(obs, actions)
        return q


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


def build_facmac():
    torch.manual_seed(0)
    model = FACMACActorCritic(obs_dim=18, n_actions=4, rnn_hidden_dim=64)
    model.eval()
    return model


def example_input_facmac():
    torch.manual_seed(0)
    return torch.randn(1, 18)


MENAGERIE_ENTRIES = [
    (
        "FACMAC (Factored Multi-Agent Centralised policy gradients, MLP actor-critic)",
        "build_facmac",
        "example_input_facmac",
        2021,
        MENAGERIE_ZOO,
    ),
]

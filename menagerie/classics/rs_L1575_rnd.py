# SOURCE: vendored from facebookresearch/impact-driven-exploration @ 877c4ea530cc0ca3902211dba4e922bf8c3ce276
# File: src/models.py + src/algos/rnd.py (net-construction wiring in main.py)
# Random Network Distillation (RND) agent, Burda et al. 2018 (arxiv 1810.12894).
# This repo's RND baseline pairs an IMPALA-style LSTM actor-critic policy
# (`MinigridPolicyNet`) with a fixed random target network and a trained
# predictor network that are both instances of `MinigridStateEmbeddingNet`
# (see main.py: `random_target_network = MinigridStateEmbeddingNet(...)`,
# `predictor_network = MinigridStateEmbeddingNet(...)`). We vendor both real
# nn.Module classes; MENAGERIE_ENTRIES exposes the policy net and the
# state-embedding net (shared arch for target/predictor) separately.

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MinigridPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(MinigridPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        init_ = lambda m: init(  # noqa: E731
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.feat_extract = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=self.observation_shape[2],
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            init_(nn.Linear(32, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.core = nn.LSTM(1024, 1024, 2)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))  # noqa: E731

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))

    def initial_state(self, batch_size):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["partial_obs"]
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float()  # / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        core_input = self.fc(x)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        if core_state == ():
            core_state = self.initial_state(B)
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


class MinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(MinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        init_ = lambda m: init(  # noqa: E731
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.feat_extract = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=self.observation_shape[2],
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
            init_(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=128,
                    kernel_size=(3, 3),
                    stride=2,
                    padding=1,
                )
            ),
            nn.ELU(),
        )

    def forward(self, inputs):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)

        return state_embedding


_OBS_SHAPE = (7, 7, 3)  # MiniGrid partial-obs H x W x C
_NUM_ACTIONS = 7


def build_rnd_policy_net():
    return MinigridPolicyNet(_OBS_SHAPE, _NUM_ACTIONS)


def example_input_rnd_policy_net():
    T, B = 2, 1
    return {
        "partial_obs": torch.randint(0, 10, (T, B, *_OBS_SHAPE)),
        "done": torch.zeros(T, B, dtype=torch.bool),
    }


def build_rnd_state_embedding_net():
    return MinigridStateEmbeddingNet(_OBS_SHAPE)


def example_input_rnd_state_embedding_net():
    T, B = 2, 1
    return torch.randint(0, 255, (T, B, *_OBS_SHAPE)).float()


MENAGERIE_ENTRIES = [
    (
        "RND_MinigridPolicyNet",
        "build_rnd_policy_net",
        "example_input_rnd_policy_net",
        2018,
        "vendored-pytorch",
    ),
    (
        "RND_StateEmbeddingNet",
        "build_rnd_state_embedding_net",
        "example_input_rnd_state_embedding_net",
        2018,
        "vendored-pytorch",
    ),
]

# SOURCE: vendored from https://github.com/kentsommer/pytorch-value-iteration-networks @ master
# (model.py: VIN, lines 1-61)
#
# Value Iteration Networks (Tamar et al. 2016, NeurIPS, "Value Iteration Networks",
# arXiv:1602.02867). The paper's own reference code (avivt/VIN) is Theano.
# kentsommer/pytorch-value-iteration-networks is the well-known community PyTorch port
# (also cross-referenced from awaelchli/pytorch-vin), widely cited as the canonical
# PyTorch VIN reimplementation. Architecture: a convolutional "reward" trunk (h -> r)
# feeds a differentiable value-iteration module built from repeated application of a
# shared-weight `q` convolution (transition-model conv over stacked reward+value maps)
# followed by a max-pool-over-actions to get the value map -- K iterations of this
# recurrent conv loop approximate K steps of the Bellman value-iteration update, which is
# VIN's defining architectural contribution (a differentiable planning module baked
# directly into a CNN's forward pass). The final Q-values are gathered at the agent's
# current (x, y) state and passed through a small FC action head. Vendored verbatim from
# model.py; only the config dataclass (constructor args unpacked directly instead).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---- model.py (VIN, vendored verbatim) ----
class VINConfig:
    """Stand-in for the argparse Namespace `config` object the real VIN(config)
    constructor expects (config.l_i / l_h / l_q), using the repo's own train.py
    defaults (--l_i 2 --l_h 150 --l_q 10), shrunk to a tiny 8x8 gridworld map."""

    def __init__(self, l_i=2, l_h=150, l_q=10):
        self.l_i = l_i
        self.l_h = l_h
        self.l_q = l_q


class VIN(nn.Module):
    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
        self.h = nn.Conv2d(
            in_channels=config.l_i,
            out_channels=config.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True,
        )
        self.r = nn.Conv2d(
            in_channels=config.l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=config.l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
        )
        self.fc = nn.Linear(in_features=config.l_q, out_features=8, bias=False)
        self.w = Parameter(torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input_view, state_x, state_y, k):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and softmaxed logits
        """
        h = self.h(input_view)  # Intermediate output
        r = self.r(h)  # Reward
        q = self.q(r)  # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        def eval_q(r, v):
            return F.conv2d(
                # Stack reward with most recent value
                torch.cat([r, v], 1),
                # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1,
            )

        # Update q and v values
        for i in range(k - 1):
            q = eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = eval_q(r, v)
        # q: (batch_sz, l_q, map_size, map_size)
        batch_sz, l_q, _, _ = q.size()
        q_out = q[torch.arange(batch_sz), :, state_x.long(), state_y.long()].view(batch_sz, l_q)

        logits = self.fc(q_out)  # q_out to actions

        return logits, self.sm(logits)


# ---- end vendored model.py ----


def build_vin():
    config = VINConfig(l_i=2, l_h=150, l_q=10)
    return VIN(config)


def example_input_vin():
    torch.manual_seed(0)
    imsize = 8
    batch_sz = 2
    # input_view: 2-channel (obstacle map + goal map) gridworld image, as produced by
    # dataset/dataset.py's GridworldData for the l_i=2 default config.
    input_view = torch.rand(batch_sz, 2, imsize, imsize)
    state_x = torch.randint(0, imsize, (batch_sz,))
    state_y = torch.randint(0, imsize, (batch_sz,))
    k = 10  # train.py default: --k 10 value-iteration steps
    return (input_view, state_x, state_y, k)


MENAGERIE_ENTRIES = [
    ("VIN", "build_vin", "example_input_vin", 2016, "vendored-pytorch"),
]

"""Model source text used by the behavioural sequences.

The source sequence writes its model to a file of its own, so the file can
be edited and deleted after capture. The argument sequence compiles its model
from text so its call sites do not move when this package is edited.
"""

from __future__ import annotations

#: The model of the source sequence, written to a scratch file.
SOURCE_MODEL_TEXT = '''"""Scratch model of the capture-core source sequence."""

import torch
from torch import nn


class SourceModel(nn.Module):
    """One linear layer and two functions."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        hidden = self.fc(x)
        hidden = torch.relu(hidden)
        return hidden + 1
'''

#: The model of the argument sequence: it mutates what it passed, after each call returns.
ARG_MODEL_SOURCE = '''
class DictIn(nn.Module):
    """A module whose forward takes a dictionary."""

    def forward(self, bundle):
        return bundle["a"] * 2


class ArgModel(nn.Module):
    """Passes a list, a dictionary and a tensor, then mutates each after the call."""

    def __init__(self):
        super().__init__()
        self.sub = DictIn()
        self.held_list = []
        self.held_dict = {}

    def forward(self, x):
        parts = [x, x + 1]
        joined = torch.cat(parts, dim=0)
        parts.append(x * 3)
        parts[0] = None
        bundle = {"a": joined}
        doubled = self.sub(bundle)
        bundle["a"] = None
        bundle["b"] = 1
        weights = torch.ones(4)
        scaled = doubled * weights
        weights.zero_()
        self.held_list = [scaled, scaled + 1]
        stacked = torch.stack(self.held_list)
        self.held_dict = bundle
        return stacked.sum(dim=0)
'''

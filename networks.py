from torch import nn
import torch
from typing import Dict

torch.TEST_ATTRIBUTE = 'networks_script'


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.linear = nn.Linear(4, 8)
        self.relu_module = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 2, 1)
        self.module_block = nn.Sequential(self.conv2, nn.ReLU(), nn.ReLU(), nn.ReLU())

    def forward(self, x):
        if x < 1:
            x = x + 1
        else:
            x = x * 2
        tensor_internal = torch.ones(2, 1, 2, 2)
        tensor_internal = tensor_internal + 1
        x = x + tensor_internal
        x = nn.functional.relu(x)
        x = self.conv1(x)
        x = x.flatten()
        x = self.linear(x)
        x = x.reshape(2, 1, 2, 2)
        x = self.conv1(x)
        x = self.relu_module(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.module_block(x)
        return x

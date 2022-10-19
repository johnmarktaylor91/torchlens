import os
import torch
from torch import nn
from torch_func_handling import orig_torch_funcs, mutate_pytorch, ignored_funcs, \
    overridable_funcs, \
    mark_tensors_in_obj
import numpy as np
from xray_utils import barcode_tensors_in_obj, pprint_tensor_record


# Now let's actually make a dummy network and make it work. Give it all the challenge points:

# 1. If-then branching
# 2. Recurrence
# 3. Both functions and modules
# 4. Internally generated tensors.

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 2, 2)
        self.linear = nn.Linear(4, 8)
        self.relu_module = nn.ReLU()

    def forward(self, x):
        if x < 1:
            x = x + 1
        else:
            x = x * 2
        tensor_internal = torch.ones(2, 1, 2, 2)
        tensor_internal = tensor_internal + 1
        x = x + tensor_internal
        x = nn.functional.relu(x)
        x = self.conv(x)
        x = x.flatten()
        x = self.linear(x)
        x = x.reshape(2, 1, 2, 2)
        x = self.conv(x)
        x = self.relu_module(x)
        return (x)


model = SimpleNetwork()
tensor_record = {'barcode_dict': {}}
orig_func_defs = []
x = torch.Tensor([.5])
x.requires_grad = True
mutate_pytorch(torch, [x], orig_func_defs, tensor_record)
mark_tensors_in_obj(x, 'xray_origin', 'input')
barcode_tensors_in_obj(x)
out = model(x)

pprint_tensor_record(tensor_record)

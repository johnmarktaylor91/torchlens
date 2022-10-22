import os
import torch
from torch import nn
from torch_func_handling import orig_torch_funcs, mutate_pytorch, ignored_funcs, \
    overridable_funcs, \
    mark_tensors_in_obj
import numpy as np
from xray_utils import barcode_tensors_in_obj, pprint_tensor_record
from networks import SimpleNetwork
import pytorch_xray as ptx

# Now let's actually make a dummy network and make it work. Give it all the challenge points:

# 1. If-then branching
# 2. Recurrence
# 3. Both functions and modules
# 4. Internally generated tensors.


model = SimpleNetwork()
hook_handles = []
hook_handles = ptx.prepare_model(model, hook_handles)
tensor_record = {'barcode_dict': {}}
orig_func_defs = []
x = torch.Tensor([.5])
x.requires_grad = True
mutate_pytorch(torch, [x], orig_func_defs, tensor_record)
mark_tensors_in_obj(x, 'xray_origin', 'input')
barcode_tensors_in_obj(x)
out = model(x)

pprint_tensor_record(tensor_record)

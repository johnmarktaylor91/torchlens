import os
import torch
from torch_func_handling import orig_torch_funcs, mutate_pytorch, ignored_funcs, \
    overridable_funcs, \
    mark_tensors_in_obj
import numpy as np
from xray_utils import barcode_tensors_in_obj, pprint_tensor_record

# Fix the print nonsense

tensor_record = {'barcode_dict': {}}
orig_func_defs = []
t = torch.ones(4, 3, requires_grad=True)
mutate_pytorch(torch, [t], orig_func_defs, tensor_record)
mark_tensors_in_obj(t, 'xray_origin', 'input')
barcode_tensors_in_obj(t)
relu1 = torch.nn.ReLU()
linear1 = torch.nn.Linear(3, 8)
print(t)
t = relu1(t)
t = t + 1
t = t[0, :]
x = torch.ones(1, requires_grad=True)
x = x + 1
t = t * x
t = t.flatten()
t = linear1(t)
pprint_tensor_record(tensor_record)
# unmutate_pytorch(torch, orig_func_defs)

torch.Tensor.__init__

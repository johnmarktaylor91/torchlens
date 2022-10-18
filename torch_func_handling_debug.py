import os
import torch
from torch_func_handling import orig_torch_funcs, mutate_pytorch, unmutate_pytorch, ignored_funcs, overridable_funcs
import numpy as np

# Fix the print nonsense

tensor_record = []
orig_func_defs = []
t = torch.ones(5, 5, requires_grad=True)
mutate_pytorch(torch, [t], orig_func_defs, tensor_record)
t = torch.from_numpy(np.ones((5, 5)))
t = t + 1
t = t + 1
t = t + 1
print(tensor_record)
unmutate_pytorch(torch, orig_func_defs)

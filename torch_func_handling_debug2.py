import os
import torch
from torch_func_handling import orig_torch_funcs, mutate_pytorch, unmutate_pytorch, ignored_funcs, \
    overridable_funcs
from tensor_tracking_funcs import initialize_history_dict
import numpy as np
from util_funcs import barcode_tensors_in_obj, pprint_tensor_record, get_tensor_memory_amount

# Fix the print nonsense

tensor_record = initialize_history_dict('all')
orig_func_defs = []
mutate_pytorch(torch, [], orig_func_defs, tensor_record)
t = torch.ones(5)
t.storage()
print(get_tensor_memory_amount(t))
unmutate_pytorch(torch, orig_func_defs)

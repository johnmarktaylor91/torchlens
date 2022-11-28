import torch

from tensor_tracking import initialize_history_dict, mutate_pytorch, unmutate_pytorch
from util_funcs import get_tensor_memory_amount

# Fix the print nonsense

tensor_record = initialize_history_dict('all')
orig_func_defs = []
mutate_pytorch(torch, [], orig_func_defs, tensor_record)
t = torch.ones(5)
t.storage()
print(get_tensor_memory_amount(t))
unmutate_pytorch(torch, orig_func_defs)

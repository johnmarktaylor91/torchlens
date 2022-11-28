import torch

from model_funcs import run_model_and_save_specified_activations
from networks import SimpleNetwork
from util_funcs import pprint_tensor_record

# Now let's actually make a dummy network and make it work. Give it all the challenge points:

# 1. If-then branching
# 2. Recurrence
# 3. Both functions and modules
# 4. Internally generated tensors.


model = SimpleNetwork()
x = torch.Tensor([.5])
x.requires_grad = True
tensor_record = run_model_and_save_specified_activations(model, x, 'exhaustive', 'all', None)

pprint_tensor_record(tensor_record)

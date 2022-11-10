import os
import torch
from torch import nn
import torchvision

import model_funcs
import tensor_tracking
from tensor_tracking import ignored_funcs, mutate_pytorch, orig_torch_funcs, overridable_funcs
import numpy as np
from util_funcs import barcode_tensors_in_obj, pprint_tensor_record, mark_tensors_in_obj
from networks import SimpleNetwork
import pytorch_xray as ptx
from model_funcs import run_model_and_save_specified_activations

# Now let's actually make a dummy network and make it work. Give it all the challenge points:

# 1. If-then branching
# 2. Recurrence
# 3. Both functions and modules
# 4. Internally generated tensors.


model = torchvision.models.AlexNet()
model = torchvision.models.ResNet()
x = torch.rand(6, 3, 256, 256)
tensor_record = run_model_and_save_specified_activations(model, x, 'exhaustive', 'all', None)

pprint_tensor_record(tensor_record)

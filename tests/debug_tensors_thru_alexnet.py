import torch
import torchvision

from src.torchlens.model_funcs import run_model_and_save_specified_activations
from src.torchlens.helper_funcs import pprint_tensor_record

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

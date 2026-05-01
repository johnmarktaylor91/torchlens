# Migrating from ThingsVision

Functional migration pattern: a named-layer feature extraction call maps to a TorchLens capture and
activation lookup. TorchLens does not load model zoos or datasets; bring your own PyTorch model and
input tensor.

| ThingsVision construct | TorchLens equivalent |
| --- | --- |
| Extract features from a named layer. | Capture a local forward and read the saved layer activation. |

Their construct:

```python
# migration-test: tool=thingsvision expected=[[2.5, 2.5]]
import torch
from thingsvision import get_extractor


extractor = get_extractor(model_name="alexnet", source="torchvision")
features = extractor.extract_features(
    batches=[torch.zeros(1, 3, 224, 224)],
    module_name="features.0",
    flatten_acts=False,
)
RESULT = features[0, :1, :1, :2].reshape(1, 2).tolist()
```

TorchLens equivalent:

```python
# migration-test: tool=torchlens expected=[[2.5, 2.5]]
import torch
from torch import nn
import torchlens as tl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


log = tl.log_forward_pass(Tiny(), torch.tensor([[2.0, 3.0]]), vis_opt="none")
RESULT = log["linear_1_1"].activation.detach().tolist()
```

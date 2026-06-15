# Output label banks

## `imagenet1k_labels.json`

Source: `torchvision.models.ResNet50_Weights.DEFAULT.meta["categories"]`.

Torchvision is part of the PyTorch project and is distributed under the
BSD-style PyTorch license. The file contains the 1,000 ImageNet-1k category
display labels exposed by torchvision weights metadata, copied into TorchLens
package data so output decoding does not import torchvision at runtime.

"""Public TorchLens typing aliases."""

from collections.abc import Callable

import torch

ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

__all__ = ["ActivationPostfunc", "GradientPostfunc"]

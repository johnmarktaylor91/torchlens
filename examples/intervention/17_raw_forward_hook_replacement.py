"""Replace a module output with a raw PyTorch forward hook.

Run with:

``python examples/intervention/17_raw_forward_hook_replacement.py``
"""

from __future__ import annotations

import torch

import torchlens as tl


class HookedMlp(torch.nn.Module):
    """Tiny MLP with a raw-hook replacement site."""

    def __init__(self) -> None:
        """Create the MLP modules."""

        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.fc2(self.relu(self.fc1(x)))


def halve_relu(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    """Return a fresh replacement tensor from a raw PyTorch hook."""

    del module, args
    return output * 0.5


def main() -> None:
    """Capture a trace with a raw forward-hook replacement."""

    model = HookedMlp()
    x = torch.randn(3, 4)
    handle = model.relu.register_forward_hook(halve_relu)
    try:
        log = tl.trace(model, x)
    finally:
        handle.remove()

    replaced = [layer.layer_label for layer in log.layer_list if layer.intervention_replaced]
    print(f"replacement layers: {replaced}")


if __name__ == "__main__":
    main()

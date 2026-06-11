"""Trace a toy transformer block that builds its mask with ``torch.vmap``."""

import torch
from torch import nn

import torchlens as tl


class VmapMaskedToyTransformer(nn.Module):
    """Tiny attention-style block with a vmapped mask builder."""

    def __init__(self, width: int) -> None:
        """Initialize the toy projection stack.

        Parameters
        ----------
        width:
            Hidden width for the toy block.
        """

        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a masked attention-style computation.

        Parameters
        ----------
        x:
            Input tensor shaped ``(tokens, width)``.

        Returns
        -------
        torch.Tensor
            Masked attention output.
        """

        keep_tokens = torch.vmap(lambda row: row.mean() > 0)(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = q @ k.T
        masked_scores = scores.masked_fill(~keep_tokens[:, None], float("-inf"))
        return torch.softmax(masked_scores, dim=-1) @ v


def main() -> None:
    """Run the example trace and draw a graph artifact."""

    torch.manual_seed(0)
    model = VmapMaskedToyTransformer(width=4).eval()
    x = torch.randn(5, 4)
    trace = tl.trace(model, x, layers_to_save="all")

    print(trace.transforms)
    print(trace["vmap_1_1"].transform_config)
    trace.draw(vis_outpath="vmap_masked_toy_transformer", vis_save_only=True)


if __name__ == "__main__":
    main()

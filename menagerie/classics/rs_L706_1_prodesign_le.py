# SOURCE: vendored from bigict/ProDESIGN-LE @ f9a9ba1 (main)
# File: pe/model/modules.py
# ProDESIGN-LE: local-environment-aware transformer for protein sequence design.
# `design.py` (repo root) constructs the real trainable network as
# `modules.Transformer(46, num_class, 256, nhead=16, nlayer=3, device=...)` -- a plain
# torch.nn.TransformerEncoder stack over per-residue local-environment features, with a
# learned input projection and mean-pooled output head. Vendored verbatim (only the module
# docstring/comments are the original repo's; no architecture changes).
from torch import nn


class Transformer(nn.Module):
    """Transformer model for protein sequence design.

    This class implements a transformer encoder architecture that takes in protein structural
    features and outputs sequence predictions. The model consists of:
    - An input projection layer
    - A stack of transformer encoder layers
    - An output projection layer

    Args:
        d_input (int): Dimensionality of input features
        d_output (int): Dimensionality of output predictions
        d_model (int): Hidden dimension size of transformer (default: 256)
        nhead (int): Number of attention heads in multi-head attention (default: 4)
        nlayer (int): Number of transformer encoder layers (default: 3)
        **kw: Additional keyword arguments passed to layer initialization
    """

    def __init__(self, d_input, d_output, d_model=256, nhead=4, nlayer=3, **kw):
        super().__init__()

        # Create a single transformer encoder layer with specified dimensions
        # This layer includes self-attention and feed-forward components
        layer = nn.TransformerEncoderLayer(
            d_model,  # Hidden dimension size
            nhead,  # Number of attention heads
            batch_first=True,  # Input format: (batch, sequence, features)
            **kw,
        )

        # Input projection layer: maps input features to transformer dimension
        self.input = nn.Linear(d_input, d_model, **kw)

        # Main transformer encoder: stack of multiple identical layers
        self.main = nn.TransformerEncoder(layer, num_layers=nlayer)

        # Output projection layer: maps transformer outputs to final predictions
        self.output = nn.Linear(d_model, d_output, **kw)

    def forward(self, feat, mask):
        """Forward pass through the transformer model.

        Args:
            feat: Input features tensor of shape (batch_size, sequence_length, feature_dim)
            mask: Boolean mask tensor of shape (batch_size, sequence_length) indicating
                  which positions should be attended to (True = attend, False = ignore)

        Returns:
            Output predictions tensor of shape (batch_size, output_dim)
        """
        # Project input features to transformer dimension
        feat = self.input(feat)

        # Process features through transformer encoder
        # Note: ~mask inverts the mask since PyTorch expects True = ignore
        logits = self.main(feat, src_key_padding_mask=~mask)

        # Average over sequence dimension to get global representation
        logits = logits.mean(dim=1)

        # Project to final output dimension
        return self.output(logits)


MENAGERIE_ZOO = "vendored-pytorch"


def build_prodesign_le():
    import torch

    torch.manual_seed(0)
    # real constructor signature from design.py main(): Transformer(46, num_class, 256,
    # nhead=16, nlayer=3, device=...); shrunk to tiny d_model/nlayer for a fast trace.
    num_class = 21  # len(rc.restypes_with_x) in the real repo (20 aa + X)
    return Transformer(46, num_class, d_model=32, nhead=4, nlayer=2)


def example_input_prodesign_le():
    import torch

    torch.manual_seed(0)
    batch, seq_len, d_input = 1, 12, 46
    feat = torch.randn(batch, seq_len, d_input)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    return (feat, mask)


MENAGERIE_ENTRIES = [
    ("ProDESIGN-LE", "build_prodesign_le", "example_input_prodesign_le", 2023, "SOURCE_AVAILABLE"),
]

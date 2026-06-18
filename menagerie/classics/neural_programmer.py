"""Neural Programmer, 2016, Neelakantan et al., "Neural Programmer".

Paper: Neelakantan 2016, "Neural Programmer: Inducing Latent Programs with Gradient Descent."
An RNN controller softly selects table columns and differentiable operations, then
updates a row-selection vector through arithmetic and comparison-style operators.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuralProgrammer(nn.Module):
    """Tiny differentiable table programmer with soft operation selection."""

    def __init__(
        self, n_cols: int = 4, question_size: int = 5, hidden_size: int = 8, n_steps: int = 3
    ) -> None:
        """Initialize controller and operation heads.

        Parameters
        ----------
        n_cols
            Number of table columns.
        question_size
            Per-token question feature width.
        hidden_size
            Controller hidden size.
        n_steps
            Number of recurrent program steps.
        """
        super().__init__()
        self.n_steps = n_steps
        self.controller = nn.GRUCell(question_size, hidden_size)
        self.op_head = nn.Linear(hidden_size, 4)
        self.col_head = nn.Linear(hidden_size, n_cols)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, packed: Tensor) -> Tensor:
        """Execute soft table operations guided by a question sequence.

        Parameters
        ----------
        packed
            Tensor of shape ``(batch, rows, cols + question_size)`` containing the
            numeric table in the first columns and question tokens in the last columns.

        Returns
        -------
        Tensor
            Concatenated soft answer features ``(sum, count, selected mean)``.
        """
        table = packed[..., : self.col_head.out_features]
        question = packed[:, : self.n_steps, self.col_head.out_features :]
        batch, rows, _ = table.shape
        h = question.new_zeros(batch, self.controller.hidden_size)
        selection = table.new_full((batch, rows), 1.0 / rows)
        context = question.mean(dim=1)
        for _ in range(self.n_steps):
            h = self.controller(context, h)
            op = torch.softmax(self.op_head(h), dim=-1)
            col = torch.softmax(self.col_head(h), dim=-1)
            chosen = torch.sum(table * col[:, None, :], dim=-1)
            threshold = self.value_head(h)
            greater = torch.sigmoid(chosen - threshold)
            less = torch.sigmoid(threshold - chosen)
            keep = selection
            reset = table.new_full((batch, rows), 1.0 / rows)
            ops = torch.stack((keep, greater, less, reset), dim=-1)
            selection = torch.sum(ops * op[:, None, :], dim=-1)
            selection = selection / selection.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        selected = torch.sum(selection.unsqueeze(-1) * table, dim=1)
        count = selection.sum(dim=-1, keepdim=True)
        total = selected.sum(dim=-1, keepdim=True)
        return torch.cat((total, count, selected), dim=-1)


MENAGERIE_ENTRIES = [("Neural Programmer (Neelakantan)", "build", "example_input", "2016", "CD")]


def build() -> nn.Module:
    """Build a small neural programmer.

    Returns
    -------
    nn.Module
        Configured programmer module.
    """
    return NeuralProgrammer()


def example_input() -> Tensor:
    """Create packed table and question examples.

    Returns
    -------
    Tensor
        Packed table/question tensor with shape ``(2, 5, 9)``.
    """
    table = torch.randn(2, 5, 4)
    question = torch.randn(2, 5, 5)
    return torch.cat((table, question), dim=-1)

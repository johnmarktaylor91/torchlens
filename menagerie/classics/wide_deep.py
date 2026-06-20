"""Wide & Deep for tabular data (pytorch-widedeep ``WideDeep``).

Cheng et al., 2016, "Wide & Deep Learning for Recommender Systems"
(arXiv:1606.07792), as packaged by jrzaurin/pytorch-widedeep. The model jointly
trains a *wide* memorization path and a *deep* generalization path and ADDS their
logits:

  out = wide(x_wide) + deeptabular(x_tab)

Faithful structure (pytorch_widedeep.models):

  * **Wide** -- a linear model implemented as an ``nn.Embedding`` over the one-hot
    / crossed feature index space (``wide_linear``): each active feature index
    looks up a ``pred_dim``-wide weight vector, the lookups are summed over the
    feature axis, and a learned bias is added. This is exactly a sparse linear
    layer over (binary + crossed) features.

  * **TabMlp** (deeptabular) -- categorical features each get an ``nn.Embedding``;
    the embeddings are concatenated with the continuous features (here passed
    through a small per-feature continuous embedding + LayerNorm, the default
    "embed_continuous" path), and the concatenation is run through an MLP
    (Linear -> activation -> dropout blocks) ending in a ``pred_dim`` output.

This reimplementation builds both paths at a modest width and returns the summed
prediction, matching ``WideDeep.forward``'s additive combination.

Source: https://github.com/jrzaurin/pytorch-widedeep
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class Wide(nn.Module):
    """Sparse linear (wide) component: summed embedding lookups + bias.

    ``input_dim`` is the size of the (binary + crossed) feature vocabulary;
    each sample provides ``n_wide`` active integer feature indices.
    """

    def __init__(self, input_dim: int, pred_dim: int = 1) -> None:
        super().__init__()
        # padding_idx=0 reserved (pytorch-widedeep convention); +1 for it.
        self.wide_linear = nn.Embedding(input_dim + 1, pred_dim, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(pred_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_wide) integer feature indices -> (B, pred_dim)
        return self.wide_linear(x.long()).sum(dim=1) + self.bias


class _ContinuousEmbeddings(nn.Module):
    """Per-feature continuous embedding (the default 'embed_continuous' path)."""

    def __init__(self, n_cont: int, embed_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_cont, embed_dim))
        self.bias = nn.Parameter(torch.zeros(n_cont, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_cont) -> (B, n_cont, embed_dim)
        out = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.norm(out)


class TabMlp(nn.Module):
    """Deep (deeptabular) component: categorical + continuous embeddings -> MLP."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        cat_embed_dim: int = 16,
        n_continuous: int = 4,
        cont_embed_dim: int = 16,
        mlp_hidden: List[int] = None,
        pred_dim: int = 1,
    ) -> None:
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [64, 32]
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(card, cat_embed_dim) for card in cat_cardinalities]
        )
        self.cont_embed = _ContinuousEmbeddings(n_continuous, cont_embed_dim)

        flat_dim = len(cat_cardinalities) * cat_embed_dim + n_continuous * cont_embed_dim
        layers: List[nn.Module] = []
        prev = flat_dim
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, pred_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        cat = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embeds)]
        cat = torch.cat(cat, dim=1)  # (B, n_cat*cat_embed_dim)
        cont = self.cont_embed(x_cont).flatten(1)  # (B, n_cont*cont_embed_dim)
        h = torch.cat([cat, cont], dim=1)
        return self.mlp(h)


class WideDeep(nn.Module):
    """Joint Wide & Deep model: out = wide(x_wide) + deeptabular(x_cat, x_cont)."""

    def __init__(
        self,
        wide_dim: int = 100,
        n_wide_active: int = 8,
        cat_cardinalities: List[int] = None,
        n_continuous: int = 4,
        pred_dim: int = 1,
    ) -> None:
        super().__init__()
        if cat_cardinalities is None:
            cat_cardinalities = [10, 8, 6, 12]
        self.n_wide_active = n_wide_active
        self.n_cat = len(cat_cardinalities)
        self.n_cont = n_continuous
        self.wide = Wide(wide_dim, pred_dim)
        self.deeptabular = TabMlp(cat_cardinalities, n_continuous=n_continuous, pred_dim=pred_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x packs: [ wide indices (n_wide_active) | cat indices (n_cat) | cont (n_cont) ]
        n_w, n_c = self.n_wide_active, self.n_cat
        x_wide = x[:, :n_w]
        x_cat = x[:, n_w : n_w + n_c]
        x_cont = x[:, n_w + n_c :]
        return self.wide(x_wide) + self.deeptabular(x_cat, x_cont)


def build() -> nn.Module:
    """Build the pytorch-widedeep WideDeep tabular model (wide + TabMlp)."""
    return WideDeep()


def example_input() -> torch.Tensor:
    """Packed tabular sample ``(16, 8+4+4)``: wide idxs + cat idxs + continuous."""
    wide_idx = torch.randint(1, 100, (16, 8)).float()
    cat_idx = torch.stack([torch.randint(0, c, (16,)) for c in [10, 8, 6, 12]], dim=1).float()
    cont = torch.randn(16, 4)
    return torch.cat([wide_idx, cat_idx, cont], dim=1)


MENAGERIE_ENTRIES = [
    (
        "Wide & Deep (tabular: sparse-linear wide + TabMlp deep, additive)",
        "build",
        "example_input",
        "2016",
        "DC",
    ),
]

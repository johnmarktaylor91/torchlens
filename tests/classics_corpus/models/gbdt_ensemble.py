"""Gradient-boosted decision-tree ensembles as TorchLens inference op graphs.

Paper: Chen and Guestrin 2016, XGBoost; Ke et al. 2017, LightGBM.

RecBole exposes LightGBM and XGBoost as external-library recommenders, so there
is no native ``torch.nn.Module`` graph to trace. This module implements the
shared inference structure faithfully: an additive ensemble of hard decision
trees, represented with vectorized PyTorch comparisons, path routing, leaf
gathering, and tree-score summation. The split decisions are non-differentiable;
TorchLens captures the forward inference op graph with ``inference_only=True``
and does not train the trees.
"""

from __future__ import annotations

import torch
import torch.nn as nn


BATCH_SIZE = 8
NUM_FEATURES = 32
NUM_TREES = 16
TREE_DEPTH = 4
NUM_OUTPUTS = 1


def _make_leaf_path_tables(depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create internal-node indices and directions for every full-tree leaf.

    Parameters
    ----------
    depth:
        Full binary tree depth. A depth of ``D`` has ``2 ** D`` leaves.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(node_indices, directions)`` with shape ``(num_leaves, depth)``.
        Directions are ``True`` for right branches and ``False`` for left
        branches.
    """

    leaf_count = 2**depth
    node_rows: list[list[int]] = []
    direction_rows: list[list[bool]] = []
    for leaf_index in range(leaf_count):
        node_index = 0
        nodes: list[int] = []
        directions: list[bool] = []
        for level in range(depth):
            bit_shift = depth - level - 1
            go_right = bool((leaf_index >> bit_shift) & 1)
            nodes.append(node_index)
            directions.append(go_right)
            node_index = (2 * node_index) + 2 if go_right else (2 * node_index) + 1
        node_rows.append(nodes)
        direction_rows.append(directions)
    return torch.tensor(node_rows, dtype=torch.long), torch.tensor(direction_rows, dtype=torch.bool)


class HardGBDTEnsemble(nn.Module):
    """Vectorized hard-routing gradient-boosted decision-tree ensemble."""

    def __init__(
        self,
        *,
        num_features: int = NUM_FEATURES,
        num_trees: int = NUM_TREES,
        depth: int = TREE_DEPTH,
        num_outputs: int = NUM_OUTPUTS,
        learning_rate: float = 0.1,
        base_score: float = 0.0,
        seed: int = 0,
    ) -> None:
        """Initialize a random hard-decision tree ensemble.

        Parameters
        ----------
        num_features:
            Number of dense input features.
        num_trees:
            Number of additive boosted trees.
        depth:
            Full binary-tree depth.
        num_outputs:
            Number of output scores per example.
        learning_rate:
            Boosting shrinkage applied to the summed tree outputs.
        base_score:
            Constant bias added after tree-score summation.
        seed:
            Random seed for deterministic buffer initialization.
        """

        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        internal_nodes = 2**depth - 1
        num_leaves = 2**depth

        split_features = torch.randint(
            low=0,
            high=num_features,
            size=(num_trees, internal_nodes),
            generator=generator,
        )
        thresholds = torch.randn(num_trees, internal_nodes, generator=generator)
        leaf_values = torch.randn(num_trees, num_leaves, num_outputs, generator=generator)
        leaf_values = leaf_values / num_trees**0.5
        path_nodes, path_directions = _make_leaf_path_tables(depth)

        self.num_features = num_features
        self.num_trees = num_trees
        self.depth = depth
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.base_score = base_score
        self.register_buffer("split_features", split_features)
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("leaf_values", leaf_values)
        self.register_buffer("path_nodes", path_nodes)
        self.register_buffer("path_directions", path_directions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the additive hard-tree ensemble.

        Parameters
        ----------
        x:
            Dense feature matrix with shape ``(batch, num_features)``.

        Returns
        -------
        torch.Tensor
            Ensemble prediction tensor with shape ``(batch, num_outputs)``.
        """

        batch_size = x.shape[0]
        feature_values = x.unsqueeze(1).expand(batch_size, self.num_trees, self.num_features)
        split_features = self.split_features.unsqueeze(0).expand(batch_size, -1, -1)
        split_values = torch.gather(feature_values, dim=2, index=split_features)
        split_go_right = torch.gt(split_values, self.thresholds.unsqueeze(0))

        path_decisions = split_go_right[:, :, self.path_nodes]
        path_directions = self.path_directions.unsqueeze(0).unsqueeze(0)
        right_reached = torch.where(
            path_directions, path_decisions, torch.logical_not(path_decisions)
        )
        routing = right_reached.to(x.dtype).prod(dim=-1)

        tree_outputs = torch.einsum("btl,tlo->bto", routing, self.leaf_values)
        ensemble_output = tree_outputs.sum(dim=1) * self.learning_rate
        return ensemble_output + self.base_score


def build_lightgbm() -> nn.Module:
    """Build a LightGBM-style hard GBDT inference ensemble.

    Returns
    -------
    nn.Module
        Random-init additive hard decision-tree ensemble.
    """

    return HardGBDTEnsemble(seed=2017)


def build_xgboost() -> nn.Module:
    """Build an XGBoost-style hard GBDT inference ensemble.

    Returns
    -------
    nn.Module
        Random-init additive hard decision-tree ensemble.
    """

    return HardGBDTEnsemble(seed=2016)


def example_input_lightgbm() -> torch.Tensor:
    """Create the LightGBM example dense feature matrix.

    Returns
    -------
    torch.Tensor
        Feature matrix with shape ``(8, 32)``.
    """

    generator = torch.Generator()
    generator.manual_seed(1717)
    return torch.randn(BATCH_SIZE, NUM_FEATURES, generator=generator)


def example_input_xgboost() -> torch.Tensor:
    """Create the XGBoost example dense feature matrix.

    Returns
    -------
    torch.Tensor
        Feature matrix with shape ``(8, 32)``.
    """

    generator = torch.Generator()
    generator.manual_seed(1616)
    return torch.randn(BATCH_SIZE, NUM_FEATURES, generator=generator)


MENAGERIE_ENTRIES = [
    ("recbole.LightGBM", "build_lightgbm", "example_input_lightgbm", "2017", "DC"),
    ("recbole.XGBoost", "build_xgboost", "example_input_xgboost", "2016", "DC"),
]

# FAITHFUL PORT of https://github.com/brain-research/deep-molecular-massspec @ main
# (molecule_predictors.py's MLPSpectraPrediction + util.py's scatter_by_anchor_indices)
# (original framework: TensorFlow 1.x / tf.estimator / tf.contrib.training.HParams)
#
# NEIMS: Neural Electron-Ionization Mass Spectrometry (Wei, Belanger, Adams, Sculley,
# "Rapid Prediction of Electron-Ionization Mass Spectrometry Using Neural Networks",
# J. Chem. Inf. Model. 2019, arXiv:1811.08545). Predicts an EI mass spectrum (a dense
# vector over integer m/z bins) from a molecule's Morgan/circular fingerprint. The
# repo (github.com/brain-research/deep-molecular-massspec, cited by the paper) is
# TF1-era code built on `tf.estimator`/`tf.layers`/`tf.contrib.training.HParams`,
# none of which exist in modern TF2 or in our torch-only environment, so the model
# cannot be vendored/run directly; it is transcribed faithfully here instead.
#
# Ported architecture (MLPSpectraPrediction, the paper's flagship/default model):
#   1. `_make_learned_features`: fingerprint -> Linear -> [residual pre-activation
#      block: BatchNorm -> activation -> Dropout -> Linear(bottleneck) -> BatchNorm
#      -> activation -> Linear] x num_hidden_layers, with a residual (+=) add each
#      block -> final BatchNorm -> activation.
#   2. `make_prediction_ops`'s bidirectional-prediction head (the paper's key
#      architectural contribution): two independent linear heads project the
#      learned features to `max_mass_spec_peak_loc` bins; the "forward" head is
#      masked to zero out bins beyond (molecular_weight + max_prediction_above_
#      molecule_mass); the "backward" head is *reversed* relative to the molecular
#      weight via `util.scatter_by_anchor_indices` (a right-aligned flip-and-shift:
#      output[i][j] = data[i][anchor_indices[i] - j + index_shift], zero-padded
#      out-of-range) so the network can predict small neutral-loss fragments (which
#      cluster near the parent mass) as if they were low-mass ions; the two
#      predictions are summed, then a final ReLU is applied (the repo's default
#      hparams use `loss='generalized_mse'`, so `final_prediction = relu(raw)`,
#      not cross-entropy/softmax).
#
# Hyperparameters here (fingerprint length, hidden width, peak count) are shrunk
# from the paper's production defaults (fp_length=4096, num_hidden_units=2000,
# max_mass_spec_peak_loc=1000) to menagerie-recipe scale; every mechanism above is
# preserved exactly.

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def scatter_by_anchor_indices(
    anchor_indices: torch.Tensor, data: torch.Tensor, index_shift: int
) -> torch.Tensor:
    """Faithful torch port of util.scatter_by_anchor_indices.

    output[i][j] = data[i][anchor_indices[i] - j + index_shift], with
    out-of-range (j) positions zeroed, matching the original TF sparse-scatter
    implementation exactly (flip-and-right-align each row of `data` around its
    own `anchor_indices[i] + index_shift`).
    """
    batch_size, num_data_columns = data.shape
    device = data.device

    col = torch.arange(num_data_columns, device=device).unsqueeze(0)  # [1, C]
    shifted_indices = anchor_indices.unsqueeze(-1) - col + index_shift  # [B, C]
    valid = (shifted_indices >= 0) & (shifted_indices < num_data_columns)

    safe_indices = shifted_indices.clamp(0, num_data_columns - 1)
    gathered = torch.gather(data, 1, safe_indices)
    return torch.where(valid, gathered, torch.zeros_like(gathered))


class ResidualBlock(nn.Module):
    """`MLPSpectraPrediction._residual_block`: pre-activation residual block."""

    def __init__(self, num_hidden_units: int, resnet_bottleneck_factor: float, dropout_rate: float):
        super().__init__()
        bottleneck_units = int(resnet_bottleneck_factor * num_hidden_units)
        self.bn1 = nn.BatchNorm1d(num_hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_hidden_units, bottleneck_units)
        self.bn2 = nn.BatchNorm1d(bottleneck_units)
        self.fc2 = nn.Linear(bottleneck_units, num_hidden_units)
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.bn1(features)
        features = self.activation(features)
        features = self.dropout(features)
        features = self.fc1(features)
        features = self.bn2(features)
        features = self.activation(features)
        return self.fc2(features)


class NEIMSMLPSpectraPrediction(nn.Module):
    """Faithful port of MLPSpectraPrediction (the paper's flagship model).

    Forward mode assumes `mode != TRAIN` (inference), matching how the real
    repo's estimator runs at prediction/eval time (`is_training=False` feeds
    into the `_batch_norm`/`dropout` calls that gate on `mode`); we express
    this simply by calling `.eval()` in `build_neims_mlp` so BatchNorm/Dropout
    use running statistics / are inert, exactly like TF's `training=False`.
    """

    def __init__(
        self,
        fp_length: int = 256,
        num_hidden_units: int = 128,
        num_hidden_layers: int = 1,
        resnet_bottleneck_factor: float = 0.5,
        dropout_rate: float = 0.25,
        max_mass_spec_peak_loc: int = 64,
        max_prediction_above_molecule_mass: int = 5,
        bidirectional_prediction: bool = True,
        gate_bidirectional_predictions: bool = False,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.max_mass_spec_peak_loc = max_mass_spec_peak_loc
        self.max_prediction_above_molecule_mass = max_prediction_above_molecule_mass
        self.bidirectional_prediction = bidirectional_prediction
        self.gate_bidirectional_predictions = gate_bidirectional_predictions

        self.input_dense = nn.Linear(fp_length, num_hidden_units)
        self.input_activation = nn.ReLU()
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(num_hidden_units, resnet_bottleneck_factor, dropout_rate)
                for _ in range(num_hidden_layers)
            ]
        )
        self.final_bn = nn.BatchNorm1d(num_hidden_units)
        self.final_activation = nn.ReLU()

        self.forward_head = nn.Linear(num_hidden_units, max_mass_spec_peak_loc)
        self.backward_head = nn.Linear(num_hidden_units, max_mass_spec_peak_loc)
        if gate_bidirectional_predictions:
            self.gate_head = nn.Linear(num_hidden_units, max_mass_spec_peak_loc)

    def _make_learned_features(self, fingerprint: torch.Tensor) -> torch.Tensor:
        layer_output = fingerprint
        if self.num_hidden_layers > 0:
            layer_output = self.input_activation(self.input_dense(layer_output))
        for block in self.residual_blocks:
            layer_output = layer_output + block(layer_output)
        layer_output = self.final_bn(layer_output)
        return self.final_activation(layer_output)

    def _mask_prediction_by_mass(
        self, raw_prediction: torch.Tensor, molecule_weight: torch.Tensor
    ) -> torch.Tensor:
        total_mass = torch.round(molecule_weight).long()
        indices = torch.arange(raw_prediction.shape[-1], device=raw_prediction.device).unsqueeze(0)
        right_of_total_mass = indices > (
            total_mass.unsqueeze(-1) + self.max_prediction_above_molecule_mass
        )
        return torch.where(right_of_total_mass, torch.zeros_like(raw_prediction), raw_prediction)

    def _reverse_prediction(
        self, raw_prediction: torch.Tensor, molecule_weight: torch.Tensor
    ) -> torch.Tensor:
        total_mass = torch.round(molecule_weight).long()
        return scatter_by_anchor_indices(
            total_mass, raw_prediction, self.max_prediction_above_molecule_mass
        )

    def forward(self, fingerprint: torch.Tensor, molecule_weight: torch.Tensor) -> torch.Tensor:
        learned_features = self._make_learned_features(fingerprint)

        if self.bidirectional_prediction:
            forward_prediction = self.forward_head(learned_features)
            forward_prediction = self._mask_prediction_by_mass(forward_prediction, molecule_weight)

            backward_prediction = self.backward_head(learned_features)
            backward_prediction = self._reverse_prediction(backward_prediction, molecule_weight)

            if self.gate_bidirectional_predictions:
                gate = torch.sigmoid(self.gate_head(learned_features))
                raw_prediction = gate * forward_prediction + (1.0 - gate) * backward_prediction
            else:
                raw_prediction = forward_prediction + backward_prediction
        else:
            raw_prediction = self.forward_head(learned_features)
            raw_prediction = self._mask_prediction_by_mass(raw_prediction, molecule_weight)

        # Default hparams use loss='generalized_mse' -> final_prediction = relu(raw)
        return torch.relu(raw_prediction)


def build_neims_mlp():
    torch.manual_seed(0)
    model = NEIMSMLPSpectraPrediction(
        fp_length=256,
        num_hidden_units=128,
        num_hidden_layers=1,
        resnet_bottleneck_factor=0.5,
        dropout_rate=0.25,
        max_mass_spec_peak_loc=64,
        max_prediction_above_molecule_mass=5,
        bidirectional_prediction=True,
        gate_bidirectional_predictions=False,
    )
    model.eval()
    return model


def example_input_neims_mlp():
    # A tiny batch of synthetic Morgan/circular-fingerprint bit-vectors (real
    # inputs come from RDKit via the repo's own `feature_utils.py`, which is a
    # data-featurization concern, not part of the model) plus per-molecule
    # weights (real values come from `fmap_constants.MOLECULE_WEIGHT`).
    torch.manual_seed(0)
    batch_size = 4
    fp_length = 256
    fingerprint = (torch.rand(batch_size, fp_length) > 0.9).float()
    molecule_weight = torch.tensor([180.0, 46.0, 342.3, 58.1], dtype=torch.float32)
    return (fingerprint, molecule_weight)


MENAGERIE_ENTRIES = [
    (
        "NEIMS (MLP spectrum predictor)",
        "build_neims_mlp",
        "example_input_neims_mlp",
        2019,
        MENAGERIE_ZOO,
    ),
]

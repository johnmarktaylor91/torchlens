# SOURCE: vendored from https://github.com/nicksungg/clarc_blended_wing_body @ main
# (models/film_model_v1.py, pointnet/point_to_parameter_model.py)
#
# BlendedNet: A Blended Wing Body Aircraft Dataset and Surrogate Model for
# Aerodynamic Predictions (Sung, Spreizer, Elrefaie, Samuel, Jones, Ahmed --
# ASME IDETC/CIE 2025, DETC2025-168977; arXiv:2509.07209). Two-stage surrogate
# model for pointwise aerodynamic-coefficient prediction on blended-wing-body
# (BWB) aircraft: (1) a permutation-invariant PointNet regressor predicts 9
# geometric design parameters from a sampled surface point cloud, then (2) a
# Feature-wise Linear Modulation (FiLM) MLP -- conditioned on those predicted
# geometric parameters plus flight conditions -- maps per-point 3D coordinates
# (+ normals) to pointwise Cp/Cfx/Cfz aerodynamic coefficients.
#
# The repo's README labels film_model_v2.py the "Final FiLM model (SIREN-style
# with sine + residuals)", but the repo's OWN reproducible training/eval
# scripts (train_model.ipynb / test_model.ipynb, embedded
# `train_with_frozen_stats.py` / eval cells) import and construct
# `film_model_v1.FiLMNet(cond_dim=13, coord_dim=6, output_dim=3, hidden_dim=256,
# num_layers=4, extra_layers=3)` exclusively -- v1 is the model that actually
# produced the paper's reported checkpoints (`film_best.pth`/`film_final.pth`).
# This staging module vendors v1, the ACTUAL FiLM architecture exercised by the
# repo's own training/eval pipeline (FiLM-modulated ReLU MLP + non-modulated
# extra layers), together with the real PointNetRegressor (shared Conv1d
# feature extractor + global max-pool + 2-layer regression head) that predicts
# the 9 geometric shape parameters feeding the FiLM condition vector -- both
# pure torch/torch.nn.functional, no exotic dependency. No architecture was
# altered; only the module-relative imports were collapsed into one file for
# menagerie staging.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# pointnet/point_to_parameter_model.py
# ---------------------------------------------------------------------------
class PointNetEncoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(latent_size)

    def forward(self, x):
        # x shape: (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))  # (B, latent_size, N)
        x = torch.max(x, dim=2, keepdim=False)[0]  # global max pooling over points
        return x


class PointNetRegressor(nn.Module):
    def __init__(self, latent_size, output_size=9):
        super().__init__()
        self.encoder = PointNetEncoder(latent_size)
        self.regressor = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        # x shape: (B, 3, N)
        latent = self.encoder(x)  # (B, latent_size)
        output = self.regressor(latent)  # (B, output_size)
        return output


# ---------------------------------------------------------------------------
# models/film_model_v1.py -- the model actually used by train/eval scripts
# ---------------------------------------------------------------------------
class FiLMModulation(nn.Module):
    """
    Maps the condition vector to scaling (gamma) and shifting (beta)
    parameters for FiLM modulation in the MLP.
    """

    def __init__(self, cond_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # For each hidden layer (except the final) we output scale+shift parameters.
        self.num_mod_params = 2 * hidden_dim * (num_layers - 1)
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_mod_params),
        )

    def forward(self, cond):
        out = self.fc(cond)  # (batch, 2*hidden_dim*(num_layers-1))
        chunk_size = self.hidden_dim * (self.num_layers - 1)
        gamma = out[:, :chunk_size]
        beta = out[:, chunk_size:]
        return gamma, beta


class ModulatedMLP(nn.Module):
    """
    MLP that maps 3D coordinates to aerodynamic coefficients.
    Applies FiLM (scale+shift) after each hidden layer's ReLU,
    then processes the output through extra (non-modulated) layers
    to increase the model's expressiveness.
    """

    def __init__(self, input_dim=3, output_dim=3, hidden_dim=256, num_layers=4, extra_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extra_layers = extra_layers

        # Build the FiLM-modulated layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Extra non-modulated layers to boost expressiveness
        self.extra = nn.ModuleList()
        for _ in range(extra_layers):
            self.extra.append(nn.Linear(hidden_dim, hidden_dim))

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, coords, gamma, beta):
        chunk_size = self.hidden_dim
        h = coords
        # FiLM-modulated part
        for i in range(self.num_layers - 1):
            h = self.layers[i](h)
            h = torch.relu(h)
            # Apply FiLM modulation
            g_i = gamma[:, i * chunk_size : (i + 1) * chunk_size]
            b_i = beta[:, i * chunk_size : (i + 1) * chunk_size]
            h = g_i * h + b_i

        # Extra non-modulated layers
        for layer in self.extra:
            h = torch.relu(layer(h))

        out = self.output_layer(h)
        return out


class FiLMNet(nn.Module):
    """
    Combines FiLMModulation and ModulatedMLP into one model.
    Takes in 3D coordinates and a condition vector and outputs predictions.
    """

    def __init__(
        self, cond_dim=14, coord_dim=3, output_dim=3, hidden_dim=256, num_layers=4, extra_layers=2
    ):
        super().__init__()
        self.modulation_net = FiLMModulation(cond_dim, hidden_dim, num_layers)
        self.mlp = ModulatedMLP(coord_dim, output_dim, hidden_dim, num_layers, extra_layers)

    def forward(self, coords, cond):
        gamma, beta = self.modulation_net(cond)
        return self.mlp(coords, gamma, beta)


# ---------------------------------------------------------------------------
# BlendedNet end-to-end -- point cloud -> PointNet params -> FiLM(coords, flight+params) -> Cp/Cfx/Cfz
# ---------------------------------------------------------------------------
class BlendedNet(nn.Module):
    """End-to-end BlendedNet surrogate: PointNetRegressor -> FiLMNet.

    Stage 1 (PointNetRegressor) predicts `n_shape_params` geometric design
    parameters from a sampled surface point cloud (B, 3, N_pts). Stage 2
    (FiLMNet) conditions on those predicted shape parameters concatenated with
    `n_flight` flight-condition scalars, and maps per-surface-point
    coordinates+normals (B, N_query, coord_dim) to pointwise (cp, cfx, cfz).
    """

    def __init__(self, latent_size=256, n_shape_params=9, n_flight=4, coord_dim=6):
        super().__init__()
        self.pointnet = PointNetRegressor(latent_size=latent_size, output_size=n_shape_params)
        self.film = FiLMNet(
            cond_dim=n_shape_params + n_flight,
            coord_dim=coord_dim,
            output_dim=3,
            hidden_dim=256,
            num_layers=4,
            extra_layers=3,
        )

    def forward(self, point_cloud, flight_cond, query_coords):
        # point_cloud: (B, 3, N_pts); flight_cond: (B, n_flight)
        shape_params = self.pointnet(point_cloud)  # (B, n_shape_params)
        cond = torch.cat([shape_params, flight_cond], dim=-1)  # (B, n_shape_params + n_flight)

        # Broadcast condition per query point: (B, N_query, coord_dim) queries share one cond vector.
        n_query = query_coords.shape[1]
        cond_expanded = cond.unsqueeze(1).expand(-1, n_query, -1).reshape(-1, cond.shape[-1])
        coords_flat = query_coords.reshape(-1, query_coords.shape[-1])

        preds = self.film(coords_flat, cond_expanded)  # (B*N_query, 3)
        return preds.view(query_coords.shape[0], n_query, 3)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------
_LATENT = 32
_N_SHAPE_PARAMS = (
    9  # B1,B2,B3,C1,C2,C3,C4,S1,S3,X3 minus one dropped -> matches repo's output_size=9
)
_N_FLIGHT = 4  # Re_L, M_inf, alpha_deg (+1 optional, matches flight_cond total width variability)
_COORD_DIM = 6  # (x, y, z) normalized coords + 3D surface normal, per dataset.py 'points' (N,6)
_N_PTS = 64  # sampled surface points fed to PointNet
_N_QUERY = 16  # query surface points evaluated per forward pass
_BATCH = 2


def build_blendednet():
    torch.manual_seed(0)
    model = BlendedNet(
        latent_size=_LATENT,
        n_shape_params=_N_SHAPE_PARAMS,
        n_flight=_N_FLIGHT,
        coord_dim=_COORD_DIM,
    )
    model.eval()
    return model


def example_input_blendednet():
    torch.manual_seed(0)
    point_cloud = torch.randn(_BATCH, 3, _N_PTS)
    flight_cond = torch.randn(_BATCH, _N_FLIGHT)
    query_coords = torch.randn(_BATCH, _N_QUERY, _COORD_DIM)
    return (point_cloud, flight_cond, query_coords)


MENAGERIE_ENTRIES = [
    ("BlendedNet", "build_blendednet", "example_input_blendednet", 2025, MENAGERIE_ZOO),
]

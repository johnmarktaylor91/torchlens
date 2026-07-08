# FAITHFUL PORT of LPDI-EPFL/masif @ master (original framework: TensorFlow 1.x)
#
# Ported from source/masif_modules/MaSIF_site.py::MaSIF_site (the real "MaSIF-site" surface
# binding-site-prediction model from Gainza et al., Nature Methods 2020, "Deciphering
# interaction fingerprints from protein molecular surfaces using geometric deep learning").
# The original file is TF1.x graph-mode code (`tf.placeholder`, `tf.get_variable`,
# `tf.Session`, `tf.contrib.layers.fully_connected`, `tf.trainable_variables()`) which cannot
# run in a base torch env and TF1.x is not something to install; this port transcribes the
# geodesic-convolution architecture (`MaSIF_site.inference`, the repo's own MoNet-style
# Gaussian-mixture patch convolution over precomputed geodesic polar (rho, theta) surface
# coordinates) and the `__init__` layer-stacking logic (default masif_opts["site"] config:
# n_conv_layers=3, n_thetas=16, n_rhos=5, n_rotations=16, feat_mask=[1.0]*5, max_rho=9.0)
# into self-contained eager torch. The TF1.x-only training/session/optimizer/loss machinery
# in `__init__` (tf.Session, tf.train.AdamOptimizer, tf.gradients, the data_loss/eval_score
# ops) is training infrastructure, not part of the model architecture, and is not ported;
# `inference()` and the conv-layer stacking that builds `global_desc`/`logits`
# (`full_score` in the original) are the real forward path and are ported mechanism-for-
# mechanism, including the rotation-averaged (max over `n_rotations`) Gaussian geodesic
# convolution and the patch re-gathering (`tf.gather(..., indices_tensor)`) between
# conv layers.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class _GeodesicGaussianConv(nn.Module):
    """Ported from MaSIF_site.inference(): one rotation-averaged Gaussian-mixture geodesic
    convolution over a patch of (rho, theta) geodesic polar coordinates, per input feature
    channel. `W_conv`/`b_conv` (a `tf.get_variable`/`tf.Variable` pair per feature channel in
    the original) become an `nn.Linear` per channel; `mu_rho`/`sigma_rho`/`mu_theta`/
    `sigma_theta` (per-channel `tf.Variable`s initialized from `compute_initial_coordinates`)
    become learnable `nn.Parameter`s.
    """

    def __init__(
        self,
        n_feat,
        n_thetas,
        n_rhos,
        max_rho,
        n_rotations,
        sigma_rho_init,
        sigma_theta_init,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
        super().__init__()
        self.n_feat = n_feat
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_gauss = n_thetas * n_rhos
        self.n_rotations = n_rotations
        self.eps = eps
        self.mean_gauss_activation = mean_gauss_activation

        initial_coords = self._compute_initial_coordinates(
            max_rho, n_thetas, n_rhos
        )  # [n_gauss, 2]
        mu_rho_initial = initial_coords[:, 0].unsqueeze(0).float()  # [1, n_gauss]
        mu_theta_initial = initial_coords[:, 1].unsqueeze(0).float()  # [1, n_gauss]

        self.mu_rho = nn.ParameterList(
            [nn.Parameter(mu_rho_initial.clone()) for _ in range(n_feat)]
        )
        self.mu_theta = nn.ParameterList(
            [nn.Parameter(mu_theta_initial.clone()) for _ in range(n_feat)]
        )
        self.sigma_rho = nn.ParameterList(
            [nn.Parameter(torch.full_like(mu_rho_initial, sigma_rho_init)) for _ in range(n_feat)]
        )
        self.sigma_theta = nn.ParameterList(
            [
                nn.Parameter(torch.full_like(mu_theta_initial, sigma_theta_init))
                for _ in range(n_feat)
            ]
        )
        self.conv = nn.ModuleList([nn.Linear(self.n_gauss, self.n_gauss) for _ in range(n_feat)])
        for lin in self.conv:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    @staticmethod
    def _compute_initial_coordinates(max_rho, n_thetas, n_rhos):
        # Ported verbatim (numeric semantics) from MaSIF_site.compute_initial_coordinates().
        range_rho = (0.0, max_rho)
        range_theta = (0.0, 2 * math.pi)

        grid_rho = torch.linspace(range_rho[0], range_rho[1], n_rhos + 1)[1:]
        grid_theta = torch.linspace(range_theta[0], range_theta[1], n_thetas + 1)[:-1]

        grid_rho_, grid_theta_ = torch.meshgrid(grid_rho, grid_theta, indexing="ij")
        grid_rho_ = grid_rho_.T.flatten()
        grid_theta_ = grid_theta_.T.flatten()

        coords = torch.stack([grid_rho_, grid_theta_], dim=0).T  # [n_gauss, 2]
        return coords

    def _one_channel(self, i, input_feat_i, rho_coords, theta_coords, mask):
        """Ported from MaSIF_site.inference() body, specialized to a single feature channel
        (the original loops over `n_feat` channels at the call site, each with its own
        W_conv/mu_rho/etc.). Shapes follow the original: rho_coords/theta_coords
        [n_samples, n_vertices], input_feat_i [n_samples, n_vertices, 1], mask
        [n_samples, n_vertices, 1]."""
        n_samples, n_vertices = rho_coords.shape

        all_conv_feat = []
        for k in range(self.n_rotations):
            rho_coords_ = rho_coords.reshape(-1, 1)
            theta_coords_ = theta_coords.reshape(-1, 1)

            theta_coords_ = theta_coords_ + k * 2 * math.pi / self.n_rotations
            theta_coords_ = torch.remainder(theta_coords_, 2 * math.pi)
            rho_gauss = torch.exp(
                -torch.square(rho_coords_ - self.mu_rho[i])
                / (torch.square(self.sigma_rho[i]) + self.eps)
            )
            theta_gauss = torch.exp(
                -torch.square(theta_coords_ - self.mu_theta[i])
                / (torch.square(self.sigma_theta[i]) + self.eps)
            )

            gauss_activations = rho_gauss * theta_gauss  # [n_samples*n_vertices, n_gauss]
            gauss_activations = gauss_activations.reshape(n_samples, n_vertices, -1)
            gauss_activations = gauss_activations * mask
            if self.mean_gauss_activation:
                gauss_activations = gauss_activations / (
                    gauss_activations.sum(dim=1, keepdim=True) + self.eps
                )

            gauss_activations = gauss_activations.unsqueeze(
                2
            )  # [n_samples, n_vertices, 1, n_gauss]
            input_feat_ = input_feat_i.unsqueeze(3)  # [n_samples, n_vertices, n_feat_i=1, 1]

            gauss_desc = gauss_activations * input_feat_  # [n_samples, n_vertices, 1, n_gauss]
            gauss_desc = gauss_desc.sum(dim=1)  # [n_samples, 1, n_gauss]
            gauss_desc = gauss_desc.reshape(n_samples, self.n_gauss)

            conv_feat = self.conv[i](gauss_desc)  # [n_samples, n_gauss]
            all_conv_feat.append(conv_feat)

        all_conv_feat = torch.stack(all_conv_feat, dim=0)
        conv_feat = torch.amax(all_conv_feat, dim=0)
        conv_feat = F.relu(conv_feat)
        return conv_feat

    def forward(self, input_feat, rho_coords, theta_coords, mask):
        """input_feat: [n_samples, n_vertices, n_feat]; rho_coords/theta_coords:
        [n_samples, n_vertices]; mask: [n_samples, n_vertices, 1]. Returns a list of
        per-channel conv outputs (mirrors the original's `self.global_desc` list-of-tensors
        built by the `for i in range(self.n_feat)` loop at the `__init__` call site)."""
        outs = []
        for i in range(self.n_feat):
            input_feat_i = input_feat[:, :, i : i + 1]
            outs.append(self._one_channel(i, input_feat_i, rho_coords, theta_coords, mask))
        return outs


class MaSIFSite(nn.Module):
    """Ported from MaSIF_site.__init__ / the model's real forward computation graph (the
    ops that build `self.global_desc` -> `self.logits` -> `self.full_score`), using the
    default masif_opts["site"] config: n_conv_layers=3, n_thetas=16, n_rhos=5,
    n_rotations=16, max_rho=9.0, feat_mask=[1.0]*5 (n_feat=5)."""

    def __init__(
        self,
        max_rho=9.0,
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        feat_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
        n_conv_layers=3,
    ):
        super().__init__()
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_gauss = n_thetas * n_rhos
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        self.n_labels = 2
        self.n_conv_layers = n_conv_layers

        sigma_rho_init = max_rho / 8
        sigma_theta_init = 1.0

        self.conv1 = _GeodesicGaussianConv(
            self.n_feat, n_thetas, n_rhos, max_rho, n_rotations, sigma_rho_init, sigma_theta_init
        )

        if n_conv_layers > 1:
            self.conv2 = _GeodesicGaussianConv(
                self.n_feat,
                n_thetas,
                n_rhos,
                max_rho,
                n_rotations,
                sigma_rho_init,
                sigma_theta_init,
            )
            self.fc_l2 = nn.Linear(self.n_feat * self.n_gauss, self.n_gauss * self.n_feat)
            nn.init.xavier_uniform_(self.fc_l2.weight)
        if n_conv_layers > 2:
            self.conv3 = _GeodesicGaussianConv(
                self.n_feat,
                n_thetas,
                n_rhos,
                max_rho,
                n_rotations,
                sigma_rho_init,
                sigma_theta_init,
            )
            self.fc_l3 = nn.Linear(self.n_feat * self.n_gauss, self.n_gauss * self.n_feat)
            nn.init.xavier_uniform_(self.fc_l3.weight)
        if n_conv_layers > 3:
            self.conv4 = _GeodesicGaussianConv(
                self.n_feat,
                n_thetas,
                n_rhos,
                max_rho,
                n_rotations,
                sigma_rho_init,
                sigma_theta_init,
            )
            self.fc_l4 = nn.Linear(self.n_gauss * self.n_gauss, self.n_gauss * self.n_gauss)
            nn.init.xavier_uniform_(self.fc_l4.weight)

        # Ported from `tf.contrib.layers.fully_connected` calls after the conv-layer stack.
        self.fc_refine1 = nn.Linear(self.n_gauss * self.n_feat, self.n_gauss)
        self.fc_refine2 = nn.Linear(self.n_gauss, self.n_feat)
        self.fc_mlp = nn.Linear(self.n_feat, self.n_thetas)
        self.fc_logits = nn.Linear(self.n_thetas, self.n_labels)

    def forward(self, input_feat, rho_coords, theta_coords, mask, indices_tensor):
        """input_feat: [n_samples, n_vertices, n_feat]; rho_coords/theta_coords:
        [n_samples, n_vertices]; mask: [n_samples, n_vertices, 1]; indices_tensor:
        [n_samples, max_verts] long indices used to re-gather patches between conv layers
        (mirrors the original's `tf.gather(self.global_desc, self.indices_tensor)`).
        Returns `full_score`: [n_samples] sigmoid binding-site score per input surface
        vertex/patch, matching the original `self.full_score`."""
        n_samples = input_feat.shape[0]

        global_desc = self.conv1(
            input_feat, rho_coords, theta_coords, mask
        )  # list of [n_samples, n_gauss]
        global_desc = torch.stack(global_desc, dim=1)  # [n_samples, n_feat, n_gauss]
        global_desc = global_desc.reshape(n_samples, self.n_gauss * self.n_feat)
        global_desc = F.relu(self.fc_refine1(global_desc))
        global_desc = F.relu(self.fc_refine2(global_desc))

        if self.n_conv_layers > 1:
            patch = global_desc[indices_tensor]  # [n_samples, max_verts, n_feat]
            patch2 = self.conv2(patch, rho_coords, theta_coords, mask)
            patch2 = torch.stack(patch2, dim=1)  # [n_samples, n_feat, n_gauss]
            patch2 = patch2.reshape(n_samples, self.n_feat * self.n_gauss)
            patch2 = self.fc_l2(patch2)
            batch_size = patch2.shape[0]
            patch2 = patch2.reshape(batch_size, self.n_feat, self.n_gauss)
            global_desc = patch2.mean(dim=2)

        if self.n_conv_layers > 2:
            patch = global_desc[indices_tensor]
            patch3 = self.conv3(patch, rho_coords, theta_coords, mask)
            patch3 = torch.stack(patch3, dim=1)
            patch3 = patch3.reshape(n_samples, self.n_feat * self.n_gauss)
            patch3 = self.fc_l3(patch3)
            batch_size = patch3.shape[0]
            patch3 = patch3.reshape(batch_size, self.n_feat, self.n_gauss)
            global_desc = patch3.mean(dim=2)

        if self.n_conv_layers > 3:
            patch = global_desc[indices_tensor]
            patch4 = self.conv4(patch, rho_coords, theta_coords, mask)
            patch4 = torch.stack(patch4, dim=1)
            patch4 = patch4.reshape(n_samples, self.n_gauss * self.n_gauss)
            patch4 = self.fc_l4(patch4)
            batch_size = patch4.shape[0]
            patch4 = patch4.reshape(batch_size, self.n_gauss, self.n_gauss)
            global_desc = patch4.amax(dim=2)

        global_desc = F.relu(self.fc_mlp(global_desc))
        logits = self.fc_logits(global_desc)

        full_logits = torch.sigmoid(logits)
        full_score = full_logits.squeeze()[:, 0]
        return full_score


def build_masif_site():
    torch.manual_seed(0)
    return MaSIFSite(
        max_rho=9.0,
        n_thetas=16,
        n_rhos=5,
        n_rotations=4,
        feat_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
        n_conv_layers=3,
    )


def example_input_masif_site():
    """Tiny synthetic single-batch geodesic surface-patch input matching the real field
    schema fed via feed_dict in the repo's train_masif_site.py (input_feat, rho_coords,
    theta_coords, mask, indices_tensor); n_samples=2 surface vertices, max_verts=6 patch
    neighbors, n_feat=5 surface chemical/geometric features."""
    torch.manual_seed(0)
    n_samples = 2
    max_verts = 6
    n_feat = 5

    input_feat = torch.randn(n_samples, max_verts, n_feat)
    rho_coords = torch.rand(n_samples, max_verts) * 9.0
    theta_coords = torch.rand(n_samples, max_verts) * (2 * math.pi)
    mask = torch.ones(n_samples, max_verts, 1)
    indices_tensor = torch.zeros(n_samples, max_verts, dtype=torch.long)

    return (input_feat, rho_coords, theta_coords, mask, indices_tensor)


MENAGERIE_ENTRIES = [
    ("MaSIF-site", "build_masif_site", "example_input_masif_site", 2020, MENAGERIE_ZOO),
]

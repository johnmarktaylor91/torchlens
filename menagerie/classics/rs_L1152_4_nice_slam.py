# SOURCE: vendored from cvg/nice-slam @ master
#   https://github.com/cvg/nice-slam
#   File: src/conv_onet/models/decoder.py (GaussianFourierFeatureTransform,
#   Nerf_positional_embedding, DenseLayer, Same, MLP, MLP_no_xyz, NICE), plus
#   the normalize_3d_coordinate() helper from src/common.py that MLP/MLP_no_xyz
#   call during grid-feature sampling. Code is kept verbatim except: (1) the
#   hardcoded `device = f'cuda:{p.get_device()}'` line in NICE.forward is
#   replaced with `device = p.device` so the model is runnable on CPU (a
#   non-architectural device-selection fix, not a change to any layer/module);
#   (2) `self.bound` (normally injected onto MLP/MLP_no_xyz externally by
#   NICE_SLAM.py's renderer before each forward call) is set as a constructor
#   buffer here since this module is traced standalone.
#
# Architecture: NICE (Neural Implicit Scalable Encoding) -- the multi-resolution
# feature-grid + coordinate-MLP occupancy decoder at the heart of NICE-SLAM
# (Zhu et al., CVPR 2022). Coarse/middle/fine/color decoder heads each sample
# a learned 3D feature grid (bilinear/trilinear grid_sample) at query points,
# Fourier/NeRF-style positional-encode the raw coordinates, and fuse both
# through a small residual MLP with one skip connection to predict occupancy
# (and RGB color for the color head).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def normalize_3d_coordinate(p, bound):
    """Normalize coordinate to [-1, 1], corresponds to the bounding box given."""
    p = p.clone().reshape(-1, 3)
    p[:, 0] = ((p[:, 0] - bound[0, 0]) / (bound[0, 1] - bound[0, 0])) * 2 - 1.0
    p[:, 1] = ((p[:, 1] - bound[1, 0]) / (bound[1, 1] - bound[1, 0])) * 2 - 1.0
    p[:, 2] = ((p[:, 2] - bound[2, 0]) / (bound[2, 1] - bound[2, 0])) * 2 - 1.0
    return p


class GaussianFourierFeatureTransform(nn.Module):
    """Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low
    Dimensional Domains": https://arxiv.org/abs/2006.10739
    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()
        if learnable:
            self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale)
        else:
            self.register_buffer("_B", torch.randn((num_input_channels, mapping_size)) * scale)

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, "Expected 2D input (got {}D input)".format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(nn.Module):
    """Nerf positional embedding."""

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires - 1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, "Expected 2D input (got {}D input)".format(x.dim())
        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        return torch.cat(output, dim=1)


class DenseLayer(nn.Linear):
    def __init__(self, in_dim, out_dim, activation="relu", *args, **kwargs):
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Same(nn.Module):
    def forward(self, x):
        return x.squeeze(0)


class MLP(nn.Module):
    """Decoder. Point coordinates used both to sample the feature grids and as MLP input."""

    def __init__(
        self,
        name="",
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        leaky=False,
        sample_mode="bilinear",
        color=False,
        skips=[2],
        grid_len=0.16,
        pos_embedding_method="fourier",
        concat_feature=False,
        bound=None,
    ):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips
        self.register_buffer(
            "bound", bound if bound is not None else torch.tensor([[-1.0, 1.0]] * 3)
        )

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])

        if pos_embedding_method == "fourier":
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25
            )
        elif pos_embedding_method == "same":
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == "nerf":
            if "color" in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(multires, log_sampling=False)
            embedding_size = multires * 6 + 3
        elif pos_embedding_method == "fc_relu":
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation="relu")

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")]
            + [
                DenseLayer(hidden_size, hidden_size, activation="relu")
                if i not in self.skips
                else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu")
                for i in range(n_blocks - 1)
            ]
        )

        if self.color:
            self.output_linear = DenseLayer(hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(hidden_size, 1, activation="linear")

        self.actvn = F.relu if not leaky else (lambda x: F.leaky_relu(x, 0.2))
        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = F.grid_sample(
            c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode
        )
        return c.squeeze(-1).squeeze(-1)

    def forward(self, p, c_grid=None):
        if self.c_dim != 0:
            c = self.sample_grid_feature(p, c_grid["grid_" + self.name]).transpose(1, 2).squeeze(0)
            if self.concat_feature:
                with torch.no_grad():
                    c_middle = (
                        self.sample_grid_feature(p, c_grid["grid_middle"])
                        .transpose(1, 2)
                        .squeeze(0)
                    )
                c = torch.cat([c, c_middle], dim=1)

        p = p.float()
        embedded_pts = self.embedder(p)
        h = embedded_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class MLP_no_xyz(nn.Module):
    """Decoder. Point coordinates only used to sample the feature grids, not as MLP input."""

    def __init__(
        self,
        name="",
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        leaky=False,
        sample_mode="bilinear",
        color=False,
        skips=[2],
        grid_len=0.16,
        bound=None,
    ):
        super().__init__()
        self.name = name
        self.no_grad_feature = False
        self.color = color
        self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips
        self.register_buffer(
            "bound", bound if bound is not None else torch.tensor([[-1.0, 1.0]] * 3)
        )

        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")]
            + [
                DenseLayer(hidden_size, hidden_size, activation="relu")
                if i not in self.skips
                else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu")
                for i in range(n_blocks - 1)
            ]
        )

        if self.color:
            self.output_linear = DenseLayer(hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(hidden_size, 1, activation="linear")

        self.actvn = F.relu if not leaky else (lambda x: F.leaky_relu(x, 0.2))
        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, grid_feature):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = F.grid_sample(
            grid_feature, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode
        )
        return c.squeeze(-1).squeeze(-1)

    def forward(self, p, c_grid, **kwargs):
        c = self.sample_grid_feature(p, c_grid["grid_" + self.name]).transpose(1, 2).squeeze(0)
        h = c
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([c, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class NICE(nn.Module):
    """Neural Implicit Scalable Encoding -- the multi-level decoder of NICE-SLAM."""

    def __init__(
        self,
        dim=3,
        c_dim=32,
        coarse_grid_len=2.0,
        middle_grid_len=0.16,
        fine_grid_len=0.16,
        color_grid_len=0.16,
        hidden_size=32,
        coarse=False,
        pos_embedding_method="fourier",
    ):
        super().__init__()
        if coarse:
            self.coarse_decoder = MLP_no_xyz(
                name="coarse",
                dim=dim,
                c_dim=c_dim,
                color=False,
                hidden_size=hidden_size,
                grid_len=coarse_grid_len,
            )

        self.middle_decoder = MLP(
            name="middle",
            dim=dim,
            c_dim=c_dim,
            color=False,
            skips=[2],
            n_blocks=5,
            hidden_size=hidden_size,
            grid_len=middle_grid_len,
            pos_embedding_method=pos_embedding_method,
        )
        self.fine_decoder = MLP(
            name="fine",
            dim=dim,
            c_dim=c_dim * 2,
            color=False,
            skips=[2],
            n_blocks=5,
            hidden_size=hidden_size,
            grid_len=fine_grid_len,
            concat_feature=True,
            pos_embedding_method=pos_embedding_method,
        )
        self.color_decoder = MLP(
            name="color",
            dim=dim,
            c_dim=c_dim,
            color=True,
            skips=[2],
            n_blocks=5,
            hidden_size=hidden_size,
            grid_len=color_grid_len,
            pos_embedding_method=pos_embedding_method,
        )

    def forward(self, p, c_grid, stage="middle", **kwargs):
        """Output occupancy/color in different stage."""
        device = p.device
        if stage == "coarse":
            occ = self.coarse_decoder(p, c_grid)
            occ = occ.squeeze(0)
            raw = torch.zeros(occ.shape[0], 4, device=device).float()
            raw[..., -1] = occ
            return raw
        elif stage == "middle":
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw = torch.zeros(middle_occ.shape[0], 4, device=device).float()
            raw[..., -1] = middle_occ
            return raw
        elif stage == "fine":
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(fine_occ.shape[0], 4, device=device).float()
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ
            return raw
        elif stage == "color":
            fine_occ = self.fine_decoder(p, c_grid)
            raw = self.color_decoder(p, c_grid)
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ
            return raw


class NICEStaged(nn.Module):
    """Staging wrapper: bundles a NICE decoder with its fixed-size input feature
    grids so the whole thing is a single-call nn.Module (matches how NICE_SLAM.py
    drives the decoder with per-stage feature-grid dicts produce by the mapper).
    """

    def __init__(self, dim=3, c_dim=8, hidden_size=16, grid_res=8, stage="middle"):
        super().__init__()
        self.stage = stage
        self.nice = NICE(
            dim=dim,
            c_dim=c_dim,
            coarse_grid_len=2.0,
            middle_grid_len=0.32,
            fine_grid_len=0.16,
            color_grid_len=0.16,
            hidden_size=hidden_size,
            coarse=False,
            pos_embedding_method="fourier",
        )
        self.c_dim = c_dim
        self.grid_res = grid_res
        self.register_buffer("grid_middle", torch.randn(1, c_dim, grid_res, grid_res, grid_res))
        self.register_buffer("grid_fine", torch.randn(1, c_dim * 2, grid_res, grid_res, grid_res))
        self.register_buffer("grid_color", torch.randn(1, c_dim, grid_res, grid_res, grid_res))

    def forward(self, p):
        c_grid = {
            "grid_middle": self.grid_middle,
            "grid_fine": self.grid_fine,
            "grid_color": self.grid_color,
        }
        return self.nice(p, c_grid, stage=self.stage)


def build_nice_slam():
    return NICEStaged()


def example_input_nice_slam():
    # [n_pts, 3] query coordinates within the [-1, 1] default bound.
    p = torch.rand(64, 3) * 2 - 1
    return (p,)


MENAGERIE_ENTRIES = [
    ("NICE-SLAM", build_nice_slam, example_input_nice_slam, 2022, "vendored-pytorch"),
]

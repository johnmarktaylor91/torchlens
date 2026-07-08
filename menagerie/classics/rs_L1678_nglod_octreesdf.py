# SOURCE: vendored from nv-tlabs/nglod @ main (sdf-net/lib/models/{OctreeSDF,BaseLOD,BaseSDF,Embedder}.py)
# https://github.com/nv-tlabs/nglod -- NGLOD ("Neural Geometric Level of Detail: Real-time
# Rendering with Implicit 3D Shapes", Takikawa et al., CVPR 2021 oral). `OctreeSDF` is the
# repo's real multi-LOD implicit-surface model: a sparse octree feature volume queried by
# trilinear `grid_sample` at each level-of-detail, summed across levels, and decoded to a
# signed distance by small per-LOD MLP heads. Classes below (`FeatureVolume`, `OctreeSDF`,
# and the `BaseLOD`/`BaseSDF` base classes plus the `positional_encoding` helper `BaseSDF`
# depends on, and the `setparam` args-bridge helper) are transcribed verbatim from the real
# repo files; only the relative `lib.*` imports were flattened into this single file (the
# repo's `PerfTimer` import in OctreeSDF.py is unused profiling scaffolding and is dropped).
# The `--num-lods/--feature-dim/--feature-size/--hidden-dim/--base-lod/...` CLI defaults
# from `lib/options.py` are reproduced in a plain namespace below (this repo builds models
# from an argparse Namespace `args`, not kwargs, so the staging build function constructs
# that same Namespace with the same defaults, just at a smaller `feature_size`/`hidden_dim`
# for fast tracing).
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from lib/utils.py (args/kwargs bridge used throughout the real models) ----
def setparam(args, param, paramstr):
    argsparam = getattr(args, paramstr, None)
    if param is not None or argsparam is None:
        return param
    else:
        return argsparam


# ---- verbatim from lib/models/Embedder.py (originally from krrish94/nerf-pytorch) ----
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


# ---- verbatim from lib/models/BaseSDF.py ----
class BaseSDF(nn.Module):
    def __init__(
        self,
        args=None,
        pos_enc: bool = None,
        ff_dim: int = None,
        ff_width: float = None,
    ):
        super().__init__()
        self.args = args
        self.pos_enc = setparam(args, pos_enc, "pos_enc")
        self.ff_dim = setparam(args, ff_dim, "ff_dim")
        self.ff_width = setparam(args, ff_width, "ff_width")

        self.input_dim = 3
        self.out_dim = 1

        if self.ff_dim > 0:
            mat = torch.randn([self.ff_dim, 3]) * self.ff_width
            self.gauss_matrix = nn.Parameter(mat)
            self.gauss_matrix.requires_grad_(False)
            self.input_dim += (self.ff_dim * 2) - 3
        elif self.pos_enc:
            self.input_dim = self.input_dim * 13

    def forward(self, x, lod=None):
        x = self.encode(x)
        return self.sdf(x)

    def freeze(self):
        for k, v in self.named_parameters():
            v.requires_grad_(False)

    def encode(self, x):
        if self.ff_dim > 0:
            x = F.linear(x, self.gauss_matrix)
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        elif self.pos_enc:
            x = positional_encoding(x)
        return x

    def sdf(self, x, lod=None):
        return None


# ---- verbatim from lib/models/BaseLOD.py ----
class BaseLOD(BaseSDF):
    def __init__(self, args):
        super().__init__(args)
        self.num_lods = args.num_lods
        self.lod = None

    def forward(self, x, lod=None):
        if lod is None:
            lod = self.lod
        x = self.encode(x)
        return self.sdf(x)

    def sdf(self, x, lod=None):
        if lod is None:
            lod = self.lod
        return None


# ---- verbatim from lib/models/OctreeSDF.py ----
class FeatureVolume(nn.Module):
    def __init__(self, fdim, fsize):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fm = nn.Parameter(torch.randn(1, fdim, fsize + 1, fsize + 1, fsize + 1) * 0.01)
        self.sparse = None

    def forward(self, x):
        N = x.shape[0]
        if x.shape[1] == 3:
            sample_coords = x.reshape(1, N, 1, 1, 3)  # [N, 1, 1, 3]
            sample = F.grid_sample(
                self.fm, sample_coords, align_corners=True, padding_mode="border"
            )[0, :, :, 0, 0].transpose(0, 1)
        else:
            sample_coords = x.reshape(1, N, x.shape[1], 1, 3)  # [N, 1, 1, 3]
            sample = F.grid_sample(
                self.fm, sample_coords, align_corners=True, padding_mode="border"
            )[0, :, :, :, 0].permute([1, 2, 0])

        return sample


class OctreeSDF(BaseLOD):
    def __init__(self, args, init=None):
        super().__init__(args)

        self.fdim = self.args.feature_dim
        self.fsize = self.args.feature_size
        self.hidden_dim = self.args.hidden_dim
        self.pos_invariant = self.args.pos_invariant

        self.features = nn.ModuleList([])
        for i in range(self.args.num_lods):
            self.features.append(FeatureVolume(self.fdim, (2 ** (i + self.args.base_lod))))
        self.interpolate = self.args.interpolate

        self.louts = nn.ModuleList([])

        self.sdf_input_dim = self.fdim
        if not self.pos_invariant:
            self.sdf_input_dim += self.input_dim

        self.num_decoder = 1 if args.joint_decoder else self.args.num_lods

        for i in range(self.num_decoder):
            self.louts.append(
                nn.Sequential(
                    nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1, bias=True),
                )
            )

    def encode(self, x):
        # Disable encoding
        return x

    def sdf(self, x, lod=None, return_lst=False):
        if lod is None:
            lod = self.lod

        # Query
        l = []  # noqa: E741 (verbatim name from the real repo's OctreeSDF.sdf)
        samples = []

        for i in range(self.num_lods):
            # Query features
            sample = self.features[i](x)
            samples.append(sample)

            # Sum queried features
            if i > 0:
                samples[i] = samples[i] + samples[i - 1]

            # Concatenate xyz
            ex_sample = samples[i]
            if not self.pos_invariant:
                ex_sample = torch.cat([x, ex_sample], dim=-1)

            if self.num_decoder == 1:
                prev_decoder = self.louts[0]
                curr_decoder = self.louts[0]
            else:
                prev_decoder = self.louts[i - 1]
                curr_decoder = self.louts[i]

            d = curr_decoder(ex_sample)

            # Interpolation mode
            if self.interpolate is not None and lod is not None:
                if i == len(self.louts) - 1:
                    return d

                if lod + 1 == i:
                    _ex_sample = samples[i - 1]
                    if not self.pos_invariant:
                        _ex_sample = torch.cat([x, _ex_sample], dim=-1)
                    _d = prev_decoder(_ex_sample)

                    return (1.0 - self.interpolate) * _d + self.interpolate * d

            # Get distance
            else:
                d = curr_decoder(ex_sample)

                # Return distance if in prediction mode
                if lod is not None and lod == i:
                    return d

                l.append(d)
        if self.training:
            self.loss_preds = l

        if return_lst:
            return l
        else:
            return l[-1]


# ---- staging build/example helpers ----
def _make_args(**overrides):
    # Defaults mirror sdf-net/lib/options.py (net_group), shrunk (feature_size, hidden_dim)
    # for fast tracing; architecture/control-flow are unchanged.
    defaults = dict(
        num_lods=2,
        base_lod=2,
        feature_dim=8,
        feature_size=2,
        hidden_dim=16,
        pos_invariant=False,
        interpolate=None,
        joint_decoder=False,
        pos_enc=False,
        ff_dim=-1,
        ff_width=16.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def build_nglod_octreesdf():
    torch.manual_seed(0)
    args = _make_args()
    model = OctreeSDF(args)
    model.lod = None  # predict at finest LOD (returns l[-1])
    return model


def example_input_nglod_octreesdf():
    torch.manual_seed(0)
    # Real repo queries the SDF at batches of 3D points in [-1, 1]^3 (grid_sample coords).
    return (torch.rand(64, 3) * 2.0 - 1.0,)


MENAGERIE_ENTRIES = [
    (
        "NGLOD-OctreeSDF",
        build_nglod_octreesdf,
        example_input_nglod_octreesdf,
        2021,
        "vendored-pytorch",
    ),
]

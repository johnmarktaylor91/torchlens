# FAITHFUL REIMPLEMENTATION from Piergiovanni, Casser, Ryoo, Angelova,
# "4D-Net for Learned Multi-Modal Alignment" (ICCV 2021, arXiv:2109.01066) (no public code)
#
# Provenance check performed before reimplementing (source-ladder rungs 1-3 all failed):
#   - The paper explicitly says "We will open source the code," but no official release
#     ever appeared under google-research/google-research or the authors' GitHub accounts.
#   - The only community repo found, chanlilong/4D_NET_pytorch, explicitly disclaims the
#     paper's actual architecture: its README states "NO TEMPORAL ELEMENT (RGB+LiDAR only,
#     no Time)" and "I may have missed out some stuff from the paper" -- i.e. it drops
#     exactly the "4D" (3D-space + time) dynamic-connection mechanism that IS the paper's
#     stated contribution, so vendoring/porting it would misrepresent the architecture.
#   - The ICCV paper (and its "Dynamic Connections" section, quoted below) gives a
#     detailed, equation-level description of the one genuinely novel mechanism, so this
#     is reimplemented faithfully from that description rather than approximated.
#
# Faithfully reimplemented mechanism (paper Sec. 3.2 "Dynamic Connections", direct quotes):
#   "Given a set of RGB feature maps, {R_i | i in [0,1,...,B]} (B being the total number of
#    blocks/feature maps in the RGB network), we can compute the projection of each pillar
#    into the 2D space and obtain a feature vector. This produces a set of feature vectors
#    F = {f_i | i in [0,1,...,B]}. We then have a learned weight w, which is a B-dimensional
#    vector. We apply softmax and then compute w x F to obtain the final feature vector...
#    This is done after each block in the Point-Pillars network, allowing many connections
#    to be learned."
#   "...we modified the connections to be dynamic ... we replace w with a linear layer with
#    B outputs, omega, which is applied to the PointPillar feature M_i[p_x,p_y] and generates
#    weights over the B RGB feature maps. omega is followed by a softmax activation function.
#    This allows the network to dynamically select which RGB block to fuse information from
#    ... Since this is done for each pillar individually, the network can learn how and where
#    to select these features based on the input."
#   Sec 3.1: the point-cloud stream is featurized with PointPillars ("we chose to use
#    PointPillars to generate these features, but other 3D point 'featurising' approaches
#    can be used"); both the point-cloud stream and the RGB stream are processed "in time"
#    (a sequence of point clouds / images), and Sec 3.3 states the dynamic connections
#    "allow for combining multiple computational towers seamlessly" -- i.e. the same
#    per-pillar dynamic-softmax-gathering mechanism is what is extended across time frames
#    to obtain the full "4D" (3D + time) alignment, which is what is reimplemented below:
#    the omega/softmax gather draws candidates from EVERY (RGB block, time frame) pair, not
#    just the current frame's blocks.
#
# Simplifications made explicit (backbones are standard, cited, non-novel components; the
# camera projection needs real calibration matrices that don't exist for a synthetic demo):
#   - PointPillars pillar featurization is the well-established Lang et al. 2019 recipe
#     (per-point MLP on point + pillar-relative offsets, then per-pillar max-pool scatter
#     onto a BEV pseudo-image), shrunk to a tiny width/grid.
#   - The RGB tower is a small plain conv-BN-ReLU stack standing in for the paper's
#     (unspecified in detail) RGB CNN backbone.
#   - Real 4D-Net projects each pillar into the image plane via the sensor's camera/LiDAR
#     calibration; without real calibration data we stand in with a resolution-matching
#     interpolation (this is the ONLY substitution for a component the paper treats as an
#     external geometric input, not a learned part of the architecture -- the dynamic
#     connection weighting/gathering mechanism itself, which IS the learned contribution,
#     is implemented exactly as described above).
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


class PillarFeatureNetLite(nn.Module):
    """Simplified PointPillars-style point-cloud pillar featurizer (Lang et al. 2019),
    which 4D-Net Sec. 3.1 explicitly names as its chosen 3D point featurizer: per-point
    MLP on (point features + pillar-relative offsets), then per-pillar max-pool scatter
    onto a BEV pseudo-image."""

    def __init__(self, in_feats=4, feat_dim=16, grid_size=(8, 8), xy_range=(-4.0, 4.0)):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.xy_range = xy_range
        self.feat_dim = feat_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(in_feats + 2, feat_dim),  # +2 pillar-relative (dx, dy) offsets
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points):
        # points: (B, N, in_feats) with points[..., 0:2] == (x, y) -> (B, feat_dim, grid_h, grid_w)
        B, N, _ = points.shape
        lo, hi = self.xy_range
        cell = (hi - lo) / self.grid_h
        xy = points[..., :2].clamp(lo, hi - 1e-4)
        gx = ((xy[..., 0] - lo) / cell).long().clamp(0, self.grid_w - 1)
        gy = ((xy[..., 1] - lo) / cell).long().clamp(0, self.grid_h - 1)
        pillar_idx = gy * self.grid_w + gx  # (B, N)
        cx = lo + (gx.float() + 0.5) * cell
        cy = lo + (gy.float() + 0.5) * cell
        offsets = torch.stack([xy[..., 0] - cx, xy[..., 1] - cy], dim=-1)  # (B, N, 2)
        point_in = torch.cat([points, offsets], dim=-1)  # (B, N, in_feats + 2)

        out_grids = []
        for b in range(B):
            feats = self.point_mlp(point_in[b])  # (N, feat_dim)
            grid = torch.zeros(
                self.grid_h * self.grid_w, self.feat_dim, device=points.device, dtype=feats.dtype
            )
            idx = pillar_idx[b].unsqueeze(-1).expand(-1, self.feat_dim)
            grid = grid.scatter_reduce(0, idx, feats, reduce="amax", include_self=False)
            out_grids.append(grid)
        grid = torch.stack(out_grids, dim=0)  # (B, grid_h * grid_w, feat_dim)
        grid = (
            grid.view(B, self.grid_h, self.grid_w, self.feat_dim).permute(0, 3, 1, 2).contiguous()
        )
        return grid


class ConvBlock(nn.Module):
    """One stride-2 conv-BN-ReLU stage of either tower's backbone."""

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DynamicConnection(nn.Module):
    """4D-Net's core novel mechanism (paper Sec. 3.2, quoted in the module header above):
    a per-pillar linear layer `omega` maps the current PointPillars feature to softmax
    weights over a bank of candidate feature maps, which are gathered (here: resolution-
    matched via interpolation, standing in for the real geometric pillar->image projection)
    and fused back into the PointPillars feature. The candidate bank spans every
    (RGB block, time frame) pair, which is what makes the connections "4D" rather than
    just multi-modal."""

    def __init__(self, pc_channels, candidate_channels):
        super().__init__()
        self.num_candidates = len(candidate_channels)
        self.omega = nn.Conv2d(pc_channels, self.num_candidates, kernel_size=1)
        self.proj = nn.ModuleList(
            [nn.Conv2d(c, pc_channels, kernel_size=1) for c in candidate_channels]
        )

    def forward(self, pc_feat, candidates):
        # pc_feat: (B, C, H, W); candidates: list of (B, C_k, H_k, W_k), len == num_candidates
        weights = torch.softmax(self.omega(pc_feat), dim=1)  # (B, num_candidates, H, W)
        out_hw = pc_feat.shape[2:]
        fused = torch.zeros_like(pc_feat)
        for k, cand in enumerate(candidates):
            resized = F.interpolate(cand, size=out_hw, mode="bilinear", align_corners=False)
            projected = self.proj[k](resized)  # (B, C, H, W)
            fused = fused + weights[:, k : k + 1] * projected
        return pc_feat + fused


class FourDNetLite(nn.Module):
    """4D-Net: fuses a point-cloud stream (PointPillars-featurized) and an RGB stream,
    both observed over T time frames, via dynamic connections applied after every
    PointPillars block, gathering from every (RGB block, time frame) candidate -- the
    paper's 3D-object-detection-over-time formulation, reduced to a small single-scale
    detection head for a fast, self-contained trace."""

    def __init__(
        self,
        num_time=2,
        grid_size=(8, 8),
        pillar_feat_dim=16,
        pc_channels=(16, 32),
        rgb_channels=(16, 32),
        num_classes=3,
    ):
        super().__init__()
        self.num_time = num_time
        self.pillar_net = PillarFeatureNetLite(
            in_feats=4, feat_dim=pillar_feat_dim, grid_size=grid_size
        )

        self.rgb_blocks = nn.ModuleList()
        in_ch = 3
        for out_ch in rgb_channels:
            self.rgb_blocks.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch

        self.pc_blocks = nn.ModuleList()
        in_ch = pillar_feat_dim
        for out_ch in pc_channels:
            self.pc_blocks.append(ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch

        # candidate bank = every (RGB block, time frame) pair -- the "4D" (space+time) part
        candidate_channels = [c for _ in range(num_time) for c in rgb_channels]
        self.dynamic_connections = nn.ModuleList(
            [DynamicConnection(pc_channels[i], candidate_channels) for i in range(len(pc_channels))]
        )

        self.head = nn.Conv2d(pc_channels[-1], num_classes, kernel_size=1)

    def forward(self, point_clouds, images):
        # point_clouds: (B, T, N, 4); images: (B, T, 3, H, W)
        T = point_clouds.shape[1]

        rgb_feats_per_t = []
        for t in range(T):
            x = images[:, t]
            feats = []
            for block in self.rgb_blocks:
                x = block(x)
                feats.append(x)
            rgb_feats_per_t.append(feats)

        # flatten the candidate pool across all (time, block) pairs
        candidates = [rgb_feats_per_t[t][i] for t in range(T) for i in range(len(self.rgb_blocks))]

        final_feats = []
        for t in range(T):
            x = self.pillar_net(point_clouds[:, t])
            for i, block in enumerate(self.pc_blocks):
                x = block(x)
                x = self.dynamic_connections[i](x, candidates)
            final_feats.append(x)

        agg = torch.stack(final_feats, dim=1).mean(dim=1)
        return self.head(agg)


# ---------------------------------------------------------------------------
# menagerie staging glue
# ---------------------------------------------------------------------------
def build_4dnet() -> nn.Module:
    return FourDNetLite(
        num_time=2,
        grid_size=(8, 8),
        pillar_feat_dim=16,
        pc_channels=(16, 32),
        rgb_channels=(16, 32),
        num_classes=3,
    )


def example_input_4dnet():
    """Real model needs TWO tensors -- a point-cloud sequence (B,T,N,4) and an RGB image
    sequence (B,T,3,H,W) -- fused via the dynamic-connection mechanism in forward()."""
    point_clouds = torch.rand(1, 2, 64, 4) * 8.0 - 4.0  # (x,y,z,intensity) in [-4,4]
    images = torch.randn(1, 2, 3, 16, 16)
    return [point_clouds, images]


MENAGERIE_ENTRIES = [
    (
        "4D-Net",
        "build_4dnet",
        "example_input_4dnet",
        2021,
        MENAGERIE_ZOO,
    ),
]

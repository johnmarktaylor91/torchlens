# SOURCE: vendored from city-super/BungeeNeRF @ main (run_nerf_helpers.py)
#
# Bungee_NeRF_block is BungeeNeRF's progressive multi-resolution NeRF radiance MLP: a base
# block plus a stack of residual blocks, each producing its own (rgb, alpha) pair that is
# stacked across "stages" -- the multi-scale growing mechanism from the paper. Vendored
# verbatim from run_nerf_helpers.py; only the unrelated ray/sampling helper functions in the
# same source file were dropped (out of scope for a single nn.Module), imports trimmed to
# what this class needs. No architectural changes.
import torch
import torch.nn.functional as F
from torch import nn


class Bungee_NeRF_baseblock(nn.Module):
    def __init__(self, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_baseblock, self).__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, net_width)] + [nn.Linear(net_width, net_width) for _ in range(3)]
        )
        self.views_linear = nn.Linear(input_ch_views + net_width, net_width // 2)
        self.feature_linear = nn.Linear(net_width, net_width)
        self.alpha_linear = nn.Linear(net_width, 1)
        self.rgb_linear = nn.Linear(net_width // 2, 3)

    def forward(self, input_pts, input_views):
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = torch.cat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = F.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h


class Bungee_NeRF_resblock(nn.Module):
    def __init__(self, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_resblock, self).__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + net_width, net_width), nn.Linear(net_width, net_width)]
        )
        self.views_linear = nn.Linear(input_ch_views + net_width, net_width // 2)
        self.feature_linear = nn.Linear(net_width, net_width)
        self.alpha_linear = nn.Linear(net_width, 1)
        self.rgb_linear = nn.Linear(net_width // 2, 3)

    def forward(self, input_pts, input_views, h):
        h = torch.cat([input_pts, h], -1)
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = torch.cat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = F.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h


class Bungee_NeRF_block(nn.Module):
    def __init__(self, num_resblocks=3, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_block, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.num_resblocks = num_resblocks

        self.baseblock = Bungee_NeRF_baseblock(
            net_width=net_width, input_ch=input_ch, input_ch_views=input_ch_views
        )
        self.resblocks = nn.ModuleList(
            [
                Bungee_NeRF_resblock(
                    net_width=net_width, input_ch=input_ch, input_ch_views=input_ch_views
                )
                for _ in range(num_resblocks)
            ]
        )

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        alphas = []
        rgbs = []
        base_rgb, base_alpha, h = self.baseblock(input_pts, input_views)
        alphas.append(base_alpha)
        rgbs.append(base_rgb)
        for i in range(self.num_resblocks):
            res_rgb, res_alpha, h = self.resblocks[i](input_pts, input_views, h)
            alphas.append(res_alpha)
            rgbs.append(res_rgb)

        output = torch.cat([torch.stack(rgbs, 1), torch.stack(alphas, 1)], -1)
        return output


MENAGERIE_ZOO = "vendored-pytorch"


def build_bungee_nerf_block():
    return Bungee_NeRF_block(num_resblocks=3, net_width=32, input_ch=63, input_ch_views=27)


def example_input_bungee_nerf_block():
    # x layout expected by Bungee_NeRF_block.forward: [pts(input_ch) | views(input_ch_views)]
    return torch.rand(8, 63 + 27)


MENAGERIE_ENTRIES = [
    (
        "bungee_nerf_block",
        "build_bungee_nerf_block",
        "example_input_bungee_nerf_block",
        2022,
        "vendored-pytorch",
    ),
]

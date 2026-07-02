# SOURCE: vendored from albertpumarola/D-NeRF @ master
# https://raw.githubusercontent.com/albertpumarola/D-NeRF/master/run_dnerf_helpers.py
#
# D-NeRF: Neural Radiance Fields for Dynamic Scenes (Pumarola et al., CVPR 2021).
# The classes below (`NeRFOriginal`, `DirectTemporalNeRF`) are the real model
# definitions from the official repo's `run_dnerf_helpers.py`, copied verbatim
# (only import/positional-encoding wiring trimmed for standalone use -- no
# architecture changes). `DirectTemporalNeRF` is the actual D-NeRF contribution:
# a canonical NeRF MLP (`NeRFOriginal`) plus a temporal deformation MLP
# (`_time`/`_time_out`) that warps query points into the canonical frame before
# the canonical NeRF is queried.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- Positional encoding (verbatim from run_dnerf_helpers.py) ---
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


# --- Canonical NeRF MLP (verbatim from run_dnerf_helpers.py) ---
class NeRFOriginal(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        input_ch_time=1,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        memory=[],
        embed_fn=None,
        output_color_ch=3,
        zero_canonical=True,
    ):
        super(NeRFOriginal, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in self.skips:
                in_channels += input_ch

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, output_color_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, _layer in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, _layer in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, torch.zeros_like(input_pts[:, :3])


# --- D-NeRF: canonical NeRF + temporal deformation MLP (verbatim) ---
class DirectTemporalNeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        input_ch_time=1,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        memory=[],
        embed_fn=None,
        zero_canonical=True,
    ):
        super(DirectTemporalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical

        self._occ = NeRFOriginal(
            D=D,
            W=W,
            input_ch=input_ch,
            input_ch_views=input_ch_views,
            input_ch_time=input_ch_time,
            output_ch=output_ch,
            skips=skips,
            use_viewdirs=use_viewdirs,
            memory=memory,
            embed_fn=embed_fn,
            output_color_ch=3,
        )
        self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, _layer in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        t = ts[0]

        assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = t[0, 0]
        if cur_time == 0.0 and self.zero_canonical:
            dx = torch.zeros_like(input_pts[:, :3])
        else:
            dx = self.query_time(input_pts, t, self._time, self._time_out)
            input_pts_orig = input_pts[:, :3]
            input_pts = self.embed_fn(input_pts_orig + dx)
        out, _ = self._occ(torch.cat([input_pts, input_views], dim=-1), t)
        return out, dx


# --- Staging build/example-input wiring (tiny sizes; real architecture unmodified) ---
_MULTIRES_XYZ = 4  # real default: 10
_MULTIRES_TIME = 2  # real default: 10
_D = 3  # real default: 8
_W = 32  # real default: 256


def build_dnerf():
    embed_fn, input_ch = get_embedder(_MULTIRES_XYZ, input_dims=3, i=0)
    _, input_ch_time = get_embedder(_MULTIRES_TIME, input_dims=1, i=0)
    model = DirectTemporalNeRF(
        D=_D,
        W=_W,
        input_ch=input_ch,
        input_ch_views=input_ch,
        input_ch_time=input_ch_time,
        output_ch=4,
        skips=[1],
        use_viewdirs=True,
        embed_fn=embed_fn,
        zero_canonical=False,
    )
    model.eval()
    return model


def example_input_dnerf():
    # x packs [encoded xyz | encoded viewdirs] along the last dim; ts packs the
    # (broadcast) encoded time embedding matching input_pts' leading dim, wrapped
    # in a length-1 outer list the way the real `render_rays` call site does
    # (`ts=[embedded_time]`).
    embed_fn, input_ch = get_embedder(_MULTIRES_XYZ, input_dims=3, i=0)
    embed_time_fn, input_ch_time = get_embedder(_MULTIRES_TIME, input_dims=1, i=0)

    n_pts = 8
    raw_pts = torch.randn(n_pts, 3)
    raw_views = torch.randn(n_pts, 3)
    raw_time = torch.full((n_pts, 1), 0.5)

    x = torch.cat([embed_fn(raw_pts), embed_fn(raw_views)], dim=-1)
    t_embedded = embed_time_fn(raw_time)
    ts = [t_embedded]
    return (x, ts)


MENAGERIE_ENTRIES = [
    ("D-NeRF", "build_dnerf", "example_input_dnerf", 2021, "vendored-pytorch"),
]

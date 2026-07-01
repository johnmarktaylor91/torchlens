# SOURCE: vendored from InhwanBae/NPSN @ main
# (npsn/model.py + npsn/utils.py)
"""NPSN: Non-Probability Sampling Network (CVPR 2022).

Vendored real nn.Module code. A plug-in quasi-Monte-Carlo sampling-location
predictor for trajectory-prediction CVAEs: a graph-attention (GAT) encoder
over pedestrian observed-trajectory nodes followed by an MLP head that
predicts `n` sampling locations (as sigmoid-squashed unit-square points that
get Box-Muller-transformed into Gaussian sample locations downstream) per
agent, used to replace naive Monte-Carlo sampling in models like Social-STGCNN
/ PECNet / SGCN with a learned low-discrepancy sampler.
"""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- npsn/utils.py (real repo code, verbatim) ------------------------------


def box_muller_transform(x: torch.FloatTensor):
    r"""Box-Muller transform"""
    shape = x.shape
    x = x.view(shape[:-1] + (-1, 2))
    z = torch.zeros_like(x, device=x.device)
    z[..., 0] = (-2 * x[..., 0].log()).sqrt() * (2 * torch.pi * x[..., 1]).cos()
    z[..., 1] = (-2 * x[..., 0].log()).sqrt() * (2 * torch.pi * x[..., 1]).sin()
    return z.view(shape)


# ---- npsn/model.py (real repo code; `get_scene_mask` in the original     --
# ---- hardcodes `device='cuda'` for a mask buffer that's only used by the --
# ---- `seq_start_end`-driven multi-scene batching path, which this        --
# ---- staging module doesn't exercise (default `mask=None` forward path,  --
# ---- exactly as the model's own `if __name__ == '__main__'` smoke test   --
# ---- in the repo uses). Fixed to build on the input tensor's device so   --
# ---- the vendored module is CPU-runnable, same class of fix as MultiPath++
# ---- and OccWorld above.) ---------------------------------------------------


class GAT(nn.Module):
    def __init__(self, in_feat=2, out_feat=64, n_head=4, dropout=0.1, skip=True):
        super(GAT, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.n_head = n_head
        self.skip = skip
        self.w = nn.Parameter(torch.Tensor(n_head, in_feat, out_feat))
        self.a_src = nn.Parameter(torch.Tensor(n_head, out_feat, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, out_feat, 1))
        self.bias = nn.Parameter(torch.Tensor(out_feat))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        nn.init.constant_(self.bias, 0)

    def forward(self, h, mask=None):
        h_prime = h.unsqueeze(1) @ self.w
        attn_src = h_prime @ self.a_src
        attn_dst = h_prime @ self.a_dst
        attn = attn_src @ attn_dst.permute(0, 1, 3, 2)
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        attn = attn * mask if mask is not None else attn
        out = (attn @ h_prime).sum(dim=1) + self.bias
        if self.skip:
            out += h_prime.sum(dim=1)
        return out, attn


class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
        super(MLP, self).__init__()
        dims = (in_feat,) + hid_feat + (out_feat,)

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = activation if activation is not None else lambda x: x
        self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layers[i](x)
        return x


class NPSN(nn.Module):
    def __init__(self, t_obs=8, s=2, n=20):
        super(NPSN, self).__init__()
        self.s, self.n = s, n
        self.input_dim = t_obs * 2
        self.hidden_dim = self.input_dim * 1
        self.output_dim = s * n

        self.graph_attention = GAT(self.input_dim, self.hidden_dim)
        self.linear = MLP(self.hidden_dim, self.output_dim, (16, 64), activation=nn.ReLU())

    def get_scene_mask(self, peds, seq_start_end, device):
        mask = torch.zeros((peds, peds), device=device)
        for start, end in seq_start_end:
            mask[start:end, start:end] = 1
        return mask

    def forward(self, x, seq_start_end=None, mask=None, global_noise=False):
        mask = (
            self.get_scene_mask(x.size(1), seq_start_end, x.device)
            if seq_start_end is not None
            else mask
        )
        node = x.reshape(x.size(0), x.size(1), -1)
        node, edge = self.graph_attention(node, mask)

        if not global_noise:
            out = self.linear(node).reshape(x.size(0), x.size(1), self.n, -1)
        else:
            node_ = torch.zeros(
                (node.size(0), seq_start_end.size(0), node.size(2)), device=x.device
            )
            for i, (start, end) in enumerate(seq_start_end):
                node_[:, i] = node[:, start:end].mean(dim=1)
            out = self.linear(node_).reshape(x.size(0), seq_start_end.size(0), self.n, -1)
        return out[..., 0 : self.s].sigmoid().clamp(min=0.01, max=0.99)

    def get_loss(self, loc, mu, cov, gt):
        loc_norm = box_muller_transform(loc).permute(2, 0, 1, 3).expand((loc.size(2),) + mu.shape)
        p_sample = mu + (torch.linalg.cholesky(cov) @ loc_norm.unsqueeze(dim=-1)).squeeze(dim=-1)

        loss_dist = (
            (p_sample.mean(dim=1) - gt.permute(0, 3, 1, 2).mean(dim=1))
            .norm(p=2, dim=-1)
            .min(dim=0)[0]
        )
        loss_dist = loss_dist.mean()

        loss_disc = (loc.unsqueeze(dim=2) - loc.unsqueeze(dim=3)).norm(p=2, dim=-1)
        loss_disc = (
            loss_disc.topk(k=2, dim=-1, largest=False, sorted=True)[0][..., 1]
            .log()
            .mul(-1)
            .mean(dim=-1)
        )
        loss_disc = loss_disc.mean()

        return loss_dist, loss_disc


# ---- staging harness: matches the repo's own `if __name__ == '__main__'` --
# ---- smoke test in npsn/model.py: NPSN(t_obs=8, s=2, n=20) on a           --
# ---- (1, 3, 2, 8) tensor -> (num_samples, num_peds, T_obs, xy) -----------


def build_npsn():
    torch.manual_seed(0)
    return NPSN(t_obs=8, s=2, n=20)


def example_input_npsn():
    torch.manual_seed(0)
    # x: (batch, num_pedestrians, xy=2, t_obs=8) observed trajectory coordinates
    return (torch.rand(size=(1, 3, 2, 8)),)


MENAGERIE_ENTRIES = [
    ("npsn", "build_npsn", "example_input_npsn", 2022, "vendored-pytorch"),
]

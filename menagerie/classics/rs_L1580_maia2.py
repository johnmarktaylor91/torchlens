# SOURCE: vendored from https://github.com/CSSLab/maia2 @ main
# (maia2/main.py: BasicBlock, ChessResNet, FeedForward, EloAwareAttention,
#  Transformer, MAIA2Model, lines 232-418)
#
# Maia-2 (Tang et al. 2024, NeurIPS 2024, "Maia-2: A Unified Model for Human-AI
# Alignment in Chess"). Official CSSLab/maia2 repo, pip-installable as `maia2`
# (`pip install maia2`); the pip wheel's `maia2/main.py` is byte-identical to the
# GitHub `main` branch (verified 2026-07-02). The model itself has no dependency
# beyond torch + einops; the surrounding pip package additionally pulls in
# `chess`/`gdown`/`yaml` for the PGN-dataset/checkpoint-download plumbing, which
# are not needed to construct or run the architecture, so only the real
# `nn.Module` classes are vendored here (verbatim) rather than depending on the
# full package. Architecture: a ChessResNet CNN over a 8x8 board-plane tensor,
# reshaped into a length-64 patch sequence and fed through an Elo-aware
# Transformer (per-layer attention queries are shifted by an embedding of both
# players' Elo bins) to produce move logits, auxiliary "side info" logits
# (piece/capture/check/from-square/to-square/legal-move indicators), and a
# scalar value head -- the model that coherently captures human chess play
# across skill levels.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

MENAGERIE_ZOO = "vendored-pytorch"


# ---- maia2/main.py (vendored verbatim) ----
class BasicBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        mid_planes = planes

        self.conv1 = torch.nn.Conv2d(
            in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(mid_planes)
        self.conv2 = torch.nn.Conv2d(
            mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)

        return out


class ChessResNet(torch.nn.Module):
    def __init__(self, block, cfg):
        super(ChessResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            cfg.input_channels, cfg.dim_cnn, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(cfg.dim_cnn)
        self.layers = self._make_layer(block, cfg.dim_cnn, cfg.num_blocks_cnn)
        self.conv_last = torch.nn.Conv2d(
            cfg.dim_cnn, cfg.vit_length, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_last = torch.nn.BatchNorm2d(cfg.vit_length)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.bn_last(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EloAwareAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, elo_dim=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.elo_query = nn.Linear(elo_dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, elo_emb):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        elo_effect = self.elo_query(elo_emb).view(x.size(0), self.heads, 1, -1)
        q = q + elo_effect

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, elo_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.elo_layers = nn.ModuleList([])
        for _ in range(depth):
            self.elo_layers.append(
                nn.ModuleList(
                    [
                        EloAwareAttention(
                            dim, heads=heads, dim_head=dim_head, dropout=dropout, elo_dim=elo_dim
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, elo_emb):
        for attn, ff in self.elo_layers:
            x = attn(x, elo_emb) + x
            x = ff(x) + x

        return self.norm(x)


class MAIA2Model(torch.nn.Module):
    def __init__(self, output_dim, elo_dict, cfg):
        super(MAIA2Model, self).__init__()

        self.cfg = cfg
        self.chess_cnn = ChessResNet(BasicBlock, cfg)

        heads = 16
        dim_head = 64
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.transformer = Transformer(
            cfg.dim_vit,
            cfg.num_blocks_vit,
            heads,
            dim_head,
            mlp_dim=cfg.dim_vit,
            dropout=0.1,
            elo_dim=cfg.elo_dim * 2,
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))

        self.fc_1 = nn.Linear(cfg.dim_vit, output_dim)
        self.fc_2 = nn.Linear(cfg.dim_vit, output_dim + 6 + 6 + 1 + 64 + 64)
        self.fc_3 = nn.Linear(128, 1)
        self.fc_3_1 = nn.Linear(cfg.dim_vit, 128)

        self.elo_embedding = torch.nn.Embedding(len(elo_dict), cfg.elo_dim)

        self.dropout = nn.Dropout(p=0.1)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

    def forward(self, boards, elos_self, elos_oppo):
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.cfg.input_channels, 8, 8)
        embs = self.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.to_patch_embedding(embs)
        x = x + self.pos_embedding
        x = self.dropout(x)

        elos_emb_self = self.elo_embedding(elos_self)
        elos_emb_oppo = self.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.transformer(x, elos_emb).mean(dim=1)

        x = self.last_ln(x)

        logits_maia = self.fc_1(x)
        logits_side_info = self.fc_2(x)
        logits_value = self.fc_3(self.dropout(torch.relu(self.fc_3_1(x)))).squeeze(dim=-1)

        return logits_maia, logits_side_info, logits_value


# ---- end vendored maia2/main.py ----


class _TinyCfg:
    """Minimal stand-in for the real `maia2.utils.Config` (YAML-driven), sized
    down for a fast trace instead of the released checkpoint's full width."""

    def __init__(self):
        self.input_channels = 18
        self.dim_cnn = 32
        self.num_blocks_cnn = 2
        self.vit_length = 8
        self.dim_vit = 32
        self.num_blocks_vit = 2
        self.elo_dim = 16


_OUTPUT_DIM = 128  # stand-in for len(get_all_possible_moves()) (real model: 1968)
_ELO_DICT_SIZE = 11  # real create_elo_dict() bin count


def build_maia2():
    cfg = _TinyCfg()
    elo_dict = {i: i for i in range(_ELO_DICT_SIZE)}
    return MAIA2Model(_OUTPUT_DIM, elo_dict, cfg)


def example_input_maia2():
    cfg = _TinyCfg()
    boards = torch.zeros(2, cfg.input_channels, 8, 8)
    elos_self = torch.zeros(2, dtype=torch.long)
    elos_oppo = torch.zeros(2, dtype=torch.long)
    return boards, elos_self, elos_oppo


MENAGERIE_ENTRIES = [
    ("Maia-2", "build_maia2", "example_input_maia2", 2024, "vendored-pytorch"),
]

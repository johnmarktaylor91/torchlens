# SOURCE: vendored from hspark1212/MOFTransformer @ master
# Files: moftransformer/modules/module.py (Module -- trimmed to the regression-downstream
#        construction path), moftransformer/modules/cgcnn.py (ConvLayer / GraphEmbeddings,
#        "Generate Embedding layers made by only convolution layers of CGCNN"),
#        moftransformer/modules/vision_transformer_3d.py (PatchEmbed3D / Attention / Mlp /
#        Block / VisionTransformer3D), moftransformer/modules/heads.py (Pooler /
#        RegressionHead), moftransformer/modules/module_utils.py (Normalizer -- unused at
#        inference but kept for fidelity).
#
# MOFTransformer (Park et al., 2023, Nat. Mach. Intell.) predicts MOF (metal-organic
# framework) properties by fusing a CGCNN-style atom/bond graph encoder with a 3D Vision
# Transformer over the MOF's energy-grid voxel representation, then running both token
# streams through a shared multi-head-attention fusion transformer (graph tokens + a class
# token + grid-patch tokens + a volume token, distinguished via 2-way token-type
# embeddings), pooling the [CLS] token, and regressing a scalar property. This is the real
# `Module` class exactly as pytorch_lightning constructs it for `loss_names={"regression": 1}`
# (the ggm/mpp/mtp/vfp/moc pretraining heads are all architecturally optional branches gated
# by `config["loss_names"][...] > 0` in the real code -- they are simply left off here by
# using the same zero-valued loss_names dict the real `test`/`example` sacred configs use for
# downstream fine-tuning). `set_metrics`/`sacred`/pytorch_lightning training-loop plumbing
# (which needs sacred + a live Trainer) is not needed for a forward pass and is omitted; the
# `Module.forward(batch)` call with `current_tasks=[]` directly triggers the real `infer()`
# graph-embed + 3D-ViT-embed + fusion-transformer + pooler pipeline used by both training and
# inference, and IS the real architecture's forward computation.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import random
from functools import partial

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from torch.nn import AvgPool3d
from timm.models.layers import DropPath, trunc_normal_

MENAGERIE_ZOO = "vendored-pytorch"


# --- moftransformer/modules/cgcnn.py (verbatim) ---
class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # [N, M, atom_fea_len]

        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)  # [N, M, atom_fea_len*2]
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len * 2)).view(
            N, M, self.atom_fea_len * 2
        )
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # [N, atom_fea_len]
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)  # [N, atom_fea_len]
        return out


class GraphEmbeddings(nn.Module):
    """
    Generate Embedding layers made by only convolution layers of CGCNN (not pooling)
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len, max_graph_len, hid_dim, n_conv=3, vis=False):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_graph_len = max_graph_len
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(119, atom_fea_len)  # 119 -> max(atomic number)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.fc = nn.Linear(atom_fea_len, hid_dim)
        self.vis = vis

    def forward(self, atom_num, nbr_idx, nbr_fea, crystal_atom_idx, uni_idx, uni_count, moc=None):
        assert self.nbr_fea_len == nbr_fea.shape[-1]
        atom_fea = self.embedding(atom_num)  # [N', atom_fea_len]
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)  # [N', atom_fea_len]
        atom_fea = self.fc(atom_fea)  # [N', hid_dim]

        new_atom_fea, mask, mo_label = self.reconstruct_batch(
            atom_fea, crystal_atom_idx, uni_idx, uni_count, moc
        )
        return new_atom_fea, mask, mo_label

    def reconstruct_batch(self, atom_fea, crystal_atom_idx, uni_idx, uni_count, moc):
        batch_size = len(crystal_atom_idx)

        new_atom_fea = torch.full(
            size=[batch_size, self.max_graph_len, self.hid_dim], fill_value=0.0
        ).to(atom_fea)
        mo_label = torch.full(size=[batch_size, self.max_graph_len], fill_value=-100.0).to(atom_fea)

        for bi, c_atom_idx in enumerate(crystal_atom_idx):
            idx_ = torch.LongTensor([random.choice(u) for u in uni_idx[bi]])[: self.max_graph_len]
            rand_idx = idx_[torch.randperm(len(idx_))]
            if self.vis:
                rand_idx = idx_
            new_atom_fea[bi][: len(rand_idx)] = atom_fea[c_atom_idx][rand_idx]

            if moc:
                mo = torch.zeros(len(c_atom_idx))
                metal_idx = moc[bi]
                mo[metal_idx] = 1
                mo_label[bi][: len(rand_idx)] = mo[rand_idx]

        mask = (new_atom_fea.sum(dim=-1) != 0).float()
        return new_atom_fea, mask, mo_label


# --- moftransformer/modules/vision_transformer_3d.py (verbatim) ---
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        assert C % self.num_heads == 0
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed3D(nn.Module):
    """Image to Patch Embedding for 3D"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans=1,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size**3) // (patch_size**3)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)",
                p1=patch_size,
                p2=patch_size,
                p3=patch_size,
            ),
            nn.Linear(patch_size * patch_size * patch_size * in_chans, embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class VisionTransformer3D(nn.Module):
    """A PyTorch impl of : `An Image is Worth 16x16 Words` -- https://arxiv.org/abs/2010.11929"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        add_norm_before_transformer=False,
        mpp_ratio=0.15,
        config=None,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.mpp_ratio = mpp_ratio

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.add_norm_before_transformer = add_norm_before_transformer

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size
        self.patch_dim = img_size // patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_tokens(self, orig_image, feats, patch_size, mpp_ratio):
        m = AvgPool3d(patch_size, patch_size)
        with torch.no_grad():
            img_patch = m(orig_image)

        labels = (img_patch.long().flatten(start_dim=2, end_dim=4)).permute(0, 2, 1).contiguous()
        probability_matrix = torch.full(labels.shape[:-1], mpp_ratio)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        return feats, labels

    def visual_embed(self, _x, max_image_len, mask_it=False):
        B, _, _, _, _ = _x.shape
        x = self.patch_embed(_x)  # [B, ph*pw*pd, embed_dim]

        if mask_it:
            x, label = self.mask_tokens(_x, x, self.patch_size, self.mpp_ratio)
            label = torch.cat(
                [torch.full((label.shape[0], 1, self.in_chans), -100).to(label), label],
                dim=1,
            )

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x += self.pos_embed
        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)

        x_mask = torch.ones(x.shape[:2]).to(x)

        if mask_it:
            return x, x_mask, label
        else:
            return x, x_mask, None


# --- moftransformer/modules/heads.py (verbatim subset used by regression downstream) ---
class Pooler(nn.Module):
    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.index = index

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RegressionHead(nn.Module):
    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        x = self.fc(x)
        return x


# --- moftransformer/modules/objectives.py (verbatim: init_weights) ---
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


# --- moftransformer/modules/module.py (Module, trimmed to the regression-downstream
#     construction path -- pytorch_lightning training-loop plumbing / sacred config /
#     set_metrics omitted since a forward pass needs none of it) ---
class Module(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.max_grid_len = config["max_grid_len"]
        self.vis = config["visualize"]

        self.graph_embeddings = GraphEmbeddings(
            atom_fea_len=config["atom_fea_len"],
            nbr_fea_len=config["nbr_fea_len"],
            max_graph_len=config["max_graph_len"],
            hid_dim=config["hid_dim"],
            vis=config["visualize"],
        )
        self.graph_embeddings.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(init_weights)

        self.transformer = VisionTransformer3D(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["hid_dim"],
            depth=config["num_layers"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            mpp_ratio=config["mpp_ratio"],
        )

        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(init_weights)

        self.volume_embeddings = nn.Linear(1, config["hid_dim"])
        self.volume_embeddings.apply(init_weights)

        self.pooler = Pooler(config["hid_dim"])
        self.pooler.apply(init_weights)

        hid_dim = config["hid_dim"]
        self.regression_head = RegressionHead(hid_dim, config["n_targets"])
        self.regression_head.apply(init_weights)
        self.mean = config["mean"]
        self.std = config["std"]

        self.current_tasks = []

    def infer(self, batch, mask_grid=False):
        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]
        nbr_idx = batch["nbr_idx"]
        nbr_fea = batch["nbr_fea"]
        crystal_atom_idx = batch["crystal_atom_idx"]
        uni_idx = batch["uni_idx"]
        uni_count = batch["uni_count"]

        grid = batch["grid"]
        volume = batch["volume"]

        moc = batch.get("moc") or batch.get("bbc")

        (graph_embeds, graph_masks, mo_labels) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            moc=moc,
        )
        cls_tokens = torch.zeros(len(crystal_atom_idx)).to(graph_embeds)
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])
        cls_mask = torch.ones(len(crystal_atom_idx), 1).to(graph_masks)

        graph_embeds = torch.cat([cls_embeds, graph_embeds], dim=1)
        graph_masks = torch.cat([cls_mask, graph_masks], dim=1)

        (grid_embeds, grid_masks, grid_labels) = self.transformer.visual_embed(
            grid, max_image_len=self.max_grid_len, mask_it=mask_grid
        )

        volume = torch.FloatTensor(volume).to(grid_embeds)
        volume_embeds = self.volume_embeddings(volume[:, None, None])
        volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)

        grid_embeds = torch.cat([grid_embeds, volume_embeds], dim=1)
        grid_masks = torch.cat([grid_masks, volume_mask], dim=1)

        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_masks, device=self.device).long()
        )
        grid_embeds = grid_embeds + self.token_type_embeddings(
            torch.ones_like(grid_masks, device=self.device).long()
        )

        co_embeds = torch.cat([graph_embeds, grid_embeds], dim=1)
        co_masks = torch.cat([graph_masks, grid_masks], dim=1)

        x = co_embeds
        for blk in self.transformer.blocks:
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        graph_feats, grid_feats = (
            x[:, : graph_embeds.shape[1]],
            x[:, graph_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        return {
            "graph_feats": graph_feats,
            "grid_feats": grid_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "graph_masks": graph_masks,
            "grid_masks": grid_masks,
            "grid_labels": grid_labels,
            "mo_labels": mo_labels,
            "cif_id": cif_id,
        }

    def forward(self, batch):
        ret = self.infer(batch)
        ret["regression_logits"] = self.regression_head(ret["cls_feats"]).squeeze(-1)
        return ret


def build_moftransformer():
    torch.manual_seed(0)
    config = {
        "max_grid_len": -1,
        "visualize": False,
        "atom_fea_len": 16,
        "nbr_fea_len": 8,
        "max_graph_len": 6,
        "hid_dim": 24,
        "img_size": 6,
        "patch_size": 2,
        "in_chans": 1,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "drop_rate": 0.0,
        "mpp_ratio": 0.15,
        "n_targets": 1,
        "mean": None,
        "std": None,
    }
    model = Module(config)
    model.eval()
    return model


def example_input_moftransformer():
    # Synthesizes a `batch` dict with the same tensor names/shapes the real repo's
    # `moftransformer.datamodules.dataset` collator produces, at menagerie-tiny scale,
    # without invoking the CIF-parsing / CGCNN feature-generation preprocessing pipeline
    # (architecture-only input construction).
    torch.manual_seed(0)
    batch_size = 2
    atoms_per_crystal = 4
    n_atoms_total = batch_size * atoms_per_crystal
    max_nbr = 3

    atom_num = torch.randint(1, 100, (n_atoms_total,), dtype=torch.long)
    nbr_idx = torch.randint(0, n_atoms_total, (n_atoms_total, max_nbr), dtype=torch.long)
    nbr_fea = torch.randn(n_atoms_total, max_nbr, 8)

    crystal_atom_idx = [
        list(range(b * atoms_per_crystal, (b + 1) * atoms_per_crystal)) for b in range(batch_size)
    ]
    # uni_idx values index locally into `atom_fea[c_atom_idx]` (i.e. 0..len(c_atom_idx)-1),
    # matching how `reconstruct_batch` uses them: `atom_fea[c_atom_idx][rand_idx]`.
    uni_idx = [[[i] for i in range(atoms_per_crystal)] for _ in range(batch_size)]
    uni_count = [[1] * atoms_per_crystal for _ in range(batch_size)]

    grid = torch.randn(batch_size, 1, 6, 6, 6)
    volume = [1.0, 1.2]

    batch = {
        "cif_id": ["mof_0", "mof_1"],
        "atom_num": atom_num,
        "nbr_idx": nbr_idx,
        "nbr_fea": nbr_fea,
        "crystal_atom_idx": crystal_atom_idx,
        "uni_idx": uni_idx,
        "uni_count": uni_count,
        "grid": grid,
        "volume": volume,
    }
    return (batch,)


MENAGERIE_ENTRIES = [
    (
        "MOFTransformer",
        "build_moftransformer",
        "example_input_moftransformer",
        2023,
        "vendored-pytorch",
    ),
]

# SOURCE: vendored from OmicsML/CellPLM @ main
#
# CellPLM (Wen et al., ICLR 2024, "CellPLM: Pre-training of Cell Language Model
# Beyond Single Cells"): a transformer-based single-cell foundation model that treats
# cells as tokens and mixes gene-level and cell-level attention via configurable
# transformer/MLP encoders, a VAE/GMVAE/split-style latent bottleneck, and NB/ZINB/MLP
# decoders. Real architecture classes copied verbatim from the official OmicsML repo
# (OmicsFormer + its embedder/encoder/decoder/latent/objective submodules), only
# trimming CLI/data-pipeline glue (scanpy/anndata loading, checkpoint IO) that is not
# part of the network itself. Instantiated here with the lightweight real code paths
# (enc_mod="mlp", dec_mod="mlp", latent_mod="none", mask_type="none", pe_type=None) so
# the forward pass stays a compact, real, traceable subset of the full model surface.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# CellPLM/utils/__init__.py (create_norm / create_activation / RMSNorm)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name, n, h=4):
    if name == "layernorm":
        return nn.LayerNorm(n)
    elif name == "batchnorm":
        return nn.BatchNorm1d(n)
    elif name == "groupnorm":
        return nn.GroupNorm(h, n)
    elif name == "rmsnorm":
        return RMSNorm(n)
    else:
        return nn.Identity()


# ---------------------------------------------------------------------------
# CellPLM/utils/sparse.py
# ---------------------------------------------------------------------------
def sparse_diag(x):
    indices = torch.arange(len(x), device=x.device).unsqueeze(0).repeat(2, 1)
    values = x
    return torch.sparse_coo_tensor(indices, values, (len(x), len(x)), device=x.device)


def sparse_normalize(x):
    size_factor = sparse_diag(1.0 / (torch.sparse.sum(x, dim=1).to_dense() + 1e-8))
    res = torch.sparse.mm(size_factor, x)
    return res


def sparse_tpm(x):
    x = sparse_normalize(x) * 1e4
    x = torch.log1p(x)
    return x


# ---------------------------------------------------------------------------
# CellPLM/utils/pe.py (only the pieces reachable with pe_type=None)
# ---------------------------------------------------------------------------
def select_pe_encoder(pe):
    raise NotImplementedError(f"Unsupported positional encoding type: {pe}")


# ---------------------------------------------------------------------------
# CellPLM/utils/mask.py (NullMaskBuilder only; mask_node_rate=0 in this recipe)
# ---------------------------------------------------------------------------
class NullMaskBuilder(nn.Module):
    def __init__(self, drop_node_rate, max_batch_size=2000):
        super().__init__()
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size

    def apply_mask(self, x_dict):
        x_dict["input_mask"] = torch.ones(
            *x_dict["x_seq"].shape, device=x_dict["x_seq"].device
        ).int()
        return x_dict


# ---------------------------------------------------------------------------
# CellPLM/embedder/omics.py
# ---------------------------------------------------------------------------
class OmicsEmbedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hid, gene_emb=None, fix_embedding=False):
        super().__init__()
        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = dict(zip(pretrained_gene_list, list(range(len(pretrained_gene_list)))))
        self.num_hid = num_hid

        if gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
        else:
            self.emb = nn.Parameter(
                torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32) * 0.005
            )
            if fix_embedding:
                self.emb.requires_grad = False

    def forward(self, x_dict, input_gene_list=None):
        if "masked_x_seq" in x_dict:
            x = x_dict["masked_x_seq"]
        else:
            x = x_dict["x_seq"]

        if "dropout" in x_dict:
            indices = x._indices().t()
            values = x._values()
            values = values.float()
            values = torch.distributions.binomial.Binomial(values, x_dict["dropout"]).sample()
            x = torch.sparse_coo_tensor(indices.t(), values, x.shape)

        x = torch.log1p(x)
        if input_gene_list is not None:
            gene_idx = torch.tensor(
                [self.gene_index[o] for o in input_gene_list if o in self.gene_index]
            ).long()
            x_dict["input_gene_mask"] = gene_idx
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError(
                    "The input gene size is not the same as the pretrained gene list. "
                    "Please provide the input gene list."
                )
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)
        feat = F.embedding(gene_idx, self.emb)
        feat = torch.sparse.mm(x, feat)
        return feat


class OmicsEmbeddingLayer(nn.Module):
    def __init__(
        self,
        gene_list,
        num_hidden,
        norm,
        activation="gelu",
        dropout=0.3,
        pe_type=None,
        cat_pe=True,
        gene_emb=None,
        inject_covariate=False,
        batch_num=None,
    ):
        super().__init__()

        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU()
        self.norm0 = create_norm(norm, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.extra_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            create_norm(norm, num_hidden),
        )
        if pe_type is not None:
            if cat_pe:
                num_emb = num_hidden // 2
            else:
                num_emb = num_hidden
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = num_hidden

        if gene_emb is None:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb)
        else:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb, gene_emb)

        if inject_covariate:
            self.cov_enc = nn.Embedding(batch_num, num_emb)
            self.inject_covariate = True
        else:
            self.inject_covariate = False

    def forward(self, x_dict, input_gene_list=None):
        x = self.feat_enc(x_dict, input_gene_list)
        if self.pe_enc is not None:
            pe_input = x_dict[self.pe_enc.pe_key]  # noqa: F841 (real code; unreachable when pe_type=None)
            pe = 0.0
            if self.inject_covariate:
                pe = pe + self.cov_enc(x_dict["batch"])
            if self.cat_pe:
                x = torch.cat([x, pe], 1)
            else:
                x = x + pe
        x = self.extra_linear(x)
        return x


# ---------------------------------------------------------------------------
# CellPLM/encoder/mlp.py + CellPLM/encoder/__init__.py (setup_encoder, "mlp" path)
# ---------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, num_hidden, num_layers, dropout, norm, covariates_dim=0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(num_hidden, num_hidden),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                    create_norm(norm, num_hidden),
                )
            )

    def forward(self, x_dict):
        x = x_dict["h"]
        for layer in self.layers:
            x = x + layer(x)
        return {"hidden": x}


def setup_encoder(
    model_type, num_hidden, num_layers, dropout, activation, norm, nhead, covariates_dim=0
):
    if model_type == "mlp":
        return MLPEncoder(
            num_hidden=num_hidden,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            covariates_dim=covariates_dim,
        )
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")


# ---------------------------------------------------------------------------
# CellPLM/decoder/mlp.py + CellPLM/decoder/__init__.py (setup_decoder, "mlp" path)
# ---------------------------------------------------------------------------
class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        dropout,
        norm,
        batch_num=0,
        dataset_num=0,
        platform_num=0,
        out_act=None,
    ):
        super().__init__()
        if out_act is None:
            out_act = nn.ReLU()
        self.layers = nn.ModuleList()
        covariate_num = batch_num + dataset_num + platform_num
        for i in range(num_layers - 1):
            dim = hidden_dim if i > 0 else in_dim
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim + covariate_num, hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                    create_norm(norm, hidden_dim),
                )
            )

        self.out_layer = [nn.Linear(hidden_dim, out_dim)]
        if out_act is not None:
            self.out_layer.append(out_act)
        self.out_layer = nn.Sequential(*self.out_layer)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.batch_num = batch_num
        self.dataset_num = dataset_num
        self.platform_num = platform_num

    def forward(self, x_dict):
        covariates = []
        if self.batch_num > 0:
            covariates.append(F.one_hot(x_dict["batch"], num_classes=self.batch_num))
        if self.dataset_num > 0:
            covariates.append(F.one_hot(x_dict["dataset"], num_classes=self.dataset_num))
        if self.platform_num > 0:
            covariates.append(F.one_hot(x_dict["platform"], num_classes=self.platform_num))
        x = x_dict["h"]
        for i, layer in enumerate(self.layers):
            x = torch.cat([x] + covariates, 1)
            x = layer(x)
        return {"recon": self.out_layer(x), "latent": x_dict["h"]}


def setup_decoder(
    model_type,
    in_dim,
    hidden_dim,
    out_dim,
    num_layers,
    dropout,
    norm,
    batch_num=0,
    dataset_num=0,
    platform_num=0,
):
    if model_type == "mlp":
        return MLPDecoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
            dataset_num=dataset_num,
            platform_num=platform_num,
        )
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")


# ---------------------------------------------------------------------------
# CellPLM/latent/__init__.py (PlaceholderLayer + LatentModel + PreLatentNorm;
# latent_mod="none" only ever exercises the placeholder identity layer)
# ---------------------------------------------------------------------------
class PlaceholderLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.is_adversarial = False

    def forward(self, x_dict):
        return x_dict["h"], torch.tensor(0.0).to(x_dict["h"].device)


class LatentModel(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.layers = nn.ModuleList([PlaceholderLayer()])
        self.alias_dict = {}

    def forward(self, x_dict):
        total_loss = 0
        for layer in self.layers:
            x_dict["h"], loss = layer(x_dict)
            total_loss += loss
        return x_dict["h"], total_loss


class DSBNNorm(nn.Module):
    def __init__(self, dim, domain_num, domain_label="dataset", eps=1e-6, flip_rate=0.3):
        super().__init__()
        self.eps = eps
        self.domain_label = domain_label
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(domain_num + 1)])
        self.flip_rate = flip_rate

    def forward(self, xdict):
        h = xdict["h"]
        h = self.bns[0](h)
        return h


class PreLatentNorm(nn.Module):
    def __init__(self, type="none", enc_hid=None, dataset_num=None):
        super().__init__()
        self.type = type
        if type not in ["none", "dsbn", "ln"]:
            raise NotImplementedError(f'"{type}" type of pre latent norm is not implemented.')
        if type == "dsbn":
            self.norm = DSBNNorm(enc_hid, dataset_num)
        elif type == "ln":
            self.norm = nn.LayerNorm(enc_hid)

    def forward(self, xdict):
        if self.type == "dsbn":
            return self.norm(xdict)
        elif self.type == "ln":
            return self.norm(xdict["h"])
        else:
            return xdict["h"]


# ---------------------------------------------------------------------------
# CellPLM/objective/autoencoder.py (ReconstructionLoss, "recon" path)
# ---------------------------------------------------------------------------
class ReconstructionLoss(nn.Module):
    def __init__(self, lib_size=None, log_norm=False, **kwargs):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.lib_size = lib_size
        self.log_norm = log_norm
        self.downstream = None

    def forward(self, out_dict, x_dict):
        y = x_dict["x_seq"].to_dense()
        if self.lib_size is not None:
            y = y / y.sum(1)[:, None] * self.lib_size
        if self.log_norm:
            y = torch.log(y + 1)
        size_factor = y.sum(1, keepdim=True)
        pred = (size_factor * out_dict["recon"] * x_dict["input_mask"])[:, x_dict["gene_mask"]]
        truth = (y * x_dict["input_mask"])[:, x_dict["gene_mask"]]
        pred = pred[x_dict["input_mask"].sum(1) > 0]
        truth = truth[x_dict["input_mask"].sum(1) > 0]
        out_dict["pred"] = pred

        return self.reconstruction_loss(pred, truth)


class Objectives(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.layers = nn.ModuleList()
        if configs is not None:
            for c in configs:
                self.layers.append(
                    ReconstructionLoss(**{k: v for k, v in c.items() if k != "type"})
                )

    def forward(self, out_dict, x_dict):
        total_loss = 0
        for layer in self.layers:
            loss = layer(out_dict, x_dict)
            total_loss += loss
        return total_loss


# ---------------------------------------------------------------------------
# CellPLM/model/cellformer.py (OmicsFormer, real class -- trimmed callers only
# support the enc_mod="mlp" / dec_mod="mlp" / latent_mod="none" / mask_type="none"
# real code paths this recipe exercises)
# ---------------------------------------------------------------------------
class OmicsFormer(nn.Module):
    def __init__(
        self,
        gene_list,
        enc_mod,
        enc_hid,
        enc_layers,
        post_latent_dim,
        dec_mod,
        dec_hid,
        dec_layers,
        out_dim,
        batch_num=0,
        dataset_num=0,
        platform_num=0,
        mask_type="none",
        model_dropout=0.1,
        activation="gelu",
        norm="layernorm",
        enc_head=8,
        mask_node_rate=0.0,
        mask_feature_rate=0.0,
        drop_node_rate=0.0,
        max_batch_size=2000,
        cat_dim=None,
        conti_dim=None,
        pe_type=None,
        cat_pe=True,
        gene_emb=None,
        latent_mod="none",
        head_type=None,
        dsbn=False,
        input_covariate=False,
        **kwargs,
    ):
        super(OmicsFormer, self).__init__()

        self.embedder = OmicsEmbeddingLayer(
            gene_list,
            enc_hid,
            norm,
            activation,
            model_dropout,
            pe_type,
            cat_pe,
            gene_emb,
            inject_covariate=input_covariate,
            batch_num=batch_num,
        )
        self.gene_set = set(gene_list)
        self.mask_type = mask_type
        self.mask_model = NullMaskBuilder(drop_node_rate, max_batch_size)
        self.encoder = setup_encoder(
            enc_mod, enc_hid, enc_layers, model_dropout, activation, norm, enc_head
        )

        self.latent = LatentModel()
        self.latent_mod = latent_mod
        if latent_mod == "none":
            post_latent_dim = enc_hid
        else:
            raise NotImplementedError(f'Latent mod "{latent_mod}" is not implemented.')

        self.head_type = head_type
        self.decoder = setup_decoder(
            dec_mod,
            post_latent_dim,
            dec_hid,
            out_dim,
            dec_layers,
            model_dropout,
            norm,
            batch_num=batch_num,
            dataset_num=dataset_num,
            platform_num=platform_num,
        )
        self.objective = Objectives([{"type": "recon"}])

        if dsbn:
            self.pre_latent_norm = PreLatentNorm("dsbn", enc_hid, dataset_num)
        else:
            self.pre_latent_norm = PreLatentNorm("ln", enc_hid)

    def forward(self, x_dict, input_gene_list=None, d_iter=False):
        if self.mask_type == "input":
            x_dict = self.mask_model.apply_mask(x_dict)
        x_dict["h"] = self.embedder(x_dict, input_gene_list)
        x_dict["h"] = self.encoder(x_dict)["hidden"]
        x_dict["h"] = self.pre_latent_norm(x_dict)
        x_dict["h"], latent_loss = self.latent(x_dict)

        out_dict = self.decoder(x_dict)
        loss = latent_loss + self.objective(out_dict, x_dict)
        out_dict["latent_loss"] = (
            latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
        )
        out_dict["target_loss"] = loss.item() - out_dict["latent_loss"]
        return out_dict, loss


def _build_example_x_dict(n_cells=4, n_genes=32, hidden=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Real CellPLM feeds a torch.sparse_coo `x_seq` (raw counts) into
    # torch.sparse.mm inside OmicsEmbedder.forward. torch.sparse.mm also accepts a
    # dense left operand (it dispatches to a dense matmul in that case), so a dense
    # x_seq here exercises the exact same real forward-pass code path while staying
    # within TorchLens' supported dense-strided-tensor input contract.
    x_seq = (
        (torch.rand(n_cells, n_genes, generator=g) < 0.3).float()
        * torch.rand(n_cells, n_genes, generator=g)
        * 5.0
    )
    input_mask = torch.ones(n_cells, n_genes, dtype=torch.int32)
    gene_mask = torch.ones(n_genes, dtype=torch.bool)
    return {
        "x_seq": x_seq,
        "input_mask": input_mask,
        "gene_mask": gene_mask,
    }


def build_cellplm_omicsformer():
    gene_list = [f"gene_{i}" for i in range(32)]
    return OmicsFormer(
        gene_list=gene_list,
        enc_mod="mlp",
        enc_hid=16,
        enc_layers=2,
        post_latent_dim=16,
        dec_mod="mlp",
        dec_hid=16,
        dec_layers=2,
        out_dim=32,
        mask_type="none",
        mask_node_rate=0.0,
        mask_feature_rate=0.0,
        pe_type=None,
        latent_mod="none",
        head_type=None,
    )


def example_input_cellplm_omicsformer():
    return _build_example_x_dict()


MENAGERIE_ENTRIES = [
    (
        "CellPLM-OmicsFormer",
        build_cellplm_omicsformer,
        example_input_cellplm_omicsformer,
        2024,
        "SOURCE_AVAILABLE",
    ),
]

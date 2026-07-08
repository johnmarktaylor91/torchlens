# SOURCE: vendored from https://github.com/A4Bio/CellVQ @ master
#
# CellVQ (Nature Communications 2026) -- a foundation model for single-cell RNA-seq
# that quantizes cell embeddings into a discrete "cell codebook" via a spherical
# soft vector-quantization layer (SoftCVQLayer), trained with a ZINB reconstruction
# objective on top of a gene-token transformer encoder.
#
# Vendored (unmodified architecture, only import paths adjusted to be self-contained
# in this single file) from:
#   - model/pretrainmodels/model.py       (AutoDiscretizationEmbedding2, Model)
#   - model/pretrainmodels/transformer.py (pytorchTransformerModule)
#   - model/pretrainmodels/select_model.py(select_module / select_model)
#   - modules/vq_modules.py               (SoftCVQLayer)
#
# The real end-to-end inference call used by the authors' own inference.py is
# `pretrainmodel.get_cellcode(x, padding_label, encoder_position_gene_ids)`, which
# this module wraps in a thin nn.Module.forward shim (CellVQTraceWrapper) -- the
# wrapper adds NO new architecture, it only forwards to the real vendored call.

import math

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ----------------------------------------------------------------------------
# modules/vq_modules.py :: SoftCVQLayer (vendored verbatim)
# ----------------------------------------------------------------------------
class SoftCVQLayer(nn.Module):
    def __init__(self, log2_num_embeddings, embedding_dim, vq_dim, condition_layer=6, sphere=True):
        super(SoftCVQLayer, self).__init__()
        self.init = True
        self.log2_num_embeddings = log2_num_embeddings
        int_range = torch.arange(0, 2**log2_num_embeddings)
        bool_vectors = (
            int_range[:, None] & (1 << torch.arange(log2_num_embeddings - 1, -1, -1))
        ) > 0

        self.register_buffer("embedding", bool_vectors.float())
        self.sphere = sphere

        hidden_dim = 1024

        if condition_layer <= 3:
            layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.ReLU()]
            for _ in range(condition_layer - 2):
                (layers.append(nn.Linear(hidden_dim, hidden_dim)),)
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, vq_dim))
            self.embedding_mlp = nn.Sequential(*layers)
        else:
            layers = [
                nn.Linear(log2_num_embeddings, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
            for _ in range(condition_layer - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, vq_dim))
            self.embedding_mlp = nn.Sequential(*layers)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)  # noqa: F841
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)

        self.init = False
        self.MSE = nn.HuberLoss(reduction="none")

    def project(self, h):
        h = self.proj(h)
        return h

    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id, level=None):
        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            embed = embed / torch.norm(embed, dim=-1, keepdim=True)
        return self.proj_inv(embed[vq_id])

    def get_code(self, h, attn_mask=None, temperature=1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h)
            embed = self.normalize(embed)

        if attn_mask is None:
            attn_mask = torch.ones_like(h[:, 0])

        h_flat = h[attn_mask == 1]
        A, _ = self.attention(h_flat, embed, temperature)
        vq_code = A.argmax(dim=-1)
        return vq_code

    def decimal2binary(self, vqids):
        return self.embedding[vqids]

    def binary2decimal(self, binary_vector):
        base = 2 ** torch.arange(binary_vector.size(-1) - 1, -1, -1, device=binary_vector.device)
        vqids = (binary_vector * base).long().sum(dim=-1)
        return vqids

    def attention(self, H, C, temperature=1):
        alpha = 1 / temperature
        distances = -2 * (alpha - 1) * (H @ C.t()).detach() - 2 * H @ C.t()
        A = F.softmax(-distances, dim=1)
        vq_code = distances.argmin(dim=-1)
        return A, vq_code

    def normalize(self, x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)

    def get_vq(self, h, attn_mask=None, temperature=1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h)
            embed = self.normalize(embed)

        if attn_mask is None:
            attn_mask = torch.ones_like(h[:, 0])

        h_flat = h[attn_mask == 1]
        A, code = self.attention(h_flat, embed, temperature)
        h_vq = embed[code]

        quantized = torch.zeros_like(h)
        quantized[attn_mask == 1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device=h.device, dtype=torch.long)
        vq_code[attn_mask == 1] = code
        quantized = self.proj_inv(quantized)
        return vq_code, quantized

    def entropy_loss(self, P, Q):
        return -torch.sum(P * torch.log(Q))

    def forward(
        self, h_in, attn_mask=None, mode="train", temperature=1, vqshortcut=False, frozen=False
    ):
        h = self.proj(h_in)

        embed = self.embedding_mlp(self.embedding)

        if self.sphere:
            h = self.normalize(h)
            embed = self.normalize(embed)

        if attn_mask is None:
            attn_mask = torch.ones_like(h[:, 0])
        h_flat = h[attn_mask == 1]

        A, code = self.attention(h_flat, embed, temperature)

        mat = embed @ embed.permute(1, 0)
        indices = torch.arange(mat.size(0))
        mat[indices, indices] = -1
        vq_loss = mat.max(dim=-1)[0].mean()

        h_vq = embed[code]

        quantized = torch.zeros_like(h)
        quantized[attn_mask == 1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device=h.device, dtype=torch.long)
        vq_code[attn_mask == 1] = code

        if vqshortcut and not frozen:
            quantized = h.clone()
            quantized[attn_mask == 1] = h_vq

        quantized = self.proj_inv(quantized)
        return quantized, vq_code, vq_loss


# ----------------------------------------------------------------------------
# model/pretrainmodels/transformer.py :: pytorchTransformerModule (vendored verbatim)
# ----------------------------------------------------------------------------
class pytorchTransformerModule(nn.Module):
    def __init__(self, max_seq_len, dim, depth, heads, ff_mult=4, norm_first=False):
        super(pytorchTransformerModule, self).__init__()

        self.max_seq_len = max_seq_len
        self.depth = depth
        layers = []
        for i in range(depth):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim * ff_mult,
                    batch_first=True,
                    norm_first=norm_first,
                )
            )

        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask):
        b, n, _, device = *x.shape, x.device  # noqa: F841
        assert n <= self.max_seq_len, (
            f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"
        )
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x


# ----------------------------------------------------------------------------
# model/pretrainmodels/model.py :: AutoDiscretizationEmbedding2, Model (vendored verbatim)
# ----------------------------------------------------------------------------
def exists(val):
    return val is not None


class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id=None, pad_token_id=None):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)

        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)

        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x == self.mask_token_id).nonzero()
        x_pad_idx = (x == self.pad_token_id).nonzero()

        x = self.mlp(x)
        x = self.LeakyReLU(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.Softmax(x)

        bin_num_idx = self.bin_num_idx.to(x.device)

        token_emb = self.emb(bin_num_idx)
        x = torch.matmul(weight, token_emb)

        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(x_mask_idx.shape[0], 1)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(x_pad_idx.shape[0], 1)

        if output_weight:
            return x, weight
        return x


class Model(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        embed_dim,
        decoder_embed_dim,
        tie_embed=False,
        bin_alpha=1.0,
        bin_num=10,
        pad_token_id=None,
        mask_token_id=None,
        level=12,
        condition_layer=6,
        latent_dim=32,
        celltype_num=831,
        tissue_num=377,
        disease_num=150,
        n_genes=19266,
    ):
        super(Model, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        self.token_emb = AutoDiscretizationEmbedding2(
            embed_dim,
            max_seq_len,
            bin_num=bin_num,
            bin_alpha=bin_alpha,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
        )
        self.pos_emb = nn.Embedding(max_seq_len + 1, embed_dim)

        self.encoder = None
        self.cell_vq = SoftCVQLayer(level, embed_dim, latent_dim, condition_layer=condition_layer)

        self.decoder = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

        self.cell_type = nn.Linear(4 * embed_dim, celltype_num)
        self.tissue = nn.Linear(4 * embed_dim, tissue_num)
        self.disease = nn.Linear(4 * embed_dim, disease_num)
        self.sex = nn.Linear(4 * embed_dim, 3)
        self.age = nn.Linear(4 * embed_dim, 184)
        self.mean = nn.Linear(decoder_embed_dim, 1)
        self.disp = nn.Linear(decoder_embed_dim, 1)
        self.pi = nn.Linear(decoder_embed_dim, 1)

        # go (gene-ontology one-hot lookup); n_genes is downsized here vs. the
        # real checkpoint's 19266 human genes so the module traces at tiny size
        self.go = torch.eye(n_genes)
        self.go = nn.Parameter(self.go, requires_grad=True)
        self.go_embed = nn.Linear(n_genes, embed_dim)

    def get_cellemb(
        self, x, padding_label, encoder_position_gene_ids, output_attentions=False, **kwargs
    ):
        geneemb = self.encode(
            x,
            padding_label,
            encoder_position_gene_ids,
            output_attentions=output_attentions,
            **kwargs,
        )
        geneemb1 = geneemb[:, -1, :]
        geneemb2 = geneemb[:, -2, :]
        geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
        geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
        geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
        geneembmax, _ = torch.max(geneemb, dim=1)
        return geneembmerge, geneembmax

    def get_cellcode(
        self, x, padding_label, encoder_position_gene_ids, output_attentions=False, **kwargs
    ):
        geneemb = self.encode(
            x,
            padding_label,
            encoder_position_gene_ids,
            output_attentions=output_attentions,
            **kwargs,
        )
        cellemb, indexes = torch.max(geneemb, dim=1)
        _, indexes2 = torch.max(geneemb[:, :-2], dim=1)
        _, cell_code, vq_loss = self.cell_vq(cellemb, temperature=1e-8, vqshortcut=False)
        return geneemb, cell_code, encoder_position_gene_ids[:, indexes2[0]]

    def encode(
        self, x, padding_label, encoder_position_gene_ids, output_attentions=False, **kwargs
    ):
        b, n, device = *x.shape, x.device  # noqa: F841
        assert n <= self.max_seq_len, (
            f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"
        )

        x = self.token_emb(torch.unsqueeze(x, 2), output_weight=0)
        if output_attentions:
            x.requires_grad_()

        position_emb = self.pos_emb(encoder_position_gene_ids)
        go_emb = self.go_embed(self.go[encoder_position_gene_ids])

        x = x + position_emb
        x = x + go_emb
        x = self.encoder(x, padding_mask=padding_label)

        return x


# ----------------------------------------------------------------------------
# model/pretrainmodels/select_model.py :: select_module / select_model (vendored verbatim)
# ----------------------------------------------------------------------------
def select_module(config, sub_config, module_name):
    if module_name == "transformer":
        return pytorchTransformerModule(
            max_seq_len=config["seq_len"],
            dim=sub_config["hidden_dim"],
            depth=sub_config["depth"],
            heads=sub_config["heads"],
        )
    else:
        raise ValueError(f"module type error: {module_name}")


def select_model(config):
    encoder_config = config["encoder"]
    decoder_config = config["decoder"]
    encoder = select_module(config, encoder_config, config["encoder"]["module_type"])
    decoder = select_module(config, decoder_config, config["decoder"]["module_type"])
    model = Model(
        num_tokens=config["n_class"],
        max_seq_len=config["seq_len"],
        embed_dim=config["encoder"]["hidden_dim"],
        decoder_embed_dim=config["decoder"]["hidden_dim"],
        bin_alpha=config["bin_alpha"],
        bin_num=config["bin_num"],
        pad_token_id=config["pad_token_id"],
        mask_token_id=config["mask_token_id"],
        level=config.get("level", 12),
        condition_layer=config.get("condition_layer", 6),
        latent_dim=config.get("latent_dim", 32),
        n_genes=config.get("n_genes", 19266),
    )
    model.encoder = encoder
    model.decoder = decoder
    return model


# ----------------------------------------------------------------------------
# TorchLens staging entry point
# ----------------------------------------------------------------------------
class CellVQTraceWrapper(nn.Module):
    """Thin forward() shim over the real CellVQ Model.get_cellcode() inference call
    (the same call used in the authors' own inference.py). No new architecture is
    introduced here -- this only adapts the tuple-of-kwargs interface into a
    positional forward() so TorchLens can trace a single call."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, padding_label, encoder_position_gene_ids):
        geneemb, cell_code, gene_id = self.model.get_cellcode(
            x=x,
            padding_label=padding_label,
            encoder_position_gene_ids=encoder_position_gene_ids,
        )
        return geneemb


def _tiny_config():
    n_genes = 32
    seq_len = 8
    return {
        "n_class": 7,
        "seq_len": seq_len,
        "bin_alpha": 1.0,
        "bin_num": 10,
        "pad_token_id": 0,
        "mask_token_id": 1,
        "level": 8,
        "condition_layer": 2,
        "latent_dim": 16,
        "n_genes": n_genes,
        "encoder": {"module_type": "transformer", "hidden_dim": 32, "depth": 2, "heads": 4},
        "decoder": {"module_type": "transformer", "hidden_dim": 32, "depth": 1, "heads": 4},
    }


def build_cellvq():
    config = _tiny_config()
    model = select_model(config)
    return CellVQTraceWrapper(model)


def example_input_cellvq():
    config = _tiny_config()
    seq_len = config["seq_len"]
    # encoder_position_gene_ids indexes BOTH self.pos_emb (size max_seq_len+1)
    # and self.go / self.go_embed (size n_genes), so its range must respect
    # the tighter of the two bounds.
    max_position_id = min(seq_len, config["n_genes"] - 1)
    batch = 1
    x = torch.rand(batch, seq_len)
    padding_label = torch.zeros(batch, seq_len, dtype=torch.bool)
    encoder_position_gene_ids = torch.randint(
        0, max_position_id + 1, (batch, seq_len), dtype=torch.long
    )
    return (x, padding_label, encoder_position_gene_ids)


MENAGERIE_ENTRIES = [
    ("CellVQ", "build_cellvq", "example_input_cellvq", 2026, "vendored-pytorch"),
]

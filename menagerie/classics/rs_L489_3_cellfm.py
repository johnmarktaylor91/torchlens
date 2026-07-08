# FAITHFUL PORT of biomed-AI/CellFM @ main (model.py, retention.py, attention.py,
# lora.py) (original framework: MindSpore)
#
# CellFM (Zeng et al., "CellFM: a large-scale foundation model pre-trained on
# transcriptomics of 100 million human cells"): an ~800M-1.5B parameter RetNet-style
# (retention, not softmax-attention) single-cell foundation model. The real repo is
# implemented entirely in MindSpore (Huawei's DL framework: `mindspore.nn.Cell`,
# `mindspore.ops.operations`, `mindspore.Parameter`) for training on Ascend NPUs;
# MindSpore is not one of this environment's installed base libraries and is not
# reasonably installable alongside the torch stack here, so the real MindSpore code
# cannot be run/vendored directly (RUNG 2 blocked). This is a line-for-line
# transcription of the real MindSpore `nn.Cell` classes into torch `nn.Module`
# equivalents -- every op the original explicitly wires up via `mindspore.ops.operations`
# (Dense/Linear, BatchMatMul, ReLU/SiLU kernels for the linear-retention Q/K, gated
# linear unit, LayerNorm/RMSNorm, LoRA-optional projections) is preserved with the same
# structure and same forward-pass order as `model.py::CellFM.construct` /
# `retention.py::RetentionLayer/MHRetention/GatedLinearUnit` /
# `attention.py::AttentionLayer` (the classification/full-attention branch used only
# when `cfg.label=True`). Traced here with the real repo's `Config` defaults
# (`lora=0`, `pad_zero=True`, `add_zero=False`, `label=False`) at drastically smaller
# `enc_dims`/`enc_nlayers`/`enc_num_heads` for fast CPU tracing; the mechanism (linear
# "retention" instead of softmax attention, SRMSNorm, gated FFN) is unchanged.
# ruff: noqa: E741 (variable names `O`/`l` mirror the original MindSpore code's
# output/seq-len naming for close correspondence to the source)
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# lora.py::lora_block
# ---------------------------------------------------------------------------
class LoraBlock(nn.Module):
    def __init__(self, in_dims, out_dims, hid_dims=16):
        super().__init__()
        self.alpha = 1
        self.A = nn.Linear(in_dims, hid_dims, bias=False)
        self.B = nn.Linear(hid_dims, out_dims, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)
        return x * self.alpha


# ---------------------------------------------------------------------------
# retention.py::SRMSNorm / MHRetention / GatedLinearUnit / RetentionLayer
# ---------------------------------------------------------------------------
class SRMSNorm(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.scale = emb_dims**-0.5
        self.eps = 1e-7

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x_norm = torch.linalg.vector_norm(x * self.scale, ord=2, dim=-1, keepdim=True)
        return (x / x_norm.clamp(min=1e-12)).to(dtype)


class Kernel(nn.Module):
    # ReLU "kernel" feature map used for Q/K in the linear-attention retention op.
    def forward(self, x):
        return F.relu(x)


class MHRetention(nn.Module):
    def __init__(self, emb_dims, num_heads, lth=None, lora=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = emb_dims // num_heads
        self.scale = self.head_dims**0.5
        self.lth = lth
        self.lora = lora
        self.kernelQ = Kernel()
        self.kernelK = Kernel()
        self.kernelU = nn.SiLU()
        beta = 1 if lth is None else (lth * 8) ** -0.25
        self.q_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.k_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        nn.init.xavier_normal_(self.q_proj.weight, gain=1)
        nn.init.xavier_normal_(self.k_proj.weight, gain=1)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)
        if self.lora > 0:
            self.lora_q = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_k = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)
        self.pre_norm = SRMSNorm(emb_dims)
        self.inner_norm = SRMSNorm(self.head_dims)

    def qkvu_compute(self, x, y):
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        u = self.u_proj(x)
        if self.lora > 0:
            q = q + self.lora_q(x)
            k = k + self.lora_k(y)
            v = v + self.lora_v(y)
            u = u + self.lora_u(x)
        return q, k, v, u

    def forward(self, x, y=None, v_pos=None, attn_mask=None, seq_mask=None):
        h = self.num_heads
        if y is None:
            q, k, v, u = self.qkvu_compute(x, x)
        else:
            q, k, v, u = self.qkvu_compute(x, y)
        b, l1, d = q.shape
        _, l2, _ = k.shape
        Q = q.reshape(b, l1, h, self.head_dims).permute(0, 2, 1, 3)
        K = k.reshape(b, l2, h, self.head_dims).permute(0, 2, 1, 3)
        V = v.reshape(b, l2, h, self.head_dims).permute(0, 2, 1, 3)
        U = u.reshape(b, l1, h, self.head_dims).permute(0, 2, 1, 3)

        Q = self.kernelQ(Q)
        K = self.kernelK(K)
        U = self.kernelU(U)
        if seq_mask is not None:
            Q = Q * seq_mask
        if attn_mask is not None:
            K = K * attn_mask
        if v_pos is not None:
            V = V * v_pos
        Q = Q / self.scale
        K = K / self.scale
        # linear retention: O = Q @ (K^T @ V), no softmax
        KV = torch.matmul(K.transpose(-2, -1), V)
        O = torch.matmul(Q, KV)
        O = self.inner_norm(O)
        O = O * U
        O = O.permute(0, 2, 1, 3).reshape(b, l1, d)
        if self.lora > 0:
            O = self.o_proj(O) + self.lora_o(O)
        else:
            O = self.o_proj(O)
        return O


class GatedLinearUnit(nn.Module):
    def __init__(self, emb_dims, lth=None, lora=0):
        super().__init__()
        beta = 1 if lth is None else (lth * 8) ** -0.25
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)
        self.lora = lora
        if self.lora > 0:
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(self, x):
        b, l, d = x.shape
        x = x.reshape(-1, d)
        if self.lora > 0:
            u = self.u_proj(x) + self.lora_u(x)
            v = self.v_proj(x) + self.lora_v(x)
            o = u * v
            o = self.o_proj(o) + self.lora_o(o)
        else:
            u = self.u_proj(x)
            v = self.v_proj(x)
            o = u * v
            o = self.o_proj(o)
        return o.reshape(b, l, -1)


class RetentionLayer(nn.Module):
    def __init__(self, emb_dims, num_heads, lth, dropout=0.0, lora=0, recompute=False):
        super().__init__()
        self.attn = MHRetention(emb_dims, num_heads, lth, lora)
        self.ffn = GatedLinearUnit(emb_dims, lth, lora)
        self.dropout = nn.Dropout(p=dropout)
        self.post_norm1 = nn.LayerNorm(emb_dims)
        self.post_norm2 = nn.LayerNorm(emb_dims)
        self.alpha = (2 * lth) ** 0.25 if lth else 1.0

    def forward(self, x, y=None, v_pos=None, attn_mask=None, seq_mask=None):
        out = self.dropout(self.attn(x, y, v_pos, attn_mask, seq_mask))
        x = self.post_norm1(x * self.alpha + out)
        out = self.dropout(self.ffn(x))
        x = self.post_norm2(x * self.alpha + out)
        return x


# ---------------------------------------------------------------------------
# attention.py::FullAttention / ffn / AttentionLayer (used only for the label/cls path)
# ---------------------------------------------------------------------------
class FullAttention(nn.Module):
    def __init__(self, emb_dims, num_heads, dropout=0.0):
        super().__init__()
        self.q_proj = nn.Linear(emb_dims, emb_dims, bias=True)
        self.k_proj = nn.Linear(emb_dims, emb_dims, bias=True)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.to_out = nn.Linear(emb_dims, emb_dims, bias=False)
        self.num_heads = num_heads
        self.head_dim = emb_dims // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, y=None, attn_mask=None, k_pos=None, v_pos=None):
        b, l1, d = x.shape
        h = self.num_heads
        q = self.q_proj(x).reshape(b, l1, h, self.head_dim).permute(0, 2, 1, 3)
        if y is None:
            y = x
        b, l2, _ = y.shape
        k = self.k_proj(y).reshape(b, l2, h, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(y).reshape(b, l2, h, self.head_dim).permute(0, 2, 1, 3)
        if k_pos is not None:
            k = k * k_pos
        if v_pos is not None:
            v = v * v_pos
        scores = torch.matmul(q, k.transpose(-2, -1)).float() * self.scale
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(b, 1, 1, -1)
            attn_mask = (1.0 - attn_mask.to(scores.dtype)) * -1e5
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1).to(x.dtype)
        attn = self.dropout(attn)
        o = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, l1, d)
        return self.to_out(o)


class FFN2(nn.Module):
    def __init__(self, emb_dims, dropout=0.0):
        super().__init__()
        self.dense1 = nn.Linear(emb_dims, 2 * emb_dims)
        self.act = nn.LeakyReLU()
        self.dense2 = nn.Linear(2 * emb_dims, emb_dims)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        b, l, d = x.shape
        x = x.reshape(-1, d)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x.reshape(b, l, d)


class AttentionLayer(nn.Module):
    def __init__(self, emb_dims, num_heads, dropout=0.0, recompute=False):
        super().__init__()
        self.attn = FullAttention(emb_dims, num_heads, dropout)
        self.ffn = FFN2(emb_dims, dropout)
        self.norm1 = nn.LayerNorm(emb_dims, eps=1e-7)
        self.norm2 = nn.LayerNorm(emb_dims, eps=1e-7)

    def forward(self, x, y=None, v_pos=None, k_pos=None, attn_mask=None):
        x = self.norm1(x + self.attn(x, y, attn_mask, k_pos, v_pos))
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# model.py::FFN / ValueEncoder / ValueDecoder / CellwiseDecoder / CellFM
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, in_dims, emb_dims, b=256):
        super().__init__()
        self.w1 = nn.Linear(in_dims, b, bias=False)
        self.act1 = nn.LeakyReLU()
        self.w3 = nn.Linear(b, b, bias=False)
        self.table = nn.Linear(b, emb_dims, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        b, l, d = x.shape
        v = x.reshape(-1, d)
        v = self.act1(self.w1(v))
        v = self.w3(v) + v * self.a
        v = torch.softmax(v, dim=-1)
        v = self.table(v)
        return v.reshape(b, l, -1)


class ValueEncoder(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.value_enc = FFN(1, emb_dims)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, emb_dims))

    def forward(self, x):
        b, l = x.shape[:2]
        if len(x.shape) == 3:
            unmask, expr = x.chunk(2, dim=-1)
            unmasked = self.value_enc(expr) * unmask
            masked = self.mask_emb * (1 - unmask)
            expr_emb = masked + unmasked
        else:
            expr = x.reshape(b, l, 1)
            unmask = torch.ones_like(expr)
            expr_emb = self.value_enc(expr)
        return expr_emb, unmask


class ValueDecoder(nn.Module):
    def __init__(self, emb_dims, dropout, zero=False):
        super().__init__()
        self.zero = zero
        self.w1 = nn.Linear(emb_dims, emb_dims, bias=False)
        self.act = nn.LeakyReLU()
        self.w2 = nn.Linear(emb_dims, 1, bias=False)
        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid(),
            )

    def forward(self, expr_emb):
        b, l, d = expr_emb.shape
        x = self.w2(self.act(self.w1(expr_emb)))
        pred = x.reshape(b, l)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).reshape(b, -1)
        return pred, zero_prob


class CellwiseDecoder(nn.Module):
    def __init__(self, in_dims, emb_dims=None, dropout=0.0, zero=False):
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.map = nn.Linear(in_dims, emb_dims, bias=False)
        self.zero = zero
        if zero:
            self.zero_logit = nn.Linear(emb_dims, emb_dims)

    def forward(self, cell_emb, gene_emb):
        b = cell_emb.shape[0]
        query = torch.sigmoid(self.map(gene_emb))
        key = cell_emb.reshape(b, -1, 1)
        pred = torch.bmm(query, key).reshape(b, -1)
        if not self.zero:
            return pred
        zero_query = self.zero_logit(gene_emb)
        zero_prob = torch.sigmoid(torch.bmm(zero_query, key)).reshape(b, -1)
        return pred, zero_prob


class CellFM(nn.Module):
    def __init__(self, n_genes, cfg):
        super().__init__()
        self.depth = cfg["enc_nlayers"]
        self.if_cls = cfg["label"]
        self.n_genes = n_genes
        self.add_zero = cfg["add_zero"] and not cfg["pad_zero"]
        self.pad_zero = cfg["pad_zero"]

        pad_dim = n_genes + 1 + (-n_genes - 1) % 8
        self.gene_emb = nn.Parameter(torch.empty(pad_dim, cfg["enc_dims"]))
        nn.init.xavier_normal_(self.gene_emb, gain=0.5)
        with torch.no_grad():
            self.gene_emb[0, :] = 0
        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg["enc_dims"]))
        nn.init.xavier_normal_(self.cls_token, gain=0.5)
        self.zero_emb = nn.Parameter(torch.zeros(1, 1, cfg["enc_dims"]))

        self.value_enc = ValueEncoder(cfg["enc_dims"])
        self.encoder = nn.ModuleList(
            [
                RetentionLayer(
                    cfg["enc_dims"],
                    cfg["enc_num_heads"],
                    cfg["enc_nlayers"],
                    cfg["enc_dropout"] * i / cfg["enc_nlayers"],
                    cfg["lora"],
                    cfg["recompute"],
                )
                for i in range(cfg["enc_nlayers"])
            ]
        )
        self.value_dec = ValueDecoder(cfg["enc_dims"], cfg["dropout"], zero=self.add_zero)
        self.cellwise_dec = CellwiseDecoder(cfg["enc_dims"], cfg["enc_dims"], zero=self.add_zero)

        if cfg["label"]:
            self.cluster_emb = nn.Parameter(torch.empty(cfg["num_cls"], cfg["enc_dims"]))
            nn.init.xavier_normal_(self.cluster_emb, gain=0.5)
            self.query = RetentionLayer(cfg["enc_dims"], cfg["enc_num_heads"], 0.5, 0, 0, False)
            self.classifier = nn.Linear(cfg["enc_dims"], 1, bias=False)
            self.proj = nn.Sequential(
                nn.Linear(cfg["enc_dims"], cfg["enc_dims"]),
                nn.LeakyReLU(),
                nn.Linear(cfg["enc_dims"], cfg["enc_dims"]),
                nn.LeakyReLU(),
                nn.Linear(cfg["enc_dims"], cfg["enc_dims"]),
            )

    def encode(self, expr, gene, zero_idx):
        b, l = gene.shape
        gene_emb = F.embedding(gene, self.gene_emb)
        expr_emb, unmask = self.value_enc(expr)
        len_scale = (torch.sum(zero_idx, dim=-1, keepdim=True) - 1).clamp(min=1e-12).rsqrt()
        len_scale = len_scale.detach().reshape(b, 1, 1, 1)
        if not self.pad_zero:
            zero_unmask = (1 - zero_idx).reshape(b, -1, 1) * unmask
            expr_emb = zero_unmask * self.zero_emb + (1 - zero_unmask) * expr_emb

        expr_emb = gene_emb + expr_emb
        cls_token = self.cls_token.expand(b, -1, -1)
        expr_emb = torch.cat([cls_token, expr_emb], dim=1)
        if self.pad_zero:
            expr_emb = expr_emb * zero_idx.reshape(b, -1, 1)
        mask_pos = torch.cat([torch.ones(b, 1, 1, dtype=unmask.dtype), unmask], dim=1).reshape(
            b, 1, -1, 1
        )
        for i in range(self.depth // 2):
            expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)
        if self.pad_zero:
            mask_pos = zero_idx.reshape(b, 1, -1, 1)
        else:
            mask_pos = None
        for i in range(self.depth // 2, self.depth):
            expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, attn_mask=mask_pos)
        return expr_emb, gene_emb

    def forward_encode(self, expr, gene, zero_idx):
        b, l = gene.shape
        emb, gene_emb = self.encode(expr, gene, zero_idx)
        cls_token, expr_emb = emb[:, 0], emb[:, 1:]
        cls_token = cls_token.reshape(b, -1)
        return expr_emb, gene_emb, cls_token

    def forward(self, masked_nzdata, nonz_gene, zero_idx):
        # Eval-mode forward (construct() with self.training=False in the real repo):
        # returns (gene-wise reconstruction, cell-wise reconstruction) predictions.
        expr_emb, gene_emb, cls_token = self.forward_encode(masked_nzdata, nonz_gene, zero_idx)
        if self.add_zero:
            gw_pred, _z_prob1 = self.value_dec(expr_emb)
            cw_pred, _z_prob2 = self.cellwise_dec(cls_token, gene_emb)
        else:
            gw_pred = self.value_dec(expr_emb)
            cw_pred = self.cellwise_dec(cls_token, gene_emb)
        return gw_pred, cw_pred


def build_cellfm():
    n_genes = 40
    cfg = {
        "lora": 0,
        "enc_dims": 32,
        "enc_nlayers": 2,
        "enc_num_heads": 4,
        "enc_dropout": 0.0,
        "dropout": 0.0,
        "recompute": False,
        "add_zero": False,
        "pad_zero": True,
        "label": False,
    }
    model = CellFM(n_genes, cfg)
    model.eval()
    return model


def example_input_cellfm():
    # Real forward signature: (masked_nzdata: (B, L), nonz_gene indices: (B, L) long,
    # zero_idx presence-mask: (B, L+1)). With pad_zero=True (real Config default),
    # value_enc sees a 2D expr tensor (no unmask/expr split), matching the real
    # `ValueEncoder.construct` `len(x.shape)==2` branch. The real data pipeline
    # (data_process.py::Prepare.attn_mask) builds `zero_idx` one element LONGER than
    # `masked_nzdata`/`nonz_gene` (`pad_len + self.pad`, pad=1) to also cover the
    # prepended cls-token slot inside `CellFM.encode`.
    batch, length = 2, 8
    n_genes = 40
    masked_nzdata = torch.rand(batch, length)
    nonz_gene = torch.randint(1, n_genes + 1, (batch, length))
    zero_idx = torch.ones(batch, length + 1)
    return masked_nzdata, nonz_gene, zero_idx


MENAGERIE_ENTRIES = [
    ("CellFM", build_cellfm, example_input_cellfm, 2024, "SOURCE_AVAILABLE"),
]

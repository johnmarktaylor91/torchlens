# SOURCE: vendored from Song-Jun-Ho/MEMTO @ b118c58ad89d703628e65698a0d0c37e160d11f5
# Files vendored (paths in upstream repo under MEMTO/model/):
#   Transformer.py, attn_layer.py, embedding.py, ours_memory_module.py
# MEMTO: Memory-guided Transformer for multivariate time-series anomaly detection
# (NeurIPS 2023, https://github.com/Song-Jun-Ho/MEMTO).
#
# Minimal, non-architectural fixes applied while vendoring (imports/paths only,
# per RUNG-2 rules):
#   - Removed relative-package imports (`.attn_layer` / `.embedding` /
#     `.ours_memory_module`) in favor of local in-module classes (single-file
#     staging module).
#   - Replaced hardcoded `.cuda()` calls in InputEmbedding.forward and
#     MemoryModule.read/update/__init__ with device-aware `.to(query.device)` /
#     `.to(x.device)` so the real model can run on CPU during tracing; the
#     original code assumed a CUDA-only environment. No layer, mechanism, or
#     control-flow path was changed -- only the hardcoded device string.
#   - MemoryModule.__init__ guarded the `phase_type == 'test'` branch (which
#     `torch.load`s a checkpoint file from disk) so construction with
#     phase_type=None (first-train phase, matching upstream's default training
#     entry point) does not require any external file.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from sklearn.cluster import KMeans  # noqa: F401  (imported by upstream module; unused at inference)


# ---------------------------------------------------------------------------
# attn_layer.py (vendored verbatim, architecture unchanged)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries : N x L x Head x d
        keys : N x L(s) x Head x d
        values : N x L x Head x d
        """
        N, L, Head, C = queries.shape

        scale = self.scale if self.scale is not None else 1.0 / sqrt(C)

        attn_scores = torch.einsum("nlhd,nshd->nhls", queries, keys)  # N x Head x L x L
        attn_weights = self.dropout(torch.softmax(scale * attn_scores, dim=-1))

        updated_values = torch.einsum("nhls,nshd->nlhd", attn_weights, values)  # N x L x Head x d

        return updated_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(
        self,
        window_size,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        mask_flag=False,
        scale=None,
        dropout=0.0,
    ):
        super(AttentionLayer, self).__init__()

        self.d_keys = d_keys if d_keys is not None else (d_model // n_heads)
        self.d_values = d_values if d_values is not None else (d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model  # d_model = C

        # Linear projections to Q, K, V
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)

        self.out_proj = nn.Linear(self.n_heads * self.d_values, self.d_model)

        self.attn = Attention(
            window_size=window_size, mask_flag=mask_flag, scale=scale, dropout=dropout
        )

    def forward(self, input):
        """
        input : N x L x C(=d_model)
        """
        N, L, _ = input.shape

        Q = self.W_Q(input).contiguous().view(N, L, self.n_heads, -1)
        K = self.W_K(input).contiguous().view(N, L, self.n_heads, -1)
        V = self.W_V(input).contiguous().view(N, L, self.n_heads, -1)

        updated_V = self.attn(Q, K, V)  # N x L x Head x d_values
        out = updated_V.view(N, L, -1)

        return self.out_proj(out)  # N x L x C(=d_model)


# ---------------------------------------------------------------------------
# embedding.py (vendored; only `.cuda()` -> `.to(x.device)` changed, see header)
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        super(TokenEmbedding, self).__init__()
        pad = 1 if torch.__version__ >= "1.5.0" else 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=3,
            padding=pad,
            padding_mode="circular",
            bias=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)

        return x


class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, device, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # upstream: `+ self.pos_embedding(x).cuda()` (CUDA-only); made device-aware.
        x = self.token_embedding(x) + self.pos_embedding(x).to(x.device)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# ours_memory_module.py (vendored; `.cuda()` -> `.to(...)`, phase_type='test'
# torch.load branch guarded to only fire when explicitly requested, see header)
# ---------------------------------------------------------------------------
class MemoryModule(nn.Module):
    def __init__(
        self,
        n_memory,
        fea_dim,
        shrink_thres=0.0025,
        device=None,
        memory_init_embedding=None,
        phase_type=None,
        dataset_name=None,
    ):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.shrink_thres = shrink_thres
        self.device = device
        self.phase_type = phase_type
        self.memory_init_embedding = memory_init_embedding

        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)

        # mem (memory items) : M x C
        # first train -> memory_initial : False / memory_init_embedding : None
        # second_train -> memory_initial : False / memory_init_embedding : kmeans item
        # test -> memory_initial: False / memory_init_embedding : vectors from second train phase
        if self.memory_init_embedding is None:
            if self.phase_type == "test":
                load_path = f"./memory_item/{dataset_name}_memory_item.pth"
                self.mem = torch.load(load_path)
                print(load_path)
                print("loading memory item vectors trained from kmeans (for test phase)")
            else:
                # first train
                self.mem = F.normalize(
                    torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1
                )
        else:
            # second train
            if self.phase_type == "second_train":
                self.mem = memory_init_embedding

    # relu based hard shrinkage function, only works for positive values
    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)

        return output

    def get_attn_score(self, query, key):
        """
        Calculating attention score with sparsity regularization
        query (initial features) : (NxL) x C or N x C -> T x C
        key (memory items): M x C
        """
        # upstream: `key.cuda()` (CUDA-only); made device-aware.
        attn = torch.matmul(query, torch.t(key.to(query.device)))  # (TxC) x (CxM) -> TxM
        attn = F.softmax(attn, dim=-1)

        if self.shrink_thres > 0:
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            # re-normalize
            attn = F.normalize(attn, p=1, dim=1)

        return attn

    def read(self, query):
        """
        query (initial features) : (NxL) x C or N x C -> T x C
        read memory items and get new robust features,
        while memory items(cluster centers) being fixed
        """
        self.mem = self.mem.to(query.device)
        attn = self.get_attn_score(query, self.mem.detach())  # T x M
        add_memory = torch.matmul(attn, self.mem.detach())  # T x C

        read_query = torch.cat((query, add_memory), dim=1)  # T x 2C

        return {"output": read_query, "attn": attn}

    def update(self, query):
        """
        Update memory items(cluster centers)
        Fix Encoder parameters (detach)
        query (encoder output features) : (NxL) x C or N x C -> T x C
        """
        self.mem = self.mem.to(query.device)
        attn = self.get_attn_score(self.mem, query.detach())  # M x T
        add_mem = torch.matmul(attn, query.detach())  # M x C

        # update gate : M x C
        update_gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem))  # M x C
        self.mem = (1 - update_gate) * self.mem + update_gate * add_mem

    def forward(self, query):
        """
        query (encoder output features) : N x L x C or N x C
        """
        s = query.data.shape
        l = len(s)  # noqa: E741 (upstream variable name, kept faithful)

        query = query.contiguous()
        query = query.view(-1, s[-1])  # N x L x C or N x C -> T x C

        # update memory items(cluster centers), while encoder parameters being fixed
        if self.phase_type != "test":
            self.update(query)

        # get new robust features, while memory items(cluster centers) being fixed
        outs = self.read(query)

        read_query, attn = outs["output"], outs["attn"]

        if l == 2:
            pass
        elif l == 3:
            read_query = read_query.view(s[0], s[1], 2 * s[2])
            attn = attn.view(s[0], s[1], self.n_memory)
        else:
            raise TypeError("Wrong input dimension")
        """
        output : N x L x 2C or N x 2C
        attn : N x L x M or N x M
        """
        return {"output": read_query, "attn": attn, "memory_init_embedding": self.mem}


# ---------------------------------------------------------------------------
# Transformer.py (vendored verbatim, architecture unchanged)
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)  # N x L x C(=d_model)


# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation="relu", dropout=0.1):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.decoder_layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        self.decoder_layer2 = nn.Conv1d(in_channels=d_ff, out_channels=c_out, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = nn.BatchNorm1d(d_ff)

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        out : reconstructed output
        """
        out = self.out_linear(x)
        return out  # N x L x c_out


class TransformerVar(nn.Module):
    # ours: shrink_thres=0.0025
    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        n_memory,
        shrink_thres=0,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        device=None,
        memory_init_embedding=None,
        memory_initial=False,
        phase_type=None,
        dataset_name=None,
    ):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Encoding
        self.embedding = InputEmbedding(
            in_dim=enc_in, d_model=d_model, dropout=dropout, device=device
        )  # N x L x C(=d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(win_size, d_model, n_heads, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.mem_module = MemoryModule(
            n_memory=n_memory,
            fea_dim=d_model,
            shrink_thres=shrink_thres,
            device=device,
            memory_init_embedding=memory_init_embedding,
            phase_type=phase_type,
            dataset_name=dataset_name,
        )

        # ours
        self.weak_decoder = Decoder(2 * d_model, c_out, d_ff=d_ff, activation="gelu", dropout=0.1)

    def forward(self, x):
        """
        x (input time window) : N x L x enc_in
        """
        x = self.embedding(x)  # embeddin : N x L x C(=d_model)
        queries = out = self.encoder(x)  # encoder out : N x L x C(=d_model)

        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = (
            outputs["output"],
            outputs["attn"],
            outputs["memory_init_embedding"],
        )

        mem = self.mem_module.mem

        if self.memory_initial:
            return {"out": out, "memory_item_embedding": None, "queries": queries, "mem": mem}
        else:
            out = self.weak_decoder(out)
            """
            out (reconstructed input time window) : N x L x enc_in
            enc_in == c_out
            """
            return {
                "out": out,
                "memory_item_embedding": memory_item_embedding,
                "queries": queries,
                "mem": mem,
                "attn": attn,
            }


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


def build_memto():
    """Tiny MEMTO (memory-guided transformer) for multivariate time-series
    anomaly detection, matching the real upstream constructor signature."""
    return TransformerVar(
        win_size=16,
        enc_in=8,
        c_out=8,
        n_memory=10,
        shrink_thres=0.0025,
        d_model=32,
        n_heads=2,
        e_layers=2,
        d_ff=32,
        dropout=0.0,
        activation="gelu",
        device="cpu",
        memory_init_embedding=None,
        memory_initial=False,
        phase_type=None,
        dataset_name=None,
    )


def example_input_memto():
    return torch.randn(2, 16, 8)


MENAGERIE_ENTRIES = [
    ("MEMTO", build_memto, example_input_memto, 2023, "reconstruction"),
]

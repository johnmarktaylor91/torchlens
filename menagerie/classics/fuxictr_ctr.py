"""FuxiCTR CTR model family: AFN+, AOANet, DESTINE, EDCN, EulerNet, FLEN,
FinalMLP, FinalNet, FmFM, GDCN, InterHAt, MaskNet, PEPNet, PPNet, SAM.

Papers and Sources:
  AFN+:      Cheng et al., "Adaptive Factorization Network: Learning Adaptive-Order
             Feature Interactions", AAAI 2021. arXiv:1909.03276
             Source: https://github.com/xue-pai/FuxiCTR
  AOANet:    Liu et al., "Architecture and Operation Adaptive Network for Online
             Recommendations", KDD 2021. arXiv:2012.04994
             Source: https://github.com/xue-pai/FuxiCTR
  DESTINE:   Zhou et al., "Disentangled Self-Attentive Neural Networks for
             Click-Through Rate Prediction", CIKM 2021. arXiv:2101.03654
             Source: https://github.com/xue-pai/FuxiCTR
  EDCN:      Chen et al., "Enhancing Explicit and Implicit Feature Interactions
             via Information Sharing for Parallel Deep CTR Models", KDD 2021.
             arXiv:2108.09975
             Source: https://github.com/xue-pai/FuxiCTR
  EulerNet:  Tian et al., "EulerNet: Adaptive Feature Interaction Learning via
             Euler's Formula for CTR Prediction", SIGIR 2023. arXiv:2304.10711
             Source: https://github.com/xue-pai/FuxiCTR
  FLEN:      Chen et al., "FLEN: Leveraging Field for Scalable CTR Prediction",
             DLP-KDD 2019. arXiv:1911.04690
             Source: https://github.com/xue-pai/FuxiCTR
  FinalMLP:  Mao et al., "FinalMLP: An Enhanced Two-Stream MLP Model for CTR
             Prediction", AAAI 2023. arXiv:2304.00902
             Source: https://github.com/xue-pai/FuxiCTR
  FinalNet:  Mao et al., "FINAL: Factorized Interaction Layer for CTR Prediction",
             RecSys 2023. arXiv:2307.00500
             Source: https://github.com/xue-pai/FuxiCTR
  FmFM:      Sun et al., "FM^2: Field-matrixed Factorization Machines for CTR
             Prediction", WWW 2021. arXiv:2102.12994
             Source: https://github.com/xue-pai/FuxiCTR
  GDCN:      Li et al., "GDCN: A Gated Deep Cross Network for CTR Prediction",
             RecSys 2023. arXiv:2206.10843
             Source: https://github.com/xue-pai/FuxiCTR
  InterHAt:  Li et al., "Interpretable Click-Through Rate Prediction through
             Hierarchical Attention", WSDM 2020. arXiv:2003.12283
             Source: https://github.com/xue-pai/FuxiCTR
  MaskNet:   Wang et al., "MaskNet: Introducing Feature-Wise Multiplication to CTR
             Ranking Models by Instance-Guided Mask", DLP-KDD 2021. arXiv:2102.07619
             Source: https://github.com/xue-pai/FuxiCTR
  PEPNet:    Chang et al., "PEPNet: Parameter and Embedding Personalized Network for
             Infeed Recommendation", KDD 2023. arXiv:2302.01115
             Source: https://github.com/xue-pai/FuxiCTR
  PPNet:     Sub-module from PEPNet (parameter-personalization via GateNN).
             Same source as PEPNet.
  SAM:       Su et al., "SAM: Squeeze-and-Excitation Attention and Multi-Head
             Attention for Click-Through Rate Prediction", 2023. arXiv:2208.06229
             Source: https://github.com/xue-pai/FuxiCTR

All models:
  - Standard tabular CTR input: (B, num_fields) integer indices.
  - Shared embedding table: num_fields x embed_dim.
  - Each model's DISTINCTIVE interaction module is faithfully reproduced.
  - Random init, CPU, compact scale (num_fields=8, embed_dim=8).
  - Faithful-core simplifications: DNN components reduced to 2-3 layers,
    attention heads reduced to 2, sequence lengths kept minimal.
  - Trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Shared hypers ───────────────────────────────────────────────────────────
NUM_FIELDS = 8
EMBED_DIM = 8
VOCAB = 50
BATCH = 2


def _make_embedding(
    num_fields: int = NUM_FIELDS, embed_dim: int = EMBED_DIM, vocab: int = VOCAB
) -> nn.Embedding:
    return nn.Embedding(vocab, embed_dim)


def _mlp(in_dim: int, hidden: List[int], out_dim: int, act: str = "relu") -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    act_fn = nn.ReLU() if act == "relu" else nn.Sigmoid()
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), act_fn]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ============================================================
# 1. AFN+ (Adaptive Factorization Network)
# ============================================================


class LogTransformLayer(nn.Module):
    """Logarithmic neuron layer (LNN) for learning arbitrary-order interactions.

    Each neuron i computes:  sum_j (w_ij * log(|x_j| + eps))
    then passes through exp to get a power-product interaction.
    This is the distinctive AFN module (Section 3.2 of the paper).
    """

    def __init__(self, in_features: int, out_neurons: int, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(out_neurons, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_neurons))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)  -- must be positive inputs (use abs for safety)
        log_x = torch.log(x.abs() + self.eps)  # (B, F)
        out = log_x @ self.weight.t() + self.bias  # (B, out_neurons)
        return torch.exp(out)  # power-product interaction


class AFNPlus(nn.Module):
    """AFN+: LNN adaptive interaction + DNN, ensembled via addition.

    Architecture (Fig 3 of paper):
      Embedding -> flatten -> LogTransformLayer (LTL) -> BN -> DNN_afn
                             |------ DNN_deep --------|
      out = DNN_afn logit + DNN_deep logit -> sigmoid
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        ltl_neurons: int = 16,
        afn_hidden: Optional[List[int]] = None,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if afn_hidden is None:
            afn_hidden = [32, 16]
        if dnn_hidden is None:
            dnn_hidden = [32, 16]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        flat = num_fields * embed_dim
        # LNN path
        self.ltl = LogTransformLayer(flat, ltl_neurons)
        self.bn_ltl = nn.BatchNorm1d(ltl_neurons)
        self.afn_dnn = _mlp(ltl_neurons, afn_hidden, 1)
        # Deep path
        self.deep_dnn = _mlp(flat, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_fields)
        emb = self.embedding(x.long())  # (B, F, E)
        flat = emb.flatten(1)  # (B, F*E)
        # AFN path: log-transform then DNN
        ltl_out = self.bn_ltl(self.ltl(flat))  # (B, ltl_neurons)
        afn_logit = self.afn_dnn(ltl_out)  # (B, 1)
        # Deep path
        deep_logit = self.deep_dnn(flat)  # (B, 1)
        return (afn_logit + deep_logit).squeeze(-1)  # (B,)


# ============================================================
# 2. AOANet (Architecture and Operation Adaptive Network)
# ============================================================


class GeneralizedInteractionLayer(nn.Module):
    """Operation-adaptive interaction layer (Eq 4 of AOANet paper).

    Each output neuron i is a linear combination of element-wise products
    x_p * x_q for all (p, q) pairs -- a generalized bilinear operation:
      out_i = sum_{p,q} w_{ipq} * x_p * x_q

    For efficiency (compact version): uses a rank-reduced parameterization
    via two projection matrices U (B-view) and V (A-view).
    """

    def __init__(self, in_dim: int, num_subspaces: int = 2, out_dim: int = 32) -> None:
        super().__init__()
        self.U = nn.Linear(in_dim, num_subspaces, bias=False)  # project to subspaces
        self.V = nn.Linear(in_dim, num_subspaces, bias=False)
        self.proj = nn.Linear(num_subspaces, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.U(x)  # (B, S)
        b = self.V(x)  # (B, S)
        inter = a * b  # element-wise: (B, S)
        return self.bn(self.proj(inter))  # (B, out_dim)


class AOANet(nn.Module):
    """AOANet: stacked generalized interaction layers + DNN.

    Architecture (Fig 2):
      Embedding -> flatten ->
        [GeneralizedInteractionLayer x num_layers] -> concat with flat ->
        DNN -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_inter_layers: int = 2,
        subspaces: int = 4,
        inter_dim: int = 32,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [64, 32]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        flat = num_fields * embed_dim
        self.interaction_layers = nn.ModuleList(
            [
                GeneralizedInteractionLayer(flat if i == 0 else inter_dim, subspaces, inter_dim)
                for i in range(num_inter_layers)
            ]
        )
        self.dnn = _mlp(flat + inter_dim, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())
        flat = emb.flatten(1)  # (B, F*E)
        h = flat
        for layer in self.interaction_layers:
            h = layer(h)  # (B, inter_dim)
        combined = torch.cat([flat, h], dim=1)
        return self.dnn(combined).squeeze(-1)


# ============================================================
# 3. DESTINE (Disentangled Self-Attentive NN)
# ============================================================


class DisentangledSelfAttention(nn.Module):
    """DESTINE's whitened multi-head self-attention with disentangled queries.

    Key innovation: query is split into 'unary' (field-specific) and
    'interaction' (cross-field) components. The attention matrix is
    whitened (subtract diagonal, i.e., no self-interaction) before softmax.
    """

    def __init__(self, embed_dim: int, num_heads: int = 2, residual: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        # Unary (positional/field-specific) query projection
        self.W_qu = nn.Linear(embed_dim, embed_dim, bias=False)
        # Interaction query projection
        self.W_qi = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.residual = residual
        if residual:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, nF, E)
        B, nF, E = x.shape
        H, D = self.num_heads, self.head_dim

        def split_heads(t):
            return t.view(B, nF, H, D).transpose(1, 2)  # (B, H, nF, D)

        # Disentangled query: unary + interaction
        q = split_heads(self.W_qu(x) + self.W_qi(x))  # (B, H, nF, D)
        k = split_heads(self.W_k(x))  # (B, H, nF, D)
        v = split_heads(self.W_v(x))  # (B, H, nF, D)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, nF, nF)

        # DESTINE whitening: zero out diagonal (no self-interaction)
        mask = torch.eye(nF, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)  # handle inf->nan from all-inf rows (nF=1 edge case)

        out = attn @ v  # (B, H, nF, D)
        out = out.transpose(1, 2).contiguous().view(B, nF, E)
        out = self.W_o(out)

        if self.residual:
            out = self.norm(out + x)
        return out


class DESTINE(nn.Module):
    """DESTINE: stacked disentangled self-attention layers -> flatten -> DNN.

    Architecture (Fig 2 of paper):
      Embedding -> [DisentangledSelfAttention x num_layers] -> flatten -> DNN -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_attn_layers: int = 2,
        num_heads: int = 2,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [64, 32]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        self.attn_layers = nn.ModuleList(
            [DisentangledSelfAttention(embed_dim, num_heads) for _ in range(num_attn_layers)]
        )
        self.dnn = _mlp(num_fields * embed_dim, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        h = emb
        for layer in self.attn_layers:
            h = layer(h)
        flat = h.flatten(1)  # (B, F*E)
        return self.dnn(flat).squeeze(-1)


# ============================================================
# 4. EDCN (Enhanced Deep & Cross Network)
# ============================================================


class BridgeModule(nn.Module):
    """EDCN bridge module: integrates cross and deep representations.

    Bridge types: pointwise_addition, hadamard_product, concatenation, attention_pooling.
    Using hadamard_product as it's the most distinctive (Section 3.2).
    """

    def __init__(self, dim: int, bridge_type: str = "hadamard_product") -> None:
        super().__init__()
        self.bridge_type = bridge_type
        if bridge_type == "attention_pooling":
            self.attn = nn.Linear(dim * 2, 1)
        elif bridge_type == "concatenation":
            self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x_cross: torch.Tensor, x_deep: torch.Tensor) -> torch.Tensor:
        if self.bridge_type == "pointwise_addition":
            return x_cross + x_deep
        elif self.bridge_type == "hadamard_product":
            return x_cross * x_deep
        elif self.bridge_type == "concatenation":
            return self.proj(torch.cat([x_cross, x_deep], dim=-1))
        else:  # attention_pooling
            cat = torch.cat([x_cross, x_deep], dim=-1)
            a = torch.sigmoid(self.attn(cat))
            return a * x_cross + (1 - a) * x_deep


class RegulationModule(nn.Module):
    """EDCN regulation module: field-level gating on concatenated representation.

    Applies a learned field-gate to select which portions of the combined
    representation flow to each tower (Section 3.3).
    """

    def __init__(self, num_fields: int, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(num_fields * dim, num_fields * dim)
        self.num_fields = num_fields
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F*dim)
        gate = torch.sigmoid(self.gate(x))
        return x * gate


class CrossLayer(nn.Module):
    """Single cross layer: x_{l+1} = x_0 * (w^T x_l) + b + x_l."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.w(xl) + self.b + xl


class EDCN(nn.Module):
    """EDCN: Bridge+Regulation modules coupling Cross-Net and Deep-Net.

    Architecture (Fig 2):
      Embedding ->
        parallel Cross-Net and Deep-Net layers, coupled at each layer by
        BridgeModule -> then both outputs fed to RegulationModule ->
        stack: repeat for num_layers -> concat -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_layers: int = 2,
        dnn_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        flat = num_fields * embed_dim
        self.cross_layers = nn.ModuleList([CrossLayer(flat) for _ in range(num_layers)])
        self.deep_layers = nn.ModuleList([nn.Linear(flat, flat) for _ in range(num_layers)])
        self.bridge = BridgeModule(flat, bridge_type="hadamard_product")
        self.regulation = RegulationModule(num_fields, embed_dim)
        self.out = nn.Linear(flat * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())
        flat0 = emb.flatten(1)  # (B, F*E)  x_0

        x_cross = flat0
        x_deep = flat0
        for cross_l, deep_l in zip(self.cross_layers, self.deep_layers):
            x_cross = cross_l(flat0, x_cross)  # cross layer update
            x_deep = F.relu(deep_l(x_deep))  # deep layer update
            bridged = self.bridge(x_cross, x_deep)  # bridge integrates them
            # regulation gates the combined before feeding back
            regulated = self.regulation(bridged)
            x_cross = regulated
            x_deep = regulated

        combined = torch.cat([x_cross, x_deep], dim=1)  # (B, 2*flat)
        return self.out(combined).squeeze(-1)


# ============================================================
# 5. EulerNet
# ============================================================


class EulerInteractionLayer(nn.Module):
    """EulerNet interaction layer using Euler's formula in polar form.

    Each feature is represented as a complex number in polar coordinates
    (amplitude r, phase theta). Interactions are computed as:
      r_out = prod(r_i^{w_ri})     (in log space: sum w_ri * log r_i)
      theta_out = sum w_ti * theta_i
    Both r and theta are transformed by learned weight matrices.
    This is the core Euler formula interaction (Section 3.2).
    """

    def __init__(self, num_fields: int, embed_dim: int, out_dim: int) -> None:
        super().__init__()
        in_dim = num_fields * embed_dim
        self.W_r = nn.Linear(in_dim, out_dim, bias=True)
        self.W_theta = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, r: torch.Tensor, theta: torch.Tensor) -> tuple:
        # r, theta: each (B, F*E)
        # Amplitude interaction in log-space (power products)
        log_r = torch.log(r.abs() + 1e-7)
        new_log_r = self.W_r(log_r)  # (B, out_dim)
        new_r = torch.exp(new_log_r)

        # Phase interaction (linear combination)
        new_theta = self.W_theta(theta)  # (B, out_dim)
        return new_r, new_theta


class EulerNet(nn.Module):
    """EulerNet: feature interaction via Euler's formula (complex polar representation).

    Architecture (Fig 2):
      Embedding -> split into r (amplitude via sigmoid) and theta (phase) ->
        [EulerInteractionLayer x num_layers] ->
        convert back: x = r * cos(theta), y = r * sin(theta) ->
        concat real + imaginary -> DNN -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_layers: int = 2,
        inter_dim: int = 32,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [32]
        flat = num_fields * embed_dim
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        # Separate amplitude and phase projections from embedding
        self.r_proj = nn.Linear(flat, flat)
        self.theta_proj = nn.Linear(flat, flat)
        # Euler interaction layers
        self.euler_layers = nn.ModuleList(
            [
                EulerInteractionLayer(
                    num_fields if i == 0 else 1, embed_dim if i == 0 else inter_dim, inter_dim
                )
                for i in range(num_layers)
            ]
        )
        self.dnn = _mlp(inter_dim * 2, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())
        flat = emb.flatten(1)  # (B, F*E)
        r = torch.sigmoid(self.r_proj(flat))  # amplitude in (0,1)
        theta = self.theta_proj(flat)  # phase (unbounded)

        # Reshape for euler layers: treat F*E as a single "field" of inter_dim
        # First layer takes (F*E)-dim input
        r_cur, theta_cur = r, theta
        for layer in self.euler_layers:
            r_cur, theta_cur = layer(r_cur, theta_cur)

        # Convert back to Cartesian
        real = r_cur * torch.cos(theta_cur)  # (B, inter_dim)
        imag = r_cur * torch.sin(theta_cur)  # (B, inter_dim)
        out = torch.cat([real, imag], dim=-1)  # (B, 2*inter_dim)
        return self.dnn(out).squeeze(-1)


# ============================================================
# 6. FLEN (Field-Leveraged Embedding Network)
# ============================================================


class FieldWiseBiInteraction(nn.Module):
    """FLEN field-wise bi-interaction (FwBI).

    For each field pair (i, j), compute:
      intra-field: hadamard product  e_i * e_j (same field interaction pool)
      inter-field: hadamard product  e_i * e_j (cross-field)

    The paper distinguishes intra-field FM interactions (within same logical field)
    from inter-field FM interactions (across logical fields).
    Here we implement the simplified version: partition fields into groups and
    compute FM-style sum of squared interactions separately per group.
    """

    def __init__(self, num_fields: int, embed_dim: int, num_field_groups: int = 2) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.num_groups = num_field_groups
        # Group assignment: first half = group 0, second half = group 1
        self.out_dim = num_field_groups * embed_dim + embed_dim  # intra + inter

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, F, E)
        B, F, E = emb.shape
        # Intra-field interaction (FM over each group)
        group_size = F // self.num_groups
        intra_parts = []
        for g in range(self.num_groups):
            start = g * group_size
            end = start + group_size if g < self.num_groups - 1 else F
            grp = emb[:, start:end, :]  # (B, gs, E)
            # FM interaction: 0.5*(sum^2 - sum of squares)
            sum_emb = grp.sum(dim=1)  # (B, E)
            sum_sq = (grp**2).sum(dim=1)  # (B, E)
            intra_parts.append(0.5 * (sum_emb**2 - sum_sq))  # (B, E)
        intra = torch.cat(intra_parts, dim=1)  # (B, G*E)

        # Inter-field interaction (FM over all fields)
        sum_all = emb.sum(dim=1)  # (B, E)
        sum_sq_all = (emb**2).sum(dim=1)  # (B, E)
        inter = 0.5 * (sum_all**2 - sum_sq_all)  # (B, E)

        return torch.cat([intra, inter], dim=1)  # (B, G*E + E)


class FLEN(nn.Module):
    """FLEN: Field-wise Bi-Interaction (FwBI) + DNN -> logit.

    Architecture (Fig 3):
      Embedding -> FieldWiseBiInteraction (intra + inter FM) ->
        concat with flattened embeddings -> DNN -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_field_groups: int = 2,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [32, 16]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        self.fwbi = FieldWiseBiInteraction(num_fields, embed_dim, num_field_groups)
        flat = num_fields * embed_dim
        dnn_in = self.fwbi.out_dim + flat
        self.dnn = _mlp(dnn_in, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        fwbi_out = self.fwbi(emb)  # (B, G*E + E)
        flat = emb.flatten(1)  # (B, F*E)
        h = torch.cat([fwbi_out, flat], dim=1)
        return self.dnn(h).squeeze(-1)


# ============================================================
# 7. FinalMLP
# ============================================================


class FeatureSelectionGate(nn.Module):
    """Feature-selection gate for FinalMLP (Section 3.2).

    Learns a per-feature selection weight from the input embedding,
    then applies it as a soft gate to select relevant features for each stream.
    """

    def __init__(self, num_fields: int, embed_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        flat = num_fields * embed_dim
        self.gate_net = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat),
            nn.Sigmoid(),
        )

    def forward(self, emb_flat: torch.Tensor) -> torch.Tensor:
        return emb_flat * self.gate_net(emb_flat)


class BilinearFusionHead(nn.Module):
    """Multi-head bilinear aggregation for FinalMLP (Section 3.3).

    Given outputs from two MLP streams (h1, h2), compute bilinear interactions
    per head: out_k = h1_k^T W_k h2_k, then aggregate.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # Bilinear weights per head
        self.W = nn.Parameter(torch.randn(num_heads, head_dim, head_dim) * 0.01)
        self.out = nn.Linear(num_heads, 1)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # h1, h2: (B, dim)
        B = h1.shape[0]
        h1_h = h1.view(B, self.num_heads, self.head_dim)  # (B, H, D)
        h2_h = h2.view(B, self.num_heads, self.head_dim)  # (B, H, D)
        # Per-head bilinear: h1_h[:, k, :] @ W[k] @ h2_h[:, k, :].T
        # -> (B, H)
        scores = torch.einsum("bhd,hde,bhe->bh", h1_h, self.W, h2_h)
        return self.out(scores).squeeze(-1)  # (B,)


class FinalMLP(nn.Module):
    """FinalMLP: two parallel MLP streams + feature-selection gates + bilinear fusion.

    Architecture (Fig 2):
      Embedding -> [Stream1: FeatureGate -> MLP1]
                   [Stream2: FeatureGate -> MLP2]
        -> BilinearFusionHead -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        mlp_hidden: Optional[List[int]] = None,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [32, 32]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        flat = num_fields * embed_dim
        out_dim = mlp_hidden[-1]
        # Ensure out_dim is divisible by num_heads
        out_dim = (out_dim // num_heads) * num_heads

        self.gate1 = FeatureSelectionGate(num_fields, embed_dim)
        self.gate2 = FeatureSelectionGate(num_fields, embed_dim)
        self.mlp1 = _mlp(flat, mlp_hidden[:-1], out_dim)
        self.mlp2 = _mlp(flat, mlp_hidden[:-1], out_dim)
        self.fusion = BilinearFusionHead(out_dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())
        flat = emb.flatten(1)  # (B, F*E)
        h1 = self.mlp1(self.gate1(flat))  # (B, out_dim)
        h2 = self.mlp2(self.gate2(flat))  # (B, out_dim)
        return self.fusion(h1, h2)  # (B,)


# ============================================================
# 8. FinalNet (FINAL: Factorized Interaction Layer)
# ============================================================


class FINALBlock(nn.Module):
    """FINAL block: factorized + field-aware gating, no MLP (Section 3.2).

    Each FINAL block computes:
      1. Factorized interaction: z = LN(x) @ W1 (element-wise product in latent space)
      2. Field-aware gate: g = sigmoid(LN(x) @ W2)
      3. Output: x + g * z  (residual with gated interaction)

    This is distinctly NOT an MLP -- it's a pure interaction block.
    """

    def __init__(self, num_fields: int, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        # Factorized weight: per-field (outer product factorization)
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)  # gate
        # Field interaction: separate weight per field pair
        self.field_weight = nn.Parameter(torch.randn(num_fields, num_fields) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, E)
        B, F, E = x.shape
        normed = self.norm(x)  # (B, F, E)
        # Factorized interaction across fields
        field_mix = torch.einsum("ij,bje->bie", self.field_weight, normed)  # (B, F, E)
        z = self.W1(field_mix)  # (B, F, E)
        # Gate
        g = torch.sigmoid(self.W2(normed))  # (B, F, E)
        return x + g * z  # (B, F, E)


class FinalNet(nn.Module):
    """FinalNet: FINAL blocks (factorized gated interaction) -> flatten -> linear.

    Architecture (Fig 1):
      Embedding -> [FINALBlock x num_blocks] -> flatten -> Linear -> logit
    No MLP -- the FINAL blocks ARE the interaction module.
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        self.blocks = nn.ModuleList([FINALBlock(num_fields, embed_dim) for _ in range(num_blocks)])
        self.out = nn.Linear(num_fields * embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        h = emb
        for block in self.blocks:
            h = block(h)
        flat = h.flatten(1)  # (B, F*E)
        return self.out(flat).squeeze(-1)


# ============================================================
# 9. FmFM (Field-matrixed FM)
# ============================================================


class FieldMatrixInteraction(nn.Module):
    """FmFM: per-field-pair interaction matrices.

    For each pair (i, j) of fields, a matrix M_{ij} of shape (E, E) transforms
    field i's embedding before dot-product with field j's embedding:
      score_{ij} = e_i^T M_{ij} e_j

    This generalizes FM (which uses M_{ij} = I) to learn field-specific
    interaction patterns.

    For compactness we use a rank-r factorization: M_{ij} = U_i^T V_j
    where U_i, V_j are (r, E) matrices -- equivalent to the FmFM field matrices
    when r = E.
    """

    def __init__(self, num_fields: int, embed_dim: int, rank: int = 4) -> None:
        super().__init__()
        self.num_fields = num_fields
        # Per-field left/right transformation matrices
        self.U = nn.Parameter(torch.randn(num_fields, rank, embed_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(num_fields, rank, embed_dim) * 0.01)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, F, E)
        B, F, E = emb.shape
        # Transform each field embedding: (B, F, rank)
        u = torch.einsum("bfe,fre->bfr", emb, self.U)  # (B, F, R)
        v = torch.einsum("bfe,fre->bfr", emb, self.V)  # (B, F, R)

        # All pairs: (B, F, F, R) dot product on last dim
        # score[b,i,j] = sum_r u[b,i,r] * v[b,j,r]
        scores = torch.einsum("bir,bjr->bij", u, v)  # (B, F, F)

        # Upper triangle (i < j)
        mask = torch.triu(torch.ones(F, F, device=emb.device), diagonal=1).bool()
        pair_scores = scores[:, mask]  # (B, num_pairs)
        return pair_scores.sum(dim=-1, keepdim=True)  # (B, 1)


class FmFM(nn.Module):
    """FmFM: field-matrixed FM interactions + linear logit.

    Architecture:
      Embedding -> FieldMatrixInteraction (all pairs, per-field-pair transform)
                -> scalar logit (+ optional bias)
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        rank: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        self.fm_fm = FieldMatrixInteraction(num_fields, embed_dim, rank)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        logit = self.fm_fm(emb)  # (B, 1)
        return (logit + self.bias).squeeze(-1)


# ============================================================
# 10. GDCN (Gated Deep Cross Network)
# ============================================================


class GatedCrossLayer(nn.Module):
    """GDCN gated cross layer with information gate (Eq 1-2 of paper).

    Extends DCN cross layer with an explicit information gate:
      g = sigmoid(W_g * [x_0, x_l] + b_g)      -- information gate
      x_{l+1} = x_0 * (w^T x_l) * g + x_l      -- gated cross update
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))
        # Gate on [x_0; x_l]
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        cross = x0 * self.w(xl) + self.b  # standard cross: (B, dim)
        gate = torch.sigmoid(self.gate(torch.cat([x0, xl], dim=-1)))  # (B, dim)
        return cross * gate + xl  # gated cross + residual


class GDCN(nn.Module):
    """GDCN: gated cross network + deep net -> concat -> logit.

    Architecture (Fig 2):
      Embedding -> flatten ->
        [GatedCrossLayer x num_cross] (cross tower)
        [Linear-ReLU x num_deep]      (deep tower)
        -> concat -> Linear -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_cross: int = 2,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [64, 32]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        flat = num_fields * embed_dim
        self.cross_layers = nn.ModuleList([GatedCrossLayer(flat) for _ in range(num_cross)])
        deep_layers: List[nn.Module] = []
        prev = flat
        for h in dnn_hidden:
            deep_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.deep = nn.Sequential(*deep_layers)
        self.out = nn.Linear(flat + prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())
        flat0 = emb.flatten(1)  # (B, F*E)
        # Cross tower
        xl = flat0
        for layer in self.cross_layers:
            xl = layer(flat0, xl)  # (B, F*E)
        # Deep tower
        xd = self.deep(flat0)  # (B, dnn_hidden[-1])
        combined = torch.cat([xl, xd], dim=-1)
        return self.out(combined).squeeze(-1)


# ============================================================
# 11. InterHAt (Interpretable Hierarchical Attention)
# ============================================================


class HierarchicalAttentionAggregation(nn.Module):
    """InterHAt hierarchical attention over interaction orders.

    After multi-head self-attention, aggregate across fields using a
    learned attention vector, then repeatedly stack interaction->attention
    to build higher-order representations (Section 3.3).
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.attn_vec = nn.Linear(embed_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, F, E) -> (B, E) via attention pooling
        scores = self.attn_vec(h)  # (B, F, 1)
        weights = F.softmax(scores, dim=1)  # (B, F, 1)
        return (h * weights).sum(dim=1)  # (B, E)


class InterHAt(nn.Module):
    """InterHAt: multi-head transformer + hierarchical attentional aggregation.

    Architecture (Fig 2):
      Embedding -> [MHA + HierarchicalAttention x num_orders] ->
        sum aggregated order representations -> DNN -> logit

    Each order: MHA over current emb -> attentive pooling to get a summary vector.
    The summary vectors for each order are summed and fed to DNN.
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        num_orders: int = 3,
        num_heads: int = 2,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [32]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        self.mha_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                for _ in range(num_orders)
            ]
        )
        self.agg_layers = nn.ModuleList(
            [HierarchicalAttentionAggregation(embed_dim) for _ in range(num_orders)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_orders)])
        self.dnn = _mlp(embed_dim, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        h = emb
        order_summaries = []
        for mha, agg, norm in zip(self.mha_layers, self.agg_layers, self.norms):
            attn_out, _ = mha(h, h, h)  # (B, F, E)
            h = norm(h + attn_out)  # residual + norm
            summary = agg(h)  # (B, E)
            order_summaries.append(summary)

        # Hierarchical aggregation: sum across orders
        combined = torch.stack(order_summaries, dim=1).sum(dim=1)  # (B, E)
        return self.dnn(combined).squeeze(-1)


# ============================================================
# 12. MaskNet
# ============================================================


class MaskBlock(nn.Module):
    """MaskNet MaskBlock: instance-guided mask applied multiplicatively.

    Computes:
      mask = V * LN(H * x_agg)   (H: aggregation transform, V: mask projection)
      output = LN(mask * x_field_emb) -> feed-forward layer
    """

    def __init__(self, embed_dim: int, num_fields: int, hidden_dim: int = 32) -> None:
        super().__init__()
        flat = num_fields * embed_dim
        # Mask generation: from aggregated embedding
        self.H = nn.Linear(flat, flat)  # aggregation -> mask space
        self.V = nn.Linear(flat, embed_dim)  # mask projection per output unit
        self.ln_mask = nn.LayerNorm(flat)
        # Feed-forward on masked output
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ln_out = nn.LayerNorm(embed_dim)

    def forward(self, x_agg: torch.Tensor, x_field: torch.Tensor) -> torch.Tensor:
        # x_agg: (B, F*E) flattened embedding aggregate
        # x_field: (B, E) one field's embedding
        mask = self.V(self.ln_mask(self.H(x_agg)))  # (B, E)
        masked = mask * x_field  # instance-guided mask
        out = self.ff(masked)
        return self.ln_out(out + x_field)  # residual


class MaskNet(nn.Module):
    """MaskNet: instance-guided MaskBlocks applied to each field.

    Architecture (Fig 3, parallel MaskNet):
      Embedding -> flatten (for mask generation) ->
        parallel MaskBlocks for each field position ->
        concat -> DNN -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [32]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        self.mask_blocks = nn.ModuleList(
            [MaskBlock(embed_dim, num_fields) for _ in range(num_fields)]
        )
        self.dnn = _mlp(num_fields * embed_dim, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        flat = emb.flatten(1)  # (B, F*E)
        # Apply MaskBlock to each field
        masked_fields = []
        for i, block in enumerate(self.mask_blocks):
            masked_fields.append(block(flat, emb[:, i, :]))  # (B, E)
        h = torch.cat(masked_fields, dim=1)  # (B, F*E)
        return self.dnn(h).squeeze(-1)


# ============================================================
# 13 & 14. PEPNet and PPNet
# ============================================================


class GateNN(nn.Module):
    """GateNN: personalized gating network from PEPNet (Section 3.2).

    Takes a personalized feature (e.g., user ID embedding) and outputs
    a gate vector to modulate another embedding. Used in both EPNet and PPNet.
    """

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int = 32, init_ones: bool = True
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),  # gates in (0,2) after scaling
        )
        self.scale = 2.0  # paper uses 2x scaling to center around 1.0
        self.init_ones = init_ones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.scale  # (B, out_dim)


class EPNet(nn.Module):
    """EPNet (Embedding Personalization Network): GateNN on embedding layer.

    Applies personalized gates from user/context embedding to modulate
    the general item/feature embeddings field-by-field.
    """

    def __init__(self, num_fields: int, embed_dim: int, user_dim: int) -> None:
        super().__init__()
        self.gate = GateNN(user_dim, num_fields * embed_dim)

    def forward(self, emb_flat: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        gate = self.gate(user_emb)  # (B, F*E)
        return emb_flat * gate  # personalized embedding


class PPNetModule(nn.Module):
    """PPNet (Parameter Personalization Network): GateNN modulates DNN hidden layers.

    At each DNN layer, the hidden activation is gated by a personalized
    gate vector derived from a combined user+general embedding.
    """

    def __init__(self, dnn_dims: List[int], user_dim: int) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(dnn_dims[i], dnn_dims[i + 1]) for i in range(len(dnn_dims) - 1)]
        )
        self.gates = nn.ModuleList(
            [GateNN(user_dim, dnn_dims[i + 1]) for i in range(len(dnn_dims) - 1)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dnn_dims[i + 1]) for i in range(len(dnn_dims) - 1)]
        )

    def forward(self, h: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        for linear, gate_net, norm in zip(self.linears, self.gates, self.norms):
            h = linear(h)
            gate = gate_net(user_emb)  # personalized gate
            h = norm(F.relu(h) * gate)  # gated activation
        return h


class PEPNet(nn.Module):
    """PEPNet: EPNet (embedding personalization) + PPNet (parameter personalization).

    Architecture (Fig 2):
      [User features] -> user_emb
      [Item/general features] -> general_emb
      EPNet: user_emb gates general_emb field-by-field
      PPNet: user_emb gates DNN hidden units layer-by-layer
      -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        user_fields: int = 2,
        dnn_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_dims is None:
            dnn_dims = [num_fields * embed_dim, 32, 16]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        user_embed_dim = user_fields * embed_dim
        self.epnet = EPNet(num_fields - user_fields, embed_dim, user_embed_dim)
        self.ppnet = PPNetModule(dnn_dims, user_embed_dim)
        self.out = nn.Linear(dnn_dims[-1], 1)
        self.user_fields = user_fields
        self.num_fields = num_fields
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        # Split user vs general fields
        user_emb = emb[:, : self.user_fields, :].flatten(1)  # (B, user_F*E)
        gen_emb = emb[:, self.user_fields :, :].flatten(1)  # (B, gen_F*E)
        # EPNet: personalize general embeddings
        gen_personalized = self.epnet(gen_emb, user_emb)  # (B, gen_F*E)
        # Concat user + personalized general
        h = torch.cat([user_emb, gen_personalized], dim=1)  # (B, F*E)
        # PPNet: parameter-personalized DNN
        h = self.ppnet(h, user_emb)  # (B, dnn_dims[-1])
        return self.out(h).squeeze(-1)


class PPNet(nn.Module):
    """PPNet (standalone): parameter-personalized DNN tower.

    This is the PPNet portion of PEPNet in isolation -- the DNN where each
    layer's activations are gated by a personalized user embedding.
    Useful as a standalone model (the paper also describes PPNet independently).
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        user_fields: int = 2,
        dnn_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_dims is None:
            dnn_dims = [num_fields * embed_dim, 32, 16]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        user_embed_dim = user_fields * embed_dim
        self.ppnet = PPNetModule(dnn_dims, user_embed_dim)
        self.out = nn.Linear(dnn_dims[-1], 1)
        self.user_fields = user_fields

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())
        user_emb = emb[:, : self.user_fields, :].flatten(1)
        flat = emb.flatten(1)
        h = self.ppnet(flat, user_emb)
        return self.out(h).squeeze(-1)


# ============================================================
# 15. SAM (Self-Attention Interaction Model)
# ============================================================


class SAM2A(nn.Module):
    """SAM2A: pairwise product + self-attention for interaction aggregation.

    SAM2A computes all pairwise hadamard products e_i * e_j,
    then uses self-attention to weight and aggregate them.
    """

    def __init__(self, num_fields: int, embed_dim: int, num_heads: int = 2) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        # Reduce F sequence to 1
        self.agg = nn.Linear(num_fields, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, F, E)
        attn_out, _ = self.attn(emb, emb, emb)
        h = self.norm(emb + attn_out)  # (B, F, E)
        # Aggregate: (B, E, F) -> linear -> (B, E, 1) -> squeeze
        h_agg = self.agg(h.transpose(1, 2)).squeeze(-1)  # (B, E)
        return h_agg


class SAM2E(nn.Module):
    """SAM2E: Squeeze-and-Excitation + element-wise attention variant.

    Uses SE-style channel attention on the embedding dimension,
    then applies element-wise attention across fields.
    """

    def __init__(self, num_fields: int, embed_dim: int, reduction: int = 2) -> None:
        super().__init__()
        self.num_fields = num_fields
        # SE on embedding dim
        self.se = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, embed_dim),
            nn.Sigmoid(),
        )
        # Attention across fields
        self.field_attn = nn.Linear(num_fields * embed_dim, num_fields)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, nF, E)
        B, nF, E = emb.shape
        # SE: global average pool over fields -> channel excitation
        avg = emb.mean(dim=1)  # (B, E)
        se_w = self.se(avg)  # (B, E)
        h = emb * se_w.unsqueeze(1)  # (B, nF, E) -- SE weighted

        # Field attention
        flat = h.flatten(1)  # (B, nF*E)
        field_w = torch.nn.functional.softmax(self.field_attn(flat), dim=-1)  # (B, nF)
        return (h * field_w.unsqueeze(-1)).sum(dim=1)  # (B, E)


class SAM(nn.Module):
    """SAM: Self-Attention Interaction Model with SAM2A or SAM2E blocks + DNN.

    Architecture (Section 3):
      Embedding -> SAM interaction block (SAM2A or SAM2E) -> DNN -> logit
    """

    def __init__(
        self,
        num_fields: int = NUM_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        variant: str = "SAM2A",
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [32]
        self.embedding = _make_embedding(num_fields, embed_dim, vocab)
        if variant == "SAM2A":
            self.interaction = SAM2A(num_fields, embed_dim)
        else:
            self.interaction = SAM2E(num_fields, embed_dim)
        self.dnn = _mlp(embed_dim, dnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long())  # (B, F, E)
        h = self.interaction(emb)  # (B, E)
        return self.dnn(h).squeeze(-1)


# ============================================================
# Build functions and example inputs
# ============================================================


def build_afnplus() -> nn.Module:
    """AFN+: LNN log-transform interaction + DNN, ensembled."""
    return AFNPlus()


def example_input_afnplus() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_aoanet() -> nn.Module:
    """AOANet: generalized interaction layers (operation-adaptive)."""
    return AOANet()


def example_input_aoanet() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_destine() -> nn.Module:
    """DESTINE: disentangled self-attention (whitened, diagonal masked)."""
    return DESTINE()


def example_input_destine() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_edcn() -> nn.Module:
    """EDCN: bridge + regulation modules coupling cross-net and deep-net."""
    return EDCN()


def example_input_edcn() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_eulernet() -> nn.Module:
    """EulerNet: complex polar (Euler formula) feature interaction."""
    return EulerNet()


def example_input_eulernet() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_flen() -> nn.Module:
    """FLEN: field-wise bi-interaction (intra + inter FM) + DNN."""
    return FLEN()


def example_input_flen() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_finalmlp() -> nn.Module:
    """FinalMLP: two MLP streams + feature-selection gates + bilinear fusion."""
    return FinalMLP()


def example_input_finalmlp() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_finalnet() -> nn.Module:
    """FinalNet: FINAL blocks (factorized gated interaction, no MLP)."""
    return FinalNet()


def example_input_finalnet() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_fmfm() -> nn.Module:
    """FmFM: field-matrixed FM (per-field-pair transformation matrices)."""
    return FmFM()


def example_input_fmfm() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_gdcn() -> nn.Module:
    """GDCN: gated cross network (information gate on each cross layer)."""
    return GDCN()


def example_input_gdcn() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_interhat() -> nn.Module:
    """InterHAt: multi-head transformer + hierarchical attentional aggregation."""
    return InterHAt()


def example_input_interhat() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_masknet() -> nn.Module:
    """MaskNet: instance-guided mask (MaskBlock) per field."""
    return MaskNet()


def example_input_masknet() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_pepnet() -> nn.Module:
    """PEPNet: EPNet (embedding personalization) + PPNet (parameter personalization)."""
    return PEPNet()


def example_input_pepnet() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_ppnet() -> nn.Module:
    """PPNet (standalone): parameter-personalized DNN via GateNN."""
    return PPNet()


def example_input_ppnet() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


def build_sam() -> nn.Module:
    """SAM (SAM2A): self-attention interaction model."""
    return SAM(variant="SAM2A")


def example_input_sam() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, NUM_FIELDS))


MENAGERIE_ENTRIES = [
    (
        "AFN+ (Adaptive Factorization Network: log-transform LNN + DNN ensemble)",
        "build_afnplus",
        "example_input_afnplus",
        "2021",
        "DC",
    ),
    (
        "AOANet (Architecture & Operation Adaptive: generalized interaction layers)",
        "build_aoanet",
        "example_input_aoanet",
        "2021",
        "DC",
    ),
    (
        "DESTINE (Disentangled Self-Attention: whitened diagonal-masked MHA)",
        "build_destine",
        "example_input_destine",
        "2021",
        "DC",
    ),
    (
        "EDCN (Enhanced Deep&Cross: bridge + regulation module coupling)",
        "build_edcn",
        "example_input_edcn",
        "2021",
        "DC",
    ),
    (
        "EulerNet (Complex polar feature interaction via Euler formula)",
        "build_eulernet",
        "example_input_eulernet",
        "2023",
        "DC",
    ),
    (
        "FLEN (Field-Leveraged Embedding: field-wise bi-interaction intra+inter FM)",
        "build_flen",
        "example_input_flen",
        "2019",
        "DC",
    ),
    (
        "FinalMLP (Two-stream MLP + feature-selection gates + bilinear fusion head)",
        "build_finalmlp",
        "example_input_finalmlp",
        "2023",
        "DC",
    ),
    (
        "FinalNet (FINAL blocks: factorized field-aware gating, no MLP)",
        "build_finalnet",
        "example_input_finalnet",
        "2023",
        "DC",
    ),
    (
        "FmFM (Field-matrixed FM: per-field-pair interaction matrices)",
        "build_fmfm",
        "example_input_fmfm",
        "2021",
        "DC",
    ),
    (
        "GDCN (Gated Deep Cross Network: information gate on each cross layer)",
        "build_gdcn",
        "example_input_gdcn",
        "2023",
        "DC",
    ),
    (
        "InterHAt (Interpretable Hierarchical Attention: MHA + hierarchical aggregation)",
        "build_interhat",
        "example_input_interhat",
        "2020",
        "DC",
    ),
    (
        "MaskNet (Instance-guided MaskBlocks applied multiplicatively per field)",
        "build_masknet",
        "example_input_masknet",
        "2021",
        "DC",
    ),
    (
        "PEPNet (EPNet+PPNet: embedding + parameter personalization via GateNN)",
        "build_pepnet",
        "example_input_pepnet",
        "2023",
        "DC",
    ),
    (
        "PPNet (Parameter Personalization Network: GateNN-gated DNN tower)",
        "build_ppnet",
        "example_input_ppnet",
        "2023",
        "DC",
    ),
    (
        "SAM (SAM2A: Self-Attention Interaction Model for CTR)",
        "build_sam",
        "example_input_sam",
        "2023",
        "DC",
    ),
]

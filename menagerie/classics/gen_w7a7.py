"""Compact faithful classics for five structural/sequence protein-model
architectures.

Sources checked (repo/model card inspected via GitHub API and HuggingFace,
base env only, no clone or pip install):
  - PepMLM: https://github.com/programmablebio/pepmlm (README + HuggingFace
    model card ``TianlaiChen/PepMLM-650M``). Chen, Quinn, Dumas et al.,
    "Target Sequence-Conditioned Design of Peptide Binders using Masked
    Language Modeling", Nature Biotechnology 2025 (arXiv 2310.03842). PepMLM
    is an ESM-2 protein language model backbone with a *masked-language-
    modeling generation* strategy: the target protein sequence is
    concatenated with an all-``<mask>`` peptide binder region appended at the
    C-terminus, and the model is trained/queried to fully reconstruct the
    masked peptide tokens conditioned bidirectionally on the (unmasked)
    target sequence -- no separate encoder/decoder, no autoregression, just
    an ESM-2-style bidirectional transformer encoder with a masked-LM head
    used in a "mask the region you want to design" generation mode.
    Reimplemented compactly as ``PepMlmEncoder``: a small ESM-2-style
    pre-LN transformer encoder (rotary-free, learned absolute positions,
    standard multi-head self-attention + GELU MLP blocks) over a
    target+peptide token sequence where the peptide suffix is masked, with a
    tied-weight masked-LM output head, matching the defining
    target-conditioned full-region masking mechanism.
  - PiFold: https://github.com/A4Bio/PiFold (``methods/prodesign_module.py``
    ``NeighborAttention``, ``EdgeMLP``, ``Context``, ``GeneralGNN``,
    ``StructureEncoder``). Gao, Tan, Gao, Wang, Wu & Li, "PiFold: Toward
    Effective and Efficient Protein Inverse Folding", ICLR 2023.
    Non-autoregressive inverse-folding GNN: each node-update layer combines
    (a) a *bias-gated neighbour attention* -- an MLP over
    ``[node_i, edge_ij, node_j]`` produces per-head attention logits used to
    softmax-aggregate value vectors from the k-nearest-neighbour edge set
    (no query/key dot product, a learned bias network instead), (b) an
    edge-update MLP that refines edge features from the pair of endpoint
    node states, and (c) a global "context" gate that broadcasts a
    graph-mean summary back onto every node/edge (the PiGNN virtual-atom
    context mechanism). Reimplemented as ``PiFoldLayer``/``PiFoldEncoder``
    operating on a dense fixed-size k-NN neighbour tensor (no torch_scatter
    dependency needed for a small graph) that preserves the bias-attention +
    edge-MLP + context-gate triple exactly, decoded to per-residue amino-acid
    logits by a linear ``MLPDecoder`` head as in the original.
  - PIPR: https://github.com/muhaochen/seq_ppi
    (``binary/model/lasagna/rcnn.py`` ``build_model``). Chen, Ju, Zhou et al.,
    "Multifaceted Protein-Protein Interaction Prediction Based on Siamese
    Residual RCNN", ISMB/ECCB 2019. Siamese residual recurrent-convolutional
    network for sequence-only protein-protein interaction prediction: two
    protein sequences are each pushed independently (tied weights) through
    five stacked "Conv1D -> BiGRU -> concat(BiGRU_out, conv_out)" residual
    RCNN blocks with intermediate max-pooling, then a final conv +
    global-average-pool collapses each sequence to a single embedding; the
    two embeddings are combined by elementwise multiplication and passed
    through a small MLP with LeakyReLU to a binary interaction logit.
    Reimplemented as ``PiprResidualRcnnBlock``/``PiprSiameseEncoder`` with
    identical tied-weight siamese-RCNN-then-multiply topology (BiGRU
    substituted 1:1 for the original CuDNNGRU).
  - PocketMiner: https://github.com/Mickdub/gvp/tree/pocket_pred
    (``src/gvp.py`` ``GVP``, ``src/models.py`` ``MQAModel``/``Encoder``/
    ``MPNNLayer``/``StructuralFeatures``). Meller, Ward, Borowsky et al.,
    "Predicting Cryptic Pocket Opening from Protein Structure with Graph
    Neural Networks", Nature Communications 2023 (Bowman lab). Residue-level
    Geometric Vector Perceptron (GVP) message-passing GNN (an ``MQAModel``
    variant, repurposed from model-quality-assessment to per-residue cryptic-
    pocket-opening classification): scalar and 3D-equivariant vector node/
    edge features are embedded by GVP layers, refined by several k-NN
    message-passing ``MPNNLayer`` GVP-conv blocks, then read out to a
    per-residue scalar (no global pooling, since PocketMiner predicts *which*
    residues open a cryptic pocket) by a final scalar-only GVP + sigmoid MLP.
    Reimplemented as ``GvpBlock``/``PocketMinerEncoder`` using the same
    scalar+vector split, vector-norm-gates-scalar / scalar-gates-vector GVP
    mixing rule, and dense-graph k-NN-style message passing, ending in a
    per-residue pocket-opening probability instead of the original's global
    pooled MQA score.
  - ProDESIGN-LE: https://github.com/bigict/ProDESIGN-LE
    (``pe/model/modules.py`` ``Transformer``, ``pe/model/preprocess.py``
    ``PreProcess``). Huang, Zhang, Zhang & Han, "ProDESIGN-LE: A Fast and
    Effective Local-Environment-based Approach for Protein Sequence Design",
    Bioinformatics 2023. Local-environment-aware transformer for structure-
    conditioned sequence design: for every target residue, its k spatially
    nearest neighbour residues are featurized with a rich per-neighbour
    descriptor (one-hot neighbour amino-acid identity, same-chain flag,
    relative-sequence-position one-hot, "comes before target" flag, and a
    relative backbone-frame orientation feature) and this local-environment
    token set (not the whole protein) is fed through a small transformer
    encoder whose pooled (mean-over-neighbourhood) output is projected to an
    amino-acid-type distribution for the target residue -- the defining
    "local environment, not whole structure or whole sequence" design
    mechanism. Reimplemented as ``LocalEnvironmentFeaturizer`` (reproducing
    the neighbour one-hot/same-chain/relative-position/orientation feature
    concatenation) feeding ``ProdesignLeTransformer`` (a
    ``TransformerEncoder`` + mean-pool + output-projection matching
    ``pe.model.modules.Transformer`` exactly).
  - ProFOLD (ProSPr / democratized AlphaFold1 clone):
    https://github.com/dellacortelab/prospr (``prospr/nn.py``
    ``ProsprNetwork``, ``Block``). AlQuraishi lab / Dell'Corte lab,
    "ProSPr: Democratized Implementation of Alphafold Protein Distance
    Prediction Network", bioRxiv 2019.830273. AlphaFold1-style 2D dilated
    residual-convolution trunk over an L x L pairwise residue feature map:
    220 bottleneck ("project-down -> dilated 3x3 conv -> project-up",
    dilation cycling 1/2/4/8) pre-activation residual blocks at two channel
    widths (256 then 128, with a 1x1 channel-reduction conv between them),
    followed by a 1x1-conv distogram head over the pair map plus two
    *anisotropic* auxiliary heads (a 64x1 conv and a 1x64 conv, i.e.
    row-pool and column-pool respectively) that read off per-residue
    secondary-structure / phi / psi / solvent-accessibility predictions from
    the two symmetric 1D projections of the pair representation -- the
    defining "distogram trunk + anisotropic 1D auxiliary heads" AF1-clone
    mechanism. Reimplemented compactly as ``ProfoldBlock``/``ProfoldNetwork``
    with a much smaller channel width/block count/crop size but the exact
    project-down/dilate/project-up residual block, two-stage channel width,
    and distogram + row-pool/column-pool auxiliary head topology intact.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# PepMLM: ESM-2-style bidirectional transformer encoder used in a masked-
# language-modeling *generation* mode -- the peptide-binder region appended
# to the target sequence is fully masked, and the model reconstructs it
# conditioned bidirectionally on the (unmasked) target tokens.
# ---------------------------------------------------------------------------


class PepMlmEncoder(nn.Module):
    """ESM-2-style masked-LM encoder used for target-conditioned peptide design."""

    def __init__(
        self,
        vocab_size: int = 25,
        max_len: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, norm_first=True, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L] target-sequence tokens with a masked peptide suffix.
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        h = self.token_embed(tokens) + self.pos_embed(positions).unsqueeze(0)
        h = self.encoder(h)
        h = self.norm(h)
        return self.lm_head(h)


def build_pepmlm() -> nn.Module:
    """Build a compact PepMLM ESM-2-style masked-LM peptide-design encoder."""

    return PepMlmEncoder().eval()


def example_input_pepmlm() -> torch.Tensor:
    """Return target+masked-peptide token ids, shape [batch, length]."""

    target_tokens = torch.randint(4, 24, (1, 40))
    mask_tokens = torch.full((1, 12), 24)
    return torch.cat([target_tokens, mask_tokens], dim=1)


# ---------------------------------------------------------------------------
# PiFold: non-autoregressive inverse-folding GNN. Each layer combines a
# bias-gated neighbour attention (MLP-produced attention logits, not a QK
# dot product), an edge-refinement MLP, and a global context gate broadcast
# back onto every node/edge -- the PiGNN "virtual atom" context mechanism.
# ---------------------------------------------------------------------------


class PiFoldLayer(nn.Module):
    """One PiFold GeneralGNN layer: bias-attention + edge MLP + context gate."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.bias_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_out = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.context_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, knn_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # node: [N, D]; edge: [N, K, D] features of each node's K neighbours;
        # knn_idx: [N, K] long indices of those neighbours into `node`.
        n, k = knn_idx.shape
        d = self.hidden_dim
        neighbor_node = node[knn_idx]  # [N, K, D]
        center_node = node.unsqueeze(1).expand(n, k, d)

        bias_logits = self.bias_mlp(torch.cat([center_node, edge, neighbor_node], dim=-1))
        bias_logits = bias_logits.view(n, k, self.num_heads, 1)
        attend = torch.softmax(bias_logits, dim=1)

        value = self.value_mlp(torch.cat([edge, neighbor_node], dim=-1))
        value = value.view(n, k, self.num_heads, d // self.num_heads)
        agg = (attend * value).sum(dim=1).reshape(n, d)
        node = self.node_norm(node + self.node_out(agg))

        edge_msg = self.edge_mlp(torch.cat([center_node, edge, neighbor_node], dim=-1))
        edge = self.edge_norm(edge + edge_msg)

        context = node.mean(dim=0, keepdim=True)
        node = node * self.context_gate(context)
        return node, edge


class PiFoldEncoder(nn.Module):
    """PiFold: stacked PiGNN layers over a dense k-NN residue graph."""

    def __init__(
        self,
        node_in: int = 21,
        edge_in: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 3,
        vocab: int = 20,
    ) -> None:
        super().__init__()
        self.node_in = nn.Linear(node_in, hidden_dim)
        self.edge_in = nn.Linear(edge_in, hidden_dim)
        self.layers = nn.ModuleList([PiFoldLayer(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Linear(hidden_dim, vocab)

    def forward(
        self, node_feat: torch.Tensor, edge_feat: torch.Tensor, knn_idx: torch.Tensor
    ) -> torch.Tensor:
        node = self.node_in(node_feat)
        edge = self.edge_in(edge_feat)
        for layer in self.layers:
            node, edge = layer(node, edge, knn_idx)
        logits = self.decoder(node)
        return F.log_softmax(logits, dim=-1)


def build_pifold() -> nn.Module:
    """Build a compact PiFold non-autoregressive inverse-folding GNN."""

    return PiFoldEncoder().eval()


def example_input_pifold() -> List[torch.Tensor]:
    """Return node features, per-neighbour edge features, and a dense knn index."""

    n, k = 12, 5
    node_feat = torch.rand(n, 21)
    edge_feat = torch.rand(n, k, 16)
    knn_idx = torch.stack([torch.randperm(n)[:k] for _ in range(n)], dim=0)
    return [node_feat, edge_feat, knn_idx]


# ---------------------------------------------------------------------------
# PIPR: siamese residual RCNN for sequence-only protein-protein interaction
# prediction. Two tied-weight towers each stack Conv1D -> BiGRU -> residual
# concat blocks; the two pooled embeddings are combined by elementwise
# multiplication before a small MLP head.
# ---------------------------------------------------------------------------


class PiprResidualRcnnBlock(nn.Module):
    """One PIPR conv -> BiGRU -> residual-concat block with max-pooling."""

    def __init__(self, in_dim: int, hidden_dim: int, pool: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, in_dim]
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, hidden]
        gru_out, _ = self.gru(conv_out)  # [B, L, 2*hidden]
        merged = torch.cat([gru_out, conv_out], dim=-1)  # residual concat
        return self.pool(merged.transpose(1, 2)).transpose(1, 2)


class PiprSiameseEncoder(nn.Module):
    """PIPR: siamese residual RCNN tower applied to two protein sequences."""

    def __init__(self, in_dim: int = 13, hidden_dim: int = 16, num_blocks: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [3 * hidden_dim] * (num_blocks - 1)
        self.blocks = nn.ModuleList(
            [PiprResidualRcnnBlock(dims[i], hidden_dim) for i in range(num_blocks)]
        )
        self.final_conv = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def tower(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x.transpose(1, 2)).transpose(1, 2)
        return x.mean(dim=1)  # global average pool

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        emb_a = self.tower(seq_a)
        emb_b = self.tower(seq_b)
        merged = emb_a * emb_b
        return self.head(merged)


def build_pipr() -> nn.Module:
    """Build a compact PIPR siamese residual RCNN PPI predictor."""

    return PiprSiameseEncoder().eval()


def example_input_pipr() -> List[torch.Tensor]:
    """Return two one-hot-embedded protein sequence tensors [batch, L, dim]."""

    seq_a = torch.rand(1, 96, 13)
    seq_b = torch.rand(1, 96, 13)
    return [seq_a, seq_b]


# ---------------------------------------------------------------------------
# PocketMiner: residue-level Geometric Vector Perceptron (GVP) message-
# passing GNN. Scalar and 3D-equivariant vector node/edge features are mixed
# by GVP layers (vector norms feed the scalar path, a learned gate scales
# the vector path); per-residue outputs (not a pooled global score) predict
# which residues participate in cryptic-pocket opening.
# ---------------------------------------------------------------------------


class Gvp(nn.Module):
    """Geometric Vector Perceptron: joint scalar+vector feature update."""

    def __init__(self, in_dims: Tuple[int, int], out_dims: Tuple[int, int]) -> None:
        super().__init__()
        si, vi = in_dims
        so, vo = out_dims
        self.vi, self.vo = vi, vo
        self.h_dim = max(vi, vo, 1)
        if vi:
            self.wh = nn.Linear(vi, self.h_dim, bias=False)
        self.ws = nn.Linear(self.h_dim + si if vi else si, so)
        if vo:
            self.wv = nn.Linear(self.h_dim, vo, bias=False)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s: [..., si] scalars; v: [..., vi, 3] vectors.
        if self.vi:
            vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)  # [..., h_dim, 3]
            vh_norm = vh.norm(dim=-1)
            s_out = F.relu(self.ws(torch.cat([s, vh_norm], dim=-1)))
        else:
            s_out = F.relu(self.ws(s))
        if self.vo:
            v_out = self.wv(vh.transpose(-1, -2)).transpose(-1, -2)
            gate = torch.sigmoid(s_out[..., : self.vo]).unsqueeze(-1)
            v_out = v_out * gate
            return s_out, v_out
        return s_out, torch.zeros(*s_out.shape[:-1], 0, 3, device=s.device)


class GvpBlock(nn.Module):
    """One GVP message-passing block over a dense k-NN residue graph."""

    def __init__(self, node_dims: Tuple[int, int], edge_dims: Tuple[int, int]) -> None:
        super().__init__()
        ns, nv = node_dims
        es, ev = edge_dims
        self.message = Gvp((2 * ns + es, 2 * nv + ev), (ns, nv))
        self.node_norm = nn.LayerNorm(ns)

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # node_s: [N,ns]; node_v: [N,nv,3]; edge_s/edge_v: [N,K,*]; knn_idx: [N,K]
        n, k = knn_idx.shape
        s_i = node_s.unsqueeze(1).expand(n, k, -1)
        s_j = node_s[knn_idx]
        v_i = node_v.unsqueeze(1).expand(n, k, -1, -1)
        v_j = node_v[knn_idx]

        msg_s_in = torch.cat([s_i, s_j, edge_s], dim=-1)
        msg_v_in = torch.cat([v_i, v_j, edge_v], dim=-2)
        msg_s, msg_v = self.message(msg_s_in, msg_v_in)

        node_s = self.node_norm(node_s + msg_s.mean(dim=1))
        node_v = node_v + msg_v.mean(dim=1)
        return node_s, node_v


class PocketMinerEncoder(nn.Module):
    """PocketMiner: GVP-GNN predicting per-residue cryptic-pocket opening."""

    def __init__(
        self,
        node_in: Tuple[int, int] = (6, 3),
        node_h: Tuple[int, int] = (32, 8),
        edge_in: Tuple[int, int] = (16, 1),
        edge_h: Tuple[int, int] = (16, 2),
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.embed_node = Gvp(node_in, node_h)
        self.embed_edge = Gvp(edge_in, edge_h)
        self.layers = nn.ModuleList([GvpBlock(node_h, edge_h) for _ in range(num_layers)])
        self.readout = Gvp(node_h, (node_h[0], 0))
        self.head = nn.Sequential(
            nn.Linear(node_h[0], node_h[0]), nn.ReLU(), nn.Linear(node_h[0], 1)
        )

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
        knn_idx: torch.Tensor,
    ) -> torch.Tensor:
        node_s, node_v = self.embed_node(node_s, node_v)
        edge_s, edge_v = self.embed_edge(edge_s, edge_v)
        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_s, edge_v, knn_idx)
        pooled_s, _ = self.readout(node_s, node_v)
        return torch.sigmoid(self.head(pooled_s)).squeeze(-1)


def build_pocketminer() -> nn.Module:
    """Build a compact PocketMiner GVP-GNN cryptic-pocket-opening predictor."""

    return PocketMinerEncoder().eval()


def example_input_pocketminer() -> List[torch.Tensor]:
    """Return scalar/vector node and edge features plus a dense knn index."""

    n, k = 14, 6
    node_s = torch.rand(n, 6)
    node_v = torch.randn(n, 3, 3)
    edge_s = torch.rand(n, k, 16)
    edge_v = torch.randn(n, k, 1, 3)
    knn_idx = torch.stack([torch.randperm(n)[:k] for _ in range(n)], dim=0)
    return [node_s, node_v, edge_s, edge_v, knn_idx]


# ---------------------------------------------------------------------------
# ProDESIGN-LE: local-environment-aware transformer for sequence design. For
# every target residue, its k nearest-neighbour residues are featurized
# (identity one-hot, same-chain flag, relative-position one-hot, "before
# target" flag, relative-orientation feature) and this local-environment
# token set is pooled by a small transformer to predict the target residue's
# amino-acid type -- the defining "local environment, not global structure"
# mechanism.
# ---------------------------------------------------------------------------


class LocalEnvironmentFeaturizer(nn.Module):
    """Featurize each target residue's k-nearest-neighbour local environment."""

    def __init__(self, num_aa: int = 20, max_relative: int = 5) -> None:
        super().__init__()
        self.num_aa = num_aa
        self.max_relative = max_relative

    def forward(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        # feat holds per-(target, neighbour) descriptors, shape [B, K, ...].
        aa_type = F.one_hot(feat["neighbor_aa"], num_classes=self.num_aa + 1).float()
        same_chain = feat["same_chain"].unsqueeze(-1).float()
        is_senpai = feat["is_before_target"].unsqueeze(-1).float()
        offset = torch.clamp(
            feat["relative_position"] + self.max_relative, 0, 2 * self.max_relative
        )
        rel_pos = F.one_hot(offset, num_classes=2 * self.max_relative + 1).float()
        orientation = feat["relative_orientation"]
        return torch.cat([aa_type, same_chain, is_senpai, rel_pos, orientation], dim=-1)


class ProdesignLeTransformer(nn.Module):
    """ProDESIGN-LE: transformer over a local-environment neighbour set."""

    def __init__(
        self,
        d_input: int,
        d_output: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        nlayer: int = 2,
    ) -> None:
        super().__init__()
        self.featurizer = LocalEnvironmentFeaturizer()
        self.input = nn.Linear(d_input, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.main = nn.TransformerEncoder(layer, num_layers=nlayer)
        self.output = nn.Linear(d_model, d_output)

    def forward(self, feat: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        env_feat = self.featurizer(feat)
        h = self.input(env_feat)
        logits = self.main(h, src_key_padding_mask=~mask)
        pooled = logits.mean(dim=1)
        return self.output(pooled)


def build_prodesign_le() -> nn.Module:
    """Build a compact ProDESIGN-LE local-environment sequence-design transformer."""

    num_aa = 20
    max_relative = 5
    d_input = (num_aa + 1) + 1 + 1 + (2 * max_relative + 1) + 6
    return ProdesignLeTransformer(d_input=d_input).eval()


def example_input_prodesign_le() -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Return a local-environment feature dict and an attend mask, batch of 1."""

    b, k = 1, 10
    feat = {
        "neighbor_aa": torch.randint(0, 20, (b, k)),
        "same_chain": torch.randint(0, 2, (b, k)).bool(),
        "is_before_target": torch.randint(0, 2, (b, k)).bool(),
        "relative_position": torch.randint(-5, 6, (b, k)),
        "relative_orientation": torch.rand(b, k, 6),
    }
    mask = torch.ones(b, k, dtype=torch.bool)
    return feat, mask


# ---------------------------------------------------------------------------
# ProFOLD (ProSPr, democratized AlphaFold1 clone): 2D dilated residual-conv
# trunk over an L x L pairwise feature map -- project-down/dilated-3x3/
# project-up bottleneck blocks at two channel widths -- feeding a distogram
# head plus anisotropic (row-pool / column-pool) 1D auxiliary heads for
# secondary structure and backbone torsion angles.
# ---------------------------------------------------------------------------


class ProfoldBlock(nn.Module):
    """One AF1-style dilated bottleneck residual block over the pair map."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        mid = channels // 2
        self.norm1 = nn.BatchNorm2d(channels)
        self.project_down = nn.Conv2d(channels, mid, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(mid)
        self.dilated = nn.Conv2d(mid, mid, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm3 = nn.BatchNorm2d(mid)
        self.project_up = nn.Conv2d(mid, channels, kernel_size=1)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(x))
        out = self.project_down(out)
        out = self.act(self.norm2(out))
        out = self.dilated(out)
        out = self.act(self.norm3(out))
        out = self.project_up(out)
        return out + identity


class ProfoldNetwork(nn.Module):
    """ProFOLD/ProSPr: dilated-conv distogram trunk + anisotropic aux heads."""

    def __init__(
        self,
        input_dim: int = 64,
        wide_channels: int = 32,
        narrow_channels: int = 16,
        num_wide_blocks: int = 2,
        num_narrow_blocks: int = 2,
        crop_size: int = 16,
        dist_bins: int = 10,
        aux_bins: int = 20,
    ) -> None:
        super().__init__()
        self.bn_in = nn.BatchNorm2d(input_dim)
        self.conv_in = nn.Conv2d(input_dim, wide_channels, kernel_size=1)
        dilations = [1, 2, 4, 8]
        wide_blocks = [
            ProfoldBlock(wide_channels, dilation=dilations[i % 4]) for i in range(num_wide_blocks)
        ]
        self.wide_blocks = nn.Sequential(*wide_blocks)
        self.channel_reduce = nn.Conv2d(wide_channels, narrow_channels, kernel_size=1)
        narrow_blocks = [
            ProfoldBlock(narrow_channels, dilation=dilations[i % 4])
            for i in range(num_narrow_blocks)
        ]
        self.narrow_blocks = nn.Sequential(*narrow_blocks)
        self.distogram_head = nn.Conv2d(narrow_channels, dist_bins, kernel_size=1)
        self.aux_row_head = nn.Conv2d(narrow_channels, aux_bins, kernel_size=(crop_size, 1))
        self.aux_col_head = nn.Conv2d(narrow_channels, aux_bins, kernel_size=(1, crop_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, input_dim, L, L] pairwise residue feature map.
        h = self.bn_in(x)
        h = self.conv_in(h)
        h = self.wide_blocks(h)
        h = self.channel_reduce(h)
        h = self.narrow_blocks(h)
        distogram = self.distogram_head(h)
        aux_row = self.aux_row_head(h).squeeze(2)  # [B, aux_bins, L]
        aux_col = self.aux_col_head(h).squeeze(3)  # [B, aux_bins, L]
        return distogram, aux_row, aux_col


def build_profold() -> nn.Module:
    """Build a compact ProFOLD/ProSPr AF1-style distogram + aux-head network."""

    return ProfoldNetwork().eval()


def example_input_profold() -> torch.Tensor:
    """Return a pairwise residue feature map, shape [batch, dim, L, L]."""

    return torch.rand(1, 64, 16, 16)


MENAGERIE_ENTRIES = [
    ("PepMLM", "build_pepmlm", "example_input_pepmlm", "2023", "BIO"),
    ("PiFold", "build_pifold", "example_input_pifold", "2023", "BIO"),
    ("PIPR", "build_pipr", "example_input_pipr", "2019", "BIO"),
    ("PocketMiner", "build_pocketminer", "example_input_pocketminer", "2023", "BIO"),
    ("ProDESIGN-LE", "build_prodesign_le", "example_input_prodesign_le", "2023", "BIO"),
    ("ProFOLD", "build_profold", "example_input_profold", "2019", "BIO"),
]

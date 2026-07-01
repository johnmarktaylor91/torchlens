"""Generation w4a18: computational-biology / bioinformatics classics batch.

Sources checked (repo_url / desc_source from the build queue, read via GitHub API /
web search -- no cloning, no pip installs beyond the base env):
  - Akita: https://github.com/HaoyeYang/akita_pytorch_replica (PyTorch replica);
    original https://github.com/calico/basenji (TF); paper
    https://www.nature.com/articles/s41592-020-0958-x (Fudenberg et al., Nat Methods 2020).
  - AlphaFold-Multimer: https://github.com/google-deepmind/alphafold (JAX/Haiku);
    paper https://www.biorxiv.org/content/10.1101/2021.10.04.463034 (Evans et al. 2021).
  - AlphaMissense: https://github.com/google-deepmind/alphamissense (JAX);
    paper arxiv:2307.03056 / Science 2023 (Cheng et al.).
  - AlphaPeptDeep: https://github.com/MannLabs/alphapeptdeep (PyTorch);
    paper arxiv:2207.07582 / Nat Commun 2022 (Zeng et al.).
  - AttentionSiteDTI: https://github.com/yazdanimehdi/AttentionSiteDTI (PyTorch);
    paper arxiv:2111.06939 / Briefings in Bioinformatics 2022 (Yazdani-Jahromi et al.).
  - BioBERT: https://github.com/dmis-lab/biobert (weights on HuggingFace as
    dmis-lab/biobert-*); paper arxiv:1901.08746 / Bioinformatics 2020 (Lee et al.).

Each model below reimplements the DISTINCTIVE architectural mechanism of its source
paper/repo compactly and faithfully, at tiny random-init dimensions suitable for a
traceable architecture atlas (not a trained-weights zoo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Akita: dilated 1D conv tower -> 1D-to-2D outer-sum -> symmetrized dilated 2D
# conv tower -> per-pixel contact-map track head. The DISTINCTIVE mechanism
# (vs. Basenji's 1D-track-only head, already present in the catalog) is the
# **2D contact map path**: after the 1D dilated tower, Akita takes the 1D
# per-bin embedding and forms a symmetric pairwise map via an outer sum
# (upper-triangular flatten in the real model; here a dense (L, L, C) map
# for a small L), then refines it with symmetrized dilated 2D convolutions
# (conv output is averaged with its own transpose every block to enforce
# the physical symmetry of a Hi-C contact matrix) before a 1x1 head predicts
# per-pixel contact-frequency tracks.
# ---------------------------------------------------------------------------


class _Dilated1DBlock(nn.Module):
    """Dilated 1D residual conv block (Basenji/Akita-style 1D trunk)."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(F.gelu(self.norm(x)))


class _SymmetricDilated2DBlock(nn.Module):
    """Dilated 2D conv block whose output is symmetrized with its transpose.

    This is Akita's key departure from a generic 2D CNN: because a Hi-C
    contact map is symmetric (contact(i, j) == contact(j, i)), each 2D
    refinement block averages its convolution output with the
    transpose-over-the-two-spatial-axes of that same output, so the
    representation is forced toward symmetry layer by layer.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.gelu(self.norm(x)))
        h_sym = 0.5 * (h + h.transpose(-1, -2))
        return x + h_sym


class Akita(nn.Module):
    """Compact Akita: dilated 1D tower -> outer-sum 2D map -> symmetric 2D tower -> contact head."""

    def __init__(
        self,
        in_channels: int = 4,
        c_1d: int = 16,
        c_2d: int = 12,
        n_1d_blocks: int = 3,
        n_2d_blocks: int = 3,
        n_tracks: int = 5,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_channels, c_1d, 11, padding=5)
        self.tower_1d = nn.ModuleList(
            [_Dilated1DBlock(c_1d, dilation=2**i) for i in range(n_1d_blocks)]
        )
        self.pool = nn.MaxPool1d(2)
        self.to_pair = nn.Linear(2 * c_1d, c_2d)
        self.tower_2d = nn.ModuleList(
            [_SymmetricDilated2DBlock(c_2d, dilation=2**i) for i in range(n_2d_blocks)]
        )
        self.head = nn.Conv2d(c_2d, n_tracks, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One-hot DNA ``(B, L, 4)`` -> predicted contact-map tracks ``(B, n_tracks, L', L')``."""
        h = self.stem(x.transpose(1, 2))
        for blk in self.tower_1d:
            h = blk(h)
        h = self.pool(h)  # (B, c_1d, L')
        h = h.transpose(1, 2)  # (B, L', c_1d)
        length = h.shape[1]
        # outer-sum 1D -> 2D: pairwise concat of bin i and bin j embeddings
        left = h.unsqueeze(2).expand(-1, -1, length, -1)
        right = h.unsqueeze(1).expand(-1, length, -1, -1)
        pair = self.to_pair(torch.cat((left, right), dim=-1))  # (B, L', L', c_2d)
        pair = pair.permute(0, 3, 1, 2)  # (B, c_2d, L', L')
        for blk in self.tower_2d:
            pair = blk(pair)
        return self.head(pair)


def build_akita() -> nn.Module:
    """Build a tiny random-init Akita contact-map predictor."""
    return Akita().eval()


def example_input_akita() -> torch.Tensor:
    """One-hot DNA window ``(1, 64, 4)``."""
    return F.one_hot(torch.randint(0, 4, (1, 64)), num_classes=4).float()


# ---------------------------------------------------------------------------
# AlphaFold-Multimer: extends AF2's Evoformer + IPA (already faithfully built
# in menagerie/classics/openfold_af2.py) with two DISTINCTIVE multimer-only
# mechanisms, which is what this build focuses on (it does NOT re-derive the
# full Evoformer -- see openfold_af2.py for that faithful reimplementation):
#   1. **Cross-chain MSA pairing**: instead of one MSA covering a single
#      chain, multimer builds a per-chain MSA stack and explicitly pairs rows
#      across chains that come from matched-species / paired hits, then
#      concatenates them along the residue axis into one "multi-chain MSA"
#      that the Evoformer consumes. We reimplement the pairing step directly:
#      chain-tagged single-sequence MSAs are combined with a learned
#      cross-chain pairing attention that aligns paired rows before concat.
#   2. **Entity-level / relative-chain positional encoding**: AF2-Multimer
#      replaces AF2's single relative-residue-index pair feature with a
#      richer encoding that also encodes relative CHAIN index and whether two
#      residues are in the same chain (`same_entity`), added as an extra pair
#      bias term feeding the pair representation before Evoformer attention.
# ---------------------------------------------------------------------------


class CrossChainMSAPairing(nn.Module):
    """Pairs MSA rows across chains via learned cross-attention before concatenation."""

    def __init__(self, c_m: int, n_head: int = 4) -> None:
        super().__init__()
        self.n_head = n_head
        self.c_head = c_m // n_head
        self.q = nn.Linear(c_m, c_m, bias=False)
        self.k = nn.Linear(c_m, c_m, bias=False)
        self.v = nn.Linear(c_m, c_m, bias=False)
        self.out = nn.Linear(c_m, c_m)

    def forward(
        self, msa_a: torch.Tensor, msa_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align chain-B MSA rows onto chain-A MSA rows (and vice versa) via cross-attention.

        msa_a: (S_a, R_a, c_m), msa_b: (S_b, R_b, c_m) -- residue-pooled per row for pairing.
        """
        pool_a = msa_a.mean(dim=1)  # (S_a, c_m) row summary
        pool_b = msa_b.mean(dim=1)  # (S_b, c_m)

        def cross(q_src: torch.Tensor, kv_src: torch.Tensor) -> torch.Tensor:
            qh = self.q(q_src).view(-1, self.n_head, self.c_head)
            kh = self.k(kv_src).view(-1, self.n_head, self.c_head)
            vh = self.v(kv_src).view(-1, self.n_head, self.c_head)
            attn = torch.einsum("qhd,khd->hqk", qh, kh) / (self.c_head**0.5)
            attn = torch.softmax(attn, dim=-1)
            return torch.einsum("hqk,khd->qhd", attn, vh).reshape(q_src.shape[0], -1)

        pair_weight_ab = self.out(cross(pool_a, pool_b))  # (S_a, c_m): how row a pairs to B
        pair_weight_ba = self.out(cross(pool_b, pool_a))  # (S_b, c_m)
        msa_a = msa_a + pair_weight_ab.unsqueeze(1)
        msa_b = msa_b + pair_weight_ba.unsqueeze(1)
        return msa_a, msa_b


class RelativeChainPositionBias(nn.Module):
    """Multimer's relative-chain / same-entity pair positional encoding."""

    def __init__(self, c_z: int, max_rel: int = 8) -> None:
        super().__init__()
        self.max_rel = max_rel
        # residue relative-position bucket, chain relative-position bucket, same-entity flag
        self.rel_res_embed = nn.Linear(2 * max_rel + 2, c_z)
        self.rel_chain_embed = nn.Linear(2 * max_rel + 2, c_z)
        self.same_entity_embed = nn.Linear(2, c_z)

    def forward(self, res_idx: torch.Tensor, chain_idx: torch.Tensor) -> torch.Tensor:
        """res_idx, chain_idx: (R,) -> pair bias (R, R, c_z)."""
        d_res = (res_idx[:, None] - res_idx[None, :]).clamp(-self.max_rel, self.max_rel)
        d_chain = (chain_idx[:, None] - chain_idx[None, :]).clamp(-self.max_rel, self.max_rel)
        same = (chain_idx[:, None] == chain_idx[None, :]).long()
        res_onehot = F.one_hot(d_res + self.max_rel, 2 * self.max_rel + 2).float()
        chain_onehot = F.one_hot(d_chain + self.max_rel, 2 * self.max_rel + 2).float()
        same_onehot = F.one_hot(same, 2).float()
        return (
            self.rel_res_embed(res_onehot)
            + self.rel_chain_embed(chain_onehot)
            + self.same_entity_embed(same_onehot)
        )


class TrianglePairUpdate(nn.Module):
    """Minimal AF2-style triangle multiplicative update (outgoing) for the pair track."""

    def __init__(self, c_z: int, c_hidden: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_z)
        self.a = nn.Linear(c_z, c_hidden)
        self.b = nn.Linear(c_z, c_hidden)
        self.out = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        a, b = self.a(z), self.b(z)
        mix = torch.einsum("ikc,jkc->ijc", a, b)
        return z + self.out(mix)


class AlphaFoldMultimer(nn.Module):
    """Compact AlphaFold-Multimer: cross-chain MSA pairing + relative-chain pair bias.

    Focuses on the two mechanisms that distinguish Multimer from single-chain AF2
    (already faithfully covered by ``openfold_af2.AlphaFold2``): cross-chain MSA
    row pairing and relative-chain/same-entity pair positional encoding, feeding a
    small triangle-update pair track and a per-residue coordinate head.
    """

    def __init__(self, c_m: int = 12, c_z: int = 12, n_token: int = 22) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_token, c_m)
        self.pairing = CrossChainMSAPairing(c_m)
        self.rel_bias = RelativeChainPositionBias(c_z)
        self.left = nn.Linear(c_m, c_z)
        self.right = nn.Linear(c_m, c_z)
        self.triangle = TrianglePairUpdate(c_z)
        self.coord_head = nn.Linear(c_m, 3)

    def forward(self, chain_a_tokens: torch.Tensor, chain_b_tokens: torch.Tensor) -> torch.Tensor:
        """chain_a_tokens: (S_a, R_a), chain_b_tokens: (S_b, R_b) integer MSA tokens."""
        msa_a = self.embed(chain_a_tokens)
        msa_b = self.embed(chain_b_tokens)
        msa_a, msa_b = self.pairing(msa_a, msa_b)
        multi_msa = torch.cat((msa_a, msa_b), dim=1)  # concat along residue axis
        seq = multi_msa[0]  # (R_a+R_b, c_m) first-row target sequence

        r_a, r_b = chain_a_tokens.shape[1], chain_b_tokens.shape[1]
        res_idx = torch.cat((torch.arange(r_a), torch.arange(r_b)))
        chain_idx = torch.cat(
            (torch.zeros(r_a, dtype=torch.long), torch.ones(r_b, dtype=torch.long))
        )

        z = self.left(seq)[:, None, :] + self.right(seq)[None, :, :]
        z = z + self.rel_bias(res_idx, chain_idx)
        z = self.triangle(z)
        s = seq + z.mean(dim=1)
        return self.coord_head(s)


def build_alphafold_multimer() -> nn.Module:
    """Build a tiny random-init AlphaFold-Multimer chain-pairing model."""
    return AlphaFoldMultimer().eval()


def example_input_alphafold_multimer() -> tuple[torch.Tensor, torch.Tensor]:
    """Two small chain MSA token tensors: chain A ``(3, 6)``, chain B ``(3, 5)``."""
    return torch.randint(0, 22, (3, 6)), torch.randint(0, 22, (3, 5))


# ---------------------------------------------------------------------------
# AlphaMissense: AlphaFold-derived structural/sequence context (single-sequence
# "MSA" pathway + pair track, à la a lightweight Evoformer trunk) feeding a
# per-residue MLP classifier over the 20 possible substituted amino acids,
# trained to predict pathogenicity of a missense variant. The DISTINCTIVE
# mechanism is the head: rather than predicting structure, AlphaMissense reuses
# AF2-style single+pair representations purely as structural CONTEXT and adds
# a **variant-conditioned classification head** that reads out, for a given
# reference position, a per-alt-amino-acid pathogenicity logit -- i.e. the
# structural trunk is frozen-in-spirit context and the novel part is the
# lightweight classifier bolted onto the per-residue embedding at the variant
# position, gated by which alternate amino acid is being scored.
# ---------------------------------------------------------------------------


class LightEvoformerTrunk(nn.Module):
    """Minimal single+pair representation trunk (AF2-style, structural context only)."""

    def __init__(self, c_s: int = 16, c_z: int = 12, n_head: int = 4) -> None:
        super().__init__()
        self.n_head = n_head
        self.c_head = c_s // n_head
        self.norm_s = nn.LayerNorm(c_s)
        self.q = nn.Linear(c_s, c_s, bias=False)
        self.k = nn.Linear(c_s, c_s, bias=False)
        self.v = nn.Linear(c_s, c_s, bias=False)
        self.pair_bias = nn.Linear(c_z, n_head, bias=False)
        self.out = nn.Linear(c_s, c_s)
        self.pair_update = nn.Linear(2 * c_s, c_z)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sn = self.norm_s(s)
        qh = self.q(sn).view(-1, self.n_head, self.c_head)
        kh = self.k(sn).view(-1, self.n_head, self.c_head)
        vh = self.v(sn).view(-1, self.n_head, self.c_head)
        bias = self.pair_bias(z).permute(2, 0, 1)  # (H, R, R)
        attn = torch.einsum("qhd,khd->hqk", qh, kh) / (self.c_head**0.5) + bias
        attn = torch.softmax(attn, dim=-1)
        o = torch.einsum("hqk,khd->qhd", attn, vh).reshape(s.shape[0], -1)
        s = s + self.out(o)
        left = s.unsqueeze(1).expand(-1, s.shape[0], -1)
        right = s.unsqueeze(0).expand(s.shape[0], -1, -1)
        z = z + self.pair_update(torch.cat((left, right), dim=-1))
        return s, z


class AlphaMissense(nn.Module):
    """Compact AlphaMissense: structural-context trunk + variant-conditioned pathogenicity head."""

    def __init__(self, c_s: int = 16, c_z: int = 12, n_block: int = 2, n_aa: int = 20) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_aa, c_s)
        self.left = nn.Linear(c_s, c_z)
        self.right = nn.Linear(c_s, c_z)
        self.trunk = nn.ModuleList([LightEvoformerTrunk(c_s, c_z) for _ in range(n_block)])
        self.alt_embed = nn.Embedding(n_aa, c_s)
        self.classifier = nn.Sequential(nn.Linear(2 * c_s, c_s), nn.ReLU(), nn.Linear(c_s, 1))

    def forward(
        self, ref_seq: torch.Tensor, variant_pos: torch.Tensor, alt_aa: torch.Tensor
    ) -> torch.Tensor:
        """ref_seq: (R,) tokens; variant_pos: () long index; alt_aa: () long alt-AA id."""
        s = self.embed(ref_seq)
        z = self.left(s)[:, None, :] + self.right(s)[None, :, :]
        for blk in self.trunk:
            s, z = blk(s, z)
        ref_context = s[variant_pos]
        alt_vec = self.alt_embed(alt_aa)
        logit = self.classifier(torch.cat((ref_context, alt_vec), dim=-1))
        return torch.sigmoid(logit)


def build_alphamissense() -> nn.Module:
    """Build a tiny random-init AlphaMissense pathogenicity classifier."""
    return AlphaMissense().eval()


def example_input_alphamissense() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference sequence ``(12,)``, variant position scalar, alt amino-acid id scalar."""
    return torch.randint(0, 20, (12,)), torch.tensor(5), torch.tensor(3)


# ---------------------------------------------------------------------------
# AlphaPeptDeep: modular BiLSTM + self-attention peptide property predictor.
# The DISTINCTIVE mechanism is the shared modular trunk (amino-acid + PTM/mod
# embedding -> BiLSTM -> self-attention pooling) reused across THREE property
# heads (retention time / RT, fragment-ion intensity, collision-cross-section
# / CCS) each fine-tuned from a common pretrained base -- i.e. multi-task
# modularity around one sequence encoder, with the fragmentation head
# additionally attending over each *position* to emit per-bond b/y-ion
# intensities rather than a single pooled scalar like RT/CCS.
# ---------------------------------------------------------------------------


class SelfAttentionPool(nn.Module):
    """Additive self-attention pooling over the sequence axis (AlphaPeptDeep-style)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x), dim=1)  # (B, L, 1)
        return (x * w).sum(dim=1)


class PeptideEncoder(nn.Module):
    """Shared modular trunk: AA + modification embedding -> BiLSTM -> self-attention."""

    def __init__(self, n_aa: int = 22, n_mod: int = 8, dim: int = 24) -> None:
        super().__init__()
        self.aa_embed = nn.Embedding(n_aa, dim)
        self.mod_embed = nn.Embedding(n_mod, dim)
        self.bilstm = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * dim, dim)
        self.pool = SelfAttentionPool(dim)

    def forward(
        self, aa_ids: torch.Tensor, mod_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.aa_embed(aa_ids) + self.mod_embed(mod_ids)
        h, _ = self.bilstm(h)
        h = self.proj(h)  # (B, L, dim) per-position features
        pooled = self.pool(h)  # (B, dim)
        return h, pooled


class AlphaPeptDeep(nn.Module):
    """Compact AlphaPeptDeep: shared BiLSTM+attention trunk with RT/CCS/MS2 heads."""

    def __init__(self, n_aa: int = 22, n_mod: int = 8, dim: int = 24, n_ion_types: int = 2) -> None:
        super().__init__()
        self.encoder = PeptideEncoder(n_aa, n_mod, dim)
        self.rt_head = nn.Linear(dim, 1)
        self.ccs_head = nn.Linear(dim, 1)
        self.ms2_head = nn.Linear(dim, n_ion_types)  # per-position b/y-ion intensities

    def forward(
        self, aa_ids: torch.Tensor, mod_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_pos, pooled = self.encoder(aa_ids, mod_ids)
        rt = self.rt_head(pooled)
        ccs = self.ccs_head(pooled)
        ms2 = torch.sigmoid(self.ms2_head(per_pos))
        return rt, ccs, ms2


def build_alphapeptdeep() -> nn.Module:
    """Build a tiny random-init AlphaPeptDeep multi-task peptide property predictor."""
    return AlphaPeptDeep().eval()


def example_input_alphapeptdeep() -> tuple[torch.Tensor, torch.Tensor]:
    """Peptide amino-acid ids ``(2, 10)`` and per-residue modification ids ``(2, 10)``."""
    return torch.randint(0, 22, (2, 10)), torch.randint(0, 8, (2, 10))


# ---------------------------------------------------------------------------
# AttentionSiteDTI: drug-target interaction prediction using attention over a
# BAG OF SUBGRAPHS. The DISTINCTIVE mechanism (vs. a generic per-atom GNN) is
# that the protein binding POCKET is decomposed into several 3D residue
# sub-pockets and the ligand into fragment subgraphs; each subgraph is
# embedded independently by a GNN + readout into one "instance" vector, and
# a **transformer self-attention layer treats the resulting set of protein
# sub-pocket instances + ligand fragment instances as a single sequence**
# (multiple-instance-learning-with-attention framing) whose pooled output
# feeds a binary DTI classifier -- i.e. cross-instance attention over
# pocket/fragment embeddings is the novel primitive, not the per-subgraph GNN.
# ---------------------------------------------------------------------------


class _SubgraphGNNReadout(nn.Module):
    """Tiny per-subgraph message-passing GNN + mean readout -> one instance vector."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, hidden)
        self.lin_neigh = nn.Linear(in_dim, hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) node feats, adj: (N, N) dense adjacency -> pooled (hidden,)."""
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        neigh = (adj @ x) / deg
        h = F.relu(self.lin_self(x) + self.lin_neigh(neigh))
        return h.mean(dim=0)


class AttentionSiteDTI(nn.Module):
    """Compact AttentionSiteDTI: per-subgraph GNN instances + cross-instance self-attention."""

    def __init__(self, node_dim: int = 12, hidden: int = 24, n_head: int = 4) -> None:
        super().__init__()
        self.protein_gnn = _SubgraphGNNReadout(node_dim, hidden)
        self.ligand_gnn = _SubgraphGNNReadout(node_dim, hidden)
        self.type_embed = nn.Embedding(2, hidden)  # 0=protein subpocket, 1=ligand fragment
        layer = nn.TransformerEncoderLayer(hidden, n_head, hidden * 2, batch_first=True)
        self.instance_attn = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(
        self,
        protein_subgraph_nodes: list[torch.Tensor],
        protein_subgraph_adjs: list[torch.Tensor],
        ligand_subgraph_nodes: list[torch.Tensor],
        ligand_subgraph_adjs: list[torch.Tensor],
    ) -> torch.Tensor:
        protein_instances = torch.stack(
            [self.protein_gnn(x, a) for x, a in zip(protein_subgraph_nodes, protein_subgraph_adjs)]
        )
        ligand_instances = torch.stack(
            [self.ligand_gnn(x, a) for x, a in zip(ligand_subgraph_nodes, ligand_subgraph_adjs)]
        )
        protein_instances = protein_instances + self.type_embed(
            torch.zeros(protein_instances.shape[0], dtype=torch.long)
        )
        ligand_instances = ligand_instances + self.type_embed(
            torch.ones(ligand_instances.shape[0], dtype=torch.long)
        )
        instances = torch.cat((protein_instances, ligand_instances), dim=0).unsqueeze(0)
        attended = self.instance_attn(instances).squeeze(0)
        pooled = attended.mean(dim=0)
        return torch.sigmoid(self.classifier(pooled))


def build_attentionsitedti() -> nn.Module:
    """Build a tiny random-init AttentionSiteDTI cross-instance-attention DTI model."""
    return AttentionSiteDTI().eval()


def example_input_attentionsitedti() -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]:
    """3 protein sub-pockets (5 nodes each) + 2 ligand fragments (4 nodes each), dense adjacency."""
    torch.manual_seed(0)
    protein_nodes = [torch.randn(5, 12) for _ in range(3)]
    protein_adjs = [(torch.rand(5, 5) > 0.5).float() for _ in range(3)]
    ligand_nodes = [torch.randn(4, 12) for _ in range(2)]
    ligand_adjs = [(torch.rand(4, 4) > 0.5).float() for _ in range(2)]
    return protein_nodes, protein_adjs, ligand_nodes, ligand_adjs


# ---------------------------------------------------------------------------
# BioBERT: BERT pretrained on PubMed abstracts + PMC full text. Architecture-
# wise BioBERT is IDENTICAL to BERT (same transformer encoder stack); its
# distinctiveness is purely the pretraining CORPUS, not a structural
# mechanism. We build it faithfully via ``transformers`` with a tiny
# BERT-family config (this is a config of an installed library model, per
# the build instructions) so the atlas captures the standard BERT encoder
# graph that BioBERT's checkpoint conforms to.
# ---------------------------------------------------------------------------


def build_biobert() -> nn.Module:
    """Build a tiny random-init BERT-architecture model standing in for BioBERT."""
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=2,
    )
    return BertModel(cfg).eval()


def example_input_biobert() -> torch.Tensor:
    """Token id sequence ``(1, 16)`` within the tiny BioBERT-config vocab."""
    return torch.randint(0, 100, (1, 16))


MENAGERIE_ENTRIES = [
    ("Akita", "build_akita", "example_input_akita", "2020", "DC"),
    (
        "AlphaFold-Multimer",
        "build_alphafold_multimer",
        "example_input_alphafold_multimer",
        "2021",
        "DC",
    ),
    ("AlphaMissense", "build_alphamissense", "example_input_alphamissense", "2023", "DC"),
    ("AlphaPeptDeep", "build_alphapeptdeep", "example_input_alphapeptdeep", "2022", "DC"),
    (
        "AttentionSiteDTI",
        "build_attentionsitedti",
        "example_input_attentionsitedti",
        "2022",
        "DC",
    ),
    ("BioBERT", "build_biobert", "example_input_biobert", "2020", "NLP"),
]

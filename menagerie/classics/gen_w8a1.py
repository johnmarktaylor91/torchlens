"""Wave 8 batch 1 menagerie classics: physics-aware catalyst GNNs, ocean-bottom
seismic phase picking, and chemical-language-model polymer informatics
(fingerprinting + multimodal NL/SMILES property prediction) plus a multi-stage
conditional diffusion model for retrosynthesis.

Sources checked (repo_url / desc_source columns of the build queue, web research
2026-07-01; no cloning, no pip installs beyond the base env):

  - PhAST: https://github.com/vict0rsch/phast ; Duval, Schmidt, Garcia-Duran,
    Miret, Malliaros, Bengio, Rolnick, arXiv:2211.12020 (Nov 2022) / JMLR 2024,
    "PhAST: Physics-Aware, Scalable, and Task-specific GNNs for Accelerated
    Catalyst Design". Confirmed from the paper abstract and JMLR/arXiv text:
    a drop-in set of four physics-informed architectural modules meant to
    retrofit any catalyst-adsorbate GNN -- (1) a tailored graph construction
    that only creates edges between adsorbate atoms and the physically nearby
    catalyst-surface atoms (instead of a generic radius graph over every atom),
    (2) richer per-atom input features combining a learned atomic-number
    embedding with physics-based scalar tags (surface vs. adsorbate
    membership, atomic radius/period/group-style descriptors), (3) an energy
    head that predicts one scalar per atom and sums only over the tagged
    "energy-relevant" atoms (a physically-motivated pooling restriction,
    instead of pooling over all atoms uniformly), and (4) a direct force head
    that predicts a per-atom 3-vector so that ``-grad(energy)`` need not be
    computed via autograd, cutting compute. Reproduced here as a compact
    message-passing GNN over a small fixed catalyst-slab + adsorbate graph
    (bipartite-tagged edges only between adsorbate and nearby-surface atoms)
    with atomic-number + physics-tag input embeddings, scatter-sum message
    passing, an energy head that pools only the adsorbate+near-surface atoms,
    and a separate per-atom direct force-vector head -- the paper's four
    hallmark ideas preserved on a small fixed graph.

  - PickBlue: https://github.com/seisbench/seisbench (model shipped in the
    SeisBench hub) ; Bornstein, Lange, Munchmeyer, Woollam, Rietbrock, Barcheck,
    Grevemeyer, Tilmann, Earth and Space Science 11, e2023EA003332 (2024),
    "PickBlue: Seismic Phase Picking for Ocean Bottom Seismometers With Deep
    Learning". Confirmed from the paper abstract: PickBlue fine-tunes the
    EQTransformer architecture (and, separately, PhaseNet) for ocean-bottom
    seismometer (OBS) data by adding a fourth input channel for the
    hydrophone trace alongside the usual 3-component (Z/N/E) geophone
    channels. Reproduced here as a compact EQTransformer-style hierarchical
    network: a shared 1D-conv + residual-block + bidirectional-LSTM encoder
    that ingests 4 input channels (3 seismometer components + 1 hydrophone --
    the paper's key OBS-specific modification), followed by a lightweight
    self-attention block, feeding three parallel decoder heads (event
    detection, P-pick, S-pick), each emitting a per-timestep probability via
    its own 1D-conv + LSTM decoder stack -- the paper's multi-task
    detector/P-picker/S-picker hierarchy preserved with the OBS hydrophone
    channel as the distinguishing input.

  - polyBERT: https://github.com/Ramprasad-Group/psmiles (repo hosting the
    PSMILES tooling around the model) ; model card
    https://huggingface.co/kuelumbus/polyBERT ; Kuenneth, Ramprasad, Nature
    Communications 14, 4099 (2023), "polyBERT: a chemical language model to
    enable fully machine-driven ultrafast polymer informatics". Confirmed
    from the paper abstract and HF model card: a DeBERTa-based encoder-only
    Transformer (disentangled relative-position attention, the DeBERTa
    hallmark distinguishing it from a plain BERT encoder) trained with masked
    language modeling on PSMILES (polymer SMILES) strings, whose pooled
    [CLS]-style output is projected to a fixed-size dense "polymer
    fingerprint" vector (600-d in the paper) used as a drop-in feature vector
    for downstream property-prediction heads. Reproduced here via
    ``transformers.DebertaV2Config``/``DebertaV2Model`` at tiny dims (the
    paper's exact backbone family) with a fingerprint projection head on the
    pooled first-token hidden state -- the paper's central "chemical language
    model -> fixed dense fingerprint" idea.

  - PolyNC: https://github.com/hkqiu/Unified_ML4Polymers ; Qiu, Xu, Guo,
    Chemical Science 15, 534 (2024), "PolyNC: a natural and chemical language
    model for the prediction of unified polymer properties". Confirmed from
    the paper abstract: PolyNC is a T5 (text-to-text transfer transformer)
    encoder-decoder that unifies polymer property prediction across many
    properties and both regression/classification into one text-to-text
    format, by combining a *natural-language* prompt/property-name string
    with a *chemical-language* (PSMILES) string as one concatenated input
    sequence, decoding the property value as generated text -- the paper's
    central multimodal NL+chemical-language-in-one-sequence idea (as opposed
    to a separate property-specific head per task). Reproduced here via
    ``transformers.T5Config``/``T5ForConditionalGeneration`` at tiny dims with
    a single shared vocabulary spanning both natural-language tokens and
    PSMILES tokens, fed as one concatenated encoder input, decoded
    autoregressively into the property-value text -- the paper's unified
    text-to-text multimodal formulation preserved on a small fixed sequence.

  - RetroDiff: https://github.com/Alsace08/RetroDiff ; anonymous/Wang et al.
    (repo attribution: Alsace08), arXiv:2311.14077 (Nov 2023) / AISTATS 2025,
    "RetroDiff: Retrosynthesis as Multi-stage Distribution Interpolation".
    Confirmed from the paper abstract and OpenReview PDF: retrosynthesis is
    cast as a conditional graph-to-graph generative task solved by a
    *two-stage* discrete diffusion process conditioned on the product graph --
    stage 1 denoises a "dummy" external-group-node distribution into the
    external functional-group nodes/types that must be attached to the
    product's reaction-center atoms, and stage 2 (conditioned on the stage-1
    output) denoises a dummy bond-distribution into the external bonds that
    splice those groups onto the product, mirroring the classical semi-
    template reaction-center-then-synthon-completion pipeline. Reproduced
    here as a compact product-graph transformer encoder (shared across both
    stages, conditioning on the fixed product graph) feeding two sequential
    discrete-diffusion-style denoising heads: a group-node-type head
    (stage 1, conditioned on the encoder output and a diffusion timestep
    embedding) and a bond-type head over (product-atom, external-group-node)
    pairs (stage 2, additionally conditioned on the stage-1 group
    predictions) -- the paper's two-stage group-then-bond diffusion
    interpolation preserved on a small fixed product graph with a fixed
    number of external group slots.

RetroExplainer (repo ``wangyu-sd/RetroExplainer``, later renamed
``MechRetro``) is already present in the catalog as ``MechRetro`` in
``menagerie/classics/gen_w7a21.py`` and is therefore SKIPPED here as a
duplicate (see ``menagerie/DISCOVER_MODELS.md`` duplicate-avoidance rule).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    DebertaV2Config,
    DebertaV2Model,
    T5Config,
    T5ForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# PhAST: physics-aware catalyst-adsorbate GNN with tailored bipartite graph
# construction, physics-tagged atom features, atom-restricted energy pooling,
# and a direct (non-autograd) force head.
# ---------------------------------------------------------------------------


class PhASTLayer(nn.Module):
    """One scatter-sum message-passing layer over a fixed catalyst graph."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.update_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU())
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """Propagate messages along the tailored adsorbate<->surface edges.

        Parameters
        ----------
        h:
            Atom features, shape ``(n_atoms, dim)``.
        edge_index:
            Directed edge list, shape ``(2, n_edges)`` with source/target atom
            indices restricted (by construction) to adsorbate-to-nearby-
            surface pairs and their reverse.

        Returns
        -------
        Tensor
            Updated atom features, shape ``(n_atoms, dim)``.
        """

        src, dst = edge_index[0], edge_index[1]
        msg_in = torch.cat([h[src], h[dst]], dim=-1)
        messages = self.message_mlp(msg_in)
        agg = torch.zeros_like(h).index_add_(0, dst, messages)
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class PhAST(nn.Module):
    """Physics-aware catalyst GNN: tagged features, restricted graph/pooling,
    plus a direct per-atom force head."""

    def __init__(
        self,
        n_elements: int = 20,
        dim: int = 24,
        n_layers: int = 3,
        n_physics_tags: int = 3,
    ) -> None:
        super().__init__()
        self.elem_embed = nn.Embedding(n_elements, dim)
        # Physics-based scalar tags: [is_adsorbate, is_surface, is_subsurface].
        self.tag_proj = nn.Linear(n_physics_tags, dim)
        self.layers = nn.ModuleList([PhASTLayer(dim) for _ in range(n_layers)])
        # Energy head predicts one scalar per atom; summed only over the
        # "energy-relevant" mask (adsorbate + near-surface atoms) rather than
        # all atoms -- PhAST's physics-restricted pooling.
        self.energy_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1)
        )
        # Direct force head: per-atom 3-vector, avoiding an autograd
        # backward-through-energy force computation.
        self.force_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 3))

    def forward(
        self,
        atomic_numbers: Tensor,
        physics_tags: Tensor,
        edge_index: Tensor,
        energy_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict total system energy and per-atom forces.

        Parameters
        ----------
        atomic_numbers:
            Long atomic-number index per atom, shape ``(n_atoms,)``.
        physics_tags:
            Per-atom physics tag vector, shape ``(n_atoms, 3)``.
        edge_index:
            Tailored adsorbate<->surface edge list, shape ``(2, n_edges)``.
        energy_mask:
            Boolean mask selecting energy-relevant atoms, shape ``(n_atoms,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(total_energy, forces)``: scalar energy and ``(n_atoms, 3)``
            direct per-atom force predictions.
        """

        h = self.elem_embed(atomic_numbers) + self.tag_proj(physics_tags)
        for layer in self.layers:
            h = layer(h, edge_index)
        per_atom_energy = self.energy_head(h).squeeze(-1)
        total_energy = (per_atom_energy * energy_mask.float()).sum()
        forces = self.force_head(h)
        return total_energy, forces


def build_phast() -> nn.Module:
    """Build a compact PhAST physics-aware catalyst GNN.

    Returns
    -------
    nn.Module
        Random-initialized PhAST in eval mode.
    """

    return PhAST().eval()


def example_input_phast() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a small fixed catalyst-slab + adsorbate graph for PhAST.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(atomic_numbers, physics_tags, edge_index, energy_mask)``.
    """

    torch.manual_seed(0)
    # 4 surface atoms + 2 adsorbate atoms.
    atomic_numbers = torch.tensor([12, 12, 12, 12, 6, 8])
    physics_tags = torch.zeros(6, 3)
    physics_tags[:4, 1] = 1.0  # is_surface
    physics_tags[4:, 0] = 1.0  # is_adsorbate
    # Tailored bipartite-style edges: adsorbate atoms (4,5) connect only to
    # the two nearest surface atoms (0,1), plus adsorbate-adsorbate bond.
    edges = [(4, 0), (0, 4), (4, 1), (1, 4), (5, 0), (0, 5), (5, 1), (1, 5), (4, 5), (5, 4)]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    energy_mask = torch.tensor([True, True, False, False, True, True])
    return atomic_numbers, physics_tags, edge_index, energy_mask


# ---------------------------------------------------------------------------
# PickBlue: EQTransformer-style hierarchical detector/P-picker/S-picker with
# a 4th hydrophone input channel for ocean-bottom seismometer data.
# ---------------------------------------------------------------------------


class PickBlueDecoderHead(nn.Module):
    """One decoder branch (detector / P-picker / S-picker)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.conv = nn.Conv1d(dim, dim // 2, kernel_size=5, padding=2)
        self.out_proj = nn.Conv1d(dim // 2, 1, kernel_size=1)

    def forward(self, h: Tensor) -> Tensor:
        """Decode shared encoder features into a per-timestep probability.

        Parameters
        ----------
        h:
            Shared encoder features, shape ``(batch, time, dim)``.

        Returns
        -------
        Tensor
            Per-timestep probability, shape ``(batch, time)``.
        """

        lstm_out, _ = self.lstm(h)
        x = F.relu(self.conv(lstm_out.transpose(1, 2)))
        prob = torch.sigmoid(self.out_proj(x)).squeeze(1)
        return prob


class PickBlue(nn.Module):
    """EQTransformer-style multi-task picker with an OBS hydrophone channel."""

    def __init__(self, n_channels: int = 4, dim: int = 16) -> None:
        super().__init__()
        # 4 input channels: Z, N, E geophone components + hydrophone -- the
        # paper's OBS-specific addition over the original 3-channel EQT/
        # PhaseNet input.
        self.conv_stem = nn.Sequential(
            nn.Conv1d(n_channels, dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)
        self.detector = PickBlueDecoderHead(dim)
        self.p_picker = PickBlueDecoderHead(dim)
        self.s_picker = PickBlueDecoderHead(dim)

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the shared encoder and three parallel decoder heads.

        Parameters
        ----------
        waveform:
            4-channel OBS waveform, shape ``(batch, 4, time)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(detection_prob, p_prob, s_prob)``, each ``(batch, time)``.
        """

        x = self.conv_stem(waveform).transpose(1, 2)  # (batch, time, dim)
        x, _ = self.bilstm(x)
        attn_out, _ = self.attn(x, x, x)
        h = x + attn_out
        return self.detector(h), self.p_picker(h), self.s_picker(h)


def build_pickblue() -> nn.Module:
    """Build a compact PickBlue OBS phase picker.

    Returns
    -------
    nn.Module
        Random-initialized PickBlue in eval mode.
    """

    return PickBlue().eval()


def example_input_pickblue() -> Tensor:
    """Create a small fixed 4-channel OBS waveform window for PickBlue.

    Returns
    -------
    Tensor
        Waveform, shape ``(1, 4, 64)``.
    """

    torch.manual_seed(0)
    return torch.randn(1, 4, 64)


# ---------------------------------------------------------------------------
# polyBERT: DeBERTa-based encoder-only chemical language model producing a
# fixed-size dense polymer fingerprint from a PSMILES token sequence.
# ---------------------------------------------------------------------------


class PolyBERT(nn.Module):
    """DeBERTa polymer-language-model fingerprint encoder."""

    def __init__(
        self,
        vocab_size: int = 96,
        hidden_size: int = 32,
        n_layers: int = 2,
        n_heads: int = 2,
        fingerprint_dim: int = 24,
        max_position: int = 64,
    ) -> None:
        super().__init__()
        cfg = DebertaV2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=hidden_size * 2,
            max_position_embeddings=max_position,
            relative_attention=True,
            position_buckets=16,
        )
        self.deberta = DebertaV2Model(cfg)
        # Projects the pooled first-token hidden state to the dense
        # fixed-size "polymer fingerprint" vector -- polyBERT's central
        # chemical-language-model -> fingerprint mapping.
        self.fingerprint_proj = nn.Linear(hidden_size, fingerprint_dim)

    def forward(self, psmiles_ids: Tensor) -> Tensor:
        """Encode a tokenized PSMILES string into a dense fingerprint vector.

        Parameters
        ----------
        psmiles_ids:
            Tokenized PSMILES ids, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Dense polymer fingerprint, shape ``(batch, fingerprint_dim)``.
        """

        out = self.deberta(input_ids=psmiles_ids).last_hidden_state
        pooled = out[:, 0, :]
        return self.fingerprint_proj(pooled)


def build_polybert() -> nn.Module:
    """Build a compact polyBERT chemical-language-model fingerprinter.

    Returns
    -------
    nn.Module
        Random-initialized PolyBERT in eval mode.
    """

    return PolyBERT().eval()


def example_input_polybert() -> Tensor:
    """Create a small fixed tokenized PSMILES sequence for polyBERT.

    Returns
    -------
    Tensor
        Token ids, shape ``(1, 20)``.
    """

    torch.manual_seed(0)
    return torch.randint(0, 96, (1, 20))


# ---------------------------------------------------------------------------
# PolyNC: T5 encoder-decoder unifying natural-language property prompts and
# PSMILES chemical-language input into one text-to-text sequence.
# ---------------------------------------------------------------------------


class PolyNC(nn.Module):
    """T5-based unified natural-language + chemical-language polymer model."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        cfg = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_model // n_heads,
            d_ff=4 * d_model,
            num_layers=n_layers,
            num_heads=n_heads,
        )
        self.t5 = T5ForConditionalGeneration(cfg)

    def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Decode a property value from one concatenated NL+PSMILES sequence.

        Parameters
        ----------
        input_ids:
            Token ids for the concatenated ``"<property prompt> <PSMILES>"``
            sequence -- PolyNC's unified multimodal text-to-text input --
            shape ``(batch, src_len)``.
        decoder_input_ids:
            Decoder token ids for the generated property-value text
            (teacher forcing), shape ``(batch, tgt_len)``.

        Returns
        -------
        Tensor
            Output vocabulary logits, shape ``(batch, tgt_len, vocab_size)``.
        """

        out = self.t5(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        return out.logits


def build_polync() -> nn.Module:
    """Build a compact PolyNC unified NL+chemical-language polymer model.

    Returns
    -------
    nn.Module
        Random-initialized PolyNC in eval mode.
    """

    return PolyNC().eval()


def example_input_polync() -> tuple[Tensor, Tensor]:
    """Create a fixed concatenated NL-prompt + PSMILES token sequence.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(input_ids, decoder_input_ids)``.
    """

    torch.manual_seed(0)
    input_ids = torch.randint(0, 128, (1, 24))
    decoder_input_ids = torch.randint(0, 128, (1, 6))
    return input_ids, decoder_input_ids


# ---------------------------------------------------------------------------
# RetroDiff: two-stage discrete-diffusion retrosynthesis -- stage 1 denoises
# external functional-group node types, stage 2 (conditioned on stage 1)
# denoises the external bonds splicing those groups onto the product graph.
# ---------------------------------------------------------------------------


class RetroDiffEncoder(nn.Module):
    """Product-graph transformer encoder shared by both diffusion stages."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, atom_feats: Tensor, adj_mask: Tensor) -> Tensor:
        """Encode the fixed product graph, masking attention to bonded pairs.

        Parameters
        ----------
        atom_feats:
            Product-atom features, shape ``(1, n_atoms, dim)``.
        adj_mask:
            Additive attention bias from the product adjacency, shape
            ``(n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Encoded product-atom features, shape ``(1, n_atoms, dim)``.
        """

        attn_out, _ = self.attn(atom_feats, atom_feats, atom_feats, attn_mask=adj_mask)
        h = self.norm1(atom_feats + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h


class RetroDiff(nn.Module):
    """Two-stage discrete-diffusion retrosynthesis: group nodes then bonds."""

    def __init__(
        self,
        in_dim: int = 10,
        dim: int = 24,
        n_group_slots: int = 3,
        n_group_types: int = 8,
        n_bond_types: int = 4,
        time_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Linear(in_dim, dim)
        self.encoder = RetroDiffEncoder(dim)
        self.time_embed = nn.Embedding(time_embed_dim, dim)
        # Stage 1: dummy-distribution group-node embeddings, denoised into
        # external functional-group types conditioned on the encoded product
        # graph and the diffusion timestep.
        self.group_query = nn.Parameter(torch.randn(n_group_slots, dim) * 0.02)
        self.group_denoise = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.group_type_head = nn.Linear(dim, n_group_types)
        # Stage 2: dummy-distribution bond logits over every
        # (product-atom, group-slot) pair, conditioned on stage-1 group
        # embeddings -- the paper's serial group-then-bond interpolation.
        self.bond_head = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, n_bond_types)
        )

    def forward(
        self, atom_feats: Tensor, adj_mask: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the two-stage group-then-bond diffusion denoising pass.

        Parameters
        ----------
        atom_feats:
            Product-atom input features, shape ``(1, n_atoms, in_dim)``.
        adj_mask:
            Additive attention bias from the product adjacency, shape
            ``(n_atoms, n_atoms)``.
        timestep:
            Diffusion timestep index, shape ``(1,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(group_type_logits, bond_type_logits)``:
            ``(n_group_slots, n_group_types)`` and
            ``(n_atoms, n_group_slots, n_bond_types)``.
        """

        h = self.atom_embed(atom_feats)
        h = self.encoder(h, adj_mask).squeeze(0)  # (n_atoms, dim)
        t_emb = self.time_embed(timestep).expand(self.group_query.size(0), -1)
        product_summary = h.mean(dim=0, keepdim=True).expand(self.group_query.size(0), -1)
        group_in = torch.cat([self.group_query + t_emb, product_summary], dim=-1)
        group_h = self.group_denoise(group_in)  # (n_group_slots, dim)
        group_logits = self.group_type_head(group_h)

        n_atoms = h.size(0)
        n_slots = group_h.size(0)
        atom_expand = h.unsqueeze(1).expand(n_atoms, n_slots, -1)
        group_expand = group_h.unsqueeze(0).expand(n_atoms, n_slots, -1)
        bond_in = torch.cat([atom_expand, group_expand], dim=-1)
        bond_logits = self.bond_head(bond_in)
        return group_logits, bond_logits


def build_retrodiff() -> nn.Module:
    """Build a compact RetroDiff two-stage diffusion retrosynthesis model.

    Returns
    -------
    nn.Module
        Random-initialized RetroDiff in eval mode.
    """

    return RetroDiff().eval()


def example_input_retrodiff() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small fixed product-molecule graph and diffusion timestep.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_feats, adj_mask, timestep)``.
    """

    torch.manual_seed(0)
    n_atoms = 6
    atom_feats = torch.randn(1, n_atoms, 10)
    adj = torch.zeros(n_atoms, n_atoms)
    chain = list(range(n_atoms))
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        adj[a, b] = adj[b, a] = 1.0
    adj.fill_diagonal_(1.0)
    adj_mask = torch.where(adj > 0, 0.0, float("-inf"))
    timestep = torch.tensor([3])
    return atom_feats, adj_mask, timestep


MENAGERIE_ENTRIES = [
    ("PhAST", "build_phast", "example_input_phast", "2022", "GRAPH"),
    ("PickBlue", "build_pickblue", "example_input_pickblue", "2023", "AUDIO"),
    ("polyBERT", "build_polybert", "example_input_polybert", "2023", "BIO"),
    ("PolyNC", "build_polync", "example_input_polync", "2024", "BIO"),
    ("RetroDiff", "build_retrodiff", "example_input_retrodiff", "2023", "BIO"),
]

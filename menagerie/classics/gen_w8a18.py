"""Wave 8 batch 18 menagerie classics: molecular generative / equivariant models.

Sources checked (reference only -- no cloning, no pip installs; reimplemented
compactly from scratch in base-env torch):

  * Dual-TB (tied two-way transformer retrosynthesis): AstraZeneca/MolecularAI
    Chemformer repo (https://github.com/MolecularAI/Chemformer); Kreutter, Rai
    & Reymond / Duigou et al. "Valid, Plausible, and Diverse Retrosynthesis
    Using Tied Two-Way Transformers with Latent Variables" (JCIM 2021); the
    same generic Chemformer BART encoder-decoder already lives in
    ``build_chemformer`` (``menagerie/classics/gen_w7a13.py``) -- Dual-TB is
    a distinct architecture (a forward and backward transformer that SHARE
    weights, trained jointly with a cycle-consistency / round-trip
    reconstruction term over the two translation directions).
  * E-NF: Satorras, Hoogeboom, Fuchs, Posner & Welling, "E(n) Equivariant
    Normalizing Flows", NeurIPS 2021, arXiv:2105.09016;
    https://github.com/vgsatorras/en_flows. Continuous-time normalizing flow
    (an ODE) over joint (coordinates, categorical-feature) atom state, whose
    velocity field is an EGNN (equivariant graph neural network) so that the
    flow's coordinate output is E(n)-equivariant while feature output is
    invariant. Log-density change is accumulated via a Hutchinson-trace
    estimate of the velocity field's divergence, integrated with a fixed-step
    Euler solver (torchlens needs a concrete, unrolled step count).
  * EDM: Hoogeboom, Satorras, Vignac & Welling, "Equivariant Diffusion for
    Molecule Generation in 3D", ICML 2022, arXiv:2203.17003;
    https://github.com/ehoogeboom/e3_diffusion_for_molecules. A DDPM-style
    diffusion model directly on joint continuous atom coordinates and
    categorical atom-type/charge features; the denoiser is an EGNN that
    predicts the added noise, conditioned on the diffusion timestep, and its
    coordinate output is made zero-CoM (translation-invariant) at every
    layer. NOTE: this is a different "EDM" than the Karras et al. SongUNet
    ``edm_unet`` already in the catalog (``menagerie/classics/ri_gan_misc.py``)
    -- same acronym, unrelated image-diffusion architecture.
  * EEGSDE: Bao, Zhao, Li, Su & Zhu, "Equivariant Energy-Guided SDE for
    Inverse Molecular Design", ICLR 2023, arXiv:2209.15408;
    https://github.com/gracezhao1997/EEGSDE. Adds an energy-guidance term to
    an existing equivariant diffusion (EDM) backbone: a separately-trained
    property-predictor energy network produces a gradient-like guidance
    signal that is added to the diffusion score/noise prediction at each
    reverse step, steering generation toward a target property value.
  * ELECTRO: Bradshaw, Kusner, Paige, Segler & Hernandez-Lobato, "A
    Generative Model For Electron Paths", ICLR 2019;
    https://github.com/john-bradshaw/electro. Models a chemical reaction as
    an autoregressive sequence of electron-pushing ("arrow-pushing") steps:
    a graph encoder produces per-atom embeddings, and at each step an
    autoregressive head scores every atom as the next electron
    source/sink, conditioned on the steps taken so far (a GRU over the
    sequence of selected-atom embeddings).
  * FLAG: Zhang & Liu, "Molecule Generation For Target Protein Binding with
    Structural Motifs", ICLR 2023, arXiv:2207.01005;
    https://github.com/zaixizhang/FLAG. Fragment/motif-based autoregressive
    ligand generation conditioned on a protein pocket: an E(3)-equivariant
    (EGNN-style) pocket+ligand context encoder produces a focal-atom
    embedding at each generation step, from which a motif-classification
    head picks the next fragment (from a small motif vocabulary) and an
    attachment head predicts its 3D attachment placement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Dual-TB -- tied two-way (forward/backward) Transformer with shared weights
# and a round-trip (cycle-consistency) reconstruction pass, for retrosynthesis.
# ---------------------------------------------------------------------------


class TiedTwoWayTransformer(nn.Module):
    """A single shared encoder-decoder Transformer run in both translation directions.

    The same weights encode reactants->product and product->reactants; a
    direction token distinguishes the two so the round trip (encode one way,
    decode the other, then translate back) can be scored for cycle
    consistency, as in Dual-TB / tied two-way retrosynthesis.
    """

    def __init__(
        self, vocab_size: int = 48, d_model: int = 32, n_heads: int = 4, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.dir_embed = nn.Embedding(2, d_model)  # 0: fwd (react->prod), 1: bwd (prod->react)
        self.pos_embed = nn.Embedding(64, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _embed(self, tokens: Tensor, direction: int) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).clamp(max=63)
        d = torch.full_like(tokens[:, :1], direction).expand(-1, tokens.shape[1])
        return self.tok_embed(tokens) + self.pos_embed(positions)[None] + self.dir_embed(d)

    def translate(self, src_tokens: Tensor, tgt_tokens: Tensor, direction: int) -> Tensor:
        """Encode ``src_tokens`` and decode ``tgt_tokens`` under the shared weights."""
        memory = self.encoder(self._embed(src_tokens, direction))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.shape[1]).to(
            tgt_tokens.device
        )
        hidden = self.decoder(self._embed(tgt_tokens, 1 - direction), memory, tgt_mask=causal_mask)
        return self.out_proj(hidden)

    def forward(self, reactant_tokens: Tensor, product_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Round trip: forward (reactants->product) then backward (product->reactants)."""
        fwd_logits = self.translate(reactant_tokens, product_tokens, direction=0)
        bwd_logits = self.translate(product_tokens, reactant_tokens, direction=1)
        return fwd_logits, bwd_logits


def build_dual_tb() -> nn.Module:
    """Build a compact Dual-TB tied two-way transformer for retrosynthesis.

    Returns
    -------
    nn.Module
        ``TiedTwoWayTransformer`` in eval mode.
    """
    return TiedTwoWayTransformer(vocab_size=48, d_model=32, n_heads=4, n_layers=2).eval()


def example_input_dual_tb() -> tuple[Tensor, Tensor]:
    """Return (reactant SMILES token ids, product SMILES token ids)."""
    batch, seq_len = 2, 12
    reactant_tokens = torch.randint(0, 48, (batch, seq_len))
    product_tokens = torch.randint(0, 48, (batch, seq_len))
    return reactant_tokens, product_tokens


# ---------------------------------------------------------------------------
# E-NF -- E(n)-equivariant continuous-time normalizing flow: an EGNN velocity
# field integrated with a fixed-step Euler solver, plus a Hutchinson-trace
# divergence estimate for the log-density change.
# ---------------------------------------------------------------------------


class ENFEquivariantLayer(nn.Module):
    """EGNN-style equivariant message-passing layer over a fully-connected atom graph."""

    def __init__(self, feat_dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, feat_dim)
        )

    def forward(self, feats: Tensor, coords: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return (feature velocity, coordinate velocity) at continuous time ``t``."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist2 = (diff**2).sum(-1, keepdim=True)
        f_i = feats.unsqueeze(2).expand(-1, -1, feats.shape[1], -1)
        f_j = feats.unsqueeze(1).expand(-1, feats.shape[1], -1, -1)
        t_edge = t.view(-1, 1, 1, 1).expand(-1, feats.shape[1], feats.shape[1], 1)
        edge_feat = self.edge_mlp(torch.cat([f_i, f_j, dist2, t_edge], dim=-1))
        coord_weight = self.coord_mlp(edge_feat)
        n_atoms = coords.shape[1]
        coord_velocity = (diff * coord_weight).sum(dim=2) / (n_atoms - 1)
        agg = edge_feat.mean(dim=2)
        feat_velocity = self.feat_mlp(torch.cat([feats, agg], dim=-1))
        return feat_velocity, coord_velocity


class EquivariantNormalizingFlow(nn.Module):
    """E(n)-equivariant CNF: EGNN velocity field + fixed-step Euler ODE integration.

    Jointly transforms atom positions (equivariant) and categorical features
    (invariant) from a base Gaussian toward the data distribution, while
    accumulating the log-density change via a Hutchinson-trace estimate of
    the velocity field's divergence at each Euler step -- the E-NF recipe.
    """

    def __init__(self, feat_dim: int = 16, hidden: int = 32, n_steps: int = 4) -> None:
        super().__init__()
        self.velocity_field = ENFEquivariantLayer(feat_dim, hidden)
        self.n_steps = n_steps

    def forward(self, feats: Tensor, coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Integrate the flow from t=0 to t=1; return (feats, coords, log_det)."""
        batch = coords.shape[0]
        dt = 1.0 / self.n_steps
        log_det = torch.zeros(batch, device=coords.device)
        noise = torch.randn_like(coords)
        for step in range(self.n_steps):
            t = torch.full((batch,), step * dt, device=coords.device)
            with torch.enable_grad():
                coords_req = coords.detach().requires_grad_(True)
                feat_v, coord_v = self.velocity_field(feats, coords_req, t)
                # Hutchinson-trace estimate of div(coord velocity) w.r.t. coords.
                vjp = torch.autograd.grad(
                    (coord_v * noise).sum(), coords_req, create_graph=self.training
                )[0]
            divergence = (vjp * noise).flatten(1).sum(-1)
            log_det = log_det - divergence * dt
            feats = feats + dt * feat_v.detach() if not self.training else feats + dt * feat_v
            coords = coords + dt * coord_v.detach() if not self.training else coords + dt * coord_v
        return feats, coords, log_det


def build_e_nf() -> nn.Module:
    """Build a compact E(n)-Equivariant Normalizing Flow (E-NF) over atom clouds.

    Returns
    -------
    nn.Module
        ``EquivariantNormalizingFlow`` in eval mode.
    """
    return EquivariantNormalizingFlow(feat_dim=16, hidden=32, n_steps=4).eval()


def example_input_e_nf() -> tuple[Tensor, Tensor]:
    """Return (atom features, atom coordinates) for a small molecule."""
    batch, n_atoms, feat_dim = 2, 8, 16
    feats = torch.randn(batch, n_atoms, feat_dim)
    coords = torch.randn(batch, n_atoms, 3)
    return feats, coords


# ---------------------------------------------------------------------------
# EDM -- E(3)-equivariant diffusion model for molecules: EGNN denoiser jointly
# predicting noise on continuous coordinates + categorical atom features,
# conditioned on the diffusion timestep, with zero-CoM coordinate updates.
# ---------------------------------------------------------------------------


class EDMEquivariantBlock(nn.Module):
    """EGNN block with a zero-center-of-mass coordinate update (EDM denoiser core)."""

    def __init__(self, feat_dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, feat_dim)
        )

    def forward(self, feats: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Update features and (zero-CoM) coordinates by one equivariant message pass."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist2 = (diff**2).sum(-1, keepdim=True)
        f_i = feats.unsqueeze(2).expand(-1, -1, feats.shape[1], -1)
        f_j = feats.unsqueeze(1).expand(-1, feats.shape[1], -1, -1)
        edge_feat = self.edge_mlp(torch.cat([f_i, f_j, dist2], dim=-1))
        coord_weight = self.coord_mlp(edge_feat)
        n_atoms = coords.shape[1]
        coord_update = (diff * coord_weight).sum(dim=2) / (n_atoms - 1)
        coord_update = coord_update - coord_update.mean(dim=1, keepdim=True)  # zero-CoM
        coords_out = coords + coord_update
        agg = edge_feat.mean(dim=2)
        feats_out = feats + self.feat_mlp(torch.cat([feats, agg], dim=-1))
        return feats_out, coords_out


class EquivariantDiffusionModel(nn.Module):
    """E(3)-equivariant molecular diffusion denoiser (EDM): predicts joint noise.

    Given noised atom coordinates and categorical (one-hot atom-type/charge)
    features at diffusion timestep ``t``, predicts the additive Gaussian
    noise on both streams so that the reverse SDE can denoise a joint
    continuous+categorical atom cloud, as in Hoogeboom et al. 2022.
    """

    def __init__(self, feat_dim: int = 10, hidden: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, feat_dim)
        )
        self.blocks = nn.ModuleList(
            [EDMEquivariantBlock(feat_dim, hidden) for _ in range(n_layers)]
        )
        self.feat_noise_head = nn.Linear(feat_dim, feat_dim)

    def forward(self, feats: Tensor, coords: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Predict (feature noise, coordinate noise) at diffusion time ``t``."""
        t_emb = self.time_embed(t.view(-1, 1, 1)).expand(-1, feats.shape[1], -1)
        h = feats + t_emb
        coords_t = coords - coords.mean(dim=1, keepdim=True)  # start from zero-CoM subspace
        for block in self.blocks:
            h, coords_t = block(h, coords_t)
        feat_noise = self.feat_noise_head(h)
        coord_noise = coords_t - coords
        return feat_noise, coord_noise


def build_edm() -> nn.Module:
    """Build a compact E(3)-Equivariant Diffusion Model (EDM) for molecule generation.

    Returns
    -------
    nn.Module
        ``EquivariantDiffusionModel`` in eval mode.
    """
    return EquivariantDiffusionModel(feat_dim=10, hidden=32, n_layers=3).eval()


def example_input_edm() -> tuple[Tensor, Tensor, Tensor]:
    """Return (categorical atom features, noised coordinates, diffusion times)."""
    batch, n_atoms, feat_dim = 2, 9, 10
    feats = torch.randn(batch, n_atoms, feat_dim)
    coords = torch.randn(batch, n_atoms, 3)
    t = torch.rand(batch)
    return feats, coords, t


# ---------------------------------------------------------------------------
# EEGSDE -- energy-guided SDE: an EDM-style diffusion backbone plus a
# separate property-predictor "energy" network whose gradient-like guidance
# is added to the backbone's noise prediction at every reverse step.
# ---------------------------------------------------------------------------


class EnergyGuidanceNet(nn.Module):
    """Small property-predictor ("energy") network read out from atom features."""

    def __init__(self, feat_dim: int = 10, hidden: int = 24) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU())
        self.readout = nn.Linear(hidden, 1)

    def energy(self, feats: Tensor, coords: Tensor) -> Tensor:
        """Scalar target-property estimate (e.g. a dipole moment surrogate) per molecule."""
        pooled = self.node_mlp(feats).mean(dim=1)
        pos_signal = coords.norm(dim=-1).mean(dim=1, keepdim=True)
        return self.readout(pooled) + pos_signal


class EEGSDEModel(nn.Module):
    """Energy-guided SDE: EDM backbone score + energy-network guidance term.

    Wraps an ``EquivariantDiffusionModel`` backbone; at each call, an
    auxiliary energy network's coordinate-gradient is computed (via
    autograd) and added to the backbone's coordinate-noise prediction, and
    its feature-gradient is added to the feature-noise prediction, steering
    generation toward a target property value -- the EEGSDE guidance recipe.
    """

    def __init__(self, feat_dim: int = 10, hidden: int = 32, guidance_scale: float = 0.5) -> None:
        super().__init__()
        self.backbone = EquivariantDiffusionModel(feat_dim=feat_dim, hidden=hidden, n_layers=2)
        self.energy_net = EnergyGuidanceNet(feat_dim=feat_dim, hidden=24)
        self.guidance_scale = guidance_scale

    def forward(self, feats: Tensor, coords: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Predict guided (feature noise, coordinate noise) at diffusion time ``t``."""
        feat_noise, coord_noise = self.backbone(feats, coords, t)
        with torch.enable_grad():
            feats_req = feats.detach().requires_grad_(True)
            coords_req = coords.detach().requires_grad_(True)
            energy = self.energy_net.energy(feats_req, coords_req).sum()
            feat_grad, coord_grad = torch.autograd.grad(
                energy, [feats_req, coords_req], create_graph=self.training
            )
        guided_feat_noise = feat_noise - self.guidance_scale * feat_grad.detach()
        guided_coord_noise = coord_noise - self.guidance_scale * coord_grad.detach()
        return guided_feat_noise, guided_coord_noise


def build_eegsde() -> nn.Module:
    """Build a compact EEGSDE energy-guided equivariant diffusion model.

    Returns
    -------
    nn.Module
        ``EEGSDEModel`` in eval mode.
    """
    return EEGSDEModel(feat_dim=10, hidden=32, guidance_scale=0.5).eval()


def example_input_eegsde() -> tuple[Tensor, Tensor, Tensor]:
    """Return (categorical atom features, noised coordinates, diffusion times)."""
    batch, n_atoms, feat_dim = 2, 8, 10
    feats = torch.randn(batch, n_atoms, feat_dim)
    coords = torch.randn(batch, n_atoms, 3)
    t = torch.rand(batch)
    return feats, coords, t


# ---------------------------------------------------------------------------
# ELECTRO -- autoregressive electron-pushing ("arrow-pushing") step model:
# a graph encoder produces atom embeddings, and a GRU-conditioned head scores
# every atom as the next electron source/sink at each step of the sequence.
# ---------------------------------------------------------------------------


class MoleculeGraphEncoder(nn.Module):
    """Small message-passing encoder producing per-atom embeddings from bonds."""

    def __init__(self, n_atom_types: int = 12, hidden: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.atom_embed = nn.Embedding(n_atom_types, hidden)
        self.msg_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU()) for _ in range(n_layers)]
        )

    def forward(self, atom_types: Tensor, adjacency: Tensor) -> Tensor:
        """Return per-atom embeddings ``(batch, n_atoms, hidden)`` from a bond adjacency matrix."""
        h = self.atom_embed(atom_types)
        for layer in self.msg_layers:
            h_i = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
            h_j = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            msg = layer(torch.cat([h_i, h_j], dim=-1))
            agg = (msg * adjacency.unsqueeze(-1)).sum(dim=2) / adjacency.sum(
                -1, keepdim=True
            ).clamp(min=1)
            h = h + agg
        return h


class ElectronPathModel(nn.Module):
    """Autoregressive electron-pushing sequence model over a reaction's atom graph.

    At each arrow-pushing step, scores every atom as the next electron
    source/sink, conditioned via a GRU on the embeddings of the atoms chosen
    at previous steps -- the ELECTRO generative model for electron paths.
    """

    def __init__(self, n_atom_types: int = 12, hidden: int = 32, n_steps: int = 3) -> None:
        super().__init__()
        self.encoder = MoleculeGraphEncoder(n_atom_types, hidden, n_layers=2)
        self.step_rnn = nn.GRUCell(hidden, hidden)
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.n_steps = n_steps

    def forward(self, atom_types: Tensor, adjacency: Tensor) -> Tensor:
        """Return per-step atom scores ``(batch, n_steps, n_atoms)`` for the electron path."""
        atom_emb = self.encoder(atom_types, adjacency)
        batch, n_atoms, hidden = atom_emb.shape
        rnn_state = atom_emb.mean(dim=1)
        step_scores = []
        current_atom_emb = rnn_state
        for _ in range(self.n_steps):
            rnn_state = self.step_rnn(current_atom_emb, rnn_state)
            query = rnn_state.unsqueeze(1).expand(-1, n_atoms, -1)
            scores = self.score_head(torch.cat([atom_emb, query], dim=-1)).squeeze(-1)
            step_scores.append(scores)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            current_atom_emb = (atom_emb * weights).sum(dim=1)
        return torch.stack(step_scores, dim=1)


def build_electro() -> nn.Module:
    """Build a compact ELECTRO autoregressive electron-pushing path model.

    Returns
    -------
    nn.Module
        ``ElectronPathModel`` in eval mode.
    """
    return ElectronPathModel(n_atom_types=12, hidden=32, n_steps=3).eval()


def example_input_electro() -> tuple[Tensor, Tensor]:
    """Return (atom type ids, symmetric bond adjacency matrix) for a small reaction graph."""
    batch, n_atoms = 2, 10
    atom_types = torch.randint(0, 12, (batch, n_atoms))
    adjacency = (torch.rand(batch, n_atoms, n_atoms) > 0.6).float()
    adjacency = torch.triu(adjacency, diagonal=1)
    adjacency = adjacency + adjacency.transpose(1, 2)
    return atom_types, adjacency


# ---------------------------------------------------------------------------
# FLAG -- fragment-based autoregressive ligand generation conditioned on a
# protein pocket: an equivariant pocket+ligand encoder yields a focal-atom
# embedding, from which a motif-classification head + attachment head pick
# the next fragment and place it in 3D.
# ---------------------------------------------------------------------------


class FLAGPocketLigandEncoder(nn.Module):
    """EGNN-style joint encoder over pocket residues + partial-ligand atoms."""

    def __init__(self, feat_dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, feat_dim)
        )

    def forward(self, feats: Tensor, coords: Tensor) -> Tensor:
        """Return updated per-atom embeddings ``(batch, n_atoms, feat_dim)``."""
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist2 = (diff**2).sum(-1, keepdim=True)
        f_i = feats.unsqueeze(2).expand(-1, -1, feats.shape[1], -1)
        f_j = feats.unsqueeze(1).expand(-1, feats.shape[1], -1, -1)
        edge_feat = self.edge_mlp(torch.cat([f_i, f_j, dist2], dim=-1))
        agg = edge_feat.mean(dim=2)
        return feats + self.node_mlp(torch.cat([feats, agg], dim=-1))


class FLAGMotifAttachment(nn.Module):
    """Fragment/motif-based autoregressive ligand generator conditioned on a pocket.

    An equivariant pocket+ligand context encoder produces a focal-atom
    embedding at the current growth frontier; a motif-classification head
    selects the next fragment from a small motif vocabulary and an
    attachment head predicts its 3D attachment offset from the focal atom
    -- the FLAG fragment-attachment recipe (Zhang & Liu 2023).
    """

    def __init__(
        self, feat_dim: int = 16, hidden: int = 32, n_layers: int = 2, n_motifs: int = 20
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FLAGPocketLigandEncoder(feat_dim, hidden) for _ in range(n_layers)]
        )
        self.motif_head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_motifs)
        )
        self.attach_head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, 3)
        )

    def forward(self, feats: Tensor, coords: Tensor, focal_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Predict (motif-type logits, 3D attachment offset) at the focal atom.

        Parameters
        ----------
        feats : Tensor
            Joint pocket+ligand per-atom categorical features, ``(batch, n_atoms, feat_dim)``.
        coords : Tensor
            Joint pocket+ligand 3D coordinates, ``(batch, n_atoms, 3)``.
        focal_idx : Tensor
            Index of the current growth-frontier atom per batch element, ``(batch,)``.
        """
        for layer in self.layers:
            feats = layer(feats, coords)
        batch_idx = torch.arange(feats.shape[0], device=feats.device)
        focal_emb = feats[batch_idx, focal_idx]
        motif_logits = self.motif_head(focal_emb)
        attach_offset = self.attach_head(focal_emb)
        return motif_logits, attach_offset


def build_flag() -> nn.Module:
    """Build a compact FLAG fragment-based pocket-conditioned ligand generator.

    Returns
    -------
    nn.Module
        ``FLAGMotifAttachment`` in eval mode.
    """
    return FLAGMotifAttachment(feat_dim=16, hidden=32, n_layers=2, n_motifs=20).eval()


def example_input_flag() -> tuple[Tensor, Tensor, Tensor]:
    """Return (joint pocket+ligand features, coordinates, focal-atom index)."""
    batch, n_atoms, feat_dim = 2, 14, 16
    feats = torch.randn(batch, n_atoms, feat_dim)
    coords = torch.randn(batch, n_atoms, 3)
    focal_idx = torch.randint(0, n_atoms, (batch,))
    return feats, coords, focal_idx


# ---------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    ("Dual-TB", "build_dual_tb", "example_input_dual_tb", "2021", "BIO"),
    ("E-NF", "build_e_nf", "example_input_e_nf", "2021", "BIO"),
    (
        "EDM (E(3) Equivariant Diffusion Model for molecules)",
        "build_edm",
        "example_input_edm",
        "2022",
        "BIO",
    ),
    ("EEGSDE", "build_eegsde", "example_input_eegsde", "2023", "BIO"),
    ("ELECTRO", "build_electro", "example_input_electro", "2019", "BIO"),
    ("FLAG (Fragment Ligand Generation)", "build_flag", "example_input_flag", "2023", "BIO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")

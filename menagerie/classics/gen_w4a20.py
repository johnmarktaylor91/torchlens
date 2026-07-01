"""Single-cell / metagenomics / histopathology / cryo-EM classics (batch w4a20).

Sources checked (paper + official repo source; no clone, no pip install --
reimplemented from scratch in base-env torch, or via installed libraries
where the source is a config of an installed architecture family):

- cNODE (compositional Neural ODE): Michael-Fabian Mata, ... , Yang-Yu Liu,
  bioRxiv 2021 (https://www.biorxiv.org/content/10.1101/2021.06.17.448886),
  https://github.com/michel-mata/cNODE.jl (official Julia repo, module
  ``cNODE.jl`` -- ``src/module/trainer.jl``, struct ``FitnessLayer`` +
  ``getModel``). Predicts steady-state relative species abundances from
  initial (normalized) composition ``z`` via a *compositional* Neural ODE:
  a single learned ``N x N`` interaction/"fitness" matrix ``W`` defines the
  vector field ``f = W @ p``; the replicator-equation-style dynamics
  ``dp/dt = p * (f - (p^T f) * 1)`` are what force the state to remain on
  the probability simplex (sums to 1) throughout integration -- the
  defining "compositional" constraint that distinguishes cNODE from a
  generic Neural ODE. The official Julia code integrates this with Tsit5
  from t=0 to t=1 and reads out the state at t=1 (``NeuralODE(...,
  saveat=1.0)``). Reimplemented here in PyTorch as an ``nn.Module`` owning
  the learned ``W`` and unrolling the exact replicator vector field with a
  fixed-step RK4 integrator from 0 to 1 (Julia's Tsit5 is an adaptive
  Runge-Kutta method; RK4 with a modest number of fixed steps is the
  standard graph-traceable stand-in), with a softmax projection of the raw
  input onto the simplex to form ``z`` for the model to evolve.

- Cobolt: Boying Gong, Yun Zhou & Elizabeth Purdom, Genome Biology 2021,
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02556-z ;
  https://github.com/epurdom/cobolt (official epurdom repo,
  ``cobolt/model/coboltmodel.py``, class ``CoboltModel`` +
  ``ProductOfExperts``). A multi-omic (paired/unpaired RNA + ATAC) VAE for
  joint cell-state topics: each modality has its own MLP encoder producing
  a per-modality ``(mu, log_var)``; a **Product-of-Experts** (PoE) combines
  the per-modality Gaussians (plus a fixed Laplace-approximate-Dirichlet
  prior "expert") into one joint posterior via precision-weighted averaging
  (``T = 1/var; pd_mu = sum(mu*T)/sum(T); pd_var = 1/sum(T)``) -- this PoE
  fusion of independently-encoded modality-specific Gaussians is Cobolt's
  namesake mechanism for handling arbitrary subsets of missing modalities.
  The fused latent is softmax'd onto the simplex (an LDA-style topic
  proportion) and each modality is reconstructed via its own linear
  "topic x gene/peak" loading matrix ``beta``. Reimplemented here with two
  modality encoders (RNA, ATAC), the exact precision-weighted
  Product-of-Experts fusion (including a learned prior-expert placeholder),
  and per-modality linear topic-loading decoders softmax-normalized over
  the vocabulary, matching ``CoboltModel.encode``/``ProductOfExperts.forward``.

- COMEBin: Ziye Wang, Shanshan Wang, Zhenmiao Zhang, Kang Ning, Lu Zhang &
  Yang Sun, Nature Communications 2024,
  https://www.nature.com/articles/s41467-023-44290-z ;
  https://github.com/ziyewang/COMEBin (official ziyewang repo,
  ``COMEBin/models/mlp2.py``, class ``EmbeddingNet`` + ``COMEBin/simclr.py``).
  A contrastive (SimCLR/InfoNCE) multi-view metagenomic contig-binning
  encoder: k-mer-frequency features and coverage-profile features are each
  augmented (multiple views) and jointly embedded through a shared "fusion"
  MLP whose input is the *concatenation* of the raw k-mer view with an
  L2-normalized coverage-model sub-embedding (``EmbeddingNet.forward``:
  ``x = cat([x, normalize(cov_model(x2))], dim=-1)``); the fused embedding
  passes through a batch-norm+dropout MLP trunk to a final embedding used
  for InfoNCE contrastive loss against other augmented views. Reimplemented
  here with a coverage-embedding sub-MLP, L2-normalized concatenation with
  the k-mer features, and a batchnorm MLP trunk producing the final
  contrastive embedding, matching the official dual-view fusion topology.

- CONCH: Ming Y. Lu, ... , Faisal Mahmood, Nature Medicine 2024,
  https://www.nature.com/articles/s41591-024-02856-4 ;
  https://github.com/mahmoodlab/CONCH (official mahmoodlab repo,
  ``conch/open_clip_custom/coca_model.py``, class ``CoCa``). A
  vision-language foundation model for histopathology trained with a joint
  **contrastive + captioning** (CoCa) objective: a ViT vision tower produces
  patch tokens; an attentional pooler distills them into a small set of
  contrastive query tokens (for CLIP-style image/text contrastive loss) and
  a larger set of caption query tokens; a causal text transformer encodes
  the caption; a **multimodal text decoder** cross-attends from text tokens
  onto the caption query tokens to autoregressively predict the next token
  (image captioning loss), so the single vision tower is trained by both
  losses simultaneously. Reimplemented here with a compact ViT patch
  encoder, a learned attentional-pooling query set producing both
  contrastive and caption embeddings, a causal text encoder
  (``nn.TransformerEncoder`` with a causal mask), and a
  ``nn.TransformerDecoder`` cross-attending onto the caption tokens to
  produce next-token logits -- matching CoCa's dual-objective, single-tower
  topology used by the official ``CoCa.forward``.

- CryoSTAR: Yilai Li, Yi Zhou, Jing Yuan, Fei Ye & Quanquan Gu, Nature
  Methods 2024, https://www.nature.com/articles/s41592-024-02486-1 ;
  https://github.com/bytedance/cryostar (official bytedance repo,
  ``cryostar/gmm/gmm.py`` -- ``batch_projection``, ``cryostar/gmm/deformer.py``
  -- ``E3Deformer``, ``projects/star/train_atom.py`` -- ``CryoEMTask``). Models
  cryo-EM heterogeneous reconstruction as a **rigid reference structure**
  (an AlphaFold/PDB-predicted atomic model, used as fixed Gaussian-mixture
  centers) that a VAE-encoded per-particle latent code *deforms*: a
  ``VAEEncoder`` maps a 2-D particle image to ``(mu, log_var)``; the
  reparameterized latent is decoded (by an MLP "deformer head") into a
  per-atom 3-D displacement field that is added to the fixed reference
  atomic coordinates (``E3Deformer.transform``: ``shift + coords``); the
  deformed Gaussian-mixture atoms are then rotated by the particle's pose
  and projected to a 2-D image by summing per-atom anisotropic 2-D
  Gaussians (``batch_projection``). Reimplemented here with a compact MLP
  VAE encoder + per-atom displacement decoder, a fixed random "reference
  structure" buffer standing in for the AlphaFold-predicted atom centers,
  and the exact separable-Gaussian ``batch_projection`` summation used by
  the official code to rasterize the deformed, rotated atoms into an image.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# cNODE
# ---------------------------------------------------------------------------


class CNODE(nn.Module):
    """Compositional Neural ODE: replicator-equation dynamics on the simplex.

    Reimplements the official ``FitnessLayer``/``getModel`` pair: a single
    learned interaction matrix defines a fitness vector field whose
    replicator-equation dynamics keep the state normalized to sum to one
    throughout integration.
    """

    def __init__(self, n_species: int = 20, n_steps: int = 8) -> None:
        """Initialize the learned interaction matrix and integration steps.

        Parameters
        ----------
        n_species : int, default=20
            Number of microbial species (state dimensionality).
        n_steps : int, default=8
            Number of fixed-step RK4 integration steps from t=0 to t=1.
        """
        super().__init__()
        self.n_species = n_species
        self.n_steps = n_steps
        self.weight = nn.Parameter(torch.randn(n_species, n_species) * 0.05)

    def _fitness_field(self, p: Tensor) -> Tensor:
        """Compute the replicator-equation vector field ``p * (f - <p, f>)``."""
        f = p @ self.weight.T
        mean_fitness = (p * f).sum(dim=-1, keepdim=True)
        return p * (f - mean_fitness)

    def forward(self, z: Tensor) -> Tensor:
        """Integrate the compositional ODE from an initial simplex state.

        Parameters
        ----------
        z : Tensor
            Raw initial species-abundance logits of shape ``(batch, n_species)``;
            projected onto the simplex via softmax before integration.

        Returns
        -------
        Tensor
            Predicted steady-state relative abundances at t=1, shape
            ``(batch, n_species)``, summing to 1 along the last dimension.
        """
        p = F.softmax(z, dim=-1)
        h = 1.0 / self.n_steps
        for _ in range(self.n_steps):
            k1 = self._fitness_field(p)
            k2 = self._fitness_field(p + 0.5 * h * k1)
            k3 = self._fitness_field(p + 0.5 * h * k2)
            k4 = self._fitness_field(p + h * k3)
            p = p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return p


def build_cnode() -> nn.Module:
    """Build a compact compositional Neural ODE.

    Returns
    -------
    nn.Module
        Random-initialized ``CNODE`` in eval mode.
    """
    return CNODE().eval()


def example_input_cnode() -> Tensor:
    """Create example raw species-abundance logits.

    Returns
    -------
    Tensor
        Random logits of shape ``(4, 20)``.
    """
    return torch.randn(4, 20)


# ---------------------------------------------------------------------------
# Cobolt
# ---------------------------------------------------------------------------


class _ProductOfExperts(nn.Module):
    """Precision-weighted Product-of-Experts Gaussian fusion."""

    def forward(self, mu: Tensor, log_var: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse per-expert Gaussians into one joint posterior.

        Parameters
        ----------
        mu : Tensor
            Per-expert means, shape ``(n_experts, batch, latent_dim)``.
        log_var : Tensor
            Per-expert log-variances, shape ``(n_experts, batch, latent_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Fused mean and variance, each shape ``(batch, latent_dim)``.
        """
        var = torch.exp(log_var)
        precision = 1.0 / var
        pd_mu = (mu * precision).sum(dim=0) / precision.sum(dim=0)
        pd_var = 1.0 / precision.sum(dim=0)
        return pd_mu, pd_var


class Cobolt(nn.Module):
    """Compact Cobolt: dual-modality VAE fused by Product-of-Experts.

    Reimplements the official ``CoboltModel.encode``/``ProductOfExperts``
    pair: independent RNA and ATAC encoders each produce a Gaussian
    posterior; a fixed prior "expert" plus the two modality experts are
    fused by precision-weighted Product-of-Experts; the fused latent is
    softmax-normalized (LDA-style topic proportions) and linearly decoded
    per modality.
    """

    def __init__(
        self,
        n_rna: int = 40,
        n_atac: int = 60,
        latent_dim: int = 8,
        hidden: int = 32,
    ) -> None:
        """Initialize per-modality encoders/decoders and the PoE fusion.

        Parameters
        ----------
        n_rna : int, default=40
            Number of RNA features (genes).
        n_atac : int, default=60
            Number of ATAC features (peaks).
        latent_dim : int, default=8
            Dimensionality of the joint topic latent.
        hidden : int, default=32
            Hidden width of each modality encoder.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.experts = _ProductOfExperts()

        self.rna_encoder = nn.Sequential(
            nn.Linear(n_rna, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU()
        )
        self.rna_mu = nn.Linear(hidden, latent_dim)
        self.rna_logvar = nn.Linear(hidden, latent_dim)

        self.atac_encoder = nn.Sequential(
            nn.Linear(n_atac, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU()
        )
        self.atac_mu = nn.Linear(hidden, latent_dim)
        self.atac_logvar = nn.Linear(hidden, latent_dim)

        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))

        self.beta_rna = nn.Parameter(torch.randn(latent_dim, n_rna) * 0.05)
        self.beta_atac = nn.Parameter(torch.randn(latent_dim, n_atac) * 0.05)

    def forward(self, rna: Tensor, atac: Tensor) -> tuple[Tensor, Tensor]:
        """Encode both modalities, fuse via PoE, and reconstruct each modality.

        Parameters
        ----------
        rna : Tensor
            RNA count matrix of shape ``(batch, n_rna)``.
        atac : Tensor
            ATAC accessibility matrix of shape ``(batch, n_atac)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Reconstructed RNA and ATAC profiles (softmax over the vocabulary),
            shapes ``(batch, n_rna)`` and ``(batch, n_atac)``.
        """
        batch = rna.shape[0]
        rna_h = self.rna_encoder(torch.log1p(rna))
        atac_h = self.atac_encoder(torch.log1p(atac))

        prior_mu = self.prior_mu.unsqueeze(0).expand(batch, -1).unsqueeze(0)
        prior_logvar = self.prior_logvar.unsqueeze(0).expand(batch, -1).unsqueeze(0)

        mu = torch.stack([prior_mu.squeeze(0), self.rna_mu(rna_h), self.atac_mu(atac_h)], dim=0)
        log_var = torch.stack(
            [prior_logvar.squeeze(0), self.rna_logvar(rna_h), self.atac_logvar(atac_h)], dim=0
        )

        pd_mu, pd_var = self.experts(mu, log_var)
        eps = torch.randn_like(pd_var)
        z = pd_mu + pd_var.sqrt() * eps

        theta = F.softmax(z, dim=-1)
        rna_recon = F.softmax(theta @ self.beta_rna, dim=-1)
        atac_recon = F.softmax(theta @ self.beta_atac, dim=-1)
        return rna_recon, atac_recon


def build_cobolt() -> nn.Module:
    """Build a compact Cobolt multi-omic PoE-VAE.

    Returns
    -------
    nn.Module
        Random-initialized ``Cobolt`` in eval mode.
    """
    return Cobolt().eval()


def example_input_cobolt() -> tuple[Tensor, Tensor]:
    """Create example paired RNA and ATAC count matrices.

    Returns
    -------
    tuple[Tensor, Tensor]
        RNA counts of shape ``(4, 40)`` and ATAC counts of shape ``(4, 60)``.
    """
    return torch.rand(4, 40) * 5.0, torch.rand(4, 60) * 3.0


# ---------------------------------------------------------------------------
# COMEBin
# ---------------------------------------------------------------------------


class COMEBinEncoder(nn.Module):
    """Compact COMEBin dual-view (k-mer + coverage) contrastive encoder.

    Reimplements the official ``EmbeddingNet.forward`` fusion: a coverage
    sub-model embeds the coverage-profile view; its L2-normalized embedding
    is concatenated with the raw k-mer-frequency view; the fused vector is
    passed through a batchnorm MLP trunk to the final contrastive embedding.
    """

    def __init__(
        self,
        n_kmer: int = 136,
        n_coverage: int = 20,
        cov_hidden: int = 16,
        hidden: int = 64,
        out_dim: int = 32,
    ) -> None:
        """Initialize the coverage sub-model and fused MLP trunk.

        Parameters
        ----------
        n_kmer : int, default=136
            Dimensionality of the k-mer-frequency view (4-mer canonical
            counts, matching the official default ``kmer_len``).
        n_coverage : int, default=20
            Dimensionality of the coverage-profile view.
        cov_hidden : int, default=16
            Output width of the coverage sub-model embedding.
        hidden : int, default=64
            Hidden width of the fused MLP trunk.
        out_dim : int, default=32
            Dimensionality of the final contrastive embedding.
        """
        super().__init__()
        self.cov_model = nn.Sequential(
            nn.Linear(n_coverage, cov_hidden),
            nn.ReLU(),
            nn.Linear(cov_hidden, cov_hidden),
        )
        fused_dim = n_kmer + cov_hidden
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, kmer: Tensor, coverage: Tensor) -> Tensor:
        """Fuse the k-mer and coverage views into a contrastive embedding.

        Parameters
        ----------
        kmer : Tensor
            K-mer-frequency features of shape ``(batch, n_kmer)``.
        coverage : Tensor
            Coverage-profile features of shape ``(batch, n_coverage)``.

        Returns
        -------
        Tensor
            Contrastive contig embedding of shape ``(batch, out_dim)``.
        """
        cov_emb = self.cov_model(coverage)
        fused = torch.cat([kmer, F.normalize(cov_emb, dim=-1)], dim=-1)
        return self.fc(fused)


def build_comebin() -> nn.Module:
    """Build a compact COMEBin dual-view contrastive encoder.

    Returns
    -------
    nn.Module
        Random-initialized ``COMEBinEncoder`` in eval mode.
    """
    return COMEBinEncoder().eval()


def example_input_comebin() -> tuple[Tensor, Tensor]:
    """Create example k-mer and coverage feature views for a batch of contigs.

    Returns
    -------
    tuple[Tensor, Tensor]
        K-mer features of shape ``(8, 136)`` and coverage features of shape
        ``(8, 20)``.
    """
    return torch.rand(8, 136), torch.rand(8, 20)


# ---------------------------------------------------------------------------
# CONCH
# ---------------------------------------------------------------------------


class _ViTPatchEncoder(nn.Module):
    """Compact ViT patch-token encoder (CONCH's CoCa vision trunk)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        width: int = 32,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        """Initialize patch embedding, position embeddings, and transformer.

        Parameters
        ----------
        img_size : int, default=32
            Input image side length.
        patch_size : int, default=8
            Square patch side length.
        width : int, default=32
            Token embedding dimensionality.
        depth : int, default=2
            Number of pre-LN transformer encoder layers.
        heads : int, default=4
            Number of self-attention heads.
        """
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, images: Tensor) -> Tensor:
        """Embed image patches and run the transformer trunk."""
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return self.blocks(x)


class CONCHCoCa(nn.Module):
    """Compact CONCH: CoCa dual-objective vision-language model.

    Reimplements the official ``CoCa.forward`` topology: a ViT vision trunk
    produces patch tokens; a learned attentional-pooling query set distills
    contrastive and captioning embeddings from those tokens; a causal text
    transformer encodes the input caption; a multimodal decoder
    cross-attends from text tokens onto the caption query tokens to produce
    next-token logits, jointly training contrastive and captioning heads
    from one shared vision tower.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        width: int = 32,
        n_caption_queries: int = 8,
        n_contrast_queries: int = 1,
        text_len: int = 12,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        """Initialize the vision trunk, attentional pooler, and text towers.

        Parameters
        ----------
        vocab_size : int, default=256
            Text token vocabulary size.
        width : int, default=32
            Shared embedding dimensionality.
        n_caption_queries : int, default=8
            Number of attentional-pool query tokens used for captioning.
        n_contrast_queries : int, default=1
            Number of attentional-pool query tokens used for contrastive loss.
        text_len : int, default=12
            Caption sequence length.
        depth : int, default=2
            Number of transformer layers in each tower.
        heads : int, default=4
            Number of attention heads.
        """
        super().__init__()
        self.width = width
        self.vision = _ViTPatchEncoder(width=width, depth=depth, heads=heads)

        self.contrast_queries = nn.Parameter(torch.randn(1, n_contrast_queries, width) * 0.02)
        self.caption_queries = nn.Parameter(torch.randn(1, n_caption_queries, width) * 0.02)
        pool_layer = nn.TransformerDecoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4, batch_first=True
        )
        self.attn_pool = nn.TransformerDecoder(pool_layer, num_layers=1)
        self.image_contrast_proj = nn.Linear(width, width)

        self.token_embed = nn.Embedding(vocab_size, width)
        self.text_pos_embed = nn.Parameter(torch.randn(1, text_len, width) * 0.02)
        text_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4, batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=depth)
        self.text_contrast_proj = nn.Linear(width, width)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4, batch_first=True
        )
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.lm_head = nn.Linear(width, vocab_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images: Tensor, text_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the joint contrastive + captioning forward pass.

        Parameters
        ----------
        images : Tensor
            Histopathology image patches of shape ``(batch, 3, H, W)``.
        text_ids : Tensor
            Caption token ids of shape ``(batch, text_len)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Normalized image contrastive embedding ``(batch, width)``,
            normalized text contrastive embedding ``(batch, width)``, and
            next-token caption logits ``(batch, text_len, vocab_size)``.
        """
        patch_tokens = self.vision(images)
        batch = patch_tokens.shape[0]

        queries = torch.cat(
            [
                self.contrast_queries.expand(batch, -1, -1),
                self.caption_queries.expand(batch, -1, -1),
            ],
            dim=1,
        )
        pooled = self.attn_pool(queries, patch_tokens)
        image_contrast = F.normalize(self.image_contrast_proj(pooled[:, 0]), dim=-1)
        caption_tokens = pooled[:, self.contrast_queries.shape[1] :]

        text_h = self.token_embed(text_ids) + self.text_pos_embed
        causal_mask = nn.Transformer.generate_square_subsequent_mask(text_ids.shape[1]).to(
            text_ids.device
        )
        text_tokens = self.text_encoder(text_h, mask=causal_mask)
        text_contrast = F.normalize(self.text_contrast_proj(text_tokens.mean(dim=1)), dim=-1)

        decoded = self.text_decoder(text_tokens, caption_tokens, tgt_mask=causal_mask)
        logits = self.lm_head(decoded)
        return image_contrast, text_contrast, logits


def build_conch() -> nn.Module:
    """Build a compact CONCH CoCa vision-language model.

    Returns
    -------
    nn.Module
        Random-initialized ``CONCHCoCa`` in eval mode.
    """
    return CONCHCoCa().eval()


def example_input_conch() -> tuple[Tensor, Tensor]:
    """Create example histopathology image patch and caption token ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        Image batch of shape ``(2, 3, 32, 32)`` and caption ids of shape
        ``(2, 12)``.
    """
    images = torch.randn(2, 3, 32, 32)
    text_ids = torch.randint(0, 256, (2, 12))
    return images, text_ids


# ---------------------------------------------------------------------------
# CryoSTAR
# ---------------------------------------------------------------------------


class CryoSTAR(nn.Module):
    """Compact CryoSTAR: VAE-deformed rigid-structure GMM cryo-EM projector.

    Reimplements the official atom-model pipeline: a ``VAEEncoder`` maps a
    2-D particle image to a latent code; an MLP deformer head decodes that
    code into a per-atom 3-D displacement added onto a fixed reference
    structure's Gaussian-mixture atom centers (``E3Deformer.transform``);
    the deformed atoms are rotated by the particle pose and rasterized to a
    2-D image via the separable-Gaussian ``batch_projection`` used by the
    official ``cryostar.gmm.gmm`` module.
    """

    def __init__(
        self,
        img_size: int = 16,
        n_atoms: int = 24,
        latent_dim: int = 6,
        hidden: int = 32,
    ) -> None:
        """Initialize the VAE encoder, deformer head, and reference structure.

        Parameters
        ----------
        img_size : int, default=16
            Side length of the (square) particle image.
        n_atoms : int, default=24
            Number of Gaussian-mixture atom centers in the reference
            structure (stands in for an AlphaFold-predicted atomic model).
        latent_dim : int, default=6
            Dimensionality of the VAE conformational latent code.
        hidden : int, default=32
            Hidden width of the encoder/decoder MLPs.
        """
        super().__init__()
        self.img_size = img_size
        self.n_atoms = n_atoms

        self.encoder = nn.Sequential(
            nn.Linear(img_size * img_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

        self.deformer = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_atoms * 3),
        )

        ref_centers = torch.randn(n_atoms, 3) * 4.0
        self.register_buffer("ref_centers", ref_centers)
        self.register_buffer("sigma", torch.full((n_atoms,), 1.0))
        self.register_buffer("amplitude", torch.ones(n_atoms))

        line = torch.linspace(-img_size / 2, img_size / 2 - 1, img_size)
        self.register_buffer("grid_line", line)

    def forward(self, image: Tensor, rot_mat: Tensor) -> Tensor:
        """Encode a particle image, deform the reference GMM, and reproject.

        Parameters
        ----------
        image : Tensor
            Input particle image of shape ``(batch, 1, img_size, img_size)``.
        rot_mat : Tensor
            Per-particle 3x3 rotation matrix, shape ``(batch, 3, 3)``.

        Returns
        -------
        Tensor
            Re-projected 2-D image of shape ``(batch, img_size, img_size)``.
        """
        batch = image.shape[0]
        flat = image.reshape(batch, -1)
        h = self.encoder(flat)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        displacement = self.deformer(z).reshape(batch, self.n_atoms, 3)
        deformed = displacement + self.ref_centers.unsqueeze(0)

        centers = torch.einsum("bij,bnj->bni", rot_mat, deformed)

        sigmas = 2.0 * self.sigma.pow(2)
        proj_x = self.grid_line.reshape(1, -1, 1) - centers[..., 0].unsqueeze(1)
        proj_x = torch.exp(-proj_x.pow(2) / sigmas.reshape(1, 1, -1))
        proj_y = self.grid_line.reshape(1, -1, 1) - centers[..., 1].unsqueeze(1)
        proj_y = torch.exp(-proj_y.pow(2) / sigmas.reshape(1, 1, -1))

        proj = torch.einsum("n,bxn,byn->bxy", self.amplitude, proj_x, proj_y)
        return proj


def build_cryostar() -> nn.Module:
    """Build a compact CryoSTAR VAE-deformed rigid-structure projector.

    Returns
    -------
    nn.Module
        Random-initialized ``CryoSTAR`` in eval mode.
    """
    return CryoSTAR().eval()


def example_input_cryostar() -> tuple[Tensor, Tensor]:
    """Create an example cryo-EM particle image and rotation matrix.

    Returns
    -------
    tuple[Tensor, Tensor]
        Particle image of shape ``(3, 1, 16, 16)`` and per-particle
        rotation matrices of shape ``(3, 3, 3)`` (identity-initialized then
        randomly perturbed via QR orthogonalization).
    """
    image = torch.randn(3, 1, 16, 16)
    raw = torch.randn(3, 3, 3)
    q, _ = torch.linalg.qr(raw)
    return image, q


MENAGERIE_ENTRIES = [
    ("cNODE", "build_cnode", "example_input_cnode", "2021", "BIO"),
    ("Cobolt", "build_cobolt", "example_input_cobolt", "2021", "BIO"),
    ("COMEBin", "build_comebin", "example_input_comebin", "2024", "BIO"),
    ("CONCH", "build_conch", "example_input_conch", "2024", "VIS"),
    ("CryoSTAR", "build_cryostar", "example_input_cryostar", "2024", "BIO"),
]

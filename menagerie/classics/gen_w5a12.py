"""Wave w5a12: four computational-biology/NLP architecture classics.

Sources checked (repo trees + primary source files fetched via GitHub API,
paper metadata from the build queue; no clone or pip install):

- SPACE (SPACE-2, "Semantic tree-structured Pre-trAining for Conversation
  undErstanding") -- AlibabaResearch/DAMO-ConvAI, ``space-2/`` (README.md,
  ``space/models``/``space/modules``), Wang et al., "SPACE-2: Tree-Structured
  Semi-Supervised Contrastive Pre-training for Task-Oriented Dialog
  Understanding", COLING 2022 / arXiv:2209.06638. A BERT-family transformer
  encoder (here a small from-scratch encoder standing in for the pretrained
  12-layer checkpoint) turns a multi-turn dialog token sequence into a pooled
  [CLS] representation; each dialog is additionally scored against a semantic
  tree structure (a small set of intent/slot schema nodes) via a learned
  multi-view score function that compares the pooled representation against
  every tree-node embedding at multiple projection "views" and combines the
  per-view relevance scores into one similarity value used to pull
  same-subtree dialogs together and push different-subtree dialogs apart in
  a semi-supervised contrastive objective -- the shared-encoder +
  multi-view-tree-score contrastive head is the distinctive mechanism.
- SPACEL -- QuKunLab/SPACEL, ``SPACEL/Spoint/base_model.py``
  (``PredictionModel``) and ``SPACEL/Splane/base_model.py``
  (``Splane_GCN``/``Splane_Disc``), Xu et al., "SPACEL: deep learning-based
  characterization of spatial transcriptome architectures", Nature
  Communications 2023. Two coupled sub-nets reproduced compactly: (1) Spoint,
  a spot-deconvolution encoder/decoder/predictor -- an MLP encoder compresses
  a spot's gene-expression vector to a latent code, a predictor head turns
  that code into a softmax cell-type-proportion simplex, and a decoder
  reconstructs the expression vector from the predicted proportions (a
  proportion-conditioned autoencoder, standing in for the paper's scVI/ZINB
  preprocessing + simplex-decoder loop); (2) Splane, a Chebyshev-basis graph
  convolutional autoencoder over the spot-adjacency graph whose encoded
  latent domain code is both L2-normalized (min-max scale then unit-norm,
  ``l2_activate`` in the source) and fed to an adversarial slice-of-origin
  discriminator that is trained to defeat batch effects -- the coupled
  GCN-autoencoder-plus-adversarial-discriminator for batch-invariant spatial
  domain identification is the distinctive mechanism.
- SpatialScope -- YangLabHKUST/SpatialScope, ``src/SCGrad/nn.py``
  (``SCGradNN``) and ``src/SCGrad/linear_modulation.py``
  (``FeatureWiseLinearModulation``/``PositionalEncoding``), Zhou et al.,
  "SpatialScope: a unified approach for integrating spatial and
  single-cell transcriptomics data using deep generative models", Nature
  Communications 2023. A WaveGrad-derived 1D-convolutional diffusion
  denoiser adapted from speech synthesis to per-spot gene-expression
  vectors: a two-stream dual U-Net (a downsampling "conditioner" stream
  reading the conditioning mean-expression signal ``mu`` and an
  upsampling "noise" stream reading the noisy expression vector) is
  coupled at every scale by Feature-wise Linear Modulation (FiLM) blocks
  whose scale/shift statistics are additionally conditioned on a
  sinusoidal encoding of the continuous diffusion noise level -- the
  noise-level-conditioned dual-stream FiLM U-Net denoiser (score network
  of a continuous-noise diffusion model) applied to gene-expression
  imputation is the distinctive mechanism.
- SPIDER3 -- yuedongyang/SPIDER2 (README.md: "iterative deep learning" for
  secondary structure / backbone torsion angles / solvent accessible surface
  area), Heffernan et al., "Capturing non-local interactions by long
  short-term memory bidirectional recurrent neural networks for improving
  prediction of protein secondary structure, backbone angles, contact
  numbers and solvent accessibility", Bioinformatics 2017 (SPIDER3;
  bioRxiv:2017.02.16.403048 preprint). A stack of bidirectional LSTM layers
  reads per-residue sequence-profile features (PSSM/HMM-style evolutionary
  features standing in here for the real profile pipeline) and produces
  five per-residue structural predictions (3-state secondary structure,
  solvent accessible surface area, and three real-valued backbone torsion
  angles via sin/cos pairs); the predicted outputs of one BiLSTM stack are
  concatenated onto the original input features and fed through a second
  independent BiLSTM stack for a further refinement pass, i.e. the model's
  own structural predictions become extra input features for an iterative
  re-prediction round -- the input-feature-plus-own-output iterative BiLSTM
  refinement loop is the distinctive mechanism (replacing SPIDER2's
  feed-forward iterative scheme with BiLSTMs was SPIDER3's contribution).

SolubleMPNN (cand_00624) is skipped: it shares the exact ProteinMPNN graph
message-passing/autoregressive-decoder architecture already reimplemented in
``menagerie/classics/proteinmpnn.py`` (dauparas/ProteinMPNN,
``soluble_model_weights/``; identical network, only the training data/weights
differ), so it is already represented in the catalog.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# SPACE (SPACE-2): tree-structured semi-supervised contrastive dialog encoder
# ---------------------------------------------------------------------------


class _TinyTransformerEncoder(nn.Module):
    """Small from-scratch transformer encoder standing in for the BERT backbone."""

    def __init__(self, vocab: int, d_model: int, n_heads: int, n_layers: int, max_len: int) -> None:
        """Build token/position embeddings and a stack of encoder layers.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        d_model:
            Hidden width.
        n_heads:
            Number of self-attention heads per layer.
        n_layers:
            Number of stacked encoder layers.
        max_len:
            Maximum supported sequence length (position embedding table size).
        """

        super().__init__()
        self.tok_embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: Tensor) -> Tensor:
        """Encode token ids into contextual hidden states ``(B, L, D)``."""

        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        h = self.tok_embed(ids) + self.pos_embed(pos)
        return self.norm(self.encoder(h))


class SpaceTreeContrastiveEncoder(nn.Module):
    """SPACE-2 dialog encoder with a tree-structured multi-view score head.

    A BERT-family encoder pools a dialog utterance sequence to a [CLS]
    vector; a multi-view score function then compares that vector against a
    fixed bank of semantic-tree-structure (STS) node embeddings under
    several learned linear "views" and combines the per-view scores into one
    tree-relevance vector used by the semi-supervised contrastive loss.
    """

    def __init__(
        self,
        vocab: int = 64,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 24,
        n_tree_nodes: int = 6,
        n_views: int = 3,
    ) -> None:
        """Initialize the encoder backbone and multi-view tree-score head.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        d_model:
            Hidden width of the transformer backbone.
        n_heads:
            Self-attention heads per encoder layer.
        n_layers:
            Number of transformer encoder layers.
        max_len:
            Maximum dialog token length.
        n_tree_nodes:
            Number of semantic-tree-structure node embeddings.
        n_views:
            Number of multi-view score projections.
        """

        super().__init__()
        self.backbone = _TinyTransformerEncoder(vocab, d_model, n_heads, n_layers, max_len)
        self.pooler = nn.Linear(d_model, d_model)
        self.tree_nodes = nn.Parameter(torch.randn(n_tree_nodes, d_model) * 0.02)
        self.view_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_views)])
        self.view_combine = nn.Linear(n_views, 1)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode dialogs and score them against the semantic tree structure.

        Parameters
        ----------
        ids:
            Token ids ``(B, L)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Pooled [CLS] representation ``(B, D)`` and multi-view tree
            relevance scores ``(B, n_tree_nodes)``.
        """

        hidden = self.backbone(ids)
        pooled = torch.tanh(self.pooler(hidden[:, 0]))
        view_scores = []
        for proj in self.view_proj:
            q = proj(pooled)
            view_scores.append(q @ self.tree_nodes.t() / math.sqrt(q.shape[-1]))
        stacked = torch.stack(view_scores, dim=-1)
        tree_scores = self.view_combine(stacked).squeeze(-1)
        return pooled, tree_scores


def build_space() -> nn.Module:
    """Construct a small SPACE-2 tree-structured contrastive dialog encoder."""

    return SpaceTreeContrastiveEncoder().eval()


def example_input_space() -> Tensor:
    """Return an example dialog token-id batch for :func:`build_space`."""

    return torch.randint(0, 64, (2, 24))


# ---------------------------------------------------------------------------
# SPACEL: Spoint deconvolution autoencoder + Splane GCN adversarial autoenc.
# ---------------------------------------------------------------------------


class SpointDeconvolution(nn.Module):
    """Spoint spot-deconvolution encoder/predictor/decoder (SPACEL sub-net).

    A gene-expression spot vector is encoded to a latent code, decoded
    forward into a softmax cell-type-proportion simplex by the predictor
    head, and the predicted proportions are in turn decoded back into a
    reconstructed expression vector -- matching ``PredictionModel`` in
    ``SPACEL/Spoint/base_model.py``.
    """

    def __init__(
        self, n_genes: int = 48, n_celltypes: int = 6, latent: int = 12, hidden: int = 32
    ) -> None:
        """Build the encoder, predictor, and decoder MLP stacks.

        Parameters
        ----------
        n_genes:
            Number of input/reconstructed gene-expression features.
        n_celltypes:
            Number of cell types in the predicted proportion simplex.
        latent:
            Latent code width.
        hidden:
            Hidden layer width shared by all three sub-nets.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, latent),
        )
        self.pred = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_celltypes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_celltypes, hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, predict cell-type proportions, and decode a reconstruction.

        Parameters
        ----------
        x:
            Spot gene-expression vectors ``(B, n_genes)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Latent code, softmax cell-type proportions, and the
            proportion-conditioned reconstruction.
        """

        z = self.encoder(x)
        proportions = F.softmax(self.pred(z), dim=-1)
        decoded = self.decoder(proportions)
        return z, proportions, decoded


class _ChebyGraphConv(nn.Module):
    """Chebyshev-basis graph convolution (mirrors Splane's ``GraphConvolution``)."""

    def __init__(self, in_features: int, out_features: int, order: int) -> None:
        """Initialize the stacked-basis weight matrix.

        Parameters
        ----------
        in_features:
            Input feature width.
        out_features:
            Output feature width.
        order:
            Number of Chebyshev polynomial bases supplied at forward time.
        """

        super().__init__()
        self.order = order
        self.weight = nn.Parameter(torch.empty(in_features * order, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, bases: Tensor) -> Tensor:
        """Apply each Chebyshev basis matrix to ``x`` then project the concatenation.

        Parameters
        ----------
        x:
            Node features ``(N, in_features)``.
        bases:
            Stacked basis (adjacency power) matrices ``(order, N, N)``.

        Returns
        -------
        Tensor
            Output node features ``(N, out_features)``.
        """

        supports = torch.cat([bases[i] @ x for i in range(self.order)], dim=-1)
        return supports @ self.weight


class SplaneDomainGCN(nn.Module):
    """Splane GCN autoencoder with an adversarial slice-of-origin discriminator.

    Mirrors ``Splane_GCN`` + ``Splane_Disc`` in ``SPACEL/Splane/base_model.py``:
    a 2-layer Chebyshev graph-convolutional encoder maps per-spot cell-type
    proportions plus their spatial neighborhood graph to a latent spatial
    domain code (min-max scaled then L2-normalized, as in ``l2_activate``), a
    matching 2-layer graph decoder reconstructs the input, and an adversarial
    MLP discriminator tries to predict which tissue slice a spot's latent
    code came from -- the discriminator gradient is used (outside this
    forward path, at training time) to make the latent domain code
    slice-invariant.
    """

    def __init__(
        self,
        feature_dims: int = 6,
        order: int = 2,
        latent: int = 8,
        hidden: int = 32,
        n_slices: int = 3,
    ) -> None:
        """Build the graph encoder/decoder and the slice discriminator.

        Parameters
        ----------
        feature_dims:
            Per-spot input feature width (cell-type proportions).
        order:
            Chebyshev basis order shared by every graph convolution.
        latent:
            Latent spatial-domain code width.
        hidden:
            Hidden width of the graph layers and discriminator.
        n_slices:
            Number of tissue slices the discriminator classifies.
        """

        super().__init__()
        self.encode_gc1 = _ChebyGraphConv(feature_dims, hidden, order)
        self.encode_gc2 = _ChebyGraphConv(hidden, latent, order)
        self.decode_gc1 = _ChebyGraphConv(latent, hidden, order)
        self.decode_gc2 = _ChebyGraphConv(hidden, feature_dims, order)
        self.disc = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_slices),
        )

    @staticmethod
    def _l2_activate(z: Tensor) -> Tensor:
        """Min-max scale each row to [0, 1] then L2-normalize (``l2_activate``)."""

        zmax = z.max(1, keepdim=True).values
        zmin = z.min(1, keepdim=True).values
        z_std = torch.nan_to_num((z - zmin) / (zmax - zmin), nan=0.0)
        return F.normalize(z_std, p=2, dim=1)

    def forward(self, x: Tensor, bases: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a spot graph, decode a reconstruction, and score slice origin.

        Parameters
        ----------
        x:
            Per-spot cell-type proportion features ``(N, feature_dims)``.
        bases:
            Chebyshev basis matrices ``(order, N, N)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Latent spatial-domain code, reconstructed features, and the
            adversarial slice-classification softmax.
        """

        h = F.leaky_relu(self.encode_gc1(x, bases))
        z = self._l2_activate(self.encode_gc2(h, bases))
        d = F.leaky_relu(self.decode_gc1(z, bases))
        recon = self.decode_gc2(d, bases)
        slice_logits = self.disc(z)
        return z, recon, F.softmax(slice_logits, dim=-1)


class SpacelDeconvDomain(nn.Module):
    """Joint SPACEL forward path: Spoint deconvolution feeding Splane domain ID.

    Chains the two coupled sub-nets so a single trace exercises both the
    proportion-conditioned deconvolution autoencoder and the graph
    autoencoder with adversarial discriminator, mirroring how SPACEL runs
    Spoint's cell-type proportions as the Splane graph node features.
    """

    def __init__(self, n_genes: int = 48, n_celltypes: int = 6, n_spots: int = 10) -> None:
        """Build the Spoint and Splane sub-nets and a fixed spot adjacency.

        Parameters
        ----------
        n_genes:
            Number of gene-expression features per spot.
        n_celltypes:
            Number of deconvolved cell types.
        n_spots:
            Number of spots in the toy spatial graph.
        """

        super().__init__()
        self.n_spots = n_spots
        self.spoint = SpointDeconvolution(n_genes=n_genes, n_celltypes=n_celltypes)
        self.splane = SplaneDomainGCN(feature_dims=n_celltypes, order=2)
        ring = torch.eye(n_spots)
        ring += ring.roll(1, dims=0) + ring.roll(-1, dims=0)
        deg = ring.sum(-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adj_norm", ring / deg)

    def forward(self, expr: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Deconvolve spots then identify batch-invariant spatial domains.

        Parameters
        ----------
        expr:
            Per-spot gene-expression matrix ``(n_spots, n_genes)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Reconstructed expression, cell-type proportions, spatial-domain
            latent code, and adversarial slice-classification softmax.
        """

        _, proportions, decoded = self.spoint(expr)
        order = self.splane.encode_gc1.order
        bases = torch.stack([torch.matrix_power(self.adj_norm, i + 1) for i in range(order)], dim=0)
        domain_z, recon, slice_probs = self.splane(proportions, bases)
        return decoded, proportions, domain_z, slice_probs


def build_spacel() -> nn.Module:
    """Construct the joint SPACEL Spoint-deconvolution + Splane-domain-ID module."""

    return SpacelDeconvDomain().eval()


def example_input_spacel() -> Tensor:
    """Return an example multi-spot gene-expression matrix for :func:`build_spacel`."""

    return torch.rand(10, 48)


# ---------------------------------------------------------------------------
# SpatialScope: WaveGrad-style dual-stream FiLM diffusion denoiser
# ---------------------------------------------------------------------------


class _NoiseLevelEncoding(nn.Module):
    """Sinusoidal encoding of a continuous diffusion noise level."""

    def __init__(self, channels: int) -> None:
        """Store the target embedding width.

        Parameters
        ----------
        channels:
            Output embedding width (even).
        """

        super().__init__()
        self.channels = channels

    def forward(self, noise_level: Tensor) -> Tensor:
        """Encode a per-sample scalar noise level into ``(B, channels)``."""

        half = self.channels // 2
        exponents = torch.arange(half, device=noise_level.device, dtype=torch.float32) / half
        exponents = (1e-4**exponents).unsqueeze(0)
        args = 5000.0 * noise_level.unsqueeze(-1) * exponents
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _FiLMBlock(nn.Module):
    """Feature-wise linear modulation conditioned on signal + noise level."""

    def __init__(self, channels: int) -> None:
        """Build the signal conv and the scale/shift projections.

        Parameters
        ----------
        channels:
            Channel width of the conditioning signal and outputs.
        """

        super().__init__()
        self.signal_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.noise_encoding = _NoiseLevelEncoding(channels)
        self.scale_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.shift_conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, noise_level: Tensor) -> tuple[Tensor, Tensor]:
        """Compute FiLM scale/shift statistics from ``x`` and the noise level.

        Parameters
        ----------
        x:
            Conditioning-stream features ``(B, C, L)``.
        noise_level:
            Per-sample scalar diffusion noise level ``(B,)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Scale and shift tensors, each ``(B, C, L)``.
        """

        h = F.leaky_relu(self.signal_conv(x), 0.2)
        pos = self.noise_encoding(noise_level).unsqueeze(-1)
        h = h + pos
        return self.scale_conv(h), self.shift_conv(h)


class SpatialScopeDiffusionDenoiser(nn.Module):
    """SpatialScope's WaveGrad-derived score network over gene-expression vectors.

    A conditioning stream (1D convs over the mean expression signal ``mu``)
    and a noise stream (1D convs over the noisy expression vector) run in
    lockstep; at every layer the conditioning stream's activations produce
    FiLM scale/shift statistics -- additionally modulated by a sinusoidal
    encoding of the continuous diffusion noise level -- which linearly
    modulate the noise stream. Mirrors ``SCGradNN`` in
    ``src/SCGrad/nn.py``, standing in for the paper's mel-spectrogram
    WaveGrad denoiser adapted to per-gene expression sequences.
    """

    def __init__(self, n_genes: int = 64, channels: int = 16, n_layers: int = 3) -> None:
        """Build the dual-stream conv/FiLM denoiser.

        Parameters
        ----------
        n_genes:
            Length of the (flattened, 1D) gene-expression sequence.
        channels:
            Hidden channel width of both streams.
        n_layers:
            Number of paired conditioning/noise conv-FiLM stages.
        """

        super().__init__()
        self.cond_in = nn.Conv1d(1, channels, 5, padding=2)
        self.noise_in = nn.Conv1d(1, channels, 3, padding=1)
        self.cond_layers = nn.ModuleList(
            [nn.Conv1d(channels, channels, 3, padding=1) for _ in range(n_layers)]
        )
        self.films = nn.ModuleList([_FiLMBlock(channels) for _ in range(n_layers)])
        self.noise_layers = nn.ModuleList(
            [nn.Conv1d(channels, channels, 3, padding=1) for _ in range(n_layers)]
        )
        self.out = nn.Conv1d(channels, 1, 3, padding=1)

    def forward(self, mu: Tensor, x_noisy: Tensor, noise_level: Tensor) -> Tensor:
        """Predict the diffusion noise residual for ``x_noisy`` given ``mu``.

        Parameters
        ----------
        mu:
            Conditioning mean gene-expression signal ``(B, n_genes)``.
        x_noisy:
            Noised gene-expression signal ``(B, n_genes)``.
        noise_level:
            Per-sample scalar diffusion noise level ``(B,)``.

        Returns
        -------
        Tensor
            Predicted noise residual ``(B, n_genes)``.
        """

        cond = F.leaky_relu(self.cond_in(mu.unsqueeze(1)), 0.2)
        noise = F.leaky_relu(self.noise_in(x_noisy.unsqueeze(1)), 0.2)
        for cond_conv, film, noise_conv in zip(self.cond_layers, self.films, self.noise_layers):
            cond = F.leaky_relu(cond_conv(cond), 0.2)
            scale, shift = film(cond, noise_level)
            noise = F.leaky_relu(noise_conv(noise), 0.2) * (scale + 1.0) + shift
        return self.out(noise).squeeze(1)


def build_spatialscope() -> nn.Module:
    """Construct the SpatialScope dual-stream FiLM diffusion denoiser."""

    return SpatialScopeDiffusionDenoiser().eval()


def example_input_spatialscope() -> tuple[Tensor, Tensor, Tensor]:
    """Return example (mu, noisy expression, noise level) inputs.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Conditioning mean signal ``(B, n_genes)``, noisy signal
        ``(B, n_genes)``, and scalar per-sample noise level ``(B,)``.
    """

    mu = torch.rand(2, 64)
    x_noisy = mu + 0.1 * torch.randn(2, 64)
    noise_level = torch.rand(2)
    return mu, x_noisy, noise_level


# ---------------------------------------------------------------------------
# SPIDER3: iterative BiLSTM protein structural-property predictor
# ---------------------------------------------------------------------------


class _Spider3Stack(nn.Module):
    """One SPIDER3 BiLSTM stack producing five per-residue structural heads."""

    def __init__(self, in_dims: int, hidden: int, n_ss: int = 3) -> None:
        """Build the BiLSTM trunk and the per-residue prediction heads.

        Parameters
        ----------
        in_dims:
            Per-residue input feature width.
        hidden:
            BiLSTM hidden width (per direction).
        n_ss:
            Number of secondary-structure classes (3-state by default).
        """

        super().__init__()
        self.lstm = nn.LSTM(in_dims, hidden, num_layers=2, batch_first=True, bidirectional=True)
        out_dim = hidden * 2
        self.ss_head = nn.Linear(out_dim, n_ss)
        self.asa_head = nn.Linear(out_dim, 1)
        self.angle_head = nn.Linear(out_dim, 6)  # 3 angles as (sin, cos) pairs

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the BiLSTM trunk and emit secondary-structure/ASA/angle predictions.

        Parameters
        ----------
        x:
            Per-residue input features ``(B, L, in_dims)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            BiLSTM hidden states ``(B, L, 2*hidden)``, secondary-structure
            logits ``(B, L, n_ss)``, solvent-accessible surface area
            ``(B, L, 1)``, and normalized sin/cos torsion-angle pairs
            ``(B, L, 6)``.
        """

        h, _ = self.lstm(x)
        ss = self.ss_head(h)
        asa = torch.sigmoid(self.asa_head(h))
        raw_angles = self.angle_head(h)
        pairs = raw_angles.view(*raw_angles.shape[:-1], 3, 2)
        angles = F.normalize(pairs, p=2, dim=-1).reshape(*raw_angles.shape[:-1], 6)
        return h, ss, asa, angles


class Spider3IterativeBiLSTM(nn.Module):
    """SPIDER3 iterative BiLSTM protein secondary-structure/torsion predictor.

    A first BiLSTM stack predicts 3-state secondary structure, solvent
    accessible surface area, and backbone torsion angles (as sin/cos pairs)
    from per-residue evolutionary-profile features; those five predictions
    are concatenated back onto the original input features and fed through
    a second, independent BiLSTM stack for one refinement iteration --
    reproducing SPIDER3's "iterative deep learning" scheme (the successor
    to SPIDER2's feed-forward iterative predictor, replacing it with
    BiLSTMs to capture long-range non-local interactions).
    """

    def __init__(self, in_dims: int = 20, hidden: int = 16, n_ss: int = 3) -> None:
        """Build the two iterative BiLSTM stacks.

        Parameters
        ----------
        in_dims:
            Per-residue raw evolutionary-profile feature width.
        hidden:
            BiLSTM hidden width (per direction) for both stacks.
        n_ss:
            Number of secondary-structure classes.
        """

        super().__init__()
        self.stack1 = _Spider3Stack(in_dims, hidden, n_ss)
        refine_dims = in_dims + n_ss + 1 + 6
        self.stack2 = _Spider3Stack(refine_dims, hidden, n_ss)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict structural properties, then refine via one iterative pass.

        Parameters
        ----------
        features:
            Per-residue evolutionary-profile features ``(B, L, in_dims)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Refined secondary-structure logits, ASA, and torsion-angle
            sin/cos pairs, each with the same leading ``(B, L, ...)`` shape
            as the first-pass heads.
        """

        _, ss1, asa1, ang1 = self.stack1(features)
        refined_input = torch.cat([features, ss1, asa1, ang1], dim=-1)
        _, ss2, asa2, ang2 = self.stack2(refined_input)
        return ss2, asa2, ang2


def build_spider3() -> nn.Module:
    """Construct the SPIDER3 iterative BiLSTM structural-property predictor."""

    return Spider3IterativeBiLSTM().eval()


def example_input_spider3() -> Tensor:
    """Return an example per-residue evolutionary-profile feature batch."""

    return torch.randn(2, 30, 20)


MENAGERIE_ENTRIES = [
    ("SPACE-2", "build_space", "example_input_space", "2022", "NLP"),
    ("SPACEL", "build_spacel", "example_input_spacel", "2023", "BIO"),
    ("SpatialScope", "build_spatialscope", "example_input_spatialscope", "2023", "BIO"),
    ("SPIDER3", "build_spider3", "example_input_spider3", "2017", "BIO"),
]

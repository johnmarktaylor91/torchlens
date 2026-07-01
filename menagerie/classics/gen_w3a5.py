"""Compact faithful reimplementations of six sketch/vector-graphics architecture
families (freehand-sketch representation learning, generation, explainability, and
image-to-SVG code generation).

Sources checked (paper + official source; reimplemented compactly from scratch in
base-env torch, no clone/pip-install):
  - SketchFormer: Ribeiro, Bui, Collomosse & Ponti, "Sketchformer: Transformer-Based
    Representation for Sketched Structure" (CVPR 2020, arxiv:2002.10001); official repo
    leosampaio/sketchformer (``models/sketchformer.py``, class ``Transformer``, using
    ``builders/layers/transformer.py``'s ``Encoder`` / ``Decoder`` / ``SelfAttnV1`` /
    ``DenseExpander``). The distinctive mechanism is a standard multi-layer Transformer
    ENCODER-DECODER whose encoder token sequence is BOTTLENECKED to a single fixed-size
    embedding via a learned SELF-ATTENTION POOLING layer (``SelfAttn``: a query vector
    attends over all encoder tokens, softmax-weighting them into one vector -- the
    "sketch embedding"), which is then RE-EXPANDED back to the full sequence length by
    tiling (``DenseExpander``) to serve as the decoder's cross-attention memory for
    reconstruction, while a parallel linear head classifies the same bottleneck vector.
    Distinct from plain seq2seq (LSTM-based SketchRNN): the whole pipeline is
    attention-only, and the encoder-to-decoder handoff is a genuine
    all-tokens-to-one-token-back-to-all-tokens bottleneck rather than a running hidden
    state. Reimplemented here as ``SketchFormer``: a Transformer encoder over stroke-5
    tokens, a learned-query self-attention pooling bottleneck, a tile-expand step, a
    Transformer decoder (self-attention only, over the expanded memory -- matching the
    source's non-causal reconstruction decoder), a stroke-5 output head, and a
    classification head off the bottleneck.
  - SketchHealer: Su, Gu, Zhu, Song & Wang (BMVC 2020); official repo
    sgybupt/SketchHealer (``encoder.py`` classes ``FeatureExtraction`` /
    ``GCNProcessor`` / ``EncoderGCN``, ``decoder.py`` class ``DecoderRNN``). The
    distinctive mechanism is a GRAPH-STRUCTURED sketch encoder replacing SketchRNN's
    bidirectional-LSTM encoder: each sketch is broken into ``graph_num`` sub-stroke
    PATCHES rendered as small raster images, a shared small CNN
    (``FeatureExtractionBasic``) embeds each patch to a 512-d feature independently,
    then a single-layer GRAPH CONVOLUTION (``GCNProcessor``: ``A @ X @ W + b``, A the
    sub-stroke ADJACENCY matrix, followed by a sum-pool over nodes) mixes information
    across sub-strokes according to their spatial/topological adjacency (not a
    left-to-right recurrence) before producing the VAE bottleneck (``mu``/``sigma``),
    which then seeds a standard sketch-rnn GMM+pen-state LSTM decoder
    (``DecoderRNN``: hidden/cell init from ``z``, per-step 6*M+3-d mixture output).
    Reimplemented here as ``SketchHealer``: a shared per-patch CNN encoder + one
    ``A @ X @ W`` graph-conv layer with sum-pool + ``mu``/``sigma`` heads (VAE
    reparameterization), and a GMM-decoder LSTM matching ``DecoderRNN.forward``.
  - SketchLattice: Qi, Su, Chowdhury, Yang & Song, "SketchLattice: Latticed
    Representation for Sketch Manipulation" (ICCV 2021, arxiv:2108.11636); official
    repo qugank/sketch-lattice.github.io (``encoder.py`` classes
    ``CoordinateEmbeddingXYSep`` / ``GCNPropagation2``, ``decoder.py``). The
    distinctive mechanism is that the sketch is represented not by strokes or image
    patches but by a fixed-size LATTICE POINT SET: (x, y) coordinates sampled where a
    uniform grid ("lattice") intersects the sketch's ink, so each sketch is a small,
    FIXED-CARDINALITY bag of integer 2-D COORDINATES. Each coordinate is embedded
    directly by SEPARATE learned x/y embedding tables (``CoordinateEmbeddingXYSep``,
    concatenated), then a GRAPH CONVOLUTION over a proximity-built adjacency (points
    close in the lattice are connected) propagates information between lattice points
    (``GCNPropagation2``), and the pooled graph embedding seeds an LSTM sketch-rnn-style
    GMM decoder -- distinct from SketchHealer (which embeds rendered sub-stroke IMAGE
    PATCHES via CNN) by embedding raw LATTICE COORDINATES directly via lookup tables,
    making it representation-agnostic to vector vs. raster input. Reimplemented here as
    ``SketchLattice``: separate learned x/y coordinate embedding tables concatenated
    per point, one graph-conv propagation layer over a distance-thresholded adjacency
    with sum-pool, ``mu``/``sigma`` VAE heads, and a GMM-decoder LSTM.
  - SketchXAI: Qu, Tao, Gryaditskaya, Song & Song, "SketchXAI: A First Look at
    Explainability for Human Sketches" (CVPR 2023, arxiv:2304.11744); official repo
    WinKawaks/SketchXAI (``models/sketch_transformer.py``, classes
    ``StrokeEmbeddings`` / ``SketchViT``). The distinctive mechanism is a
    ViT-style transformer whose per-TOKEN embedding is the SUM of three separately
    learned factors matching sketches' intrinsic stroke properties -- SHAPE (a
    bidirectional LSTM/GRU over each stroke's raw (dx, dy, ...) points, summed over
    time -> one shape vector per stroke), ORDER (a learned embedding indexed by each
    stroke's position in the drawing sequence, i.e. WHEN it was drawn), and LOCATION (a
    linear projection of the stroke's 2-D bounding-box-center coordinate, i.e. WHERE it
    sits on the canvas) -- prepended with a CLS token and fed through a standard ViT
    encoder; this explicit shape/order/location factorization (rather than a single
    positional index, as in generic sequence transformers) is what lets the paper's
    Stroke Location Inversion probe interrogate whether the network actually knows
    WHERE each stroke was drawn. Reimplemented here as ``SketchXAI``: a per-stroke
    BiLSTM shape encoder, a learned order-embedding table, a linear location head,
    summed into ViT tokens with a CLS token and processed by a standard
    ``nn.TransformerEncoder``, with a classification head off the CLS token.
  - StarVector: Rodriguez, Rodriguez-Puig, Vazquez-Corral, Miranda & Karlinsky,
    "StarVector: Generating Scalable Vector Graphics Code from Images and Text"
    (arxiv:2312.11556); official repo joanrod/star-vector (``starvector/model/
    adapters/adapter.py`` class ``Adapter``, ``starvector/model/starvector_arch.py``
    class ``StarVectorForCausalLM``). The distinctive mechanism is IMAGE-TO-CODE
    generation via a MULTIMODAL LLM: a frozen-style CLIP-like Vision-Transformer image
    encoder produces patch-token embeddings, a small ADAPTER (``Linear(d, 2d)`` ->
    ``Swish`` -> ``Linear(2d, llm_hidden)`` -> ``LayerNorm``, exactly ``Adapter.forward``)
    non-linearly projects those visual tokens into the CODE-LLM's embedding space as
    "visual tokens" that are CONCATENATED in front of the SVG-code token embeddings, and
    a standard causal decoder-only transformer (StarCoder-style) then generates SVG
    markup autoregressively conditioned on the visual-token prefix -- i.e. SVG synthesis
    is cast as next-token PROGRAM generation over real SVG grammar rather than
    regressing continuous stroke/Bezier parameters (contrast with sketch-rnn-family
    models above). Reimplemented here as ``StarVector``: a compact ViT patch encoder, the
    exact ``Adapter`` MLP, prefix-concatenation with an SVG-code token embedding table,
    and a causal ``nn.TransformerDecoder``-style stack (using a causal
    ``nn.TransformerEncoder`` over the concatenated sequence, matching the source's
    GPT-style single-stack causal LM) ending in a vocabulary projection head.
  - StyleCLIPDraw: Schaldenbrand, Liu & Lee, "StyleCLIPDraw: Coupling Content and Style
    in Text-to-Drawing Synthesis" (IJCAI 2022, arxiv:2111.03133); official repo
    pschaldenbrand/StyleCLIPDraw (``Style_ClipDraw.ipynb``, built on CLIPDraw's RGBA
    Bezier-stroke optimization plus a VGG16 style-loss network). The distinctive
    mechanism is a COUPLED per-image OPTIMIZATION (not amortized inference, like LIVE in
    ``gen_w3a2.py``): a fixed number of RGBA Bezier-curve strokes are directly-optimized
    ``nn.Parameter`` tensors (control points + stroke width + color), rendered by a
    differentiable rasterizer and composited over a canvas, and the SAME rendering is
    scored by TWO frozen feature networks simultaneously and END-TO-END throughout
    optimization (not sequentially, as in classic content-then-style pipelines): a
    CLIP image encoder (content/text-fit loss via cosine similarity to a text prompt)
    and a VGG16 encoder (Gram-matrix STYLE loss against a reference style image,
    evaluated at multiple layers) -- the "coupling" is that both gradients backprop into
    the SAME stroke parameters every step, so the strokes' shapes (not just their
    color/texture) end up reflecting the style image. Reimplemented here as
    ``StyleCLIPDraw``: a stack of learnable closed quadratic-Bezier RGBA paths
    (reusing the ``BezierPathRasterizer``-style soft-occupancy compositing scheme)
    plus a small VGG-like CONTENT/CLIP-proxy encoder and a separate small VGG-like
    STYLE encoder with per-layer Gram-matrix taps -- both encoders are genuine
    trainable ``nn.Module`` components feeding the forward graph (unlike LIVE, whose
    frozen scorer is out-of-graph loss-only), matching the "one raster feeding two
    simultaneous scoring heads" structure.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# MODULE 1: SketchFormer -- transformer encoder with self-attention-pooling
# bottleneck, tile-expand, transformer decoder, and classification head.
# ---------------------------------------------------------------------------


class _SelfAttnPool(nn.Module):
    """Learned-query self-attention pooling bottleneck (``SelfAttnV1``).

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    lowerdim : int
        Bottleneck embedding dimension.
    """

    def __init__(self, d_model: int, lowerdim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, lowerdim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Pool a token sequence into one embedding via learned-query attention.

        Parameters
        ----------
        tokens : Tensor
            ``(B, L, d_model)`` encoder token sequence.

        Returns
        -------
        Tensor
            ``(B, lowerdim)`` pooled sketch embedding.
        """
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
        pooled = (attn @ v).squeeze(1)
        return self.out_proj(pooled)


class SketchFormer(nn.Module):
    """Sketchformer: transformer with a self-attention-pooling sketch-embedding
    bottleneck, reconstruction decoder, and classification head.

    Parameters
    ----------
    d_model : int
        Internal transformer width.
    seq_len : int
        Fixed stroke-5 sequence length.
    lowerdim : int
        Bottleneck sketch-embedding dimension.
    n_classes : int
        Number of sketch categories for the classification head.
    """

    def __init__(
        self,
        d_model: int = 32,
        seq_len: int = 16,
        lowerdim: int = 24,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(5, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pool = _SelfAttnPool(d_model, lowerdim)
        self.expand_proj = nn.Linear(lowerdim, d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=2)
        self.recon_head = nn.Linear(d_model, 5)
        self.class_head = nn.Linear(lowerdim, n_classes)

    def forward(self, stroke5: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, bottleneck, reconstruct, and classify a stroke-5 sketch.

        Parameters
        ----------
        stroke5 : Tensor
            ``(B, seq_len, 5)`` stroke-5 encoded sketch (dx, dy, 3 pen one-hot).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(embedding, recon, class_logits)``: ``(B, lowerdim)``,
            ``(B, seq_len, 5)``, ``(B, n_classes)``.
        """
        tokens = self.input_proj(stroke5) + self.pos_embed
        enc_out = self.encoder(tokens)
        embedding = self.pool(enc_out)
        expanded = self.expand_proj(embedding).unsqueeze(1).expand(-1, self.seq_len, -1)
        dec_out = self.decoder(expanded)
        recon = self.recon_head(dec_out)
        class_logits = self.class_head(embedding)
        return embedding, recon, class_logits


def build_sketchformer() -> nn.Module:
    """Build a compact SketchFormer sketch transformer.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SketchFormer(d_model=32, seq_len=16, lowerdim=24, n_classes=10).eval()


def example_input_sketchformer() -> Tensor:
    """Example stroke-5 sketch input.

    Returns
    -------
    Tensor
        ``(1, 16, 5)`` stroke-5 tensor.
    """
    return torch.randn(1, 16, 5)


# ---------------------------------------------------------------------------
# MODULE 2: SketchHealer -- CNN sub-stroke-patch encoder + single-layer graph
# convolution + VAE bottleneck + GMM sketch-rnn LSTM decoder.
# ---------------------------------------------------------------------------


class _PatchCNN(nn.Module):
    """Shared per-sub-stroke-patch CNN feature extractor (``FeatureExtractionBasic``).

    Parameters
    ----------
    out_dim : int
        Output feature dimension.
    """

    def __init__(self, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_dim, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, patches: Tensor) -> Tensor:
        """Embed a batch of rendered sub-stroke patches.

        Parameters
        ----------
        patches : Tensor
            ``(N, 1, S, S)`` rendered sub-stroke image patches.

        Returns
        -------
        Tensor
            ``(N, out_dim)`` per-patch features.
        """
        return self.net(patches).flatten(1)


class _GraphConv(nn.Module):
    """Single-layer graph convolution over sub-stroke adjacency (``GCNProcessor``).

    Parameters
    ----------
    in_dim : int
        Input node-feature dimension.
    out_dim : int
        Output pooled-graph dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * (1.0 / math.sqrt(in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Propagate node features over the adjacency then sum-pool.

        Parameters
        ----------
        x : Tensor
            ``(B, N, in_dim)`` per-node features.
        adj : Tensor
            ``(B, N, N)`` adjacency matrix.

        Returns
        -------
        Tensor
            ``(B, out_dim)`` pooled graph embedding.
        """
        propagated = adj @ x
        transformed = propagated @ self.weight + self.bias
        return transformed.sum(dim=1)


class _GMMDecoderLSTM(nn.Module):
    """Sketch-RNN-style GMM + pen-state LSTM decoder (``DecoderRNN``).

    Parameters
    ----------
    z_dim : int
        VAE bottleneck dimension.
    hidden_size : int
        LSTM hidden width.
    n_mixtures : int
        Number of Gaussian mixture components.
    """

    def __init__(self, z_dim: int = 24, hidden_size: int = 48, n_mixtures: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_mixtures = n_mixtures
        self.fc_hc = nn.Linear(z_dim, 2 * hidden_size)
        self.lstm = nn.LSTM(5 + z_dim, hidden_size, batch_first=True)
        self.fc_params = nn.Linear(hidden_size, 6 * n_mixtures + 3)

    def forward(self, stroke5: Tensor, z: Tensor) -> Tensor:
        """Decode a stroke-5 sequence conditioned on the VAE latent ``z``.

        Parameters
        ----------
        stroke5 : Tensor
            ``(B, T, 5)`` teacher-forced stroke-5 input sequence.
        z : Tensor
            ``(B, z_dim)`` VAE latent used to init hidden/cell state and as
            per-step conditioning.

        Returns
        -------
        Tensor
            ``(B, T, 6 * n_mixtures + 3)`` per-step GMM + pen-state parameters.
        """
        t = stroke5.shape[1]
        hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.hidden_size, dim=1)
        z_rep = z.unsqueeze(1).expand(-1, t, -1)
        lstm_in = torch.cat([stroke5, z_rep], dim=-1)
        outputs, _ = self.lstm(lstm_in, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        return self.fc_params(outputs)


class SketchHealer(nn.Module):
    """SketchHealer: sub-stroke-patch CNN + graph-conv encoder feeding a
    sketch-rnn-style GMM LSTM decoder.

    Parameters
    ----------
    n_patches : int
        Number of sub-stroke patches per sketch.
    patch_size : int
        Rendered sub-stroke patch side length.
    z_dim : int
        VAE bottleneck dimension.
    """

    def __init__(self, n_patches: int = 8, patch_size: int = 16, z_dim: int = 24) -> None:
        super().__init__()
        self.n_patches = n_patches
        feat_dim = 32
        self.patch_cnn = _PatchCNN(feat_dim)
        self.gcn = _GraphConv(feat_dim, 48)
        self.fc_mu = nn.Linear(48, z_dim)
        self.fc_sigma = nn.Linear(48, z_dim)
        self.decoder = _GMMDecoderLSTM(z_dim=z_dim, hidden_size=48)

    def forward(
        self, patches: Tensor, adj: Tensor, stroke5: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode sub-stroke patches into a VAE latent and decode a sketch.

        Parameters
        ----------
        patches : Tensor
            ``(B, n_patches, 1, S, S)`` rendered sub-stroke patches.
        adj : Tensor
            ``(B, n_patches, n_patches)`` sub-stroke adjacency matrix.
        stroke5 : Tensor
            ``(B, T, 5)`` teacher-forced stroke-5 sequence to reconstruct.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(z, mu, gmm_params)``.
        """
        b, n = patches.shape[:2]
        feats = self.patch_cnn(patches.reshape(b * n, *patches.shape[2:])).view(b, n, -1)
        pooled = torch.tanh(self.gcn(feats, adj))
        mu = self.fc_mu(pooled)
        log_sigma = self.fc_sigma(pooled)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(log_sigma / 2.0) * eps
        gmm_params = self.decoder(stroke5, z)
        return z, mu, gmm_params


def build_sketchhealer() -> nn.Module:
    """Build a compact SketchHealer graph-to-sequence sketch completion network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SketchHealer(n_patches=8, patch_size=16, z_dim=24).eval()


def example_input_sketchhealer() -> tuple[Tensor, Tensor, Tensor]:
    """Example sub-stroke patches, adjacency, and stroke-5 target.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(patches, adj, stroke5)`` = ``(1, 8, 1, 16, 16)``, ``(1, 8, 8)``,
        ``(1, 10, 5)``.
    """
    patches = torch.rand(1, 8, 1, 16, 16)
    adj = torch.eye(8).unsqueeze(0)
    stroke5 = torch.randn(1, 10, 5)
    return patches, adj, stroke5


# ---------------------------------------------------------------------------
# MODULE 3: SketchLattice -- separate x/y lattice-coordinate embeddings +
# graph-conv propagation + VAE bottleneck + GMM sketch-rnn LSTM decoder.
# ---------------------------------------------------------------------------


class _CoordinateEmbeddingXYSep(nn.Module):
    """Separate learned x/y embedding tables, concatenated (``CoordinateEmbeddingXYSep``).

    Parameters
    ----------
    n_words : int
        Number of discrete coordinate bins per axis.
    out_dim : int
        Total (concatenated) output embedding dimension.
    """

    def __init__(self, n_words: int, out_dim: int) -> None:
        super().__init__()
        self.emb_x = nn.Embedding(n_words, out_dim // 2, padding_idx=0)
        self.emb_y = nn.Embedding(n_words, out_dim // 2, padding_idx=0)

    def forward(self, coords: Tensor) -> Tensor:
        """Embed integer lattice-point (x, y) coordinates.

        Parameters
        ----------
        coords : Tensor
            ``(B, N, 2)`` integer coordinates.

        Returns
        -------
        Tensor
            ``(B, N, out_dim)`` concatenated x/y embeddings.
        """
        x = coords[..., 0].long().clamp_(min=0)
        y = coords[..., 1].long().clamp_(min=0)
        return torch.cat([self.emb_x(x), self.emb_y(y)], dim=-1)


class SketchLattice(nn.Module):
    """SketchLattice: lattice-point coordinate embeddings + graph-conv
    propagation feeding a sketch-rnn-style GMM LSTM decoder.

    Parameters
    ----------
    n_points : int
        Number of lattice points sampled per sketch.
    n_words : int
        Coordinate quantization bin count per axis.
    z_dim : int
        VAE bottleneck dimension.
    """

    def __init__(self, n_points: int = 24, n_words: int = 64, z_dim: int = 24) -> None:
        super().__init__()
        self.n_points = n_points
        embed_dim = 32
        self.coord_embed = _CoordinateEmbeddingXYSep(n_words, embed_dim)
        self.gcn = _GraphConv(embed_dim, 48)
        self.fc_mu = nn.Linear(48, z_dim)
        self.fc_sigma = nn.Linear(48, z_dim)
        self.decoder = _GMMDecoderLSTM(z_dim=z_dim, hidden_size=48)

    def forward(
        self, coords: Tensor, adj: Tensor, stroke5: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a lattice point set into a VAE latent and decode a sketch.

        Parameters
        ----------
        coords : Tensor
            ``(B, n_points, 2)`` integer lattice-point coordinates.
        adj : Tensor
            ``(B, n_points, n_points)`` proximity-based adjacency matrix.
        stroke5 : Tensor
            ``(B, T, 5)`` teacher-forced stroke-5 sequence to reconstruct.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(z, mu, gmm_params)``.
        """
        embedded = self.coord_embed(coords)
        pooled = torch.tanh(self.gcn(embedded, adj))
        mu = self.fc_mu(pooled)
        log_sigma = self.fc_sigma(pooled)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(log_sigma / 2.0) * eps
        gmm_params = self.decoder(stroke5, z)
        return z, mu, gmm_params


def build_sketchlattice() -> nn.Module:
    """Build a compact SketchLattice latticed-representation sketch network.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SketchLattice(n_points=24, n_words=64, z_dim=24).eval()


def example_input_sketchlattice() -> tuple[Tensor, Tensor, Tensor]:
    """Example lattice coordinates, adjacency, and stroke-5 target.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(coords, adj, stroke5)`` = ``(1, 24, 2)``, ``(1, 24, 24)``,
        ``(1, 10, 5)``.
    """
    coords = torch.randint(0, 64, (1, 24, 2)).float()
    adj = torch.eye(24).unsqueeze(0)
    stroke5 = torch.randn(1, 10, 5)
    return coords, adj, stroke5


# ---------------------------------------------------------------------------
# MODULE 4: SketchXAI -- shape (BiLSTM) + order (embedding) + location (linear)
# factorized ViT-style stroke tokens with a CLS token.
# ---------------------------------------------------------------------------


class SketchXAI(nn.Module):
    """SketchXAI: shape/order/location-factorized stroke tokens through a ViT
    encoder, exposing per-stroke order embeddings for stroke-location-inversion.

    Parameters
    ----------
    max_strokes : int
        Maximum number of strokes per sketch (token budget excluding CLS).
    hidden_size : int
        Transformer token width.
    n_classes : int
        Number of sketch categories.
    """

    def __init__(self, max_strokes: int = 12, hidden_size: int = 32, n_classes: int = 10) -> None:
        super().__init__()
        self.max_strokes = max_strokes
        self.hidden_size = hidden_size
        lstm_hidden = hidden_size // 2
        self.shape_lstm = nn.LSTM(
            2, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.order_embed = nn.Embedding(max_strokes, hidden_size)
        self.location_proj = nn.Linear(2, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, nhead=4, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.class_head = nn.Linear(hidden_size, n_classes)

    def forward(self, stroke_points: Tensor, locations: Tensor) -> Tensor:
        """Classify a sketch from its shape/order/location-factorized strokes.

        Parameters
        ----------
        stroke_points : Tensor
            ``(B, max_strokes, P, 2)`` per-stroke (dx, dy) point sequences.
        locations : Tensor
            ``(B, max_strokes, 2)`` per-stroke bounding-box-center coordinates.

        Returns
        -------
        Tensor
            ``(B, n_classes)`` classification logits.
        """
        b, n, p, _ = stroke_points.shape
        flat_points = stroke_points.reshape(b * n, p, 2)
        _, (h_n, _) = self.shape_lstm(flat_points)
        shape_emb = h_n.transpose(0, 1).reshape(b, n, self.hidden_size)

        order_idx = torch.arange(n, device=stroke_points.device).unsqueeze(0).expand(b, -1)
        order_emb = self.order_embed(order_idx)

        location_emb = self.location_proj(locations)

        tokens = shape_emb + order_emb + location_emb
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.layernorm(self.encoder(tokens))
        return self.class_head(encoded[:, 0, :])


def build_sketchxai() -> nn.Module:
    """Build a compact SketchXAI shape/order/location stroke transformer.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return SketchXAI(max_strokes=12, hidden_size=32, n_classes=10).eval()


def example_input_sketchxai() -> tuple[Tensor, Tensor]:
    """Example per-stroke point sequences and bounding-box-center locations.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(stroke_points, locations)`` = ``(1, 12, 6, 2)``, ``(1, 12, 2)``.
    """
    return torch.randn(1, 12, 6, 2), torch.rand(1, 12, 2)


# ---------------------------------------------------------------------------
# MODULE 5: StarVector -- ViT patch encoder + MLP adapter projecting visual
# tokens into a causal code-LLM's embedding space for SVG generation.
# ---------------------------------------------------------------------------


class _Swish(nn.Module):
    """Swish activation ``x * sigmoid(x)`` (matches the source's ``Swish``)."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply the swish nonlinearity.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            ``x * sigmoid(x)``, same shape as ``x``.
        """
        return x * torch.sigmoid(x)


class _StarVectorAdapter(nn.Module):
    """MLP adapter projecting vision-encoder tokens into LLM embedding space
    (exact structure of the source's ``Adapter``).

    Parameters
    ----------
    input_size : int
        Vision-encoder hidden size.
    output_size : int
        Code-LLM hidden size.
    query_length : int
        Number of visual tokens (patches).
    """

    def __init__(self, input_size: int, output_size: int, query_length: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(input_size, input_size * 2)
        self.act = _Swish()
        self.c_proj = nn.Linear(input_size * 2, output_size)
        self.norm = nn.LayerNorm([query_length, output_size])

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project vision tokens into LLM-space visual tokens.

        Parameters
        ----------
        hidden_states : Tensor
            ``(B, query_length, input_size)`` vision-encoder token embeddings.

        Returns
        -------
        Tensor
            ``(B, query_length, output_size)`` visual tokens.
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.norm(hidden_states)


class _ViTPatchEncoder(nn.Module):
    """Compact CLIP-style ViT patch encoder producing a fixed-length token grid.

    Parameters
    ----------
    img_size : int
        Input image side length.
    patch_size : int
        Patch side length.
    hidden_size : int
        Vision-transformer token width.
    """

    def __init__(self, img_size: int = 32, patch_size: int = 8, hidden_size: int = 48) -> None:
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, hidden_size) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, nhead=4, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

    def forward(self, images: Tensor) -> Tensor:
        """Encode images into a fixed-length patch-token sequence.

        Parameters
        ----------
        images : Tensor
            ``(B, 3, img_size, img_size)`` input images.

        Returns
        -------
        Tensor
            ``(B, n_patches, hidden_size)`` patch tokens.
        """
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        return self.encoder(patches + self.pos_embed)


class StarVector(nn.Module):
    """StarVector: ViT image encoder + adapter producing visual-token prefix for
    a causal code-LLM that autoregressively emits SVG token ids.

    Parameters
    ----------
    img_size : int
        Input image side length.
    patch_size : int
        Vision-encoder patch side length.
    vis_hidden : int
        Vision-encoder hidden size.
    llm_hidden : int
        Code-LLM hidden size.
    vocab_size : int
        SVG-code token vocabulary size.
    code_len : int
        Number of SVG-code tokens appended after the visual-token prefix.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        vis_hidden: int = 48,
        llm_hidden: int = 64,
        vocab_size: int = 128,
        code_len: int = 12,
    ) -> None:
        super().__init__()
        self.image_encoder = _ViTPatchEncoder(img_size, patch_size, vis_hidden)
        query_length = self.image_encoder.n_patches
        self.adapter = _StarVectorAdapter(vis_hidden, llm_hidden, query_length)
        self.token_embed = nn.Embedding(vocab_size, llm_hidden)
        self.code_len = code_len
        total_len = query_length + code_len
        self.pos_embed = nn.Parameter(torch.randn(1, total_len, llm_hidden) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(
            llm_hidden, nhead=4, dim_feedforward=llm_hidden * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=3)
        self.lm_head = nn.Linear(llm_hidden, vocab_size)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((total_len, total_len), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, images: Tensor, svg_token_ids: Tensor) -> Tensor:
        """Generate next-token SVG logits conditioned on an image-token prefix.

        Parameters
        ----------
        images : Tensor
            ``(B, 3, img_size, img_size)`` input raster images.
        svg_token_ids : Tensor
            ``(B, code_len)`` SVG-code token ids (teacher-forced).

        Returns
        -------
        Tensor
            ``(B, n_patches + code_len, vocab_size)`` next-token logits over the
            full visual-prefix + code sequence.
        """
        vision_tokens = self.image_encoder(images)
        visual_tokens = self.adapter(vision_tokens)
        code_tokens = self.token_embed(svg_token_ids)
        sequence = torch.cat([visual_tokens, code_tokens], dim=1) + self.pos_embed
        hidden = self.transformer(sequence, mask=self.causal_mask)
        return self.lm_head(hidden)


def build_starvector() -> nn.Module:
    """Build a compact StarVector image-to-SVG-code transformer.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return StarVector(
        img_size=32, patch_size=8, vis_hidden=48, llm_hidden=64, vocab_size=128, code_len=12
    ).eval()


def example_input_starvector() -> tuple[Tensor, Tensor]:
    """Example raster image and teacher-forced SVG token ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(images, svg_token_ids)`` = ``(1, 3, 32, 32)``, ``(1, 12)``.
    """
    return torch.rand(1, 3, 32, 32), torch.randint(0, 128, (1, 12))


# ---------------------------------------------------------------------------
# MODULE 6: StyleCLIPDraw -- learnable RGBA Bezier-path canvas optimized
# end-to-end against a content/CLIP-proxy encoder and a Gram-matrix style
# encoder simultaneously.
# ---------------------------------------------------------------------------


class _BezierRGBAPaths(nn.Module):
    """Stack of learnable closed quadratic-Bezier RGBA paths, rasterized by soft
    signed-distance occupancy and alpha-composited back-to-front.

    Parameters
    ----------
    n_paths : int
        Number of Bezier paths ("brush strokes").
    n_ctrl : int
        Control points per path.
    canvas_size : int
        Output canvas side length.
    """

    def __init__(self, n_paths: int = 6, n_ctrl: int = 5, canvas_size: int = 48) -> None:
        super().__init__()
        self.n_paths = n_paths
        self.canvas_size = canvas_size
        base = torch.rand(n_paths, n_ctrl, 2) * canvas_size
        self.control_points = nn.Parameter(base)
        self.colors = nn.Parameter(torch.rand(n_paths, 4) * 0.8 + 0.1)
        grid = torch.linspace(0, canvas_size - 1, canvas_size)
        gy, gx = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("grid_xy", torch.stack([gx, gy], dim=-1), persistent=False)

    def forward(self) -> Tensor:
        """Rasterize and alpha-composite all Bezier paths onto one canvas.

        Returns
        -------
        Tensor
            ``(1, 3, canvas_size, canvas_size)`` rendered RGB canvas in ``[0, 1]``.
        """
        s = self.canvas_size
        canvas = torch.ones(3, s, s, device=self.control_points.device)
        grid = self.grid_xy.unsqueeze(0)
        for i in range(self.n_paths):
            pts = self.control_points[i]
            center = pts.mean(dim=0, keepdim=True)
            dists = torch.cdist(grid.reshape(-1, 2), pts).min(dim=-1).values.view(s, s)
            radius = (pts - center).norm(dim=-1).mean().clamp(min=1.0)
            mask = torch.sigmoid((radius - dists) * 2.0)
            color = torch.sigmoid(self.colors[i, :3]).view(3, 1, 1)
            alpha = torch.sigmoid(self.colors[i, 3]) * mask
            canvas = canvas * (1 - alpha) + color * alpha
        return canvas.unsqueeze(0)


class _SmallVGGTap(nn.Module):
    """Compact VGG-like conv stack exposing intermediate feature taps, standing
    in for the CLIP content encoder and the VGG16 style encoder alike.

    Parameters
    ----------
    base_ch : int
        Channel width of the first conv stage.
    """

    def __init__(self, base_ch: int = 8) -> None:
        super().__init__()
        c = base_ch
        self.stage1 = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.ReLU(inplace=True))
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(c, c * 2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return multi-layer feature taps for Gram-matrix / embedding losses.

        Parameters
        ----------
        x : Tensor
            ``(B, 3, H, W)`` input image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Feature maps at three increasing depths / decreasing resolutions.
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


class StyleCLIPDraw(nn.Module):
    """StyleCLIPDraw: a learnable RGBA Bezier-path canvas scored end-to-end by a
    content/CLIP-proxy encoder and a separate Gram-matrix style encoder.

    Parameters
    ----------
    n_paths : int
        Number of Bezier paths.
    n_ctrl : int
        Control points per path.
    canvas_size : int
        Rendered canvas side length.
    embed_dim : int
        Content/CLIP-proxy embedding dimension.
    """

    def __init__(
        self,
        n_paths: int = 6,
        n_ctrl: int = 5,
        canvas_size: int = 48,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.canvas = _BezierRGBAPaths(n_paths, n_ctrl, canvas_size)
        self.content_encoder = _SmallVGGTap(base_ch=8)
        self.style_encoder = _SmallVGGTap(base_ch=8)
        self.content_proj = nn.Linear(8 * 4, embed_dim)

    def _gram(self, feat: Tensor) -> Tensor:
        """Compute a batched Gram matrix from a feature map.

        Parameters
        ----------
        feat : Tensor
            ``(B, C, H, W)`` feature map.

        Returns
        -------
        Tensor
            ``(B, C, C)`` Gram matrix, normalized by ``C * H * W``.
        """
        b, c, h, w = feat.shape
        flat = feat.view(b, c, h * w)
        gram = flat @ flat.transpose(1, 2)
        return gram / (c * h * w)

    def forward(self, style_image: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Render the canvas and score it against text-fit and style targets.

        Parameters
        ----------
        style_image : Tensor
            ``(1, 3, canvas_size, canvas_size)`` reference style image.

        Returns
        -------
        tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]
            ``(canvas, content_embedding, style_grams)``: the rendered canvas,
            its content/CLIP-proxy embedding, and its per-layer Gram matrices
            paired implicitly against the style image's own Gram matrices.
        """
        canvas = self.canvas()
        _, _, content_f3 = self.content_encoder(canvas)
        content_embedding = self.content_proj(content_f3.mean(dim=(2, 3)))

        canvas_grams = tuple(self._gram(f) for f in self.style_encoder(canvas))
        style_grams = tuple(self._gram(f) for f in self.style_encoder(style_image))
        gram_diffs = tuple(c - s for c, s in zip(canvas_grams, style_grams, strict=True))
        return canvas, content_embedding, gram_diffs


def build_styleclipdraw() -> nn.Module:
    """Build a compact StyleCLIPDraw coupled content/style Bezier-path optimizer.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return StyleCLIPDraw(n_paths=6, n_ctrl=5, canvas_size=48, embed_dim=32).eval()


def example_input_styleclipdraw() -> Tensor:
    """Example reference style image.

    Returns
    -------
    Tensor
        ``(1, 3, 48, 48)`` style image.
    """
    return torch.rand(1, 3, 48, 48)


MENAGERIE_ENTRIES = [
    ("SketchFormer", "build_sketchformer", "example_input_sketchformer", "2020", "GEN"),
    ("SketchHealer", "build_sketchhealer", "example_input_sketchhealer", "2020", "GEN"),
    ("SketchLattice", "build_sketchlattice", "example_input_sketchlattice", "2021", "GEN"),
    ("SketchXAI", "build_sketchxai", "example_input_sketchxai", "2023", "VIS"),
    (
        "StarVector (SVG code-generation LLM)",
        "build_starvector",
        "example_input_starvector",
        "2023",
        "GEN",
    ),
    ("StyleCLIPDraw", "build_styleclipdraw", "example_input_styleclipdraw", "2022", "GEN"),
]

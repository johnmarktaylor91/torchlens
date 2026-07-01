"""Compact faithful reimplementations for bioinformatics/cheminformatics rows.

Sources checked (GitHub API contents + raw file reads, no clone/pip-install):

* DeepPINK -- https://github.com/younglululu/DeepPINK (NeurIPS 2018,
  http://papers.neurips.cc/paper/8085-deeppink-reproducible-feature-selection-in-deep-neural-networks.pdf).
  Read ``run_withKnockoff_all.py``. The repo's reference code is Keras, but
  the paper's architecture is simple and explicit: a per-feature
  "filter" layer of two stacked ``LocallyConnected1D`` layers applied to the
  ``(original, knockoff)`` pair for every feature (extracting a linear
  contrast ``Z_j - tilde Z_j`` per feature without mixing across features),
  followed by a small feed-forward MLP over the filtered feature vector to a
  sigmoid output. Reimplemented the locally-connected filter as a grouped
  ``nn.Conv1d`` with ``groups=p`` (one independent 2->1->1 filter per
  feature, exactly the per-feature weight-sharing pattern of
  ``LocallyConnected1D`` with a single spatial position), followed by the
  same MLP head.
* DeepPPISP -- https://github.com/CSUBioGroup/DeepPPISP (Bioinformatics
  2020, https://academic.oup.com/bioinformatics/article/36/4/1110/5573399).
  Read ``models/deep_ppi.py``. Multi-scale text-CNN over the full sequence
  (three parallel ``Conv2d`` branches with kernel heights 13/15/17 spanning
  the full feature width, each followed by a max-pool over the sequence
  length) fused with hand windowed "local" features around the query
  residue, then a 2-layer MLP classifier head with a final sigmoid.
* DeepPurpose (encoders) --
  https://github.com/kexinhuang12345/DeepPurpose (Bioinformatics 2020,
  https://academic.oup.com/bioinformatics/article/36/22-23/5545/6020256).
  Read ``DeepPurpose/encoders.py`` and ``DeepPurpose/model_helper.py``. The
  package is a modular drug<->protein encoder zoo; its flagship, most
  distinctive encoder is the dual-tower BERT-style ``transformer`` encoder
  (``Embeddings`` = token + learned positional embedding + LayerNorm,
  ``Encoder_MultipleLayers`` = stacked pre-norm-free self-attention +
  feed-forward blocks) applied independently to a drug substructure token
  sequence and a protein subsequence token sequence, each pooled to its
  ``[CLS]``-style first token and concatenated into a joint interaction MLP
  that predicts a drug-target interaction score.
* DeepRT (DeepRTplus) -- https://github.com/horsepurve/DeepRTplus (arXiv
  1705.05368). Read ``capsule_network_emb.py``. Peptide retention-time
  regressor: a learned amino-acid embedding, an initial ``Conv2d`` feature
  extractor over the embedded sequence, a "primary capsule" conv layer that
  reshapes into 8-D capsule vectors with a squash nonlinearity, and a
  dynamic-routing ``CapsuleLayer`` (agreement routing between primary and
  output capsules) whose output capsule lengths / pose are read out through
  a small linear head to predict scalar retention time.
* DeepSequence -- https://github.com/debbiemarkslab/DeepSequence (Nature
  Methods 2018, https://www.nature.com/articles/s41592-018-0138-4). Read
  ``DeepSequence/model.py``. The reference implementation is Theano; the
  paper's architecture is a variational autoencoder over one-hot multiple
  sequence alignment columns: an MLP encoder producing a diagonal Gaussian
  latent, and a "Bayesian" MLP decoder whose final weight matrix is
  factorized into a low-rank dictionary of ``n_patterns`` sequence profiles
  (a per-position, per-pattern scale mixture) that is convolved with the
  penultimate hidden layer and reshaped back into a per-position amino-acid
  categorical distribution (softmax over the alphabet at each alignment
  column). Reimplemented the sparse/patterned final decoder layer as an
  explicit low-rank factorization: hidden -> (n_patterns x alphabet_size)
  logits per position via a shared linear projection, tiled across
  positions and reduced over patterns.
* DeepSignal -- https://github.com/bioinfomaticsCSU/deepsignal
  (Bioinformatics 2019,
  https://academic.oup.com/bioinformatics/article/35/22/4586/5474907). Read
  ``deepsignal/model.py`` and ``deepsignal/layers.py``. The reference
  implementation is TensorFlow 1.x; reimplemented in torch preserving the
  joint two-branch design: a "base" branch (k-mer embedding concatenated
  with per-base mean/std/count signal statistics) run through a stacked
  bidirectional LSTM, and a "signal" branch (raw per-base nanopore signal
  windows) run through an Inception-style multi-branch 1D CNN (parallel
  1x1 / 1x3 / 1x5 / pooled-projection branches concatenated channel-wise),
  with the two branch outputs concatenated and passed through a joint MLP
  classifier for 5mC/6mA methylation calling.

All models below use small random-init dimensions; they are architecture
catalog entries, not trained-weight replicas.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# DeepPINK -- knockoff-augmented per-feature filter + MLP for FDR-controlled
# feature selection
# ---------------------------------------------------------------------------


class DeepPinkFilterLayer(nn.Module):
    """Per-feature locally-connected filter over (original, knockoff) pairs.

    Applies two independent scalar weights per feature to its
    ``(Z_j, tilde Z_j)`` pair (no cross-feature mixing), mirroring the
    stacked ``LocallyConnected1D`` "filter" module from the reference Keras
    code. Implemented as a ``groups=p`` ``Conv1d`` so every feature owns its
    own private 2->1 linear filter.
    """

    def __init__(self, n_features: int) -> None:
        """Build the grouped per-feature filter.

        Parameters
        ----------
        n_features:
            Number of original features ``p`` (each with a knockoff twin).
        """

        super().__init__()
        self.n_features = n_features
        self.filter = nn.Conv1d(2 * n_features, n_features, kernel_size=1, groups=n_features)

    def forward(self, z: Tensor, z_knockoff: Tensor) -> Tensor:
        """Filter each feature's original/knockoff pair to one scalar.

        Parameters
        ----------
        z:
            Original feature values, shape ``(batch, n_features)``.
        z_knockoff:
            Knockoff feature values, shape ``(batch, n_features)``.

        Returns
        -------
        Tensor
            Filtered per-feature statistics, shape ``(batch, n_features)``.
        """

        # Interleave (z_j, zk_j) pairs per feature so the grouped conv sees
        # each feature's own 2-channel slice.
        stacked = torch.stack([z, z_knockoff], dim=2)  # (batch, p, 2)
        interleaved = stacked.reshape(z.shape[0], -1, 1)  # (batch, 2p, 1)
        filtered = self.filter(interleaved)  # (batch, p, 1)
        return filtered.squeeze(-1)


class DeepPink(nn.Module):
    """Knockoff-augmented DNN for reproducible feature selection (DeepPINK)."""

    def __init__(self, n_features: int = 20, hidden_dim: int = 16) -> None:
        """Build the filter layer and downstream MLP classifier.

        Parameters
        ----------
        n_features:
            Number of original (non-knockoff) features.
        hidden_dim:
            Hidden width of the MLP classifier head.
        """

        super().__init__()
        self.filter_layer = DeepPinkFilterLayer(n_features)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, z_knockoff: Tensor) -> Tensor:
        """Predict an outcome probability from original + knockoff features.

        Parameters
        ----------
        z:
            Original feature values, shape ``(batch, n_features)``.
        z_knockoff:
            Knockoff feature values, shape ``(batch, n_features)``.

        Returns
        -------
        Tensor
            Sigmoid probabilities, shape ``(batch, 1)``.
        """

        filtered = self.filter_layer(z, z_knockoff)
        logits = self.mlp(filtered)
        return torch.sigmoid(logits)


def build_deeppink() -> nn.Module:
    """Build a compact random-init DeepPINK model."""

    return DeepPink(n_features=20, hidden_dim=16).eval()


def example_input_deeppink() -> tuple[Tensor, Tensor]:
    """Return an (original, knockoff) feature pair for DeepPINK."""

    z = torch.randn(4, 20)
    z_knockoff = torch.randn(4, 20)
    return z, z_knockoff


# ---------------------------------------------------------------------------
# DeepPPISP -- multi-scale sequence CNN + local window features for
# protein-protein interaction site prediction
# ---------------------------------------------------------------------------


class DeepPpispConvsLayer(nn.Module):
    """Three parallel full-width text-CNN branches over the sequence axis."""

    def __init__(self, seq_len: int, feat_width: int, hidden_channels: int = 8) -> None:
        """Build the three parallel conv+pool branches.

        Parameters
        ----------
        seq_len:
            Length of the padded input sequence.
        feat_width:
            Per-residue feature width (sequence + dssp + pssm channels
            concatenated).
        hidden_channels:
            Number of output channels per branch.
        """

        super().__init__()
        kernels = (13, 15, 17)
        self.branches = nn.ModuleList()
        for k in kernels:
            pad = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv2d(1, hidden_channels, kernel_size=(k, feat_width), padding=(pad, 0)),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=(seq_len, 1), stride=1),
            )
            self.branches.append(branch)

    def forward(self, x: Tensor) -> Tensor:
        """Run all three branches and concatenate their pooled outputs.

        Parameters
        ----------
        x:
            Input feature map, shape ``(batch, 1, seq_len, feat_width)``.

        Returns
        -------
        Tensor
            Flattened concatenated branch features, shape ``(batch, C)``.
        """

        outs = [branch(x) for branch in self.branches]
        cat = torch.cat(outs, dim=1)
        return cat.reshape(cat.shape[0], -1)


class DeepPpisp(nn.Module):
    """Multi-scale CNN + local window features for PPI site prediction."""

    def __init__(
        self,
        seq_len: int = 40,
        seq_dim: int = 20,
        dssp_dim: int = 9,
        pssm_dim: int = 20,
        window_size: int = 3,
        hidden_channels: int = 8,
    ) -> None:
        """Build the sequence embedding, multi-scale CNN, and MLP head.

        Parameters
        ----------
        seq_len:
            Padded protein sequence length.
        seq_dim:
            One-hot sequence feature width.
        dssp_dim:
            Secondary-structure (DSSP) feature width.
        pssm_dim:
            Position-specific scoring matrix feature width.
        window_size:
            Half-width of the local residue window (local features span
            ``2 * window_size + 1`` residues).
        hidden_channels:
            Channel width of each CNN branch.
        """

        super().__init__()
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        feat_width = seq_dim + dssp_dim + pssm_dim

        self.seq_embed = nn.Sequential(
            nn.Linear(seq_len * seq_dim, seq_len * seq_dim),
            nn.ReLU(),
        )
        self.multi_cnn = DeepPpispConvsLayer(seq_len, feat_width, hidden_channels)

        local_dim = (2 * window_size + 1) * feat_width
        cnn_out_dim = hidden_channels * 3
        input_dim = cnn_out_dim + local_dim

        self.dnn1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.dnn2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, seq: Tensor, dssp: Tensor, pssm: Tensor, local_features: Tensor) -> Tensor:
        """Predict a per-residue interaction-site probability.

        Parameters
        ----------
        seq:
            One-hot sequence features, shape ``(batch, 1, seq_len, seq_dim)``.
        dssp:
            DSSP features, shape ``(batch, 1, seq_len, dssp_dim)``.
        pssm:
            PSSM features, shape ``(batch, 1, seq_len, pssm_dim)``.
        local_features:
            Flattened local window features, shape ``(batch, local_dim)``.

        Returns
        -------
        Tensor
            Sigmoid interaction-site probabilities, shape ``(batch, 1)``.
        """

        b = seq.shape[0]
        flat = seq.reshape(b, self.seq_len * self.seq_dim)
        embedded = self.seq_embed(flat).reshape(b, 1, self.seq_len, self.seq_dim)

        features = torch.cat((embedded, dssp, pssm), dim=3)
        cnn_features = self.multi_cnn(features)

        joined = torch.cat((cnn_features, local_features), dim=1)
        hidden = self.dnn1(joined)
        hidden = self.dnn2(hidden)
        return self.out_layer(hidden)


def build_deepppisp() -> nn.Module:
    """Build a compact random-init DeepPPISP model."""

    return DeepPpisp(
        seq_len=40, seq_dim=20, dssp_dim=9, pssm_dim=20, window_size=3, hidden_channels=8
    ).eval()


def example_input_deepppisp() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (seq, dssp, pssm, local_features) tensors for DeepPPISP."""

    seq = torch.randn(2, 1, 40, 20)
    dssp = torch.randn(2, 1, 40, 9)
    pssm = torch.randn(2, 1, 40, 20)
    local_features = torch.randn(2, (2 * 3 + 1) * (20 + 9 + 20))
    return seq, dssp, pssm, local_features


# ---------------------------------------------------------------------------
# DeepPurpose -- dual-tower BERT-style transformer encoder for drug-target
# interaction prediction
# ---------------------------------------------------------------------------


class DeepPurposeEmbeddings(nn.Module):
    """Token + learned positional embedding, matching ``DeepPurpose.Embeddings``."""

    def __init__(self, vocab_size: int, hidden_size: int, max_position: int) -> None:
        """Build the token and position embedding tables.

        Parameters
        ----------
        vocab_size:
            Number of distinct substructure/subsequence tokens.
        hidden_size:
            Embedding / model hidden width.
        max_position:
            Maximum sequence length supported by the positional table.
        """

        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed a batch of token-id sequences.

        Parameters
        ----------
        input_ids:
            Integer token ids, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Embedded + position-added + normalized sequence, shape
            ``(batch, seq_len, hidden_size)``.
        """

        seq_len = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        )
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(self.layer_norm(embeddings))


class DeepPurposeTransformerLayer(nn.Module):
    """One pre-LayerNorm-free self-attention + feed-forward block."""

    def __init__(self, hidden_size: int, n_heads: int, intermediate_size: int) -> None:
        """Build the self-attention and feed-forward sublayers.

        Parameters
        ----------
        hidden_size:
            Model hidden width.
        n_heads:
            Number of self-attention heads.
        intermediate_size:
            Width of the feed-forward hidden layer.
        """

        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Run one attention + feed-forward block with residual add-norm.

        Parameters
        ----------
        x:
            Input sequence, shape ``(batch, seq_len, hidden_size)``.
        key_padding_mask:
            Boolean mask, ``True`` at padded positions to ignore.

        Returns
        -------
        Tensor
            Updated sequence, shape ``(batch, seq_len, hidden_size)``.
        """

        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.attn_norm(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        return x


class DeepPurposeTransformerTower(nn.Module):
    """Embeddings + stacked transformer layers + first-token pooling."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        intermediate_size: int,
        max_position: int,
    ) -> None:
        """Build one drug or protein transformer tower.

        Parameters
        ----------
        vocab_size:
            Number of distinct tokens for this modality.
        hidden_size:
            Model hidden width.
        n_layers:
            Number of stacked transformer layers.
        n_heads:
            Number of self-attention heads per layer.
        intermediate_size:
            Feed-forward hidden width per layer.
        max_position:
            Maximum sequence length supported.
        """

        super().__init__()
        self.embed = DeepPurposeEmbeddings(vocab_size, hidden_size, max_position)
        self.layers = nn.ModuleList(
            [
                DeepPurposeTransformerLayer(hidden_size, n_heads, intermediate_size)
                for _ in range(n_layers)
            ]
        )

    def forward(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        """Encode a token sequence and pool the first token.

        Parameters
        ----------
        input_ids:
            Integer token ids, shape ``(batch, seq_len)``.
        mask:
            Float mask, ``1.0`` for valid tokens and ``0.0`` for padding,
            shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Pooled representation of the first token, shape
            ``(batch, hidden_size)``.
        """

        hidden = self.embed(input_ids)
        key_padding_mask = mask < 0.5
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask)
        return hidden[:, 0]


class DeepPurposeTransformerDTI(nn.Module):
    """Dual-tower transformer drug-target interaction predictor."""

    def __init__(
        self,
        drug_vocab_size: int = 64,
        protein_vocab_size: int = 26,
        hidden_size: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        intermediate_size: int = 64,
        drug_max_len: int = 50,
        protein_max_len: int = 100,
    ) -> None:
        """Build the drug tower, protein tower, and interaction MLP.

        Parameters
        ----------
        drug_vocab_size:
            Number of distinct drug substructure tokens.
        protein_vocab_size:
            Number of distinct protein subsequence tokens.
        hidden_size:
            Shared transformer hidden width.
        n_layers:
            Number of transformer layers per tower.
        n_heads:
            Number of self-attention heads per layer.
        intermediate_size:
            Feed-forward hidden width per layer.
        drug_max_len:
            Maximum drug token sequence length.
        protein_max_len:
            Maximum protein token sequence length.
        """

        super().__init__()
        self.drug_encoder = DeepPurposeTransformerTower(
            drug_vocab_size, hidden_size, n_layers, n_heads, intermediate_size, drug_max_len
        )
        self.protein_encoder = DeepPurposeTransformerTower(
            protein_vocab_size, hidden_size, n_layers, n_heads, intermediate_size, protein_max_len
        )
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        drug_ids: Tensor,
        drug_mask: Tensor,
        protein_ids: Tensor,
        protein_mask: Tensor,
    ) -> Tensor:
        """Predict a drug-target interaction score.

        Parameters
        ----------
        drug_ids:
            Drug substructure token ids, shape ``(batch, drug_max_len)``.
        drug_mask:
            Drug validity mask, shape ``(batch, drug_max_len)``.
        protein_ids:
            Protein subsequence token ids, shape ``(batch, protein_max_len)``.
        protein_mask:
            Protein validity mask, shape ``(batch, protein_max_len)``.

        Returns
        -------
        Tensor
            Interaction score, shape ``(batch, 1)``.
        """

        drug_repr = self.drug_encoder(drug_ids, drug_mask)
        protein_repr = self.protein_encoder(protein_ids, protein_mask)
        joined = torch.cat((drug_repr, protein_repr), dim=1)
        return self.interaction_mlp(joined)


def build_deeppurpose_encoders() -> nn.Module:
    """Build a compact random-init DeepPurpose dual-tower transformer DTI model."""

    return DeepPurposeTransformerDTI(
        drug_vocab_size=64,
        protein_vocab_size=26,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        intermediate_size=64,
        drug_max_len=20,
        protein_max_len=30,
    ).eval()


def example_input_deeppurpose_encoders() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (drug_ids, drug_mask, protein_ids, protein_mask) for DeepPurpose."""

    drug_ids = torch.randint(0, 64, (2, 20))
    drug_mask = torch.ones(2, 20)
    protein_ids = torch.randint(0, 26, (2, 30))
    protein_mask = torch.ones(2, 30)
    return drug_ids, drug_mask, protein_ids, protein_mask


# ---------------------------------------------------------------------------
# DeepRT (DeepRTplus) -- embedding CNN + dynamic-routing capsule network for
# peptide retention time prediction
# ---------------------------------------------------------------------------


def _squash(x: Tensor, dim: int = -1) -> Tensor:
    """Squash a capsule vector's length into ``[0, 1)`` while keeping direction.

    Parameters
    ----------
    x:
        Capsule pose vectors along ``dim``.
    dim:
        Dimension holding each capsule's pose vector.

    Returns
    -------
    Tensor
        Squashed capsule vectors, same shape as ``x``.
    """

    squared_norm = (x**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)


class DeepRtPrimaryCapsules(nn.Module):
    """Conv feature extractor producing a grid of primary capsule vectors.

    Matches the reference ``CapsuleLayer(num_route_nodes=-1)`` branch: a
    bank of ``pose_dim`` independent conv branches, each producing
    ``branch_channels`` output channels; stacking the branches along a new
    last axis turns "which branch" into the capsule pose-vector dimension,
    while the spatial (channel x width) positions become the route-node
    axis.
    """

    def __init__(self, in_channels: int, branch_channels: int, pose_dim: int) -> None:
        """Build the parallel per-pose-dimension conv branches.

        Parameters
        ----------
        in_channels:
            Input channel count from the first conv layer.
        branch_channels:
            Output channel count of each branch (becomes part of the
            route-node axis after flattening).
        pose_dim:
            Number of parallel branches, i.e. the capsule pose-vector
            dimensionality.
        """

        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 9), stride=1)
                for _ in range(pose_dim)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Extract and squash primary capsule pose vectors.

        Parameters
        ----------
        x:
            Input feature map, shape ``(batch, in_channels, h, w)``.

        Returns
        -------
        Tensor
            Primary capsules, shape ``(batch, n_route_nodes, pose_dim)``.
        """

        outs = [branch(x).reshape(x.shape[0], -1, 1) for branch in self.branches]
        cat = torch.cat(outs, dim=-1)
        return _squash(cat)


class DeepRtRoutingCapsules(nn.Module):
    """Dynamic-routing-by-agreement capsule layer."""

    def __init__(
        self,
        n_capsules: int,
        n_route_nodes: int,
        in_channels: int,
        out_channels: int,
        n_iter: int = 3,
    ) -> None:
        """Build the routing weight tensor.

        Parameters
        ----------
        n_capsules:
            Number of output capsules.
        n_route_nodes:
            Number of input (primary) capsule nodes to route from.
        in_channels:
            Pose-vector dimensionality of each input capsule.
        out_channels:
            Pose-vector dimensionality of each output capsule.
        n_iter:
            Number of dynamic-routing iterations.
        """

        super().__init__()
        self.n_iter = n_iter
        self.route_weights = nn.Parameter(
            0.01 * torch.randn(n_capsules, n_route_nodes, in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Route primary capsules to output capsules by iterative agreement.

        Parameters
        ----------
        x:
            Primary capsules, shape ``(batch, n_route_nodes, in_channels)``.

        Returns
        -------
        Tensor
            Output capsule pose vectors, shape
            ``(batch, n_capsules, out_channels)``.
        """

        # priors: (n_capsules, batch, n_route_nodes, 1, out_channels)
        priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
        logits = torch.zeros(*priors.shape, device=x.device, dtype=x.dtype)
        outputs = priors.sum(dim=2, keepdim=True)
        for i in range(self.n_iter):
            probs = F.softmax(logits, dim=2)
            outputs = _squash((probs * priors).sum(dim=2, keepdim=True))
            if i != self.n_iter - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits
        # outputs: (n_capsules, batch, 1, 1, out_channels) -> (batch, n_capsules, out_channels)
        outputs = outputs.squeeze(3).squeeze(2)
        return outputs.permute(1, 0, 2)


class DeepRt(nn.Module):
    """Embedding CNN + capsule-routing peptide retention-time regressor."""

    def __init__(
        self,
        vocab_size: int = 22,
        emb_size: int = 8,
        seq_len: int = 30,
        conv_channels: int = 16,
        primary_branch_channels: int = 4,
        primary_pose_dim: int = 8,
        n_output_capsules: int = 4,
        output_capsule_dim: int = 6,
    ) -> None:
        """Build the embedding, conv stem, and capsule stack.

        Parameters
        ----------
        vocab_size:
            Number of distinct amino-acid tokens.
        emb_size:
            Amino-acid embedding width.
        seq_len:
            Padded peptide sequence length.
        conv_channels:
            Channel width of the first conv layer.
        primary_branch_channels:
            Output channel count of each primary-capsule conv branch.
        primary_pose_dim:
            Number of parallel primary-capsule branches, i.e. the primary
            capsule pose-vector dimensionality.
        n_output_capsules:
            Number of output (digit-style) capsules.
        output_capsule_dim:
            Pose-vector dimensionality of each output capsule.
        """

        super().__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # First conv spans the full embedding "height" (like the reference
        # (28, 9)-height-spanning conv1) and slides only along the sequence
        # (width) axis.
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=(emb_size, 9), stride=1)
        conv1_len = seq_len - 9 + 1
        # Primary capsules slide a further width-9 conv along the same
        # sequence axis (height collapsed to 1 after conv1).
        self.primary_capsules = DeepRtPrimaryCapsules(
            conv_channels, primary_branch_channels, primary_pose_dim
        )
        primary_len = conv1_len - 9 + 1
        n_route_nodes = primary_len * primary_branch_channels
        self.routing_capsules = DeepRtRoutingCapsules(
            n_output_capsules, n_route_nodes, primary_pose_dim, output_capsule_dim
        )
        self.readout = nn.Linear(n_output_capsules * output_capsule_dim, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        """Predict a scalar retention time from a peptide token sequence.

        Parameters
        ----------
        tokens:
            Integer amino-acid token ids, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Predicted retention time, shape ``(batch, 1)``.
        """

        emb = self.embedding(tokens)  # (batch, seq_len, emb_size)
        x = emb.transpose(1, 2).unsqueeze(1)  # (batch, 1, emb_size, seq_len)
        x = F.relu(self.conv1(x))  # (batch, C, 1, conv1_len)
        primary = self.primary_capsules(x)  # (batch, n_route_nodes, primary_dim)
        out_capsules = self.routing_capsules(primary)  # (batch, n_out_capsules, out_dim)
        flat = out_capsules.reshape(out_capsules.shape[0], -1)
        return self.readout(flat)


def build_deeprt() -> nn.Module:
    """Build a compact random-init DeepRT (DeepRTplus) model."""

    return DeepRt(
        vocab_size=22,
        emb_size=8,
        seq_len=30,
        conv_channels=16,
        primary_branch_channels=4,
        primary_pose_dim=8,
        n_output_capsules=4,
        output_capsule_dim=6,
    ).eval()


def example_input_deeprt() -> Tensor:
    """Return a batch of tokenized peptide sequences for DeepRT."""

    return torch.randint(0, 22, (2, 30))


# ---------------------------------------------------------------------------
# DeepSequence -- variational autoencoder with a low-rank patterned decoder
# for protein variant effect prediction
# ---------------------------------------------------------------------------


class DeepSequenceVAE(nn.Module):
    """VAE over one-hot MSA columns with a low-rank patterned decoder.

    The decoder's final layer is factorized as a small dictionary of
    ``n_patterns`` per-position amino-acid profiles, mirroring the sparse
    "SVI" decoder weight structure in the reference Theano implementation
    (see ``DeepSequence/model.py``): rather than one huge
    ``hidden -> (L * alphabet)`` matrix, the model produces
    ``n_patterns`` independent ``(hidden -> L * alphabet)`` maps and sums
    them (a per-position, per-pattern combination), which is what
    ``decoder_architecture[-1] *= n_patterns`` implements upstream of a
    linear read-out.
    """

    def __init__(
        self,
        seq_len: int = 25,
        alphabet_size: int = 21,
        encoder_hidden: tuple[int, int] = (32, 32),
        decoder_hidden: int = 16,
        n_latent: int = 4,
        n_patterns: int = 4,
    ) -> None:
        """Build the MLP encoder and low-rank patterned MLP decoder.

        Parameters
        ----------
        seq_len:
            Number of alignment columns.
        alphabet_size:
            Number of amino-acid symbols (plus gap).
        encoder_hidden:
            Widths of the two encoder hidden layers.
        decoder_hidden:
            Width of the decoder's penultimate hidden layer.
        n_latent:
            Number of latent dimensions.
        n_patterns:
            Number of low-rank decoder "patterns" combined per position.
        """

        super().__init__()
        self.seq_len = seq_len
        self.alphabet_size = alphabet_size
        self.n_patterns = n_patterns
        in_dim = seq_len * alphabet_size

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, encoder_hidden[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden[0], encoder_hidden[1]),
            nn.ReLU(),
        )
        self.to_mu = nn.Linear(encoder_hidden[1], n_latent)
        self.to_logsigma = nn.Linear(encoder_hidden[1], n_latent)

        self.decoder_hidden = nn.Sequential(
            nn.Linear(n_latent, decoder_hidden),
            nn.ReLU(),
        )
        # Low-rank patterned final layer: one (hidden -> L*alphabet) map per
        # pattern, later reduced by summation across patterns.
        self.pattern_weight = nn.Linear(decoder_hidden, n_patterns * seq_len * alphabet_size)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a batch of one-hot MSA rows to a diagonal Gaussian.

        Parameters
        ----------
        x:
            One-hot alignment rows, shape ``(batch, seq_len, alphabet_size)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(mu, logsigma)`` of the approximate posterior.
        """

        flat = x.reshape(x.shape[0], -1)
        hidden = self.encoder(flat)
        return self.to_mu(hidden), self.to_logsigma(hidden)

    def decode(self, z: Tensor) -> Tensor:
        """Decode a latent sample to per-position amino-acid logits.

        Parameters
        ----------
        z:
            Latent sample, shape ``(batch, n_latent)``.

        Returns
        -------
        Tensor
            Per-position amino-acid log-probabilities, shape
            ``(batch, seq_len, alphabet_size)``.
        """

        hidden = self.decoder_hidden(z)
        patterned = self.pattern_weight(hidden)
        patterned = patterned.reshape(z.shape[0], self.n_patterns, self.seq_len, self.alphabet_size)
        logits = patterned.sum(dim=1)
        return F.log_softmax(logits, dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the encode -> reparameterize -> decode VAE pipeline.

        Parameters
        ----------
        x:
            One-hot alignment rows, shape ``(batch, seq_len, alphabet_size)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(log_probs, mu, logsigma)``.
        """

        mu, logsigma = self.encode(x)
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        log_probs = self.decode(z)
        return log_probs, mu, logsigma


def build_deepsequence() -> nn.Module:
    """Build a compact random-init DeepSequence VAE model."""

    return DeepSequenceVAE(
        seq_len=25,
        alphabet_size=21,
        encoder_hidden=(32, 32),
        decoder_hidden=16,
        n_latent=4,
        n_patterns=4,
    ).eval()


def example_input_deepsequence() -> Tensor:
    """Return a batch of one-hot MSA rows for DeepSequence."""

    idx = torch.randint(0, 21, (2, 25))
    return F.one_hot(idx, num_classes=21).float()


# ---------------------------------------------------------------------------
# DeepSignal -- joint BiLSTM (base/event) + Inception-CNN (signal) model for
# nanopore DNA methylation calling
# ---------------------------------------------------------------------------


class DeepSignalInceptionBlock(nn.Module):
    """One Inception-style multi-branch 1D conv block over the signal axis."""

    def __init__(self, in_channels: int, times: int = 4) -> None:
        """Build the four parallel branches.

        Parameters
        ----------
        in_channels:
            Input channel count.
        times:
            Base channel-width multiplier for each branch (mirrors the
            reference ``times`` parameter).
        """

        super().__init__()
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, times * 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(times * 3),
            nn.ReLU(),
        )
        self.branch_1x1 = nn.Sequential(
            nn.Conv1d(in_channels, times * 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(times * 3),
            nn.ReLU(),
        )
        self.branch_1x3 = nn.Sequential(
            nn.Conv1d(in_channels, times * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(times * 2),
            nn.ReLU(),
            nn.Conv1d(times * 2, times * 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(times * 3),
            nn.ReLU(),
        )
        self.branch_1x5 = nn.Sequential(
            nn.Conv1d(in_channels, times * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(times * 2),
            nn.ReLU(),
            nn.Conv1d(times * 2, times * 3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(times * 3),
            nn.ReLU(),
        )
        self.out_channels = times * 3 * 4

    def forward(self, x: Tensor) -> Tensor:
        """Run all four branches and concatenate along the channel axis.

        Parameters
        ----------
        x:
            Input signal features, shape ``(batch, in_channels, length)``.

        Returns
        -------
        Tensor
            Concatenated branch features, shape ``(batch, out_channels, length)``.
        """

        outs = [self.branch_pool(x), self.branch_1x1(x), self.branch_1x3(x), self.branch_1x5(x)]
        return torch.cat(outs, dim=1)


class DeepSignal(nn.Module):
    """Joint base (BiLSTM) + signal (Inception-CNN) methylation classifier."""

    def __init__(
        self,
        base_num: int = 17,
        signal_num: int = 120,
        vocab_size: int = 16,
        emb_size: int = 8,
        rnn_hidden: int = 32,
        rnn_layers: int = 2,
        n_classes: int = 2,
    ) -> None:
        """Build the event (BiLSTM) branch, signal (Inception) branch, and joint head.

        Parameters
        ----------
        base_num:
            Number of bases (k-mer positions) in the event branch input.
        signal_num:
            Number of raw signal samples across the whole read window.
        vocab_size:
            Number of distinct base tokens.
        emb_size:
            Base-token embedding width.
        rnn_hidden:
            Hidden width of each LSTM direction.
        rnn_layers:
            Number of stacked BiLSTM layers.
        n_classes:
            Number of methylation-call classes.
        """

        super().__init__()
        self.base_embedding = nn.Embedding(vocab_size, emb_size)
        self.event_rnn = nn.LSTM(
            emb_size + 3,
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.signal_stem = nn.Conv1d(1, 16, kernel_size=1)
        self.signal_inception = DeepSignalInceptionBlock(16, times=4)
        self.signal_pool = nn.AdaptiveAvgPool1d(1)

        joint_dim = 2 * rnn_hidden + self.signal_inception.out_channels
        self.joint_mlp = nn.Sequential(
            nn.Linear(joint_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, base_ids: Tensor, base_stats: Tensor, signal: Tensor) -> Tensor:
        """Predict methylation-call logits from base and raw-signal inputs.

        Parameters
        ----------
        base_ids:
            Integer base-token ids, shape ``(batch, base_num)``.
        base_stats:
            Per-base mean/std/count statistics, shape
            ``(batch, base_num, 3)``.
        signal:
            Raw per-read nanopore signal, shape ``(batch, 1, signal_num)``.

        Returns
        -------
        Tensor
            Methylation-call logits, shape ``(batch, n_classes)``.
        """

        base_emb = self.base_embedding(base_ids)
        event_in = torch.cat((base_emb, base_stats), dim=-1)
        _, (h_n, _) = self.event_rnn(event_in)
        event_repr = torch.cat((h_n[-2], h_n[-1]), dim=-1)

        signal_feat = self.signal_stem(signal)
        signal_feat = self.signal_inception(signal_feat)
        signal_repr = self.signal_pool(signal_feat).squeeze(-1)

        joined = torch.cat((event_repr, signal_repr), dim=-1)
        return self.joint_mlp(joined)


def build_deepsignal() -> nn.Module:
    """Build a compact random-init DeepSignal model."""

    return DeepSignal(
        base_num=17,
        signal_num=120,
        vocab_size=16,
        emb_size=8,
        rnn_hidden=32,
        rnn_layers=2,
        n_classes=2,
    ).eval()


def example_input_deepsignal() -> tuple[Tensor, Tensor, Tensor]:
    """Return (base_ids, base_stats, signal) tensors for DeepSignal."""

    base_ids = torch.randint(0, 16, (2, 17))
    base_stats = torch.randn(2, 17, 3)
    signal = torch.randn(2, 1, 120)
    return base_ids, base_stats, signal


MENAGERIE_ENTRIES = [
    ("DeepPINK", "build_deeppink", "example_input_deeppink", "2018", "BIO"),
    ("DeepPPISP", "build_deepppisp", "example_input_deepppisp", "2020", "BIO"),
    (
        "DeepPurpose encoders",
        "build_deeppurpose_encoders",
        "example_input_deeppurpose_encoders",
        "2020",
        "BIO",
    ),
    ("DeepRT", "build_deeprt", "example_input_deeprt", "2017", "BIO"),
    ("DeepSequence", "build_deepsequence", "example_input_deepsequence", "2018", "BIO"),
    ("DeepSignal", "build_deepsignal", "example_input_deepsignal", "2019", "BIO"),
]

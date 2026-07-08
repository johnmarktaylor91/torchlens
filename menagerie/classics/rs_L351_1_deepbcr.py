# FAITHFUL PORT of liulab/deepbcr @ bitbucket.org/liulab/deepbcr (default branch,
# cloned 2026-07-01; original framework: TensorFlow 1.5.0 / tf.contrib era)
"""DeepBCR: peptide encoding network (PEN) for B-cell-receptor repertoires
(Hu & Liu, "DeepBCR: Deep learning framework for cancer-type classification
and binding affinity estimation using B cell receptor repertoires",
bioRxiv 731158, 2019). Paper's "Code availability" link points to
https://bitbucket.org/liulab/deepbcr (not GitHub) -- real source, not a
paper-only reimplementation.

The original `src/deep_bcr.py` is written against TensorFlow 1.5 / tf.contrib
(``tf.placeholder``, ``tf.Session``, ``tf.get_variable``, ``tf.contrib.layers
.variance_scaling_initializer``) and cannot run in a modern base torch/TF2
env, so this is a FAITHFUL PORT (rung 3), not a vendor. Every mechanism in
the real ``GeneSwitchModelFast.hidden_layers`` + ``DeepBCR.logits`` forward
pass (the "Peptide Encoding Network" shown in the paper's Fig. 1a: amino-acid
encoding layer -> k-mer motif layer -> immunoglobulin-isotype (gene-switch)
layer -> max-pooling over k-mers -> motif layer -> output layer) is
transcribed layer-for-layer into torch, including the isotype-count masking
via ``sign(counts)`` and the max-pool-over-instances trick used to make the
"multi-instance repertoire" bag permutation-invariant. Only the TF-Session
plumbing (placeholders, savers, summaries) and the non-classification run
modes (Linear/Multiple-Linear/Cox-PH regression, only used for the
survival-analysis side experiments, not the flagship cancer-type
classification task in the paper) are dropped; the classification path
(``run_mode='Classification'``, the model actually evaluated in Fig. 1b/1c)
is preserved in full.

Original real code (bitbucket.org/liulab/deepbcr, src/deep_bcr.py):

    AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'

    class GeneSwitchModelFast(GeneSwitchModel):
        def hidden_layers(self, features, counts, keep_prob):
            # features: (B, M, K) int amino-acid indices, M = max k-mers/sample
            # counts:   (B, M, C) isotype gene counts (C = # Ig isotype groups)
            with tf.name_scope('pre-process'):
                counts = tf.reshape(counts, [batch_size*max_kmer, gene_num])
                valid = tf.reduce_sum(counts, axis=1) > 0
                scores = tf.reshape(tf.to_int32(features), [batch_size*max_kmer, kmer_size])
                scores = tf.boolean_mask(scores, valid)
            with tf.name_scope('encoding_layer'):
                scores = tf.gather(self.weights0, tf.reshape(scores, [-1]))  # (B*M*K, E)
            with tf.name_scope('kmer_layer'):
                scores = tf.matmul(tf.reshape(scores, [-1, K*E]), W1) + b1
                scores = tf.nn.relu(scores)                                  # (B*M, N)
            with tf.name_scope('gene_layer'):
                scores = tf.matmul(scores[:,:,None], tf.sign(counts[:,None,:]))  # (B*M, N, C)
            with tf.name_scope('max_pooling'):
                scores = tf.segment_max(scores, batch_index)                 # (B, N, C)
            with tf.name_scope('dropout_layer'):
                scores = tf.nn.dropout(scores, keep_prob)
            with tf.name_scope('motif_layer'):
                scores = tf.reduce_sum(scores * W2, axis=2) + b2
                scores = tf.nn.relu(scores)                                  # (B, N)
            with tf.name_scope('dropout_layer'):
                scores = tf.nn.dropout(scores, keep_prob)
            return scores

    class DeepBCR(GeneSwitchModelFast):
        def logits(self, features, counts, keep_prob):
            scores = self.hidden_layers(features, counts, keep_prob)
            # run_mode == 'Classification':
            scores = tf.matmul(scores, W3) + b3                             # (B, num_labels)
            return scores

Real hyperparameters used against the actual TCGA BCR repertoires
(``src/tcga_bcr.py``): k-mer size K=6, isotype groups C=8 (IGHM|IGHD, IGHG1,
IGHG2/4, IGHG3, IGHA1/2, IGK, IGL, Others -- see ``get_isotype_map()``),
default ``num_motifs`` (N) = 30, ``num_labels`` = 11 TCGA cancer types
(Fig. 1). Toy sizes below keep the same shapes/mechanisms at trace scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"


class DeepBCR(nn.Module):
    """Torch port of the real ``DeepBCR`` (``GeneSwitchModelFast`` subclass,
    run_mode='Classification') peptide-encoding-network forward pass.

    Input is a repertoire "bag": for each sample, up to ``max_kmer`` k-mers
    (each of length ``kmer_size``, amino-acid-index encoded) together with a
    ``(max_kmer, gene_num)`` isotype-count matrix indicating which Ig
    constant-gene group each k-mer's parent BCR read was observed with (zero
    row = padding). The network is a *multi-instance* max-pooling model:
    every k-mer is scored independently, entries are masked into isotype
    channels by (sign of) isotype count, and the bag is collapsed with a
    max over the k-mer axis -- mirroring ``tf.segment_max`` in the original.
    """

    def __init__(self, num_motifs=30, kmer_size=6, gene_num=8, num_labels=11, encode_size=None):
        super().__init__()
        amino_acid = len(AA_LIST)
        encode_size = encode_size if encode_size is not None else amino_acid

        self.num_motifs = num_motifs
        self.kmer_size = kmer_size
        self.gene_num = gene_num
        self.num_labels = num_labels
        self.encode_size = encode_size

        # layer0: amino-acid encoding layer (real code: self.weights0, a
        # (amino_acid, encode_size) embedding gathered per amino-acid index)
        self.weights0 = nn.Parameter(torch.empty(amino_acid, encode_size))

        # layer1: k-mer motif layer (real code: self.weights1/self.biases1)
        self.weights1 = nn.Parameter(torch.empty(kmer_size * encode_size, num_motifs))
        self.biases1 = nn.Parameter(torch.full((num_motifs,), 0.1))

        # layer motif: immunoglobulin isotype ("gene switch") layer
        # (real code: self.weights2/self.biases2, shape (num_motifs, gene_num))
        self.weights2 = nn.Parameter(torch.empty(num_motifs, gene_num))
        self.biases2 = nn.Parameter(torch.full((num_motifs,), 0.1))

        # output layer (real code: self.weights3/self.biases3,
        # run_mode == 'Classification' branch)
        self.weights3 = nn.Parameter(torch.empty(num_motifs, num_labels))
        self.biases3 = nn.Parameter(torch.full((num_labels,), 0.1))

        self.dropout = nn.Dropout(p=0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        # Real code: tf.contrib.layers.variance_scaling_initializer(
        #   factor=1.0, mode='FAN_IN', uniform=True) == torch's default
        # kaiming_uniform_ family (fan_in, uniform); use it for parity.
        for w in (self.weights0, self.weights1, self.weights2, self.weights3):
            nn.init.kaiming_uniform_(w, a=1.0, mode="fan_in", nonlinearity="linear")

    def hidden_layers(self, features, counts):
        """
        Args:
            features: (B, M, K) long amino-acid-index k-mer matrix.
            counts:   (B, M, C) isotype gene-count matrix (0 rows == padding).
        Returns:
            (B, N) pooled repertoire-level motif scores.
        """
        batch_size, max_kmer, kmer_size = features.shape
        gene_num = counts.shape[-1]

        # pre-process: flatten to (B*M, ...); mask out fully-zero (padding) rows
        counts_flat = counts.reshape(batch_size * max_kmer, gene_num)
        valid = counts_flat.sum(dim=1) > 0
        feats_flat = features.reshape(batch_size * max_kmer, kmer_size).long()

        # batch index per (flattened) k-mer instance, needed for the
        # segment-max pooling step below (mirrors tf.segment_max's idx)
        batch_idx = torch.arange(batch_size, device=features.device).repeat_interleave(max_kmer)

        feats_valid = feats_flat[valid]
        counts_valid = counts_flat[valid]
        batch_idx_valid = batch_idx[valid]

        # encoding_layer: amino-acid embedding gather -> (n_valid*K, E)
        scores = self.weights0[feats_valid.reshape(-1)]

        # kmer_layer: (n_valid, K*E) @ (K*E, N) + b1 -> ReLU -> (n_valid, N)
        scores = scores.reshape(-1, kmer_size * self.encode_size)
        scores = torch.matmul(scores, self.weights1) + self.biases1
        scores = F.relu(scores)

        # gene_layer: outer product with sign(isotype counts) -> (n_valid, N, C)
        gene_sign = torch.sign(counts_valid.to(scores.dtype))
        scores = scores.unsqueeze(2) * gene_sign.unsqueeze(1)

        # max_pooling: segment-max over k-mer instances within each sample,
        # per (motif, isotype-channel) cell -> (B, N, C). Padding samples
        # with no valid k-mers get an all -inf pool -> reset to 0 below,
        # matching the original's implicit "no evidence -> no signal".
        pooled = scores.new_full((batch_size, self.num_motifs, gene_num), float("-inf"))
        if scores.numel() > 0:
            idx = batch_idx_valid.view(-1, 1, 1).expand_as(scores)
            pooled = pooled.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)

        # dropout_layer
        pooled = self.dropout(pooled)

        # motif_layer: weighted sum across isotype channels -> ReLU -> (B, N)
        motif = (pooled * self.weights2.unsqueeze(0)).sum(dim=2) + self.biases2
        motif = F.relu(motif)

        # dropout_layer
        motif = self.dropout(motif)
        return motif

    def forward(self, features, counts):
        scores = self.hidden_layers(features, counts)
        # output_layer, run_mode == 'Classification'
        logits = torch.matmul(scores, self.weights3) + self.biases3
        return logits


def build_deepbcr():
    # Toy scale of the real TCGA-run hyperparameters (kmer_size=6, gene_num=8
    # isotype groups, num_motifs=30 default, num_labels=11 TCGA cancer types
    # in the paper) shrunk for a fast trace.
    return DeepBCR(num_motifs=8, kmer_size=6, gene_num=8, num_labels=11, encode_size=8)


def example_input_deepbcr():
    batch, max_kmer, kmer_size, gene_num = 3, 5, 6, 8
    aa_size = len(AA_LIST)
    features = torch.randint(0, aa_size, (batch, max_kmer, kmer_size))
    # random isotype-count rows; zero out a couple to exercise the padding path
    counts = torch.randint(0, 3, (batch, max_kmer, gene_num)).float()
    counts[0, -1] = 0
    counts[-1, -2:] = 0
    return (features, counts)


MENAGERIE_ENTRIES = [
    ("DeepBCR", "build_deepbcr", "example_input_deepbcr", 2019, "ported"),
]

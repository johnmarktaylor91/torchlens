# FAITHFUL PORT of https://github.com/harrylclc/LTR-DG @ master (original framework: TensorFlow 1.x)
#   Ported file: code/gan/DecompAtt.py :: DecompAtt (the shared base class subclassed by both
#   Generator.py and Discriminator.py in the repo's GAN-based distractor-ranking pipeline).
#
#   LTR-DG (Liang et al. 2018, "Distractor Generation for Multiple Choice Questions Using
#   Learning to Rank") ranks candidate multiple-choice distractors against the question+answer
#   pair using a Decomposable Attention model for NLI (Parikh et al. 2016,
#   http://arxiv.org/pdf/1606.01933v1.pdf): question/answer/distractor token sequences are
#   embedded, cross-attended via a bilinear soft-alignment (premise<->hypothesis attention
#   weights from a feedforward-projected dot product), the aligned representations are
#   concatenated with the raw embeddings and fed through a second feedforward network,
#   aggregated by summation over the sequence dim, and finally scored through a feedforward
#   classifier + a cosine-similarity auxiliary signal, combined into one logit.
#
#   The original code is graph-mode TF1 (`tf.placeholder`, `tf.get_variable`,
#   `tf.contrib.layers.l2_regularizer`, `tf.variable_scope(..., reuse=True)`) and cannot run in
#   this base env (no legacy TF1 / tf.contrib available). The architecture is fully specified
#   in the original file, so it is transcribed faithfully below:
#     - `feedforward_3d` (two size-preserving 1x1-conv "linear" layers with ReLU, implemented
#       here as two size-preserving nn.Linear layers applied per-token -- a 1x1 conv over the
#       sequence dim is algebraically the same per-token linear projection as an nn.Linear
#       applied position-wise, which is how PyTorch implementations of this exact model
#       conventionally realize the TF `feedforward_3d` conv trick) with the same weight-shared
#       "F"/"G" scope pattern (shared weights are reused across premise+hypothesis calls, and
#       between answer/hypothesis and distractor/negative branches, exactly mirroring
#       `tf.variable_scope(..., reuse=True)`)
#     - bilinear dot-product cross-attention + softmax normalization along each sequence axis
#       (`hypothesis_softmax`, `premise_softmax`)
#     - concatenation of raw embeddings with attended counterparts (`v1`, `v2`), a second
#       feedforward network (`G`), sum-pooling aggregation, then `final_representation`
#       feedforward + linear scoring head (`output_w`/`output_b`) combined with a cosine
#       similarity auxiliary term (`logits2`) via a final learned linear combination
#       (`combine_w`/`combine_b`)
#   Batch-normalization layers present in the original (`tf.layers.batch_normalization`) are
#   kept as `nn.BatchNorm1d` over the feature dim, matching the original's per-feature
#   normalization semantics.
#
#   Traced entry point here is the classification/pretraining branch of the model (the
#   "premise vs. hypothesis(=distractor)" scoring path shared by Generator and Discriminator);
#   the reinforcement-learning REINFORCE loss terms (`gan_loss`, `reward`) that Generator.py and
#   Discriminator.py add on top are training-time-only auxiliary losses, not part of the
#   DecompAtt architecture itself.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class FeedForward3D(nn.Module):
    """Port of DecompAtt.feedforward_3d: two size-preserving per-token linear+ReLU layers
    (implemented in the original as 1x1 conv2d over the [batch, length, 1, dim] tensor,
    which is algebraically a position-wise linear projection -- the standard PyTorch
    equivalent of a TF 1x1-conv "feedforward_3d" trick)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim)
        self.w2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, length, in_dim]
        features = F.relu(self.w1(x))
        features = F.relu(self.w2(features))
        return features


class FeedForward2D(nn.Module):
    """Port of DecompAtt.feedforward_2d: two square hidden linear+ReLU layers."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden1 = F.relu(self.hidden1(x))
        gate_output = F.relu(self.hidden2(hidden1))
        return gate_output


class DecompAtt(nn.Module):
    """Port of code/gan/DecompAtt.py :: DecompAtt (Decomposable Attention model for NLI,
    Parikh et al. 2016), adapted from TF1 graph-mode placeholders to a traceable
    torch.nn.Module taking tensor inputs directly."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        sequence_length_q: int,
        sequence_length_a: int,
        dropout_keep_prob: float = 1.0,
    ):
        super().__init__()
        assert hidden_size == embedding_size, "size should be the same"
        self.sequence_length_q = sequence_length_q
        self.sequence_length_a = sequence_length_a
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_keep_prob = dropout_keep_prob

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # "F" scope: shared feedforward for premise and hypothesis
        self.F = FeedForward3D(embedding_size, embedding_size)
        self.F_bn = nn.BatchNorm1d(embedding_size)

        # "G" scope: shared feedforward for v1/v2 (concat of raw + attended embeddings)
        self.G = FeedForward3D(embedding_size * 2, hidden_size)
        self.G_bn = nn.BatchNorm1d(hidden_size)

        self.final_representation = FeedForward2D(hidden_size * 2)
        self.final_bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(p=1.0 - dropout_keep_prob)

        self.output_proj = nn.Linear(hidden_size * 2, 1)
        self.combine_proj = nn.Linear(2, 1)

    @staticmethod
    def _apply_bn(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, length, dim] -> BatchNorm1d expects [batch, dim, length]
        return bn(x.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def _cosine(q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pooled_len_1 = torch.sqrt(torch.sum(q * q, dim=1))
        pooled_len_2 = torch.sqrt(torch.sum(a * a, dim=1))
        pooled_mul_12 = torch.sum(q * a, dim=1)
        return pooled_mul_12 / (pooled_len_1 * pooled_len_2 + 1e-8)

    def forward(
        self, input_x_1: torch.Tensor, input_x_2: torch.Tensor, input_x_3: torch.Tensor
    ) -> torch.Tensor:
        # input_x_1: question   [batch, seq_q]
        # input_x_2: answer     [batch, seq_a]
        # input_x_3: distractor/hypothesis candidate  [batch, seq_a]
        premise_inputs = self.embedding(input_x_1)  # [batch, seq_q, d]
        answer_inputs = self.embedding(input_x_2)  # [batch, seq_a, d]
        hypothesis_inputs = self.embedding(input_x_3)  # [batch, seq_a, d]
        answer_inputs_sum = torch.sum(answer_inputs, dim=1)
        hypothesis_inputs_sum = torch.sum(hypothesis_inputs, dim=1)

        # shared "F" feedforward, applied to premise and hypothesis (weight-tied)
        premise_F = self._apply_bn(self.F_bn, self.F(premise_inputs))
        hypothesis_F = self._apply_bn(self.F_bn, self.F(hypothesis_inputs))

        # cross-attention: normalize along sequence_length_a (for premise) / sequence_length_q (for hypothesis)
        dot1 = torch.matmul(premise_F, hypothesis_F.transpose(1, 2))  # [batch, seq_q, seq_a]
        hypothesis_softmax = F.softmax(dot1, dim=-1)  # normalize along seq_a
        dot2 = dot1.transpose(1, 2)  # [batch, seq_a, seq_q]
        premise_softmax = F.softmax(dot2, dim=-1)  # normalize along seq_q

        betas = torch.matmul(hypothesis_softmax, hypothesis_inputs)  # [batch, seq_q, d]
        alphas = torch.matmul(premise_softmax, premise_inputs)  # [batch, seq_a, d]

        v1 = torch.cat([premise_inputs, betas], dim=2)  # [batch, seq_q, 2d]
        v2 = torch.cat([hypothesis_inputs, alphas], dim=2)  # [batch, seq_a, 2d]

        # shared "G" feedforward, applied to v1 and v2 (weight-tied)
        v1 = self._apply_bn(self.G_bn, self.G(v1))
        v2 = self._apply_bn(self.G_bn, self.G(v2))

        v1_sum = torch.sum(v1, dim=1)  # [batch, hidden]
        v2_sum = torch.sum(v2, dim=1)  # [batch, hidden]
        v = torch.cat([v1_sum, v2_sum], dim=1)  # [batch, 2*hidden]

        final_representation = self.final_bn(self.final_representation(v))
        final_representation = self.dropout(final_representation)

        logits1 = self.output_proj(final_representation)  # [batch, 1]
        logits2 = self._cosine(answer_inputs_sum, hypothesis_inputs_sum).unsqueeze(1)  # [batch, 1]

        logits = self.combine_proj(torch.cat([logits1, logits2], dim=1)).squeeze(-1)  # [batch]
        score = torch.sigmoid(logits)
        return score


_VOCAB_SIZE = 32
_EMBED_DIM = 16
_SEQ_LEN_Q = 6
_SEQ_LEN_A = 6


def build_ltr_dg_decompatt():
    model = DecompAtt(
        vocab_size=_VOCAB_SIZE,
        embedding_size=_EMBED_DIM,
        hidden_size=_EMBED_DIM,
        sequence_length_q=_SEQ_LEN_Q,
        sequence_length_a=_SEQ_LEN_A,
        dropout_keep_prob=1.0,
    )
    model.eval()
    return model


def example_input_ltr_dg_decompatt():
    batch = 2  # BatchNorm1d in eval mode still needs a defined running-stat shape; batch>1 keeps torch's BN happy in eval
    input_x_1 = torch.randint(0, _VOCAB_SIZE, (batch, _SEQ_LEN_Q))
    input_x_2 = torch.randint(0, _VOCAB_SIZE, (batch, _SEQ_LEN_A))
    input_x_3 = torch.randint(0, _VOCAB_SIZE, (batch, _SEQ_LEN_A))
    return (input_x_1, input_x_2, input_x_3)


MENAGERIE_ENTRIES = [
    (
        "MCQ Distractor Generation (DG-Net)",
        build_ltr_dg_decompatt,
        example_input_ltr_dg_decompatt,
        2018,
        "MENAGERIE_ZOO",
    ),
]

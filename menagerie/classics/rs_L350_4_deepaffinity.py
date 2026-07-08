# FAITHFUL PORT of Shen-Lab/DeepAffinity @ debca4c9f01991a37594e1b67f3dbb46a29e82d1 (original framework: TensorFlow 1.x + tflearn)
# https://raw.githubusercontent.com/Shen-Lab/DeepAffinity/debca4c9f01991a37594e1b67f3dbb46a29e82d1/Joint_models/joint_attention/joint_warm_start/joint-Model.py
#
# Karimi, Wu, Wang & Shen (Bioinformatics 2019), "DeepAffinity: interpretable
# deep learning of compound-protein affinity through unified recurrent and
# convolutional neural networks". The repo's real model-construction code
# (`Joint_models/joint_attention/joint_warm_start/joint-Model.py`) is
# genuine, runnable TF1.x + `tflearn` (`tflearn.gru`, `tflearn.embedding`,
# `tf.tensordot`/`tf.slice` graph ops) -- TF1's `tflearn` API is long EOL
# and not a base lib here, so this is a faithful architectural PORT, not a
# vendor. The seq2seq sub-directories in the same repo are separate
# autoencoder-pretraining code (uses `tensorflow.contrib`, even more
# unusable); the joint-attention model below is the paper's headline
# unified predictor and is transcribed directly from its real script.
#
# Every mechanism below is a direct transcription of the real script
# (protein/compound branch construction, lines ~316-433 of the source),
# not a paraphrase from the paper text:
#   - Protein branch: token embedding (`vocab_size_protein=76`, embed dim
#     == `GRU_size_prot=256`) -> 2 stacked GRU layers (`tflearn.gru`, both
#     width 256, `return_seq=True`), each layer's warm-started initial
#     state (an unrelated pretraining detail, dropped here -- default
#     zero init is used since only the architecture is ported).
#   - Compound/drug branch: token embedding (`vocab_size_compound=68`,
#     embed dim == `GRU_size_drug=128`) -> 2 stacked GRU layers (width
#     128, `return_seq=True`).
#   - Pairwise attention (`alphas_pair`): a learned linear map `W` project
#     the protein GRU output (`GRU_size_prot` -> `GRU_size_drug`) via
#     `tf.tensordot`, then for every batch element the transformed protein
#     sequence and the drug sequence are matched via `tanh(V @ drug^T + b)`
#     (`b` is a learned `[protein_MAX_size, comp_MAX_size]` bias -- the
#     original script computes this per-example in a Python for-loop over
#     `batch_size`, equivalent to a single batched
#     `torch.tanh(torch.bmm(...) + b)`), softmax-normalized over the
#     flattened (protein_pos, drug_pos) grid to give `alphas_pair`.
#   - Context vector (`Attn`): two more learned projections `U_prot`
#     (`U_size=256` <- `GRU_size_prot`) and `U_drug` (`U_size=256` <-
#     `GRU_size_drug`) plus a learned bias `B` (`[U_size]`) combine the
#     projected protein states, projected drug states, and the attention
#     weights `alphas_pair` into a single `[batch, U_size]` context vector
#     via a weighted sum over drug positions (the script's `for i in
#     range(comp_MAX_size)` loop, vectorized here as a single weighted
#     `einsum`/matmul over the same broadcasting the loop computes).
#   - Reshape context to `[batch, U_size, 1]` (tflearn channels-last:
#     steps=U_size, channels=1) -> `Conv1d(1->64, kernel=4, stride=2,
#     padding='same')` + `LeakyReLU` -> `MaxPool1d(kernel=4, stride=4,
#     padding='same')` -> flatten (`pool_2 = reshape(pool_1, [-1,
#     64*32])`, matching `U_size=256` conv'd+pooled down to a
#     64*32=2048-dim vector: SAME padding gives conv out_len=128, pool
#     out_len=32) ->
#     `Linear(2048,600)+LeakyReLU+Dropout(0.8)` ->
#     `Linear(600,300)+LeakyReLU+Dropout(0.8)` -> `Linear(300,1)` linear
#     output (real script's `fully1`/`fully2`/`fully3`, `Dropout` rate
#     0.8 = keep_prob in tflearn convention -> `nn.Dropout(p=0.2)` in
#     torch convention (torch's `p` is the DROP probability, tflearn's
#     `dropout()` second arg is the KEEP probability) -- inert in eval
#     mode regardless, kept for architectural fidelity).
#
# Only mechanical staging/translation edits (TF1/tflearn -> torch, not
# architectural changes):
#   - The real script's Python for-loops over `range(batch_size)` and
#     `range(comp_MAX_size)` are graph-unrolling constructs specific to
#     TF1's static-batch-size graph mode; both are vectorized here into
#     equivalent batched tensor ops (`torch.bmm`/broadcasting) that compute
#     the identical mathematical quantities for an arbitrary batch size.
#   - Dropped the entire data-loading / vocabulary / warm-start-weight
#     pipeline (`prepare_data`, `read_labels`, `read_initial_state_weigths`,
#     the six `*_ic50`/`*_smile`/`*_sps` dataset loads, and the
#     `tf.global_variables()` weight-collection/checkpoint-restore blocks)
#     -- none of that is part of the model architecture itself.
#   - Added `build_deepaffinity_joint_attention()` /
#     `example_input_deepaffinity_joint_attention()` staging entry points
#     at the script's real default sizes (`comp_MAX_size=100`,
#     `protein_MAX_size=152`, `vocab_size_compound=68`,
#     `vocab_size_protein=76`, `GRU_size_prot=256`, `GRU_size_drug=128`,
#     `U_size=256`), batch=2 (script default `batch_size=64`, reduced for
#     a small trace; the module is batch-size-parametric).

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_same_1d(x, kernel_size, stride, pad_value=0.0):
    """Replicate tflearn/TF1's default 'same' padding for conv_1d/max_pool_1d.

    tflearn's `conv_1d`/`max_pool_1d` both default to `padding='same'`
    (TF's SAME padding: out_len = ceil(in_len / stride)), which torch's
    `nn.Conv1d`/`nn.MaxPool1d` do not support directly (they only take a
    fixed symmetric pad). This computes the equivalent explicit pad.
    """
    in_len = x.shape[-1]
    out_len = -(-in_len // stride)  # ceil division
    pad_total = max((out_len - 1) * stride + kernel_size - in_len, 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(x, (pad_left, pad_right), value=pad_value)


class DeepAffinityJointAttention(nn.Module):
    """DeepAffinity's joint-attention RNN-CNN compound-protein affinity model.

    Faithful port of the real `joint-Model.py` graph-construction script
    (protein/drug GRU encoders + pairwise attention + conv/FC regression
    head), vectorized from TF1's per-example for-loops into batched torch
    ops.
    """

    def __init__(
        self,
        vocab_size_protein=76,
        vocab_size_compound=68,
        protein_max_size=152,
        comp_max_size=100,
        gru_size_prot=256,
        gru_size_drug=128,
        u_size=256,
    ):
        super().__init__()
        self.protein_max_size = protein_max_size
        self.comp_max_size = comp_max_size
        self.gru_size_prot = gru_size_prot
        self.gru_size_drug = gru_size_drug
        self.u_size = u_size

        # Protein branch: embedding + 2-layer GRU (tflearn.embedding + 2x tflearn.gru).
        self.prot_embd = nn.Embedding(vocab_size_protein, gru_size_prot)
        self.prot_gru_1 = nn.GRU(gru_size_prot, gru_size_prot, batch_first=True)
        self.prot_gru_2 = nn.GRU(gru_size_prot, gru_size_prot, batch_first=True)

        # Drug/compound branch: embedding + 2-layer GRU.
        self.drug_embd = nn.Embedding(vocab_size_compound, gru_size_drug)
        self.drug_gru_1 = nn.GRU(gru_size_drug, gru_size_drug, batch_first=True)
        self.drug_gru_2 = nn.GRU(gru_size_drug, gru_size_drug, batch_first=True)

        # Pairwise attention parameters (real script's Attn_W_prot/Attn_b_prot).
        self.attn_w = nn.Parameter(torch.randn(gru_size_prot, gru_size_drug) * 0.1)
        self.attn_b = nn.Parameter(torch.randn(protein_max_size, comp_max_size) * 0.1)

        # Context-vector projection parameters (real script's Attn_U_prot/Attn_U_drug/Attn_B).
        self.u_prot = nn.Parameter(torch.randn(u_size, gru_size_prot) * 0.1)
        self.u_drug = nn.Parameter(torch.randn(u_size, gru_size_drug) * 0.1)
        self.attn_bias = nn.Parameter(torch.randn(u_size) * 0.1)

        # Conv + FC regression head (real script's conv1/pool1/fully1/fully2/fully3).
        # tflearn's Attn_reshape=[-1, U_size, 1] is (batch, steps=U_size,
        # channels=1) in tflearn's channels-last convention, so conv1 has
        # in_channels=1 and operates over a length-U_size sequence.
        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, stride=2)
        self.conv1_act = nn.LeakyReLU()
        # tflearn's conv_1d/max_pool_1d both default to 'same' padding;
        # SAME padding is applied manually in forward() since torch's
        # Conv1d/MaxPool1d only support fixed symmetric padding natively.
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.fully1 = nn.Linear(64 * 32, 600)
        self.fully1_act = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fully2 = nn.Linear(600, 300)
        self.fully2_act = nn.LeakyReLU()
        self.drop3 = nn.Dropout(0.2)
        self.fully3 = nn.Linear(300, 1)

    def forward(self, protein_tokens, drug_tokens):
        # protein_tokens: (batch, protein_max_size) int64 token ids.
        # drug_tokens: (batch, comp_max_size) int64 token ids.
        prot_embd = self.prot_embd(protein_tokens)
        prot_gru_1, _ = self.prot_gru_1(prot_embd)
        prot_gru_2, _ = self.prot_gru_2(prot_gru_1)  # (batch, protein_max_size, gru_size_prot)

        drug_embd = self.drug_embd(drug_tokens)
        drug_gru_1, _ = self.drug_gru_1(drug_embd)
        drug_gru_2, _ = self.drug_gru_2(drug_gru_1)  # (batch, comp_max_size, gru_size_drug)

        # V = prot_gru_2 @ W  -- real script's tf.tensordot(prot_gru_2, W, axes=[[2],[0]]).
        v = torch.matmul(prot_gru_2, self.attn_w)  # (batch, protein_max_size, gru_size_drug)

        # Real script's per-example for-loop:
        #   temp[i] = tanh(V[i] @ drug_gru_2[i]^T + b)
        # vectorized over the batch with bmm.
        vu = torch.tanh(torch.bmm(v, drug_gru_2.transpose(1, 2)) + self.attn_b)
        vu_flat = vu.reshape(vu.shape[0], -1)
        alphas_flat = torch.softmax(vu_flat, dim=-1)
        alphas_pair = alphas_flat.reshape(-1, self.protein_max_size, self.comp_max_size)

        # prod_drug/prod_prot -- real script's tf.tensordot projections into U_size space.
        prod_drug = torch.matmul(drug_gru_2, self.u_drug.t())  # (batch, comp_max_size, u_size)
        prod_prot = torch.matmul(prot_gru_2, self.u_prot.t())  # (batch, protein_max_size, u_size)

        # Real script's `for i in range(comp_MAX_size)` loop accumulating:
        #   Attn += sum_over_protein_pos(alphas_pair[:, :, i:i+1] * (prod_drug[:, i:i+1, :] + prod_prot + B))
        # vectorized: broadcast prod_drug over protein positions, add prod_prot + bias,
        # weight by alphas_pair, and sum over both comp and protein positions at once.
        combined = (
            prod_drug.unsqueeze(1) + prod_prot.unsqueeze(2) + self.attn_bias
        )  # (batch, protein_max_size, comp_max_size, u_size)
        weighted = alphas_pair.unsqueeze(-1) * combined
        attn = weighted.sum(dim=(1, 2))  # (batch, u_size)

        # Conv/pool/FC regression head. tflearn's Attn_reshape=[-1, U_size, 1]
        # (batch, steps=u_size, channels=1) -> torch channels-first
        # (batch, channels=1, steps=u_size).
        x = attn.unsqueeze(1)  # (batch, 1, u_size)
        # SAME padding for conv1 (kernel=4, stride=2): out_len = ceil(in/stride).
        x = _pad_same_1d(x, kernel_size=4, stride=2)
        x = self.conv1(x)
        x = self.conv1_act(x)
        # SAME padding for pool1 (kernel=4, stride=4).
        x = _pad_same_1d(x, kernel_size=4, stride=4, pad_value=float("-inf"))
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fully1(x)
        x = self.fully1_act(x)
        x = self.drop2(x)
        x = self.fully2(x)
        x = self.fully2_act(x)
        x = self.drop3(x)
        x = self.fully3(x)
        return x


def build_deepaffinity_joint_attention():
    return DeepAffinityJointAttention()


def example_input_deepaffinity_joint_attention():
    # Real script defaults: protein_MAX_size=152, comp_MAX_size=100,
    # vocab_size_protein=76, vocab_size_compound=68. batch=2 (script uses 64).
    protein_tokens = torch.randint(0, 76, (2, 152))
    drug_tokens = torch.randint(0, 68, (2, 100))
    return protein_tokens, drug_tokens


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepAffinity-JointAttention",
        "build_deepaffinity_joint_attention",
        "example_input_deepaffinity_joint_attention",
        2019,
        "ported",
    ),
]

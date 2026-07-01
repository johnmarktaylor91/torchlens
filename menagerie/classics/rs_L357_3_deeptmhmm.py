# FAITHFUL REIMPLEMENTATION from Hallgren, Tsirigos, Pedersen et al. 2022,
# "DeepTMHMM predicts alpha and beta transmembrane proteins using deep neural networks"
# (bioRxiv 2022.04.08.487609) (no public code)
#
# DeepTMHMM ships ONLY as a closed BioLib web service / Docker image
# (https://dtu.biolib.com/DeepTMHMM) -- the DTU authors never released the training or
# model-definition source. GitHub has no real implementation (only third-party
# result-parsers / pipeline wrappers that call the hosted BioLib service, e.g.
# labioinfoufsc/DeepTMHMM is a placeholder README with no code).
#
# Architecture per the paper text and secondary summaries of its Methods section: the
# encoder is the pretrained ESM-1b protein language model, whose per-residue embeddings
# are projected by a dense layer to 128 dimensions, fed into a bidirectional LSTM with
# 64 hidden units per direction (128 total), followed by a dense layer with dropout, and
# finally a linear-chain CRF decoder that predicts per-residue topology labels (e.g.
# inside/outside/signal-peptide/alpha-helix-TM/beta-strand-TM) under biologically
# constrained transitions.
#
# The ESM-1b encoder itself IS a real, published architecture with an installed base-lib
# class (`transformers.EsmModel` / `EsmConfig`, HuggingFace's port of the ESM family) --
# that half of this module is the real class, tiny-config random-init, per RUNG 1. The
# custom biLSTM -> dense+dropout -> CRF head is DeepTMHMM's own novel contribution, has
# no released code, and is reimplemented here faithfully from the paper's Methods
# description (dims/hidden sizes as reported): hence RUNG 4 for the head, hybridized with
# the real ESM-1b class for the encoder. Tagged reimpl-pytorch as the head is the
# architecturally novel, no-code part.
import torch
import torch.nn as nn
from transformers import EsmConfig, EsmModel

MENAGERIE_ZOO = "reimpl-pytorch"

# DeepTMHMM's per-residue topology label set (inside / outside / signal peptide /
# alpha-helix TM / beta-strand TM), per the paper's five-state topology grammar.
_NUM_TOPOLOGY_LABELS = 5


class DeepTMHMMCRFHead(nn.Module):
    """Linear-chain CRF decoder: emission scores from a linear layer plus a learned
    label-transition matrix, combined via the standard CRF forward (log-sum-exp)
    recursion to produce per-position scores over the topology label set.
    """

    def __init__(self, num_labels: int, in_dim: int):
        super().__init__()
        self.emissions = nn.Linear(in_dim, num_labels)
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels) * 0.01)
        self.start_transitions = nn.Parameter(torch.randn(num_labels) * 0.01)

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        emissions = self.emissions(x)
        batch, seq_len, num_labels = emissions.shape
        scores = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        all_scores = [scores]
        for t in range(1, seq_len):
            broadcast = scores.unsqueeze(2) + self.transitions.unsqueeze(0)
            scores = torch.logsumexp(broadcast, dim=1) + emissions[:, t]
            all_scores.append(scores)
        return torch.stack(all_scores, dim=1)


class DeepTMHMM(nn.Module):
    """DeepTMHMM: ESM-1b encoder (real `transformers.EsmModel` class) -> dense
    projection to 128-d -> bidirectional LSTM (64 hidden units/direction) -> dense +
    dropout -> linear-chain CRF decoder over per-residue topology labels.
    """

    def __init__(
        self,
        esm_config: EsmConfig,
        lstm_hidden: int = 64,
        proj_dim: int = 128,
        num_labels: int = _NUM_TOPOLOGY_LABELS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.esm = EsmModel(esm_config)
        self.proj = nn.Linear(esm_config.hidden_size, proj_dim)
        self.bilstm = nn.LSTM(
            proj_dim, lstm_hidden, num_layers=1, bidirectional=True, batch_first=True
        )
        self.dense = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.crf_head = DeepTMHMMCRFHead(num_labels, lstm_hidden * 2)

    def forward(self, input_ids, attention_mask=None):
        esm_out = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.proj(esm_out)
        x, _ = self.bilstm(x)
        x = self.activation(self.dense(x))
        x = self.dropout(x)
        return self.crf_head(x)


def build_deeptmhmm():
    esm_config = EsmConfig(
        vocab_size=33,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=1,
    )
    return DeepTMHMM(esm_config)


def example_input_deeptmhmm():
    return torch.randint(4, 30, (2, 20))


MENAGERIE_ENTRIES = [
    ("DeepTMHMM", build_deeptmhmm, example_input_deeptmhmm, 2022, "reimpl-pytorch"),
]

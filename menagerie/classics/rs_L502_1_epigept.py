# SOURCE: vendored from ZjGaothu/EpiGePT @ main (model/EpiGePT.py)
#
# EpiGePT (Gao et al., Genome Biology 2024) predicts genome-wide epigenomic
# signals (chromatin accessibility / histone marks) from DNA sequence
# conditioned on transcription-factor (TF) binding profiles. The real model
# is a Convmodule (5-layer Conv1d+MaxPool1d stack over one-hot sequence) whose
# output is concatenated with a TF-embedding track and fed into a real
# HuggingFace `transformers.BertModel` (with `inputs_embeds=`) transformer
# encoder, followed by a small linear "Multitaskmodule" head. This vendors
# the real Convmodule / Multitaskmodule / EpiGePT classes verbatim from
# model/EpiGePT.py; only the hardcoded config-module constants
# (CHANNEL_SIZE, SEQUENCE_DIM, TF_DIM, NUM_LAYER, NUM_HEAD, NUM_SIGNALS from
# the repo's model/config.py) are inlined as constructor args instead of
# import-time globals, and the PyTorch-Lightning training/data-loading glue
# (configure_optimizers/*_step/setup/*_dataloader, which needs the repo's
# private GenomicData dataset) is dropped -- forward-pass architecture is
# unchanged.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

MENAGERIE_ZOO = "vendored-pytorch"

# real model/config.py defaults (kept as the real values; only NUM_SIGNALS's
# use as the multitask output width and TF_DIM are architecture-defining)
CHANNEL_SIZE = 4  # one-hot ACGT channels
SEQUENCE_DIM = 128  # real: Convmodule out_channels / conv-derived token dim
TF_DIM = 40  # real: TF-binding-profile embedding width appended per-token
NUM_LAYER = 2  # real default uses more layers; kept small for a tiny trace
NUM_HEAD = 2
NUM_SIGNALS = 245  # real: number of epigenomic signal tracks predicted


class Convmodule(nn.Module):
    """Convolution Module.
    The convolution module is made up of "num_cb" conv+pooling blocks.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=stride
        )
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=stride
        )
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=96, kernel_size=5, padding=2, stride=stride
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(
            in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=stride
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(
            in_channels=128, out_channels=out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.pool5 = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        return x


class Multitaskmodule(nn.Module):
    """Multi-task prediction module.
    This module is mainly made up with linear layer.
    """

    def __init__(self, SEQUENCE_DIM, TF_DIM, NUM_SIGNALS):
        super().__init__()
        self.linear = nn.Linear(SEQUENCE_DIM + TF_DIM, NUM_SIGNALS)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return x


class EpiGePT(nn.Module):
    """Initialize layers to build EpiGePT model.
    Args:
        word_num: size of the vocabulary of the transformer module.
        sequence_dim: dimension of the token embedding from the output of the Convolution module.
        tf_dim: dimension of the TF embedding.
        batch_size: batch size for training.
    """

    def __init__(self, word_num, sequence_dim, tf_dim, batch_size):
        super().__init__()
        self.word_num = word_num
        self.sequence_dim = sequence_dim
        self.tf_dim = tf_dim
        self.batch_size = batch_size
        self.convmodule = Convmodule(CHANNEL_SIZE, SEQUENCE_DIM)
        self.config_encoder = BertConfig(
            vocab_size=word_num,
            hidden_size=SEQUENCE_DIM + TF_DIM,
            num_hidden_layers=NUM_LAYER,
            num_attention_heads=NUM_HEAD,
            intermediate_size=1024,
            output_hidden_states=False,
            output_attentions=False,
            max_position_embeddings=1000,
        )  # shape (bs, inp_len, inp_len)

        self.transformermodule = BertModel(config=self.config_encoder)
        # Linear layer for multi-task prediction
        self.multitaskmodule = Multitaskmodule(SEQUENCE_DIM, TF_DIM, NUM_SIGNALS)

    def forward(self, batch_inputs_seq, batch_inputs_tf):
        x = self.convmodule(batch_inputs_seq)
        x = x.transpose(1, 2)
        x = torch.cat([x, batch_inputs_tf], dim=2)
        x = self.transformermodule(inputs_embeds=x)
        output = self.multitaskmodule(x[0])
        return output


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Real usage feeds a one-hot sequence window (batch, CHANNEL_SIZE=4, L) through
# the 5-stage Convmodule (stride-1 conv + fixed-size maxpools: /4/4/2/2/2 =
# /128 total) to get SEQUENCE_DIM output tokens, concatenated per-token with a
# TF_DIM-wide TF-embedding track, then run through the real BertModel encoder.
# Use a tiny sequence length that is a multiple of 128 so the conv/pool stack
# yields a small positive token count (measured empirically: pool kernels of
# 4/4/2/2/2 compound to a /128 length reduction).
_TINY_SEQ_LEN = 256  # -> Convmodule token count = 256 // 128 = 2
_TOKEN_COUNT = 2


def build_epigept():
    return EpiGePT(word_num=100, sequence_dim=SEQUENCE_DIM, tf_dim=TF_DIM, batch_size=1)


def example_input_epigept():
    batch_inputs_seq = torch.randn(2, CHANNEL_SIZE, _TINY_SEQ_LEN)
    batch_inputs_tf = torch.randn(2, _TOKEN_COUNT, TF_DIM)
    return (batch_inputs_seq, batch_inputs_tf)


MENAGERIE_ENTRIES = [
    ("EpiGePT", build_epigept, example_input_epigept, 2024, "SOURCE_AVAILABLE"),
]

# SOURCE: vendored from kishwarshafin/pepper @ master
# https://raw.githubusercontent.com/kishwarshafin/pepper/master/pepper/modules/python/models/simple_model.py
#
# PEPPER (Shafin et al. 2021, Nature Methods "Haplotype-aware variant calling
# with PEPPER-Margin-DeepVariant enables high accuracy in nanopore long-reads")
# -- the RNN-based genome-inference module of the PEPPER-Margin-DeepVariant
# nanopore variant-calling / assembly-polishing pipeline. The production model
# (`ModelHandler.get_new_gru_model` in `ModelHander.py`, used by
# `train_models.py`/`predict.py`) is `TransducerGRU`: a stacked bidirectional
# GRU encoder-decoder transducer over per-column pileup-image feature vectors,
# copied verbatim from the real `simple_model.py` (the older attention-based
# `Seq2Seq_atn` CNN+GRU model in the same repo is legacy/unused by current
# training scripts). No architectural code was rewritten; only checkpoint
# save/load and CUDA/distributed plumbing (irrelevant to the traced
# architecture) were dropped.
#
# Upstream license: kishwarshafin/pepper (see repo LICENSE); random init only.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class TransducerGRU(nn.Module):
    def __init__(
        self,
        image_channels,
        image_features,
        gru_layers,
        hidden_size,
        num_classes,
        bidirectional=True,
    ):
        super(TransducerGRU, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = gru_layers
        self.num_classes = num_classes
        self.gru_encoder = nn.GRU(
            image_features,
            hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.gru_decoder = nn.GRU(
            2 * hidden_size,
            hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dense1 = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x, hidden):
        hidden = hidden.transpose(0, 1).contiguous()
        self.gru_encoder.flatten_parameters()
        x_out, hidden_out = self.gru_encoder(x, hidden)
        self.gru_decoder.flatten_parameters()
        x_out, hidden_final = self.gru_decoder(x_out, hidden_out)

        x_out = self.dense1(x_out)

        hidden_final = hidden_final.transpose(0, 1).contiguous()
        return x_out, hidden_final

    def init_hidden(self, batch_size, num_layers, bidirectional=True):
        num_directions = 1
        if bidirectional:
            num_directions = 2

        return torch.zeros(batch_size, num_directions * num_layers, self.hidden_size)


class PepperTransducerStep(nn.Module):
    """Thin wrapper providing a single concrete-tensor forward: builds the
    zero initial hidden state internally (as `predict.py` does per chunk) so
    the module has one (x, ) call signature suitable for tracing."""

    def __init__(
        self, image_channels=1, image_features=10, gru_layers=1, hidden_size=128, num_classes=5
    ):
        super().__init__()
        self.transducer = TransducerGRU(
            image_channels=image_channels,
            image_features=image_features,
            gru_layers=gru_layers,
            hidden_size=hidden_size,
            num_classes=num_classes,
            bidirectional=True,
        )

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.transducer.init_hidden(
            batch_size, self.transducer.num_layers, bidirectional=True
        )
        out, _ = self.transducer(x, hidden)
        return out


def build_pepper_transducer():
    model = PepperTransducerStep(
        image_channels=1, image_features=10, gru_layers=1, hidden_size=64, num_classes=5
    )
    model.eval()
    return model


def example_input_pepper_transducer():
    # (batch, seq_len, image_features) pileup-column feature sequence.
    return (torch.randn(1, 16, 10),)


MENAGERIE_ENTRIES = [
    (
        "PEPPER-TransducerGRU",
        "build_pepper_transducer",
        "example_input_pepper_transducer",
        2021,
        "vendored-pytorch",
    ),
]

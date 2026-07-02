# RUNG 1 -- real library model: SpeechBrain's `speechbrain.lobes.models.MetricGAN`
# module, an installed base lib, ships the actual MetricGAN-U generator
# architecture (`EnhancementGenerator`) with no modification needed.
#
# MetricGAN-U (Fu et al., "MetricGAN-U: Unsupervised Speech Enhancement/
# Dereverberation Based Only on Batch Enhanced Metric", ICASSP 2022) reuses
# the exact same generator/discriminator network architecture as the
# supervised MetricGAN(+) line of work -- its contribution is the *training
# objective* (an unsupervised proxy-metric loss estimated by a DNSMOS-style
# no-reference quality predictor, used in place of a clean-reference PESQ
# target) and *usage* (no parallel clean/noisy training pairs needed), not
# any change to the enhancement network itself. SpeechBrain's own
# `recipes/DNS/enhancement/MetricGAN-U/` recipe wires this same
# `EnhancementGenerator` class up with the unsupervised proxy-metric loss.
# Since the architecture is unmodified from the shared MetricGAN family and
# is a real class in an installed base lib (`speechbrain==1.1.0`), this is
# RUNG 1 (real library model), not a vendor/port.
#
# `EnhancementGenerator.forward(x, lengths)` takes two positional arguments
# (a batched spectrogram-magnitude tensor and a SpeechBrain-style relative
# `lengths` tensor consumed internally by `speechbrain.nnet.RNN.LSTM`), so
# per the menagerie recipe constraint (one concrete-tensor input only) this
# is emitted as a MODULE rather than a TSV recipe row.

import torch
from speechbrain.lobes.models.MetricGAN import EnhancementGenerator

MENAGERIE_ZOO = "vendored-pytorch"

# The real class's `linear1` layer is hardcoded to `xavier_init_layer(400, 300, ...)`
# in the installed source (not parameterized by `hidden_size`), so `hidden_size`
# must stay at the real default of 200 (bidirectional LSTM -> 2*200=400) for the
# constructed graph to be shape-valid; `input_size` is the real default too
# (257 = n_fft//2 + 1 for a 512-point STFT, matching MetricGAN-U's magnitude-
# spectrogram front end).
_INPUT_SIZE = 257
_HIDDEN_SIZE = 200


def build_metricganu():
    torch.manual_seed(0)
    model = EnhancementGenerator(
        input_size=_INPUT_SIZE,
        hidden_size=_HIDDEN_SIZE,
        num_layers=2,
        dropout=0,
    )
    model.eval()
    return model


def example_input_metricganu():
    torch.manual_seed(0)
    x = torch.randn(2, 10, _INPUT_SIZE)
    lengths = torch.tensor([1.0, 1.0])
    return (x, lengths)


MENAGERIE_ENTRIES = [
    ("MetricGAN-U", "build_metricganu", "example_input_metricganu", 2022, MENAGERIE_ZOO),
]

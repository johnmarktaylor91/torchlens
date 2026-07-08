# FAITHFUL PORT of kristpapadopoulos/seriesnet @ master (original framework: TensorFlow/Keras)
# https://github.com/kristpapadopoulos/seriesnet/blob/master/model/seriesnet.py
# "SeriesNet": Dilated Causal Convolutional Neural Network for Time Series
# Predictions (Papadopoulos 2018), based on WaveNet (van den Oord et al.
# 2016) and "Conditional Time Series Forecasting with Convolutional Neural
# Networks" (Borovykh, Bohte, Oosterlee 2017). This is also the queue's
# named PyTorch-compatible stand-in for the CANDIDATE repo
# pedrolarben/ElectricDemandForecasting-DL, which is itself TF/Keras-only
# (uses `keras-tcn`) -- neither repo has runnable PyTorch code, so the
# architecture is ported here rather than vendored.
#
# The real repo's `model/seriesnet.py` is TF2 `tf.keras.Model`/`Layer`
# subclasses (`DcCnnBlock`, `SeriesNet`) built entirely from `Conv1D`
# (`padding='causal'`), `Activation('selu'|'relu')`, and `Add()`. Every
# mechanism is transcribed FAITHFULLY into torch:
#   - Keras `Conv1D(..., padding='causal', dilation_rate=d)` == torch
#     `nn.Conv1d(..., dilation=d)` preceded by a manual LEFT-only pad of
#     `(kernel_size - 1) * d` samples (`F.pad(x, (pad, 0))`), since torch's
#     `Conv1d` only supports symmetric padding natively.
#   - Each `DcCnnBlock(num_filter, filter_length, dilation, l2_layer_reg)`:
#     dilated causal conv (`layer_out`, linear activation, no bias) -> SELU
#     -> two parallel 1x1 convs (`skip_out`, `network_in`), both linear/
#     no-bias -> residual add of the block'input with `network_in` ->
#     `(network_out, skip_out)`, exactly mirroring `DcCnnBlock.call`.
#     `l2_layer_reg` (Keras `kernel_regularizer=l2(...)`) is a *training-time
#     loss* term with zero effect on the forward graph, so it is dropped
#     (there is no torch forward-pass analog to carry over).
#   - `SeriesNet.call`: 7 stacked `DcCnnBlock`s with dilations
#     1,2,4,8,16,32,64 chained on their first ("a"/residual) output; the 7
#     skip ("b") outputs are summed, ReLU'd, and passed through a final
#     1x1 linear/no-bias `Conv1d(1, 1)` -- transcribed 1:1 as `SeriesNet`.
#     Two `nn.Dropout` calls (blocks 6 and 7's skip outputs) are kept but
#     the model is built and traced in `eval()` mode (dropout is a no-op),
#     matching a "predict" style forward pass.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DcCnnBlock(nn.Module):
    """
    Dilated Causal Convolutional Neural Network Block for SeriesNet,
    predicting time series. Port of the Keras `DcCnnBlock` layer.

    Args:
      num_filter: int - the number of convolution filters in DcCnnBlock
      filter_length: int - the length of convolution filters in DcCnnBlock
      dilation: int - the lookback window (dilation rate) for the block
    """

    def __init__(self, num_filter, filter_length, dilation):
        super().__init__()
        self.filter_length = filter_length
        self.dilation = dilation
        self.causal_pad = (filter_length - 1) * dilation

        self.layer_out = nn.Conv1d(
            1, num_filter, kernel_size=filter_length, dilation=dilation, bias=False
        )
        self.act = nn.SELU()
        self.skip_out = nn.Conv1d(num_filter, 1, kernel_size=1, bias=False)
        self.network_in = nn.Conv1d(num_filter, 1, kernel_size=1, bias=False)

    def forward(self, inputs):
        # inputs: (B, 1, T)
        residual = inputs
        x = F.pad(inputs, (self.causal_pad, 0))
        layer_out = self.act(self.layer_out(x))
        skip_out = self.skip_out(layer_out)
        network_in = self.network_in(layer_out)
        network_out = residual + network_in
        return network_out, skip_out


class SeriesNet(nn.Module):
    """
    Dilated Causal Convolutional Neural Network for Time Series
    Predictions, based on:
    [1] A. van den Oord et al., "Wavenet: A generative model for raw audio,"
        arXiv preprint arXiv:1609.03499, 2016.
    [2] A. Borovykh, S. Bohte, and C. W. Oosterlee, "Conditional Time Series
        Forecasting with Convolutional Neural Networks," arXiv:1703.04691
        [stat], Mar. 2017.

    Port of the Keras `SeriesNet` model (7 stacked dilated-causal-conv
    blocks with dilations 1,2,4,8,16,32,64, summed skip connections).

    Args:
        num_filter: int - the number of convolution filters in each block
        filter_length: int - the length of convolution filters
        dropout: float - dropout fraction applied to the last two skip outputs
    """

    def __init__(self, num_filter, filter_length, dropout):
        super().__init__()
        self.block1 = DcCnnBlock(num_filter, filter_length, 1)
        self.block2 = DcCnnBlock(num_filter, filter_length, 2)
        self.block3 = DcCnnBlock(num_filter, filter_length, 4)
        self.block4 = DcCnnBlock(num_filter, filter_length, 8)
        self.block5 = DcCnnBlock(num_filter, filter_length, 16)
        self.block6 = DcCnnBlock(num_filter, filter_length, 32)
        self.block7 = DcCnnBlock(num_filter, filter_length, 64)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.out = nn.Conv1d(1, 1, kernel_size=1, bias=False)

    def forward(self, inputs):
        l1a, l1b = self.block1(inputs)
        l2a, l2b = self.block2(l1a)
        l3a, l3b = self.block3(l2a)
        l4a, l4b = self.block4(l3a)
        l5a, l5b = self.block5(l4a)
        l6a, l6b = self.block6(l5a)
        l6b = self.dropout(l6b)  # dropout used to limit influence of earlier data
        l7a, l7b = self.block7(l6a)
        l7b = self.dropout(l7b)  # dropout used to limit influence of earlier data
        l8 = l1b + l2b + l3b + l4b + l5b + l6b + l7b
        l9 = self.act(l8)
        l10 = self.out(l9)
        return l10


MENAGERIE_ZOO = "ported-pytorch"


def build_seriesnet():
    torch.manual_seed(0)
    model = SeriesNet(num_filter=8, filter_length=2, dropout=0.1)
    model.eval()
    return model


def example_input_seriesnet():
    torch.manual_seed(0)
    return (torch.randn(2, 1, 64),)


MENAGERIE_ENTRIES = [
    ("SeriesNet", "build_seriesnet", "example_input_seriesnet", 2018, MENAGERIE_ZOO),
]

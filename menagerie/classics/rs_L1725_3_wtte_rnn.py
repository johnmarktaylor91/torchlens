# SOURCE: vendored from ajlien/wtte-torch @ af09fcd931f296be5a1ccdc8d431aca3f07b33a1
#
# The queued repo (ragulpr/wtte-rnn, the original WTTE-RNN reference implementation) ships
# a Keras/TensorFlow package (`python/wtte/`) with no PyTorch code -- but the WTTE-RNN
# architecture (a recurrent encoder feeding a 2-unit Dense head whose outputs are mapped to
# the (alpha, beta) parameters of a Weibull hazard distribution via a dedicated activation
# layer) is independently reimplemented as REAL, running PyTorch in `ajlien/wtte-torch`
# (`wtte/network.py`), which explicitly cites the same upstream Martinsson/ragulpr example
# as its architecture reference ("Default architecture based on Egil Martinsson's example
# at https://github.com/ragulpr/wtte-rnn/blob/master/examples/keras/simple_example.ipynb").
# `WeibullActivation`, `WtteNetwork`, and `WtteRnnNetwork` below are copied verbatim from
# that repo's `wtte/network.py` (the `StubModel` pretraining helper, which plays no role in
# the traced forward architecture, is omitted).
#
# Repo: https://github.com/ajlien/wtte-torch @ master (af09fcd)
# File vendored: wtte/network.py (WeibullActivation, WtteNetwork, WtteRnnNetwork)
# Upstream architecture reference: https://github.com/ragulpr/wtte-rnn @ master,
#       examples/keras/simple_example.ipynb

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MENAGERIE_ZOO = "vendored-pytorch"


class WeibullActivation(nn.Module):
    """Activation function to get elementwise alpha and regularized beta
    :param init_alpha: Float initial alpha value
    :param max_beta_value: Float maximum beta value
    """

    def __init__(self, init_alpha=1.0, max_beta_value=5.0, scalefactor=1.0):
        super(WeibullActivation, self).__init__()
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
        self.scalefactor = scalefactor

    def forward(self, x):
        a, b = torch.split(x, 1, dim=-1)
        a = self.init_alpha * torch.exp(a) * self.scalefactor
        b = self.max_beta_value * torch.sigmoid(b) * self.scalefactor
        return torch.cat((a, b), -1)


class WtteNetwork(nn.Module):
    """A deep neural network that receives a sequence as input and estimates the parameters of a
    Weibull distribution, conditional on inputs, describing the time to next event of the process.
    """

    def __init__(self, submodel, submodel_out_features, init_alpha=1.0, max_beta_value=5.0):
        """
        :param submodel: nn.Module with the architecture of the model
        :param submodel_out_features: Int with the dimension of the output from the last layer of the submodel
        :param init_alpha: Float initial alpha value
        :param max_beta_value: Float maximum beta value
        """
        super(WtteNetwork, self).__init__()
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
        self.submodel = submodel
        self.linear = nn.Linear(submodel_out_features, 2)
        self.activation = WeibullActivation(
            init_alpha=init_alpha,
            max_beta_value=max_beta_value,
            scalefactor=1.0 / np.log(submodel_out_features),
        )

    def forward(self, x):
        y = self.submodel(x)
        y = self.linear(y)
        y = self.activation(y)
        return y


class WtteRnnNetwork(WtteNetwork):
    """A network with recurrent layers that estimates the Weibull time to event distribution parameters.
    Default architecture based on Egil Martinsson's example at
    https://github.com/ragulpr/wtte-rnn/blob/master/examples/keras/simple_example.ipynb
    """

    def __init__(
        self,
        input_size,
        rnn_layer=nn.GRU,
        rnn_layer_options={"hidden_size": 20, "num_layers": 1},
        init_alpha=1.0,
        max_beta_value=5.0,
    ):
        """Specify an RNN for WTTE modeling.
        :param input_size: Int sequence length provided
        :param rnn_layer: Class with the nn.Module representing the recurrent layer - consider nn.GRU or nn.LSTM
        :param rnn_layer_options: Dict with parameters for RNN hidden layer
        :param init_alpha: Float initial alpha value
        :param max_beta_value: Float maximum beta value
        """
        rnn = rnn_layer(input_size=input_size, batch_first=True, **rnn_layer_options)
        super(WtteRnnNetwork, self).__init__(
            submodel=rnn,
            submodel_out_features=rnn.hidden_size,
            init_alpha=init_alpha,
            max_beta_value=max_beta_value,
        )

    def forward(self, x):
        y, _ = self.submodel(x)
        y, _ = pad_packed_sequence(y, batch_first=True, padding_value=0)
        y = self.linear(y)
        y = self.activation(y)
        return y


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_INPUT_SIZE = 8
_HIDDEN_SIZE = 6
_SEQ_LEN = 10
_BATCH = 4


def build_wtte_rnn():
    return WtteRnnNetwork(
        input_size=_INPUT_SIZE,
        rnn_layer=nn.GRU,
        rnn_layer_options={"hidden_size": _HIDDEN_SIZE, "num_layers": 1},
    )


def example_input_wtte_rnn():
    x = torch.randn(_BATCH, _SEQ_LEN, _INPUT_SIZE)
    lens = torch.arange(_SEQ_LEN, _SEQ_LEN - _BATCH, -1)
    return pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=True)


MENAGERIE_ENTRIES = [
    (
        "WTTE-RNN",
        build_wtte_rnn,
        example_input_wtte_rnn,
        2016,
        "SOURCE_AVAILABLE",
    ),
]

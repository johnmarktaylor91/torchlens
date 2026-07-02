# FAITHFUL PORT of gm-spacagna/deep-ttf @ master (original framework: Keras/TensorFlow)
#
# The queued repo (gm-spacagna/deep-ttf, "Deep Weibull Network" / DeepTTF) ships only a
# Jupyter notebook (`notebooks/Keras-WTT-RNN Engine failure.ipynb`); its `deep-ttf/__init__.py`
# package file is empty, so there is no vendorable .py module. The notebook's model-building
# cell (verbatim, Keras) is:
#
#     model = Sequential()
#     model.add(Masking(mask_value=mask_value, input_shape=(None, n_features)))
#     model.add(GRU(20, activation='tanh', recurrent_dropout=0.25))
#     model.add(Dense(2))
#     model.add(Lambda(wtte.output_lambda,
#                       arguments={"init_alpha": init_alpha,
#                                  "max_beta_value": 100.0,
#                                  "alpha_kernel_scalefactor": 0.5}))
#
# i.e. a Weibull-Time-To-Event RNN (Martinsson 2016, WTTE-RNN) head: a masked GRU encoder
# feeding a 2-unit Dense layer whose two outputs are transformed into the (alpha, beta)
# parameters of a Weibull hazard distribution via the WTTE-RNN package's `output_lambda`
# elementwise activation. `output_lambda` (transcribed faithfully from
# https://github.com/ragulpr/wtte-rnn python/wtte/wtte.py, function `output_lambda`) is:
#
#     a, b = x[..., 0], x[..., 1]
#     a = init_alpha * exp(scalefactor * a)
#     b = max_beta_value * sigmoid(scalefactor * b - log(max_beta_value - 1.0))
#
# This port transcribes that exact layer stack into torch: `nn.GRU(batch_first=True)` in
# place of Keras `GRU` (Keras masking is reproduced by zero-padding the input, which is the
# training-time convention `build_data()` in the same notebook already uses via `mask_value`
# fill -- masking itself only affects loss computation over padded steps, not the layer
# graph, so it does not appear as a distinct traced op) followed by `nn.Linear(20, 2)` and
# the `WeibullOutputActivation` module implementing `output_lambda` verbatim above.
#
# Repo: https://github.com/gm-spacagna/deep-ttf @ master,
#       notebooks/Keras-WTT-RNN Engine failure.ipynb (model-definition + `activate`/
#       `weibull_loglik_discrete` cells)
# Upstream WTTE-RNN activation source: https://github.com/ragulpr/wtte-rnn @ master,
#       python/wtte/wtte.py, function `output_lambda`

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class WeibullOutputActivation(nn.Module):
    """Faithful port of wtte.wtte.output_lambda (ragulpr/wtte-rnn).

    Elementwise transform of a 2-unit Dense output into (alpha, beta) Weibull
    hazard-distribution parameters, matching the DeepTTF notebook's Lambda layer
    call: `wtte.output_lambda(x, init_alpha=init_alpha, max_beta_value=100.0,
    alpha_kernel_scalefactor=0.5)`.
    """

    def __init__(
        self, init_alpha: float = 1.0, max_beta_value: float = 100.0, scalefactor: float = 0.5
    ):
        super().__init__()
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
        self.scalefactor = scalefactor
        # `output_lambda` shifts beta to start around 1.0 whenever max_beta_value > 1.05
        # (true here, 100.0), matching the original's `_shift = log(max_beta_value - 1.0)`.
        self._shift = math.log(max_beta_value - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[..., 0]
        b = x[..., 1]

        if self.scalefactor is not None:
            a = self.scalefactor * a
            b = self.scalefactor * b

        a = self.init_alpha * torch.exp(a)
        b = b - self._shift
        b = self.max_beta_value * torch.sigmoid(b)

        return torch.stack([a, b], dim=-1)


class DeepWeibullNetwork(nn.Module):
    """Faithful port of the DeepTTF (Deep Weibull Network) Keras model.

    A GRU-based Weibull-Time-To-Event RNN: predicts the (alpha, beta) parameters of a
    Weibull time-to-failure distribution from a masked window of sensor time series.
    """

    def __init__(
        self,
        n_features: int = 17,
        gru_units: int = 20,
        recurrent_dropout: float = 0.25,
        init_alpha: float = 1.0,
        max_beta_value: float = 100.0,
        alpha_kernel_scalefactor: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        # torch.nn.GRU has no native `recurrent_dropout`; Keras recurrent_dropout zeroes
        # the recurrent (hidden-to-hidden) connections at each timestep. torch's `dropout`
        # kwarg only applies between stacked layers, so for this single-layer GRU we keep
        # the value as a documented architectural parameter (no-op at 1 layer, matching
        # torch's own single-layer-dropout no-op semantics) rather than silently dropping
        # the field.
        self.recurrent_dropout = recurrent_dropout
        self.gru = nn.GRU(input_size=n_features, hidden_size=gru_units, batch_first=True)
        self.dense = nn.Linear(gru_units, 2)
        self.weibull_activation = WeibullOutputActivation(
            init_alpha=init_alpha,
            max_beta_value=max_beta_value,
            scalefactor=alpha_kernel_scalefactor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, n_features) -- Keras `Masking` zero-fills unobserved lookback
        # steps upstream of the GRU (see `build_data()`/`mask_value` in the source
        # notebook); the traced graph below is the layer stack the Masking layer feeds.
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        ab = self.dense(last_step)
        return self.weibull_activation(ab)


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_N_FEATURES = 17
_TIME_STEPS = 12


def build_deep_weibull_network():
    return DeepWeibullNetwork(n_features=_N_FEATURES, gru_units=20)


def example_input_deep_weibull_network():
    return torch.randn(4, _TIME_STEPS, _N_FEATURES)


MENAGERIE_ENTRIES = [
    (
        "Deep Weibull Network",
        build_deep_weibull_network,
        example_input_deep_weibull_network,
        2017,
        "PORT",
    ),
]

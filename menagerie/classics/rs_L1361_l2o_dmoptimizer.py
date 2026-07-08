# FAITHFUL PORT of https://github.com/thetianshuhuang/l2o @ master (cf71d728)
# (policies/deepmind_2016.py :: DMOptimizer; policies/architectures.py ::
#  BaseCoordinateWisePolicy call convention) (original framework: TensorFlow 2 / Keras)
#
# "Learning to learn by gradient descent by gradient descent" (Andrychowicz et al.,
# NeurIPS 2016) -- the l2o repo's DMOptimizer policy. The l2o repo is a TensorFlow-2/
# Keras framework (tf.keras.Model policies wrapped by CoordinateWiseOptimizer /
# HierarchicalOptimizer, tf.distribute training strategies, curriculum-learning
# unroll machinery) that cannot reasonably be installed alongside this torch-only
# base env; a second public PyTorch/TF2-Keras rewrite of the same paper
# (DeepStudio-TW/LSTM-optimizer) was checked and is also TensorFlow-2/Keras only,
# so no runnable-in-base-env code exists.
#
# The DMOptimizer *architecture* itself, however, is simple and fully specified by
# the real init_layers/call code in policies/deepmind_2016.py (reproduced in the
# docstring below for traceability): a stack of tf.keras.layers.LSTMCell layers
# (default sizes (20, 20)) applied coordinate-wise to a single scalar gradient
# value per parameter element, followed by a Dense(1) that emits a per-element
# learning-rate-scaled parameter delta. This module is a faithful torch port of
# that exact recurrence using nn.LSTMCell (Keras' LSTMCell and torch's nn.LSTMCell
# are both single-timestep cells) chained per the real `for i, layer in
# enumerate(self.recurrent)` / `x = layer(x, states[...])` loop, followed by the
# real `self.delta = Dense(1, ...)` output head, and the real
# `learning_rate * delta` scaling in `call()`.
#
# Real init_layers/call (policies/deepmind_2016.py), for reference:
#
#     def init_layers(self, learning_rate=0.1, layers=(20, 20), **kwargs):
#         self.learning_rate = learning_rate
#         self.recurrent = [
#             LSTMCell(hsize, name="recurrent_{}".format(i), **kwargs)
#             for i, hsize in enumerate(layers)]
#         self.delta = Dense(1, input_shape=(layers[-1],), name="delta")
#
#     def call(self, param, inputs, states, global_state, training=False):
#         states_new = {}
#         x = tf.reshape(inputs, [-1, 1])
#         for i, layer in enumerate(self.recurrent):
#             hidden_name = "lstm_{}".format(i)
#             x, states_new[hidden_name] = layer(
#                 x, states[hidden_name], training=training)
#         x = self.delta(x, training=training)
#         return self.learning_rate * tf.reshape(x, param.shape), states_new

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DMOptimizer(nn.Module):
    """Faithful torch port of the l2o repo's DMOptimizer policy
    (Andrychowicz et al. 2016 "Learning to learn by gradient descent by
    gradient descent" meta-optimizer network).

    Applied coordinate-wise: every scalar element of the optimizee's flattened
    gradient tensor is an independent "batch" item fed through the same shared
    LSTM stack, exactly as the real `tf.reshape(inputs, [-1, 1])` / per-element
    processing in `DMOptimizer.call`. Returns the per-element parameter delta
    (learning_rate-scaled), reshaped back to the gradient's original shape.
    """

    def __init__(self, learning_rate=0.1, layers=(20, 20)):
        super().__init__()
        self.learning_rate = learning_rate
        self.layer_sizes = tuple(layers)

        cells = []
        in_size = 1
        for hsize in self.layer_sizes:
            cells.append(nn.LSTMCell(in_size, hsize))
            in_size = hsize
        self.recurrent = nn.ModuleList(cells)
        self.delta = nn.Linear(self.layer_sizes[-1], 1)

    def get_initial_state(self, n_elements, device=None, dtype=torch.float32):
        """Zero (h, c) hidden state for every LSTMCell, batched over
        n_elements (one state slot per scalar gradient element -- mirrors the
        real get_initial_state's `batch_size = tf.size(var)`)."""
        states = []
        for cell in self.recurrent:
            h = torch.zeros(n_elements, cell.hidden_size, device=device, dtype=dtype)
            c = torch.zeros(n_elements, cell.hidden_size, device=device, dtype=dtype)
            states.append((h, c))
        return states

    def forward(self, param, grad, states=None):
        """
        Args:
            param (Tensor): the optimizee parameter tensor (used only for its
                shape, mirroring the real `tf.reshape(x, param.shape)`).
            grad (Tensor): gradient tensor, same shape as `param`.
            states (list[(Tensor, Tensor)] | None): per-layer LSTMCell (h, c)
                state pairs; created fresh (zeros) if None.

        Returns:
            (delta, states_new): parameter update (same shape as `param`) and
            the updated per-layer hidden states.
        """
        n_elements = grad.numel()
        if states is None:
            states = self.get_initial_state(n_elements, device=grad.device, dtype=grad.dtype)

        x = grad.reshape(-1, 1)
        states_new = []
        for cell, (h, c) in zip(self.recurrent, states):
            h, c = cell(x, (h, c))
            states_new.append((h, c))
            x = h
        x = self.delta(x)
        return self.learning_rate * x.reshape(param.shape), states_new


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing.
#
# The real repo trains DMOptimizer against arbitrary optimizee problems (MLPs,
# CNNs, quadratics -- see problems/networks.py); the meta-optimizer network
# itself is optimizee-shape-agnostic since it runs coordinate-wise per gradient
# element. We trace it directly on a small (param, grad) pair, matching the
# real `call(param, inputs, states, global_state)` signature (global_state is
# unused by the coordinate-wise policy per BaseCoordinateWisePolicy, so it is
# dropped from this port's forward signature).
# ---------------------------------------------------------------------------
_LAYERS = (8, 8)
_PARAM_SHAPE = (6, 5)


def build_l2o_dmoptimizer():
    torch.manual_seed(0)
    model = DMOptimizer(learning_rate=0.1, layers=_LAYERS)
    model.eval()
    return model


def example_input_l2o_dmoptimizer():
    torch.manual_seed(0)
    param = torch.randn(_PARAM_SHAPE)
    grad = torch.randn(_PARAM_SHAPE)
    return (param, grad)


MENAGERIE_ENTRIES = [
    (
        "L2O-DMOptimizer",
        "build_l2o_dmoptimizer",
        "example_input_l2o_dmoptimizer",
        2016,
        MENAGERIE_ZOO,
    ),
]

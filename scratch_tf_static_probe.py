import tempfile
from typing import Any

import tensorflow as tf
from tensorflow import keras

import torchlens as tl


class StaticDenseModel(keras.Model):
    def __init__(self) -> None:
        super().__init__(name="static_dense_model")
        self.dense = keras.layers.Dense(
            2,
            activation="relu",
            kernel_initializer=keras.initializers.Constant([[1.0, -1.0], [2.0, 0.5]]),
            bias_initializer=keras.initializers.Constant([0.25, -0.5]),
            name="dense",
        )

    def call(self, x: Any) -> Any:
        return self.dense(x)


model = StaticDenseModel()
x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
model(x)
with tempfile.TemporaryDirectory() as d:
    tf.saved_model.save(model, d)
    loaded = tf.saved_model.load(d)

    trace = tl.trace(loaded, x, backend="tf", save=tl.func("relu"))
    for op in trace.layer_list:
        print(
            op.label,
            "| func_name=",
            op.func_name,
            "| is_input=",
            op.is_input,
            "| is_output=",
            op.is_output,
            "| has_saved_activation=",
            op.has_saved_activation,
            "| out is None=",
            op.out is None,
        )

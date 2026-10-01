import sys
import tempfile

import tensorflow as tf

sys.path.insert(0, "tests/backends")
sys.path.insert(0, "tests")
from test_tf_static import StaticDenseModel  # noqa: E402

import torchlens as tl  # noqa: E402

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

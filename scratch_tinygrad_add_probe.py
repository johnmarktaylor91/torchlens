from tinygrad import Tensor

import torchlens as tl


def add_one(value):
    return value + 1.0


trace = tl.trace(add_one, Tensor([1.0, 2.0, 3.0]), backend="tinygrad")
for op in trace.layer_list:
    print(op.label, "func_name=", op.func_name, "layer_type=", op.layer_type)

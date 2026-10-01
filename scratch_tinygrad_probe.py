from typing import Any

from tinygrad import Tensor

import torchlens as tl


def add_one(value: Any) -> Any:
    return value + 1.0


trace = tl.trace(add_one, Tensor([1.0, 2.0, 3.0]), backend="tinygrad")
for op in trace.layer_list:
    print(op.label, "| func_name=", repr(op.func_name), "| layer_type=", repr(op.layer_type))

print()


def _tiny_block(x: Any) -> Any:
    y = x + 1.0
    z = y.relu()
    return (z * 2.0).sum()


trace2 = tl.trace(_tiny_block, Tensor([1.0, -2.0, 3.0]), backend="tinygrad")
result = trace2.validate_forward_pass([_tiny_block(Tensor([1.0, -2.0, 3.0]))])
print("validate_forward_pass result:", result)
print("validation_replay_status:", trace2.validation_replay_status)

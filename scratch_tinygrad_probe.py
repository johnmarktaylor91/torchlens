from typing import Any

from tinygrad import Tensor

import torchlens as tl


def _tiny_block(x: Any) -> Any:
    y = x + 1.0
    z = y.relu()
    return (z * 2.0).sum()


x = Tensor([1.0, -2.0, 3.0])
trace = tl.trace(_tiny_block, x, backend="tinygrad")
result = trace.validate_forward_pass([_tiny_block(x)])
print("validate_forward_pass result:", result)
print("validation_replay_status:", trace.validation_replay_status)
for op in trace.layer_list:
    print(op.label, "| layer_label=", op.layer_label, "| parents=", op.parents)

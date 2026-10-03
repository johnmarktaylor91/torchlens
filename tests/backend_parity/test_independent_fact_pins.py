"""Hand-pinned independent facts for the backend-parity golden suite.

b9 R75-4: the golden suites are self-referential by construction -- they
regenerate expected output THROUGH the code under test, so they detect
CHANGE, not correctness. The mitigant (modeled on ``capture_oracle``'s
hand-pins) is at least ONE fact per suite written down from first
principles, never from a regeneration run: if capture or the projection
machinery starts lying, the goldens re-pin around it but these do not.

Every expected value below is hand-derived from the model definition alone.
Updating one requires arguing from the MODEL, not from observed output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torchlens as tl


def test_linear_trace_facts_hold_from_first_principles():
    """A bare ``nn.Linear`` capture matches its hand-derived facts."""

    torch.manual_seed(0)
    model = nn.Linear(3, 2).eval()
    x = torch.ones(1, 3)
    log = tl.trace(model, x)
    try:
        # One module, one weight + one bias parameter: from the definition.
        assert log.backend == "torch"
        assert log.model_class_name == "Linear"
        assert log.num_modules == 1
        # Exactly one compute op (the linear kernel); input and output nodes
        # exist besides it.
        compute_funcs = [op.func_name for op in log.compute_ops]
        assert compute_funcs == ["linear"]
        assert len(log.input_layers) == 1
        assert len(log.output_layers) == 1
        # Output shape (1, 2) follows from Linear(3, 2) on a (1, 3) input.
        out_layer = log.layer_dict_all_keys[log.output_layers[0]]
        assert tuple(out_layer.out.shape) == (1, 2)
        # y = W @ x + b with x = ones: y == W.sum(dim=1) + b, computed OUTSIDE
        # capture from the model's own parameters.
        expected = model.weight.detach().sum(dim=1) + model.bias.detach()
        linear_op = next(op for op in log.compute_ops if op.func_name == "linear")
        assert torch.allclose(linear_op.out.squeeze(0), expected, atol=1e-6)
        # The linear op consumed the model's two parameters. Unconditional
        # (b9-opus R75-2): the former `if params is not None:` guard made this
        # pin vacuously green if the attribute disappeared or read None.
        assert hasattr(linear_op, "params"), "Op.params attribute is gone"
        params = linear_op.params
        assert params is not None, "linear op recorded no consumed parameters"
        assert len(tuple(params)) == 2
    finally:
        log.cleanup()

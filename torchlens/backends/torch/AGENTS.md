# backends/torch/ - Agent Notes

## Wrapper Boundaries
- `wrappers.py` owns persistent torch/function decoration. `_logging_enabled` must stay
  the runtime gate; wrappers remain installed after first capture.
- `torch.func` / functorch transform builders return boundary callables. They should attach
  transform metadata, lazy source locations, and replay callables without tracing inside the
  transformed function.
- Direct-call transform wrappers such as `torch.autograd.functional.jacobian` follow the same
  boundary-node contract.

## Provenance
- `model_prep.py` tags registered buffers and plain module tensor attributes with buffer
  addresses before capture.
- `ops.py` records `unattributed_tensor_args` only for tensor arguments with no TorchLens
  input/op/buffer provenance. This is warn-first diagnostics, not a parent-edge substitute.
- Foreign tensors in output position keep existing output binding behavior and should not
  create provenance warnings.

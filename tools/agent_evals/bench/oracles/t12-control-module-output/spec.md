# T12: record one module's output for 100 inputs (control task)

## Prompt to the agent

You are given a small convolutional network and 100 inputs. Record the output
of the module named `block2.bn` (the second batch-normalisation layer, in
evaluation mode) for every input, in input order, and save the recorded
activations.

Inputs (in `--inputs`):

- `model.py`: defines the network and `build_model(weight_seed)`, which returns
  it in eval mode with deterministic weights, and `make_stimuli(stimulus_seed, n, shape)`,
  which returns the inputs as a float32 array of shape `[n, 3, 32, 32]`.
- `config.json`: `weight_seed`, `stimulus_seed`, `n_inputs`, `input_shape`, `module`.

Required output (in `--outputs`):

- `activations.npz` with one array `activations` of shape `[100, 64, 16, 16]`,
  float32, row i holding the module's output for input i. Run the inputs in
  any batch size; the result must not depend on it.

Run on CPU in float32. Do not alter the model's mode or weights.

## Oracle and pass bars

The grader rebuilds the model and the inputs from the seeds, computes the
module output itself, and compares.

| Bar | Pass |
|---|---|
| shape | array present with shape `[n, 64, 16, 16]` |
| exact | max absolute difference <= 1e-4 (float32 batch-size and kernel noise is about 1e-5 at this activation scale; every seeded wrong solution is off by more than 7) |

A held-out run with a different `stimulus_seed` must also pass. This task is
the control: a correct solution exists in every arm, so no arm should win on
correctness here. Success rate and overhead differences are still recorded.

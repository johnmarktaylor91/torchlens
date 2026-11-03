# T10: elementwise gradient times activation at every intermediate

For 16 seeded images, compute the elementwise product of each intermediate
activation with the derivative of an objective with respect to that activation.
The objective is the sum over the batch of class-3 logits (zero-based class
index). Run `model.py`'s `build_model(weight_seed)` on `make_inputs(stimulus_seed)`
in CPU float32 and evaluation mode. Seeds are in `config.json`, and every
random draw uses `numpy.random.default_rng`.

Record these intermediates in this exact computation order. Each shape includes
the batch dimension; no reduction is allowed:
- `stem`: first 1x1 convolution, `[16,4,16,16]`.
- `conv1`: 3x3 padded convolution using the stored weight, `[16,6,16,16]`.
- `relu1`: rectification of conv1, `[16,6,16,16]`.
- `pool1`: 2x2 max pooling, `[16,6,8,8]`.
- `conv2`: second 3x3 padded convolution, `[16,6,8,8]`.
- `relu2`: rectification of conv2, `[16,6,8,8]`.
- `add1`: sum of pool1 and relu2, `[16,6,8,8]`.
- `ln1`: layer normalization over all 384 nonbatch entries, `[16,6,8,8]`.
- `flat`: flatten of ln1, `[16,384]`.
- `fc`: final affine class scores, `[16,5]`.

Write `grad_act.npz`: one float32 array under each id above, containing
`activation * d(objective)/d(activation)` at each element. Keep input order.
Use the standard `solve.py --inputs DIR --outputs DIR` interface.

The oracle reconstructs the forward independently, retains all intermediate
gradients, runs backward, and multiplies the tensors. Every id must be present
with the correct shape and finite values. Relative Frobenius error must be
strictly below `1e-4` for EACH id; the maximum is reported. A held-out run
uses different weight and input seeds and the same bars.

Pitfalls: many computations are functions rather than nn modules. Module-only
forward hooks omit them. Use the gradient of the batch SUM, not its mean, and
multiply by the tensor itself rather than reporting gradients alone. The
residual sum sends gradients through both of its input branches.

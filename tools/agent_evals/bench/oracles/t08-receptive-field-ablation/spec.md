# T8: geometric and gradient receptive fields, followed by input ablation

For each of the six spatial units listed in `config.json`, determine which
input pixels can influence it by network connectivity, which have nonzero
local sensitivity on the supplied images, and the causal effect of removing
its entire geometric input field. Preserve the listed unit order. Rows and
columns are zero-based, and `channel` selects the channel at that spatial unit.

Build `model.py`'s network with `build_model(weight_seed)` and use
`make_inputs(stimulus_seed)`: eight images of shape `[1,64,64]`. Every random
draw uses `numpy.random.default_rng`. Run CPU float32 in evaluation mode.
The network is: 5x5 convolution (stride 1, padding 2), ReLU, 2x2 max pooling
(stride 2), 3x3 convolution (stride 2, padding 1), ReLU, 3x3 convolution
(dilation 2, stride 1, padding 2), ReLU, global average pooling, linear readout.
The ids `relu1`, `pool1`, `relu2`, `relu3` name the activations after the
corresponding steps, with spatial sizes 64,32,16,16. Weights stay fixed.

Write these files:
- `geometric.npz`, boolean `masks` `[6,64,64]`: exact possible pixel support
  from kernel sizes, strides, padding and dilation alone. Ignore learned zero
  weights, rectification gates and which element wins a max pool. Clip at
  every intermediate image boundary, since padded values are constants.
- `gradient.npz`, boolean `masks` `[6,64,64]`: for each unit, compute its
  activation derivative with respect to each input pixel on each of the eight
  images, take absolute values, average these eight absolute derivatives,
  then compare strictly greater than `1e-8`. Use the input values unchanged.
- `ablation.npz`, float32 `effects` `[6]`: mean class-0 logit AFTER zeroing all
  input pixels in the geometric field minus mean class-0 logit BEFORE
  ablation. Zero those pixels in every input channel, separately for each unit.

Use `solve.py --inputs DIR --outputs DIR`. A held-out run changes weight and
image seeds and uses the same unit list.

The independent oracle propagates exact sets of pixel indices through the
layer arithmetic, calculates gradients image by image, and runs input
ablations directly. Shapes and types must match. Geometric masks must be
exactly equal (the count of differing pixels is reported). The minimum
per-unit gradient-mask intersection-over-union must be at least .95 (two empty
masks have IoU 1). Every ablation effect must have error at most 1% of the
oracle magnitude when that magnitude is at least .01, otherwise at most
`1e-4` absolute. The maximum error divided by that allowance is reported.

Pitfalls: the edge units have asymmetric clipped fields, so swapping row and
column changes the answer. A gradient on one image does not give the support
averaged over eight. Dilation and padding both matter. Zeroing the selected
internal unit measures a different experiment from zeroing its input field.
Autograd and forward hooks may be used to calculate the sensitivities.

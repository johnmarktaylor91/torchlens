# T7: layer-wise representational similarity analysis of a CNN and a vision transformer

## Prompt to the agent

Compute representational dissimilarity matrices (RDMs) for a convolutional
network and a vision transformer at every intermediate computation listed in
`layers.json`, including the residual additions inside the residual blocks and
transformer blocks (these are sums computed inside the blocks, not the output
of any named submodule). Then find, for each network, the layer whose RDM best
matches the category structure of the stimuli.

Inputs (in `--inputs`):

- `config.json`: the two networks, the stimulus seed and the normalisation
  constants. The CNN is torchvision's `resnet18` with the `IMAGENET1K_V1`
  weights; the transformer is `facebook/deit-tiny-patch16-224` loaded with the
  `transformers` library for image classification. Both run on CPU in float32
  and in evaluation mode.
- `stimuli.py`: `make_stimuli(seed, n_categories, per_category, size)` returns
  `(images, labels)`; images are float32 in [0, 1], shape `[N, 3, 224, 224]`,
  ordered category-major; `labels[i]` is the category of image `i`. Normalise
  each channel with the mean and std in `config.json` before feeding either
  network (no resizing or cropping; images are already 224 x 224).
- `layers.json`: for each network, the ordered list of layer ids and a plain
  description of the computation each id refers to. Record the full tensor
  produced by that computation for each stimulus (all channels and positions,
  or all tokens and features).

Definitions:

- Feature vector of a layer for stimulus i: the recorded tensor flattened to
  one dimension.
- RDM of a layer: the `N x N` matrix `1 - r_ij`, where `r_ij` is the Pearson
  correlation between the feature vectors of stimuli i and j, computed in
  float64. Diagonal zero.
- Category RDM: `0` where two stimuli share a category, `1` otherwise.
- Match of a layer to the category RDM: the Pearson correlation between the
  upper triangles (excluding the diagonal) of the layer RDM and the category RDM.
- Best layer of a network: the layer id with the highest match.

Required outputs (in `--outputs`):

- `rdms_cnn.npz` and `rdms_vit.npz`: one float64 array per layer id, keyed by
  the id string, shape `[N, N]`.
- `best_layer.json`: `{"cnn": "<layer id>", "vit": "<layer id>"}`.

Pitfalls the solution must get right: the two ReLUs inside a residual block
are the same module applied twice; the residual addition is performed in place
on the second batch norm's output, so a recorded tensor that is not copied is
later overwritten; the transformer's residual additions are plain tensor sums
inside the block's forward.

## Oracle and pass bars

The grader extracts every listed layer with its own independent code (explicit
re-computation of each block from its submodules, checked against the module
graph) on the same stimuli and compares.

| Bar | Pass |
|---|---|
| layers present | every id in `layers.json` has an array of shape `[N, N]` |
| rdm agreement | Pearson correlation of upper triangles between agent RDM and oracle RDM >= 0.999 for every layer (minimum over layers is reported) |
| best layer | agent's best layer equals the oracle's for both networks |

The held-out run uses a different stimulus seed and the same bars.

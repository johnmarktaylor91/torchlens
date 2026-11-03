# Feature visualisation: optimise an input for a TorchLens site

Feature visualisation by optimisation starts from noise and changes the input,
step by step, until a chosen part of the network responds strongly (Olah,
Mordvintsev and Schubert, 2017, "Feature Visualization", Distill,
https://distill.pub/2017/feature-visualization/). The image is not stored as
pixels: it is a scaled Fourier spectrum in decorrelated colour, squashed into
the valid pixel range, and each step sees a randomly shifted and rescaled copy
(Mordvintsev et al., 2018, "Differentiable Image Parameterizations", Distill,
https://distill.pub/2018/differentiable-parameterizations/). Both ideas come
with constants from the articles' reference code, lucid
(https://github.com/tensorflow/lucid); they are recipe code on this page, not
part of TorchLens.

TorchLens supplies the site. The objective can read any op a trace shows, by
its label, including ops that are not a module's output; a channel or a single
unit of that op; or an SAE latent, by passing the op's output through the SAE's
encoder. The loop runs the model through a capture-free binding,
`tl.when(site, action).bind(model)`, whose action keeps the site's output with
its gradient, so a step costs about a plain forward instead of a full trace.

## A model with a planted detector

The first convolution's channel 0 is set to a known pattern: diagonal stripes
of light and dark. Its output, and the multiply of a SiLU written out in
`forward` (an op, not a module output), are the sites below. A trace lists
every op label; a selector built from the op's function and module must match
exactly the op you picked:

```python
# recipe-test: tool=torchlens expected=(('conv2d_1_1',), ('mul_1_3',))
import torch
from torch import nn

import torchlens as tl

rows, cols = torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij")
stripes = torch.where(((rows + cols) // 2) % 2 == 0, 1.0, -1.0)
PATTERN = torch.stack([stripes, stripes, stripes])  # [3, 5, 5], entries +1 and -1


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.fc = nn.Linear(8, 10)
        with torch.no_grad():
            self.conv1.weight[0] = PATTERN  # channel 0: the planted detector
            self.conv1.bias[0] = 0.0

    def forward(self, x):
        h = self.conv1(x)
        gated = h * torch.sigmoid(h)  # SiLU written out
        return self.fc(self.conv2(gated).mean(dim=(2, 3)))


torch.manual_seed(0)
cnn = TinyCNN().eval().requires_grad_(False)  # only the input is optimised
trace = tl.trace(cnn, torch.rand(1, 3, 16, 16))
print(trace.layer_labels)

conv_site = tl.func("conv2d") & tl.in_module("conv1")
gate_site = tl.func("mul")
RESULT = (trace.find_sites(conv_site).labels(), trace.find_sites(gate_site).labels())
```

```text
['input_1', 'conv2d_1_1', 'sigmoid_1_2', 'mul_1_3', 'conv2d_2_4', 'mean_1_5', 'linear_1_6', 'output_1']
```

## The building blocks

`SiteReader` keeps the output of one site on every call, with its gradient.
`FourierImage` is lucid's `fft_image` and `to_valid_rgb`: a spectrum scaled by
one over frequency, inverted, mixed by the square root of ImageNet's colour
covariance and passed through a sigmoid. `jitter` and `random_scale` are the
transformations; `visualise` is the loop, Adam on the parameterisation's
tensors. `random_reference` is what the objective reads on images of uniform
random pixels, the yardstick the sections below compare against.

```python
# recipe-test: tool=torchlens expected=True
import torch.nn.functional as F


class SiteReader:
    """Run a model capture-free and keep one site's output, with its gradient."""

    def __init__(self, model, site):
        self.values = []
        self.bound = tl.when(site, self._keep).bind(model)

    def _keep(self, out, *, ctx):
        self.values.append(out)
        return out  # unchanged: the binding only reads

    def __call__(self, x):
        self.values.clear()
        self.bound(x)
        if len(self.values) != 1:
            raise ValueError(f"the site fired {len(self.values)} times; narrow the selector")
        return self.values[0]


# lucid's color_correlation_svd_sqrt, normalised by its largest column norm.
COLOR_SQRT = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
COLOR_SQRT = COLOR_SQRT / COLOR_SQRT.norm(dim=0).max()


class FourierImage:
    """An RGB image parameterised by a scaled spectrum in decorrelated colour."""

    def __init__(self, size, *, decay=1.0, sd=0.01, seed=0):
        height, width = size
        freqs = torch.sqrt(
            torch.fft.fftfreq(height)[:, None] ** 2 + torch.fft.rfftfreq(width)[None, :] ** 2
        )
        self.scale = 1.0 / freqs.clamp(min=1.0 / max(height, width)) ** decay
        noise = torch.randn(3, *freqs.shape, 2, generator=torch.Generator().manual_seed(seed))
        self.spectrum = (sd * noise).requires_grad_(True)
        self.size = size

    def parameters(self):
        return [self.spectrum]

    def value(self):
        spectrum = torch.view_as_complex(self.spectrum) * self.scale
        decorrelated = torch.fft.irfft2(spectrum, s=self.size, norm="ortho") / 4.0
        rgb = torch.einsum("dc,chw->dhw", COLOR_SQRT, decorrelated)
        return torch.sigmoid(rgb)[None]


def jitter(image, pixels, generator):
    dy, dx = (int(v) for v in torch.randint(-pixels, pixels + 1, (2,), generator=generator))
    return torch.roll(image, shifts=(dy, dx), dims=(2, 3))


def random_scale(image, factors, generator):
    factor = factors[int(torch.randint(len(factors), (1,), generator=generator))]
    return F.interpolate(image, scale_factor=factor, mode="bilinear", align_corners=False)


def visualise(reader, objective, parameterisation, *, steps=128, lr=0.05, transforms=(), seed=0):
    """Gradient ascent on objective(site output) over the parameterised input."""
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.Adam(parameterisation.parameters(), lr=lr)
    for _ in range(steps):
        x = parameterisation.value()
        for transform in transforms:
            x = transform(x, generator)
        loss = -objective(reader(x))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        x = parameterisation.value()
        return x, float(objective(reader(x)))


def random_reference(reader, objective, sample, *, count=8, seed=100):
    """Mean absolute objective over `count` random inputs drawn by `sample`."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        values = [abs(float(objective(reader(sample(generator))))) for _ in range(count)]
    return sum(values) / count


def random_image(generator):
    return torch.rand(1, 3, 16, 16, generator=generator)


RESULT = callable(visualise)
```

## A unit of any op: the planted detector

The objective is the centre unit of channel 0 at `conv2d_1_1`, whose receptive
field is the 5 by 5 patch at rows and columns 6 to 10. A single unit pins a
position, so this run uses no transformations: shifting the image would move
the patch away from the unit. The optimised patch should be the planted
pattern. The decorrelated colour basis gives luminance contrast most of its
scale, as natural images have; a colour-opponent version of the same stripes
(green against red and blue) reached a correlation of only about 0.73 after 512
steps in this model.

```python
# recipe-test: tool=torchlens expected=(True, True)
conv = SiteReader(cnn, conv_site)


def centre_unit(out):
    return out[0, 0, 6, 6]


image, value = visualise(conv, centre_unit, FourierImage((16, 16), seed=1))
reference = random_reference(conv, centre_unit, random_image)
patch = image[0, :, 6:11, 6:11]
correlation = float(torch.corrcoef(torch.stack([patch.flatten(), PATTERN.flatten()]))[0, 1])
print(f"unit {value:.3g}, random {reference:.3g} ({value / reference:.3g}x)")
print(f"correlation with the planted pattern {correlation:.3f}")
RESULT = (value >= 5 * reference, correlation >= 0.8)
```

```text
unit 37.6, random 1.74 (21.6x)
correlation with the planted pattern 1.000
```

## A functional op, with transformations

The same loop on `mul_1_3`, the SiLU's multiply, which no module returns. The
objective is again the centre unit of channel 0, and each step sees the image
shifted by up to one pixel and rescaled by 0.95, 1 or 1.05: the
transformation-robustness recipe of the 2017 article, at a size a 5 by 5
receptive field tolerates. Larger shifts move the patch away from the unit.

```python
# recipe-test: tool=torchlens expected=True
gate = SiteReader(cnn, gate_site)
transforms = (
    lambda x, g: jitter(x, 1, g),
    lambda x, g: random_scale(x, (0.95, 1.0, 1.05), g),
)
image, value = visualise(gate, centre_unit, FourierImage((16, 16), seed=2), transforms=transforms)
reference = random_reference(gate, centre_unit, random_image)
print(f"gated unit {value:.3g}, random {reference:.3g} ({value / reference:.3g}x)")
RESULT = value >= 5 * reference
```

```text
gated unit 25, random 1.34 (18.7x)
```

## Another kind of model: a unit of a transformer MLP

Nothing above is specific to images. A small transformer over input
embeddings: the input is a sequence of six embeddings, each kept at the norm
four (the width's square root), and the objective is unit 5 of the GELU in the
second block's MLP, a functional op, averaged over the sequence.

```python
# recipe-test: tool=torchlens expected=(('gelu_2_32',), True)
class Block(nn.Module):
    def __init__(self, width=16, heads=2):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.up, self.down = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, tokens, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class TinyTransformer(nn.Module):
    def __init__(self, width=16, depth=2, vocab=32):
        super().__init__()
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.norm, self.unembed = nn.LayerNorm(width), nn.Linear(width, vocab)

    def forward(self, embeddings):
        x = embeddings
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.norm(x))


class Embeddings:
    """A sequence of input embeddings, each held at norm sqrt(width)."""

    def __init__(self, tokens=6, width=16, *, seed=0):
        noise = torch.randn(1, tokens, width, generator=torch.Generator().manual_seed(seed))
        self.raw = noise.requires_grad_(True)

    def parameters(self):
        return [self.raw]

    def value(self):
        return self.raw / self.raw.norm(dim=-1, keepdim=True) * self.raw.shape[-1] ** 0.5


def random_embeddings(generator):
    noise = torch.randn(1, 6, 16, generator=generator)
    return noise / noise.norm(dim=-1, keepdim=True) * 4.0


torch.manual_seed(0)
lm = TinyTransformer().eval().requires_grad_(False)
mlp_site = tl.func("gelu") & tl.in_module("blocks.1")
labels = tl.trace(lm, random_embeddings(torch.Generator().manual_seed(0))).find_sites(mlp_site).labels()

mlp = SiteReader(lm, mlp_site)


def mlp_unit(out):
    return out[0, :, 5].mean()


embeddings, value = visualise(mlp, mlp_unit, Embeddings(seed=1))
reference = random_reference(mlp, mlp_unit, random_embeddings)
print(f"MLP unit {value:.3g}, random {reference:.3g} ({value / reference:.3g}x)")
RESULT = (labels, value >= 5 * reference)
```

```text
MLP unit 1.78, random 0.0721 (24.7x)
```

## An SAE latent

An SAE feature is a direction at a site, read through the SAE's encoder. The
stand-in SAE below has eight latents at the first block's output, each reading
one of eight orthonormal directions above a threshold of 1, as a trained SAE's
encoder bias does. The objective is latent 2; latent 5 is the control. Any SAE
with `encode` and `decode` works in its place, an SAELens SAE among them (see
[the pretrained-SAE recipe](pretrained_saes.md)). `tl.bridge.sae.splice` then
reads every latent on the optimised input:

```python
# recipe-test: tool=torchlens expected=(True, True)
from torchlens.bridge.sae import splice


class StandInSAE:
    def __init__(self, width=16, latents=8, seed=3):
        noise = torch.randn(width, latents, generator=torch.Generator().manual_seed(seed))
        directions = torch.linalg.qr(noise)[0]  # orthonormal columns
        self.W_enc, self.b_enc = directions, torch.full((latents,), -1.0)
        self.W_dec, self.b_dec = directions.T, torch.zeros(width)

    def encode(self, x):
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, latents):
        return latents @ self.W_dec + self.b_dec


sae = StandInSAE()
residual = SiteReader(lm, tl.module("blocks.0"))


def latent_2(out):
    return sae.encode(out)[0, :, 2].mean()


embeddings, _ = visualise(residual, latent_2, Embeddings(seed=2))
reference = random_reference(residual, latent_2, random_embeddings)
latents = splice(lm, embeddings, site="blocks.0", sae=sae).latents[0].mean(dim=0)
planted, control = float(latents[2]), float(latents[5])
print(f"latent 2 {planted:.3g}, random {reference:.3g} ({planted / reference:.3g}x)")
print(f"control latent 5 {control:.3g}")
RESULT = (planted >= 5 * reference, control <= 0.1 * planted)
```

```text
latent 2 0.965, random 0.00217 (444x)
control latent 5 0
```

## Real use: a pretrained torchvision model

Not run here: the weights download. A torchvision ImageNet model expects
normalised pixels, so a small wrapper normalises first; the wrapper shifts
module addresses by one level (`model.layer3` for `layer3`). The image is
larger, so the transformations are lucid's standard ones: pad, jitter by 8,
scale by 0.95 to 1.05, jitter by 4.

```text
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class Normalised(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


net = Normalised(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)).eval().requires_grad_(False)
reader = SiteReader(net, tl.module("model.layer3"))
transforms = (
    lambda x, g: F.pad(x, (12, 12, 12, 12), value=0.5),
    lambda x, g: jitter(x, 8, g),
    lambda x, g: random_scale(x, tuple(1 + (i - 5) / 100 for i in range(11)), g),
    lambda x, g: jitter(x, 4, g),
)
image, value = visualise(
    reader, lambda out: out[:, 7].mean(), FourierImage((128, 128)), steps=512, transforms=transforms
)
```

`image` is a `[1, 3, 128, 128]` tensor in `[0, 1]`; save it with
`torchvision.utils.save_image(image, "layer3-channel7.png")`. Any op works as
the site: run `tl.trace(net, torch.rand(1, 3, 224, 224))`, pick a label from
`trace.layer_labels`, and select it by its function and module as above.

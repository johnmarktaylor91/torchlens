# Prisma CLIP SAEs: ViT-Prisma loads, TorchLens splices

ViT-Prisma publishes sparse autoencoders for vision transformers, CLIP among
them, on the Hugging Face Hub (for example
`Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05`,
trained on the residual stream after block 11 of OpenCLIP's ViT-B/32). ViT-Prisma
loads the SAE; TorchLens traces the vision model, splices the SAE's
reconstruction in at the block it was trained on with
`torchlens.bridge.sae.splice` and reports what changed, as the
pretrained-SAE recipe (`docs/recipes/pretrained_saes.md`) does for language
models.

```text
pip install torchlens open_clip_torch timm
pip install "vit-prisma @ git+https://github.com/Prisma-Multimodal/ViT-Prisma@771494b8181a0f7779fab19adb5342688f95c724"
```

## Real use: a Prisma CLIP SAE

Not run here: the SAE (about 300 MB) and the CLIP weights download from the
Hub. A Prisma SAE's `forward` returns the reconstruction and the latents
together with its training losses, so a small adapter gives `splice` the
`encode` and `decode` pair it expects (`PrismaCodec`, defined and run in the
stand-in section below):

```text
import open_clip
import torch
from huggingface_hub import hf_hub_download
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.sae import SparseAutoencoder

from torchlens.bridge.sae import splice

repo = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05"
with torch.serialization.safe_globals([VisionModelSAERunnerConfig, torch.device]):
    sae = SparseAutoencoder.load_from_pretrained(hf_hub_download(repo, "weights.pt")).eval()

clip, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="datacomp_xl_s13b_b90k"
)
vision = clip.visual.eval()
images = torch.stack([preprocess(image) for image in my_images])  # your PIL images
result = splice(vision, images, site="transformer.resblocks.11", sae=PrismaCodec(sae))
result.fraction_variance_explained_pooled, result.output_delta_max
```

Prisma's `hook_resid_post` at layer `L` is the output of OpenCLIP's
`transformer.resblocks.L` (`blocks.L` in a timm ViT); `hook_mlp_out` is the
output of that block's `mlp`.

## The class token

A ViT's activations are `[images, tokens, features]` with the class token at
position 0. The Prisma CLIP SAEs above were trained on every token
(`cls_token_only: false` in their `config.json`), so the recipe splices every
position, the class token included, which is `splice`'s default.
`exclude_bos=True` treats position 0 as a language model's BOS token: it keeps
the class token clean and leaves it out of every metric, which suits an SAE
trained on patch tokens only.

## The same steps on a stand-in

Offline: a two-block timm ViT with random weights and a small ViT-Prisma
`SparseAutoencoder` with seeded weights, saved in Prisma's format and loaded
back with `SparseAutoencoder.load_from_pretrained`, as a downloaded one is.
Prisma pickles its config into the file, so the config class is allow-listed
for torch's weights-only loader instead of switching that check off.

```python
# recipe-test: tool=prisma expected=('SparseAutoencoder', 64, (2, 17, 16))
import contextlib
import io
import tempfile
from pathlib import Path

import torch
from timm.models.vision_transformer import VisionTransformer
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.sae import SparseAutoencoder

torch.manual_seed(0)
vit = VisionTransformer(
    img_size=32, patch_size=8, embed_dim=16, depth=2, num_heads=2, num_classes=10
).eval()
images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))

with contextlib.redirect_stdout(io.StringIO()):  # Prisma prints its config and the save path
    cfg = VisionModelSAERunnerConfig(
        d_in=16, expansion_factor=4, normalize_activations=None, b_dec_init_method="zeros",
        log_to_wandb=False, device="cpu",
    )
    built = SparseAutoencoder(cfg)
    with torch.no_grad():
        for parameter in built.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=torch.Generator().manual_seed(2)))
    path = str(Path(tempfile.mkdtemp()) / "weights.pt")
    built.save_model(path)
    with torch.serialization.safe_globals([VisionModelSAERunnerConfig, torch.device]):
        sae = SparseAutoencoder.load_from_pretrained(path).eval()


class PrismaCodec:
    """A ViT-Prisma SparseAutoencoder as the encode/decode pair splice expects."""

    def __init__(self, sae):
        self.sae = sae

    def encode(self, x):
        return self.sae(x)[1]  # forward returns (reconstruction, latents, losses...)

    def decode(self, latents):
        out = latents @ self.sae.W_dec + self.sae.b_dec
        return self.sae.run_time_activation_norm_fn_out(out)


with torch.no_grad():
    block_out = vit.blocks[1](vit.blocks[0](vit._pos_embed(vit.patch_embed(images))))
RESULT = (type(sae).__name__, sae.d_sae, tuple(block_out.shape))
```

## Splice at the block

The site is the output of block 1, `[images, 17 tokens, 16 features]`: the
class token and 16 patches. The splice's reconstruction equals Prisma's own
forward on the same activations:

```python
# recipe-test: tool=prisma expected=(True, True, True, False)
import torchlens as tl
from torchlens.bridge.sae import splice

trace = tl.trace(vit, images, capture=tl.options.CaptureOptions(intervention_ready=True))
result = splice(trace, site="blocks.1", sae=PrismaCodec(sae))
with torch.no_grad():
    own_reconstruction, own_latents = sae(result.clean_site_out)[:2]
RESULT = (
    bool(torch.allclose(result.clean_site_out, block_out, atol=1e-6)),
    bool(torch.allclose(result.reconstruction, own_reconstruction, atol=1e-6)),
    bool(torch.equal(result.latents, own_latents)),
    result.zero_variance,
)
```

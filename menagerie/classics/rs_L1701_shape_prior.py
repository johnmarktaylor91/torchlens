# SOURCE: real class from installed base lib `diffusers` (no architectural
# modification) -- `diffusers.models.transformers.prior_transformer.PriorTransformer`.
# https://github.com/huggingface/diffusers -- diffusers/models/transformers/prior_transformer.py
# This is the ACTUAL diffusion-prior network used by OpenAI's Shap-E
# (https://github.com/openai/shap-e, arXiv:2305.02463) as shipped in HuggingFace's
# `openai/shap-e` checkpoint / `diffusers.ShapEPipeline`. `ShapEPipeline.__call__`
# (diffusers/pipelines/shap_e/pipeline_shap_e.py) runs the reverse-diffusion denoising
# loop entirely through `self.prior(...)`, a plain `PriorTransformer` instance
# configured with Shap-E's hyperparameters (encoder_hid_proj_type=None,
# added_emb_type=None, norm_in_type="layer", time_embed_act_fn="gelu") -- the same
# generic prior-transformer class diffusers also uses for unCLIP/Kandinsky. Shap-E's
# other novel contribution -- the ShapERenderer / NeRF+STF MLP + mesh decoder in
# diffusers/pipelines/shap_e/renderer.py -- performs volumetric ray marching
# (`render_rays`) rather than a single static forward graph, so the traced entry point
# here is the actual generative prior network (the part that does the text/image ->
# 3D-latent diffusion denoising), constructed at a tiny size for fast tracing but with
# the real, unmodified `PriorTransformer` architecture and Shap-E's real kwargs.
import torch

MENAGERIE_ZOO = "vendored-pytorch"


def build_shape_prior():
    from diffusers import PriorTransformer

    torch.manual_seed(0)
    return PriorTransformer(
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=2,
        embedding_dim=32,
        num_embeddings=1024,
        additional_embeddings=0,
        time_embed_act_fn="gelu",
        norm_in_type="layer",
        encoder_hid_proj_type=None,
        added_emb_type=None,
    )


class ShapEPriorForward(torch.nn.Module):
    """Thin forward-only wrapper: PriorTransformer.forward returns a dataclass and
    needs a fixed timestep + proj_embedding companion tensor, matching how
    ShapEPipeline.__call__ invokes `self.prior(...)` inside its denoising loop."""

    def __init__(self):
        super().__init__()
        self.prior = build_shape_prior()

    def forward(self, sample):
        timestep = torch.tensor([5])
        proj_embedding = torch.randn(sample.shape[0], 32)
        return self.prior(
            sample, timestep=timestep, proj_embedding=proj_embedding
        ).predicted_image_embedding


def build_shape_prior_wrapper():
    torch.manual_seed(0)
    return ShapEPriorForward()


def example_input_shape_prior():
    torch.manual_seed(0)
    return (torch.randn(1, 1024, 32),)


MENAGERIE_ENTRIES = [
    ("Shap-E", build_shape_prior_wrapper, example_input_shape_prior, 2023, "vendored-pytorch"),
]

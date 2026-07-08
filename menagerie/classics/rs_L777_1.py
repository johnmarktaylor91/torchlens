# SOURCE: vendored from ibm-research/GP-MoLFormer-Uniq (HuggingFace Hub, trust_remote_code
# custom modeling file) @ sha 6eca879581e2302b4e1ab07bb02908636bddb4a2
# https://huggingface.co/ibm-research/GP-MoLFormer-Uniq (modeling_molformer.py +
# configuration_molformer.py). GP-MolFormer (IBM, arXiv:2405.04912) is a 1.1B-parameter
# generative molecular language model: MolformerForCausalLM, a decoder-only transformer
# with rotary position embeddings and *linear* (kernelized random-feature) causal
# self-attention in place of quadratic softmax attention. This is a genuinely custom
# architecture (not a stock GPT2/BERT/etc config) shipped as HF `trust_remote_code`
# modeling code, so it is vendored here verbatim (relative import fixed only).
import importlib.util
import sys
from pathlib import Path

import torch

_STAGING_DIR = Path(__file__).resolve().parent


def _load_sibling(module_name: str, filename: str):
    """Load a sibling staging file as a top-level module (staging files are loaded via
    spec_from_file_location without a package parent, so relative imports don't resolve;
    this mirrors that same file-based loading for the vendored dependency modules)."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _STAGING_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_gpmolformer_configuration = _load_sibling(
    "_gpmolformer_configuration", "_gpmolformer_configuration.py"
)
_gpmolformer_modeling = _load_sibling("_gpmolformer_modeling", "_gpmolformer_modeling.py")

MolformerConfig = _gpmolformer_configuration.MolformerConfig
MolformerForCausalLM = _gpmolformer_modeling.MolformerForCausalLM

MENAGERIE_ZOO = "vendored-pytorch"


def build_gp_molformer():
    """Tiny random-init GP-MoLFormer (MolformerForCausalLM) for tracing."""
    config = MolformerConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        num_random_features=8,
        is_decoder=True,
        use_cache=False,
    )
    model = MolformerForCausalLM(config)
    model.eval()
    return model


def example_input_gp_molformer():
    torch.manual_seed(0)
    return torch.randint(0, 64, (1, 16))


MENAGERIE_ENTRIES = [
    (
        "GP-MolFormer",
        build_gp_molformer,
        example_input_gp_molformer,
        2024,
        "vendored-pytorch",
    ),
]

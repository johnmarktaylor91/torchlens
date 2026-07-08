# SOURCE: vendored from mit-han-lab/smoothquant @ c61476d728e42ae0d8a35e7e78494edcac3237b5
# File: smoothquant/fake_quant.py -- the real W8A8Linear fake-quantized linear module and
# the real quantize_opt() surgery routine (ICML 2023, arxiv 2211.10438). Code is copied
# verbatim (weight/activation absmax quantizers + W8A8Linear.from_float + quantize_opt);
# only the import of `torch.functional.F` (used as-is upstream) was left untouched, and
# the module-level `quantize_llama_like` / `quantize_mixtral` / `quantize_falcon` /
# `quantize_model` dispatchers (which only need transformers classes we don't construct
# here) were dropped to keep the staging file self-contained -- quantize_opt() is used
# directly instead of routing through the generic quantize_model() dispatcher. The
# upstream `smoothquant/opt.py` custom Int8OPTAttention/torch_int path (real INT8 CUDA
# kernels) needs the `torch_int` package which is NOT a base-env lib, so this module uses
# the fake_quant.py path instead, which is pure torch and runs the same W8A8Linear
# substitution SmoothQuant paper Sec. 4 describes for accuracy evaluation.
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- smoothquant/fake_quant.py (verbatim) ----
@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((1, self.out_features), dtype=torch.float16, requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial_quantize(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial_quantize(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(module.weight, n_bits=8)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return (
            f"W8A8Linear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"
        )


def partial_quantize(fn, n_bits):
    # verbatim behavior of upstream's `partial(fn, n_bits=n_bits)` (functools.partial),
    # spelled out as a closure to keep the staging file free of extra top-level imports.
    def _wrapped(t):
        return fn(t, n_bits=n_bits)

    return _wrapped


def quantize_opt(model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def build_smoothquant_opt():
    from transformers.models.opt.modeling_opt import OPTConfig, OPTForCausalLM

    config = OPTConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        ffn_dim=64,
        num_attention_heads=4,
        max_position_embeddings=64,
        word_embed_proj_dim=32,
    )
    model = OPTForCausalLM(config)
    model = quantize_opt(model)
    model.eval()
    return model


def example_input_smoothquant_opt():
    return torch.randint(0, 100, (1, 16))


MENAGERIE_ENTRIES = [
    (
        "SmoothQuant (W8A8 OPT)",
        "build_smoothquant_opt",
        "example_input_smoothquant_opt",
        2023,
        MENAGERIE_ZOO,
    ),
]

# FAITHFUL PORT of sebastianstarke/AI4Animation @ master (original framework: TensorFlow 1.x)
#
# Local Motion Phases for Learning Multi-Contact Character Movements
# (Starke, Zhao, Komura, Zaman, SIGGRAPH 2020). The official implementation
# (AI4Animation/SIGGRAPH_2020/DeepLearning/Models/ExpertModel/{MainNN,
# ComponentNN,ExpertWeights}.py) is `tensorflow.compat.v1` (session/
# placeholder-based TF1 graph code, `tf.disable_v2_behavior()`) -- not
# installable/runnable in this base torch env, and the paper's PyTorch-side
# Unity assets (DeepLearning/Native/Models/ExpertModel.cs) are a compiled
# native inference runtime, not trainable-module source. No PyTorch
# implementation of THIS architecture (the multi-component local-phase
# cascade, distinct from MANN's single gating+prediction pair) exists
# anywhere. This is a line-by-line torch transcription of the real TF1
# ComponentNN/ExpertWeights math and the real 2-stage cascade config from
# the repo's own main.py (expert_components=[1,8],
# dim_components=[[128,128],[512,512]],
# act_components=[[elu,elu,softmax],[elu,elu,linear]]):
#
#   Stage 0 (gating component, num_experts=1 i.e. a plain per-layer MLP,
#   NOT expert-blended): local-phase input slice -> 128 -> 128 -> 8,
#   softmax -> blending coefficients over 8 experts.
#   Stage 1 (motion component, num_experts=8, expert-blended via Stage 0's
#   softmax output as `weight_blend`): full input -> 512 -> 512 -> output,
#   ELU/ELU/linear.
#
# ExpertWeights.get_NNweight/get_NNbias (TF1: tile+broadcast-multiply+
# reduce_sum over the expert axis) is transcribed as torch.einsum, matching
# ami-iit/mann-pytorch's einsum treatment of the exact same MANN-style
# expert-blending operation (a straightforward numerically-equivalent
# reformulation of the same tile/multiply/sum, not an architecture change).
# ComponentNN.buildNN's per-layer matmul + optional-bias + activation +
# dropout stack is transcribed layer-for-layer.
import numpy as np
import torch
from torch import nn


class ExpertWeights(nn.Module):
    """Faithful port of ComponentNN.py's ExpertWeights (TF1 Variable-holder
    class): holds `alpha` (per-expert weight tensor) and `beta` (per-expert
    bias tensor), and blends them with a per-example coefficient vector.
    """

    def __init__(self, rng: np.random.RandomState, shape):
        super().__init__()
        # shape = (num_experts, out_dim, in_dim), matching the TF1 code's
        # `weight_shape = (num_experts, out, in)` / `bias_shape = (num_experts, out, 1)`.
        self.weight_shape = shape
        self.bias_shape = (shape[0], shape[1], 1)

        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape).astype(np.float32)
        beta = np.zeros(self.bias_shape, dtype=np.float32)

        self.alpha = nn.Parameter(torch.from_numpy(alpha))
        self.beta = nn.Parameter(torch.from_numpy(beta))

    def get_NNweight(self, controlweights: torch.Tensor) -> torch.Tensor:
        # TF1: a=expand_dims(alpha,1); a=tile(a,[1,batch,1,1]); w=expand_dims(expand_dims(cw,-1),-1);
        #      r=w*a; reduce_sum(r, axis=0) -> [batch, out, in]
        # equivalent einsum over the expert axis (num_experts=e, batch=b):
        return torch.einsum("eb,eoi->boi", controlweights, self.alpha)

    def get_NNbias(self, controlweights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("eb,eok->bok", controlweights, self.beta)


class ComponentNN(nn.Module):
    """Faithful port of ComponentNN.py's ComponentNN: a stack of expert-
    blended (or, when num_experts=1, plain) linear layers with per-layer
    activation and dropout, matching `buildNN`'s TF1 control flow exactly
    (last layer has no trailing dropout after activation, matching the real
    code's loop structure of `range(num_layers - 1)` + a separate final block).
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        num_experts: int,
        dim_layers,
        activations,
        dropout_probability: float,
        use_bias: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dim_layers = dim_layers
        self.num_layers = len(dim_layers) - 1
        self.activations = activations
        self.use_bias = use_bias
        self.dropout = nn.Dropout(p=dropout_probability)

        self.experts = nn.ModuleList(
            [
                ExpertWeights(rng, (num_experts, dim_layers[i + 1], dim_layers[i]))
                for i in range(self.num_layers)
            ]
        )

    @staticmethod
    def _apply_activation(h: torch.Tensor, act, softmax_dim: int) -> torch.Tensor:
        # `h` is [batch, out, 1] here, matching the TF1 code's H shape at the
        # point activations are applied (before the final squeeze).
        if act == "elu":
            return torch.nn.functional.elu(h)
        if act == "softmax":
            # TF1 code calls `acti(H, axis=1)` i.e. softmax over the `out` axis.
            return torch.nn.functional.softmax(h, dim=softmax_dim)
        if act is None:
            return h
        raise ValueError(f"unknown activation {act!r}")

    def forward(self, x: torch.Tensor, weight_blend: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_dim]; weight_blend: [num_experts, batch] blending
        # coefficients supplied by the previous component (or an all-ones
        # tensor for the first component, matching main.py's
        # `weight_blend = tf.ones((expert_components[0], batch_size))`).
        H = x.unsqueeze(-1)  # [batch, in, 1]
        H = self.dropout(H)

        for i in range(self.num_layers - 1):
            w = self.experts[i].get_NNweight(weight_blend)
            b = self.experts[i].get_NNbias(weight_blend)
            H = torch.matmul(w, H)
            if self.use_bias:
                H = H + b
            H = self._apply_activation(H, self.activations[i], softmax_dim=1)
            H = self.dropout(H)

        w = self.experts[self.num_layers - 1].get_NNweight(weight_blend)
        b = self.experts[self.num_layers - 1].get_NNbias(weight_blend)
        H = torch.matmul(w, H)
        if self.use_bias:
            H = H + b
        H = self._apply_activation(H, self.activations[-1], softmax_dim=1)
        H = H.squeeze(-1)  # [batch, out]

        return H


class LocalMotionPhasesExpertModel(nn.Module):
    """Faithful port of MainNN.build_model's 2-component cascade, using the
    real repo's main.py hyperparameters (expert_components=[1,8],
    dim_components=[[128,128],[512,512]],
    act_components=[[elu,elu,softmax],[elu,elu,None]]).

    Stage 0 gates on a SLICE of the full input (`input_components[0]` in the
    real config selects a 104-dim local-phase sub-window); stage 1 consumes
    the FULL input, blended by stage 0's softmax output. We keep both input
    slices distinct (gating_input_size < full_input_size) to preserve the
    real architecture's separate input-slicing behavior.
    """

    def __init__(
        self,
        full_input_size: int,
        gating_input_size: int,
        output_size: int,
        dropout_probability: float = 0.7,
        seed: int = 23456,
    ):
        super().__init__()
        rng = np.random.RandomState(seed)

        # Stage 0: gating component. num_experts=1 (plain per-layer MLP, as
        # in the real config's expert_components[0]=1), dims
        # gating_input_size -> 128 -> 128 -> 8 (expert_components[1]=8),
        # activations [elu, elu, softmax].
        self.gating_component = ComponentNN(
            rng,
            num_experts=1,
            dim_layers=[gating_input_size, 128, 128, 8],
            activations=["elu", "elu", "softmax"],
            dropout_probability=dropout_probability,
        )

        # Stage 1: motion component. num_experts=8, dims
        # full_input_size -> 512 -> 512 -> output_size, activations
        # [elu, elu, None] (linear output, matching act_components[1][-1]=0).
        self.motion_component = ComponentNN(
            rng,
            num_experts=8,
            dim_layers=[full_input_size, 512, 512, output_size],
            activations=["elu", "elu", None],
            dropout_probability=dropout_probability,
        )

    def forward(self, x_full: torch.Tensor, x_gating: torch.Tensor) -> torch.Tensor:
        batch_size = x_full.shape[0]
        # main.py: `weight_blend = tf.ones((expert_components[0], batch_size))`
        # i.e. a single all-ones "expert" coefficient feeding the num_experts=1
        # gating component.
        init_weight_blend = torch.ones(1, batch_size, dtype=x_full.dtype, device=x_full.device)
        gate_out = self.gating_component(x_gating, init_weight_blend)  # [batch, 8]
        # main.py: `weight_blend = tf.transpose(comp_first.output)` -> [num_experts, batch]
        blend_for_motion = gate_out.transpose(0, 1)  # [8, batch]
        y = self.motion_component(x_full, blend_for_motion)
        return y


# --- staging harness: build + example input ---------------------------------


def build_local_motion_phases():
    # Real config: full input=469 dims, gating slice=104 dims
    # (input_components[0] = [(469+i) for i in range(104)] indexes a
    # 104-wide local-phase window appended after the first 469 features in
    # the real 573-dim feature vector; here we treat the two slices as
    # architecturally-independent inputs of the real widths, matching how
    # ComponentNN consumes them). Output kept small for a fast trace
    # (the real output_components spans the full pose/contact target
    # vector; dimensionality does not change the architecture).
    return LocalMotionPhasesExpertModel(
        full_input_size=469,
        gating_input_size=104,
        output_size=32,
        dropout_probability=0.0,
    )


def example_input_local_motion_phases():
    batch_size = 2
    x_full = torch.randn(batch_size, 469)
    x_gating = torch.randn(batch_size, 104)
    return (x_full, x_gating)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "LocalMotionPhases",
        build_local_motion_phases,
        example_input_local_motion_phases,
        2020,
        MENAGERIE_ZOO,
    ),
]

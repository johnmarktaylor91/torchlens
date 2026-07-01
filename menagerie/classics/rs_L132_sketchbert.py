# SOURCE: vendored from avalonstrel/SketchBERT @ master
# (models/SketchTransformer/models/networks.py)
"""Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by
Self-supervised Learning of Sketch Gestalt.

Lin, Hangyu, et al. arXiv:2005.09159. Official PyTorch implementation confirmed (repo README:
"the sketchbert encoder(transformer part)" is exactly `SketchTransformerModel` below, loaded
via `which_pretrained: ['enc_net']`).

Sketch-BERT generalizes BERT to vector-format sketches: a stroke-point sequence
(here `sketch_embed_type='linear'`, a per-point (dx, dy, pen-state) 5-D vector; the repo also
supports a `'discrete'` embedding variant for tokenized coordinates, omitted here as it needs
`length_to_mask` from a segment-level attention path not used by the base 'bert' model_type)
is linearly embedded, given a learned/sinusoidal position embedding, refined through a small
FC stack (`SketchEmbeddingRefineNetwork`, an ALBERT-style factorized-embedding upsampler), and
passed through a stack of standard (pre-LN-free, BERT-style) multi-head self-attention +
feed-forward transformer layers (`SketchEncoder`). `model_type='albert'` selects the
parameter-shared `SketchALEncoder` variant (all layers reuse one `SketchLayer`); this file
vendors both. Only the classes needed to build+run `SketchTransformerModel` are vendored
verbatim from the real repo file; unrelated GAN/VAE/CNN/segment-attention heads defined
alongside it in the same source file are omitted (not needed to reproduce the Sketch-BERT
encoder itself).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/SketchTransformer/models/networks.py (verbatim, encoder-path classes)
# ---------------------------------------------------------------------------
def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SketchLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(SketchLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}
NORM2FN = {"BN1d": nn.BatchNorm1d, "BN2d": nn.BatchNorm2d, "LN": nn.LayerNorm}


class SketchSelfAttention(nn.Module):
    """
    Implementation for self attention in Sketch.
    The input will be a K-Dim feature.
    Input Parameters:
        config[dict]:
            hidden_dim[int]: The dimension of input hidden embeddings in the self attention, hidden diension is equal to the output dimension
            num_heads[int]: The number of heads
            attention_probs[float]: probability parameter for dropout
    """

    def __init__(self, num_heads, hidden_dim, attention_dropout_prob):
        super(SketchSelfAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads)
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = int(self.hidden_dim / self.num_heads)
        self.all_head_dim = self.head_dim * self.num_heads
        self.scale_factor = math.sqrt(self.head_dim)

        self.query = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.key = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.value = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.dropout = nn.Dropout(attention_dropout_prob)
        self.multihead_output = None

    def transpose_(self, x):
        """
        Transpose Function for simplicity.
        """
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        output_attentions=False,
        keep_multihead_output=False,
    ):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        multi_query = self.transpose_(query)
        multi_key = self.transpose_(key)
        multi_value = self.transpose_(value)

        attention_scores = torch.matmul(multi_query, multi_key.transpose(-1, -2))
        attention_scores = attention_scores / self.scale_factor
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_states = torch.matmul(attention_probs, multi_value)

        if keep_multihead_output:
            self.multihead_output = context_states
            self.multihead_output.retain_grad()

        context_states = context_states.permute(0, 2, 1, 3)
        context_states = context_states.contiguous().view(context_states.size()[:-2] + (-1,))

        if output_attentions:
            return context_states, attention_probs
        return context_states


class SketchOutput(nn.Module):
    def __init__(self, input_dim, output_dim, attention_norm_type, output_dropout_prob):
        super(SketchOutput, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if attention_norm_type not in NORM2FN:
            raise ValueError("The attention normalization is not in standard normalization types.")
        self.norm = NORM2FN[attention_norm_type](output_dim)
        self.dropout = nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, input_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_states)
        return hidden_states


class SketchMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        attention_norm_type,
        attention_dropout_prob,
        hidden_dropout_prob,
    ):
        super(SketchMultiHeadAttention, self).__init__()
        self.attention = SketchSelfAttention(num_heads, hidden_dim, attention_dropout_prob)
        self.output = SketchOutput(hidden_dim, hidden_dim, attention_norm_type, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):
        input_states = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask, head_mask=head_mask)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        output_states = self.output(hidden_states, input_states)
        if output_attentions:
            return output_states, attention_probs

        return output_states


class SketchIntermediate(nn.Module):
    def __init__(self, hidden_dim, inter_dim, inter_activation):
        super(SketchIntermediate, self).__init__()
        self.fc = nn.Linear(hidden_dim, inter_dim)
        self.activation = ACT2FN[inter_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(next(self.fc.parameters()).device)
        inter_states = self.fc(hidden_states.contiguous())
        inter_states = self.activation(inter_states)
        return inter_states


class SketchLayer(nn.Module):
    """
    A transformer layer for sketch bert
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        inter_dim,
        attention_norm_type,
        inter_activation,
        attention_dropout_prob,
        hidden_dropout_prob,
        output_dropout_prob,
    ):
        super(SketchLayer, self).__init__()
        self.attention = SketchMultiHeadAttention(
            num_heads, hidden_dim, attention_norm_type, attention_dropout_prob, hidden_dropout_prob
        )
        self.inter_layer = SketchIntermediate(hidden_dim, inter_dim, inter_activation)
        self.output = SketchOutput(inter_dim, hidden_dim, attention_norm_type, output_dropout_prob)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):
        hidden_states = self.attention(hidden_states, attention_mask, head_mask)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        inter_states = self.inter_layer(hidden_states)
        output_states = self.output(inter_states, hidden_states)

        if output_attentions:
            return output_states, attention_probs

        return output_states


def setting2dict(paras, setting):
    paras["num_heads"] = setting[0]
    paras["hidden_dim"] = setting[1]
    paras["inter_dim"] = setting[2]


class SketchEncoder(nn.Module):
    """
    layers_setting[list]: [[12, ], []]
    """

    def __init__(
        self,
        layers_setting,
        attention_norm_type,
        inter_activation,
        attention_dropout_prob,
        hidden_dropout_prob,
        output_dropout_prob,
    ):
        super(SketchEncoder, self).__init__()
        layer_paras = {
            "attention_norm_type": attention_norm_type,
            "inter_activation": inter_activation,
            "attention_dropout_prob": attention_dropout_prob,
            "hidden_dropout_prob": hidden_dropout_prob,
            "output_dropout_prob": output_dropout_prob,
        }
        self.layers = []
        for layer_setting in layers_setting:
            setting2dict(layer_paras, layer_setting)
            self.layers.append(SketchLayer(**layer_paras))
        self.layers = nn.ModuleList(self.layers)

    def forward(
        self,
        input_states,
        attention_mask,
        head_mask=None,
        output_all_states=False,
        output_attentions=False,
        keep_multihead_output=False,
    ):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states


class SketchALEncoder(nn.Module):
    """
    A Lite BERT: Parameter Sharing
    layers_setting[list]: [[12, ], []]
    """

    def __init__(
        self,
        layers_setting,
        attention_norm_type,
        inter_activation,
        attention_dropout_prob,
        hidden_dropout_prob,
        output_dropout_prob,
    ):
        super(SketchALEncoder, self).__init__()
        layer_paras = {
            "attention_norm_type": attention_norm_type,
            "inter_activation": inter_activation,
            "attention_dropout_prob": attention_dropout_prob,
            "hidden_dropout_prob": hidden_dropout_prob,
            "output_dropout_prob": output_dropout_prob,
        }
        setting2dict(layer_paras, layers_setting[0])
        self.sketch_layer = SketchLayer(**layer_paras)
        self.layers = []
        for layer_setting in layers_setting:
            self.layers.append(self.sketch_layer)

    def forward(
        self,
        input_states,
        attention_mask,
        head_mask=None,
        output_all_states=False,
        output_attentions=False,
        keep_multihead_output=False,
    ):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states


class SketchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SketchEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, input_states):
        return self.embedding(input_states)


class SketchDiscreteEmbedding(nn.Module):
    """
    max_size[tuple](x_length, y_length)
    """

    def __init__(self, max_size, type_size, hidden_dim, pool_type):
        super(SketchDiscreteEmbedding, self).__init__()
        self.x_embedding = nn.Embedding(2 * max_size[0] + 2, hidden_dim // 2)
        self.y_embedding = nn.Embedding(2 * max_size[1] + 2, hidden_dim // 2)
        self.type_embedding = nn.Embedding(type_size + 1, hidden_dim)
        assert pool_type in ["sum", "con"]
        self.pool_type = pool_type

    def forward(self, input_states):
        input_states = input_states.to(dtype=torch.long)
        input_states = input_states + 1
        x_hidden = self.x_embedding(input_states[:, :, 0])
        y_hidden = self.y_embedding(input_states[:, :, 1])
        axis_hidden = torch.cat([x_hidden, y_hidden], dim=2)

        type_hidden = self.type_embedding(input_states[:, :, 2])

        if self.pool_type == "sum":
            return axis_hidden + type_hidden
        elif self.pool_type == "con":
            return torch.cat([axis_hidden, type_hidden], dim=2)


class SketchSinPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchSinPositionEmbedding, self).__init__()
        self.pos_embedding_matrix = torch.zeros(max_length, pos_hidden_dim)
        pos_vector = torch.arange(max_length).view(max_length, 1).type(torch.float)
        dim_vector = torch.arange(pos_hidden_dim).type(torch.float) + 1.0
        self.pos_embedding_matrix[:, ::2] = torch.sin(
            pos_vector / (dim_vector[::2] / 2).view(1, -1)
        )
        self.pos_embedding_matrix[:, 1::2] = torch.cos(
            pos_vector / ((dim_vector[1::2] - 1) / 2).view(1, -1)
        )

    def forward(self, position_labels):
        return self.pos_embedding_matrix[position_labels.view(-1), :].view(
            position_labels.size(0), position_labels.size(1), -1
        )


class SketchLearnPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchLearnPositionEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_length, pos_hidden_dim)

    def forward(self, position_labels):
        return self.pos_embedding(position_labels)


class SketchEmbeddingRefineNetwork(nn.Module):
    """
    The module to upsample the embedding feature, idea from the ALBERT: Factorized Embedding
    """

    def __init__(self, out_dim, layers_dim):
        super(SketchEmbeddingRefineNetwork, self).__init__()
        self.layers = []
        layers_dim = layers_dim.copy()
        layers_dim.append(out_dim)

        for i in range(len(layers_dim) - 1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_state):
        x = input_state
        for layer in self.layers:
            x = layer(x)
        return x


class SketchTransformerModel(nn.Module):
    """
    Input:
        layers_setting[list]
        input_dim[int]
        max_length[int]
        position_type[str]
        attention_norm_type[str]
        inter_activation[str]
        attention_dropout_prob[float]
        hidden_dropout_prob[float]
        output_dropout_prob[float]
    """

    def __init__(
        self,
        model_type,
        layers_setting,
        embed_layers_setting,
        input_dim,
        max_length,
        max_size,
        type_size,
        position_type,
        segment_type,
        sketch_embed_type,
        embed_pool_type,
        attention_norm_type,
        inter_activation,
        attention_dropout_prob,
        hidden_dropout_prob,
        output_dropout_prob,
    ):
        super(SketchTransformerModel, self).__init__()
        self.layers_setting = layers_setting
        self.num_hidden_layers = len(layers_setting)
        self.embed_pool_type = embed_pool_type
        assert sketch_embed_type in ["linear", "discrete"]

        if sketch_embed_type == "linear":
            self.embedding = SketchEmbedding(input_dim, embed_layers_setting[0])
        elif sketch_embed_type == "discrete":
            self.embedding = SketchDiscreteEmbedding(
                max_size, type_size, embed_layers_setting[0], embed_pool_type
            )
        assert position_type in ["sin", "learn", "none"]

        if position_type == "sin":
            self.pos_embedding = SketchSinPositionEmbedding(max_length, embed_layers_setting[0])
        elif position_type == "learn":
            self.pos_embedding = SketchLearnPositionEmbedding(max_length, embed_layers_setting[0])
        else:
            self.pos_embedding = None
        if segment_type == "learn":
            self.segment_embedding = SketchLearnPositionEmbedding(
                max_length, embed_layers_setting[0]
            )
        else:
            self.segment_embedding = None

        self.embed_refine_net = SketchEmbeddingRefineNetwork(
            layers_setting[0][1], embed_layers_setting
        )

        assert model_type in ["albert", "bert"]
        if model_type == "albert":
            self.encoder = SketchALEncoder(
                layers_setting,
                attention_norm_type,
                inter_activation,
                attention_dropout_prob,
                hidden_dropout_prob,
                output_dropout_prob,
            )
        elif model_type == "bert":
            self.encoder = SketchEncoder(
                layers_setting,
                attention_norm_type,
                inter_activation,
                attention_dropout_prob,
                hidden_dropout_prob,
                output_dropout_prob,
            )

    def get_pos_states(self, input_states):
        return (
            torch.arange(input_states.size(1))
            .view(1, -1)
            .repeat(input_states.size(0), 1)
            .to(device=input_states.device)
        )

    def forward(
        self,
        input_states,
        attention_mask,
        segments=None,
        head_mask=None,
        output_all_states=False,
        output_attentions=False,
        keep_multihead_output=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(input_states.size(0), input_states.size(1))
        if len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype, device=input_states.device
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype, device=input_states.device
            )
        else:
            head_mask = None

        input_states = self.embedding(input_states)

        if self.pos_embedding is not None:
            pos_states = self.pos_embedding(self.get_pos_states(input_states))
            input_states = input_states + pos_states.to(device=input_states.device)

        if self.segment_embedding is not None and segments is not None:
            segment_states = self.segment_embedding(segments)
            input_states = input_states + segment_states
        input_states = self.embed_refine_net(input_states)
        output_states = self.encoder(
            input_states,
            attention_mask,
            head_mask,
            output_all_states,
            output_attentions,
            keep_multihead_output,
        )

        if output_attentions:
            output_states, attention_probs = output_states
            return output_states[-1], attention_probs

        return output_states[-1]


# ---------------------------------------------------------------------------
# Menagerie build/example helpers
# ---------------------------------------------------------------------------
def build_sketchbert():
    # Tiny config: 2 transformer layers, 4 heads, 64-d hidden -- mirrors the repo's
    # `sketch_transformer.yml` shape (linear embedding of (dx, dy, pen-state) points,
    # sinusoidal position embedding, BERT-style encoder) at a menagerie-appropriate scale.
    layers_setting = [[4, 64, 256], [4, 64, 256]]
    model = SketchTransformerModel(
        model_type="bert",
        layers_setting=layers_setting,
        embed_layers_setting=[64],
        input_dim=5,
        max_length=64,
        max_size=(128, 128),
        type_size=4,
        position_type="sin",
        segment_type="none",
        sketch_embed_type="linear",
        embed_pool_type="sum",
        attention_norm_type="LN",
        inter_activation="gelu",
        attention_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        output_dropout_prob=0.1,
    )
    model.eval()
    return model


def example_input_sketchbert():
    return (torch.randn(1, 32, 5), torch.ones(1, 32))


MENAGERIE_ENTRIES = [
    ("Sketch-BERT", build_sketchbert, example_input_sketchbert, 2020, MENAGERIE_ZOO),
]

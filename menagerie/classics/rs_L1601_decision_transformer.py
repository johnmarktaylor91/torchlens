# SOURCE: vendored from https://github.com/kzl/decision-transformer @ master
# (gym/decision_transformer/models/decision_transformer.py, gym/decision_transformer/models/model.py,
#  gym/decision_transformer/models/trajectory_gpt2.py: Attention, MLP, Block, GPT2PreTrainedModel,
#  GPT2Model, TrajectoryModel, DecisionTransformer)
#
# Decision Transformer (Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas, Mordatch;
# NeurIPS 2021, "Decision Transformer: Reinforcement Learning via Sequence Modeling"). Offline RL
# reframed as return-conditioned sequence modeling: a causal GPT-2 transformer autoregressively
# predicts actions conditioned on a stacked (Return-to-go, state, action) token sequence. This is
# the official kzl/decision-transformer repo, and its `DecisionTransformer` module composes
# real torch/`transformers` classes -- vendored verbatim.
#
# The repo's own `trajectory_gpt2.GPT2Model` is a light fork of HuggingFace's GPT2Model with the
# absolute positional-embedding table (`wpe`) removed from the forward pass (comment: "note: the
# only difference between this GPT2Model and the default Huggingface version is that the
# positional embeddings are removed (since we'll add those ourselves)" -- DecisionTransformer adds
# its own `embed_timestep` in place of GPT2's positions). `Attention`/`MLP`/`Block`/
# `GPT2PreTrainedModel`/`GPT2Model` are vendored verbatim from that file. Two import paths were
# updated to match current `transformers` (>=4.31 moved `Conv1D` from
# `transformers.modeling_utils` to `transformers.pytorch_utils`; no other transformers API
# changed) -- no architecture/layer was altered. `GPT2DoubleHeadsModel`/`SequenceSummary` (unused
# by DecisionTransformer, which only instantiates `GPT2Model`) and the TF-checkpoint loader
# `load_tf_weights_in_gpt2` (requires `tensorflow`, dead code on the traced forward path) are
# dropped as genuinely unreachable from `DecisionTransformer.forward`.
#
# `TrajectoryModel`/`DecisionTransformer` are vendored verbatim from
# decision_transformer/models/{model,decision_transformer}.py: three linear embedding heads
# (return-to-go, state, action) plus a learned per-timestep embedding are summed, interleaved into
# (R_1, s_1, a_1, R_2, s_2, a_2, ...) token order, LayerNorm'd, and fed through the causal
# GPT2Model; separate linear heads predict the next state/action/return from the transformer's
# per-token outputs.

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.pytorch_utils import Conv1D

MENAGERIE_ZOO = "vendored-pytorch"


# ---- trajectory_gpt2.py (vendored, Conv1D import path updated for current transformers) ----
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            mask = self.bias[:, :, ns - nd : ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            assert hasattr(self, "q_attn"), (
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(
                hidden_size, n_ctx, config, scale, is_cross_attention=True
            )
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            assert hasattr(self, "crossattention"), (
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                "layers by setting `config.add_cross_attention=True`"
            )
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and
    loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPT2Model(GPT2PreTrainedModel):
    """Vendored verbatim from decision_transformer/models/trajectory_gpt2.py: HuggingFace's
    GPT2Model with the `wpe` positional-embedding lookup removed from `forward` (kept as a dead
    attribute, matching the real file's commented-out `# self.wpe = ...` / `# position_embeds =
    self.wpe(position_ids)`) since DecisionTransformer supplies its own timestep embedding."""

    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # NOTE: current `transformers` GPT2Config renamed `n_ctx` -> `n_positions` (same
        # semantic: max sequence length used to size the causal-mask buffer); attribute name
        # updated to match, no architectural change.
        self.h = nn.ModuleList(
            [Block(config.n_positions, config, scale=True) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        self.use_layers = None

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        # position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds  # + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.use_layers is not None and i >= self.use_layers:
                break

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# ---- end vendored trajectory_gpt2.py ----


# ---- model.py (vendored verbatim) ----
class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        return torch.zeros_like(actions[-1])


# ---- end vendored model.py ----


# ---- decision_transformer.py (vendored verbatim) ----
class DecisionTransformer(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs,
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])

        return state_preds, action_preds, return_preds


# ---- end vendored decision_transformer.py ----


def build_decision_transformer():
    return DecisionTransformer(
        state_dim=17,
        act_dim=6,
        hidden_size=32,
        max_length=8,
        max_ep_len=1000,
        n_layer=2,
        n_head=2,
        n_inner=4 * 32,
        activation_function="relu",
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )


def example_input_decision_transformer():
    batch_size, seq_length, state_dim, act_dim = 2, 8, 17, 6
    states = torch.randn(batch_size, seq_length, state_dim)
    actions = torch.randn(batch_size, seq_length, act_dim)
    rewards = torch.randn(batch_size, seq_length, 1)
    returns_to_go = torch.randn(batch_size, seq_length, 1)
    timesteps = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length).long()
    return states, actions, rewards, returns_to_go, timesteps


MENAGERIE_ENTRIES = [
    (
        "Decision Transformer",
        "build_decision_transformer",
        "example_input_decision_transformer",
        2021,
        "vendored-pytorch",
    ),
]

# SOURCE: vendored from licong-lin/in-context-rl @ main
# (https://github.com/licong-lin/in-context-rl, net.py)
#
# Causal decision-transformer for in-context reinforcement learning /
# algorithm distillation: a GPT2-style autoregressive backbone (either a
# custom from-scratch relu-attention DecoderTransformerBackbone or HF's
# GPT2Model) trained on RL learning histories (bandit trajectories) to
# imitate an algorithm's action-selection policy in-context.
#
# Only imports/relative-path bookkeeping was touched (device selection made
# input-driven instead of a module-level CUDA probe, and the unused
# `convert_to_tensor`/argparse plumbing from the original file was dropped
# since MENAGERIE just needs the nn.Module graph). The Transformer/
# DecoderTransformerBackbone architecture is untouched from the source.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

MENAGERIE_ZOO = "vendored-pytorch"


def get_activation(activation="relu"):
    if activation == "relu":
        return F.relu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError


class DecoderTransformerBackbone(nn.Module):
    def __init__(
        self,
        config,
        activation="relu",
        normalize_attn=True,
        mlp=True,
        layernorm=True,
        positional_embedding=True,
    ):
        super(DecoderTransformerBackbone, self).__init__()
        self.n_positions = config.n_positions
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp
        self.positional_embedding = positional_embedding

        # positional embeddings
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        self.wpe.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # layers
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for i in range(self.n_layer):
            self._queries.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._keys.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._values.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd]))
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(self.n_embd, self.n_embd),
                    nn.ReLU(),
                    nn.Linear(self.n_embd, self.n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))

        # pre-compute decoder attention mask
        with torch.no_grad():
            self.mask = torch.zeros(1, self.n_positions, self.n_positions)
            for i in range(self.n_positions):
                if self.normalize_attn:
                    self.mask[0, i, : (i + 1)].fill_(1.0 / (i + 1))
                else:
                    self.mask[0, i, : (i + 1)].fill_(1.0)

    def forward(self, inputs_embeds=None, position_ids=None, return_hidden_states=False):
        assert inputs_embeds is not None

        hidden_states = []
        N = inputs_embeds.shape[1]
        H = inputs_embeds

        if self.positional_embedding:
            if position_ids is None:
                input_shape = H.size()[:-1]
                position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=H.device)
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            position_embeds = self.wpe(position_ids)
            H = H + position_embeds
        hidden_states.append(H)

        for q, k, v, ln1, mlp, ln2 in zip(
            self._queries,
            self._keys,
            self._values,
            self._lns_1,
            self._mlps,
            self._lns_2,
        ):
            query = q(H)
            key = k(H)
            value = v(H)

            # compute attention weights
            attn_weights = self.activation(torch.einsum("bid,bjd->bij", query, key)) * self.mask[
                :, :N, :N
            ].to(H.device)

            H = H + torch.einsum("bij,bjd->bid", attn_weights, value)
            if self.layernorm:
                H = ln1(H)

            if self.mlp:
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)

            hidden_states.append(H)

        if return_hidden_states:
            return H, hidden_states
        return H


class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config["test"]
        self.horizon = self.config["horizon"]
        self.n_embd = self.config["n_embd"]
        self.n_layer = self.config["n_layer"]
        self.n_head = self.config["n_head"]
        self.state_dim = self.config["state_dim"]
        self.action_dim = self.config["action_dim"]
        self.dropout = self.config["dropout"]

        self.act_num = self.config["act_num"]
        self.dim = self.config["dim"]
        self.act_type = self.config["act_type"]

        gpt2_config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,  # previously 1
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        if self.act_type == "relu":
            self.transformer = DecoderTransformerBackbone(gpt2_config)
        else:
            self.transformer = GPT2Model(gpt2_config)

        self.embed_transition = nn.Linear(
            3 + self.dim * self.act_num + self.action_dim, self.n_embd
        )

        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        query_states = x["query_states"][:, None, :]

        if self.test:
            batch_size = 1
        else:
            batch_size = x["action_set"].size(dim=0)

        current_horizon = x["context_actions"].size(dim=1)

        action_set_seq = x["action_set"]

        action_set_seq_0 = action_set_seq.reshape(batch_size, 1, -1)
        zero_action_set = torch.zeros_like(action_set_seq_0)
        action_set_seq = torch.cat([action_set_seq_0, zero_action_set], axis=1)
        action_set_seq = action_set_seq.repeat(1, current_horizon, 1)

        device = query_states.device
        action_seq = torch.zeros(
            (batch_size, 2 * current_horizon, self.action_dim + 1), device=device
        )

        # odd numbers for the action set, even numbers for action actions & rewards
        action_seq[:, 1::2, :] = torch.cat([x["context_actions"], x["context_rewards"]], dim=2)
        one_seq = torch.ones((batch_size, 2 * current_horizon, 1), device=device)
        pos_seq = torch.arange(1, 2 * current_horizon + 1, dtype=torch.float32, device=device)
        pos_seq = pos_seq.reshape(1, -1, 1).repeat(batch_size, 1, 1)

        seq = torch.cat([action_set_seq, action_seq, one_seq, pos_seq], dim=2)

        if self.test:
            action_set_seq_test = torch.zeros((seq.size(dim=0), 1, seq.size(dim=2)), device=device)
            action_set_seq_test[:, :, : action_set_seq_0.size(dim=2)] = torch.clone(
                action_set_seq_0
            )
            action_set_seq_test[:, :, -2] = torch.ones_like(action_set_seq_test[:, :, -2])
            action_set_seq_test[:, :, -1] = (1 + 2 * current_horizon) * torch.ones_like(
                action_set_seq_test[:, :, -1]
            )
            seq = torch.cat([seq, action_set_seq_test], axis=1)

        stacked_inputs = self.embed_transition(seq)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)

        if self.act_type == "relu":
            preds = self.pred_actions(transformer_outputs)  # for relu
        else:
            preds = self.pred_actions(transformer_outputs["last_hidden_state"])  # for gpt2

        if self.test:
            return preds[:, -1, :]
        return preds[:, 0::2, :]


def build_algorithm_distillation():
    """Tiny bandit-setting Algorithm Distillation transformer (relu backbone)."""
    dim = 4
    act_num = 3
    horizon = 5
    config = {
        "test": True,
        "horizon": horizon,
        "n_embd": 16,
        "n_layer": 2,
        "n_head": 2,
        "state_dim": dim,
        "action_dim": act_num,
        "dropout": 0.0,
        "act_num": act_num,
        "dim": dim,
        "act_type": "relu",
    }
    return Transformer(config)


def example_input_algorithm_distillation():
    batch_size = 1
    dim = 4
    act_num = 3
    horizon = 5
    return {
        "query_states": torch.randn(batch_size, dim),
        "context_actions": torch.randn(batch_size, horizon, act_num),
        "context_rewards": torch.randn(batch_size, horizon, 1),
        "action_set": torch.randn(batch_size, dim * act_num),
    }


MENAGERIE_ENTRIES = [
    (
        "Algorithm Distillation (AD) agent",
        "build_algorithm_distillation",
        "example_input_algorithm_distillation",
        2022,
        MENAGERIE_ZOO,
    ),
]

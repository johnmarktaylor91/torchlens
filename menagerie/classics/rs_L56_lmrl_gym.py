# FAITHFUL PORT of abdulhaim/LMRL-Gym @ main (original framework: JAX/Flax)
#
# https://github.com/abdulhaim/LMRL-Gym
# https://raw.githubusercontent.com/abdulhaim/LMRL-Gym/main/LLM_RL/algorithms/value_rl_base/gpt2/generation.py
# https://raw.githubusercontent.com/abdulhaim/LMRL-Gym/main/LLM_RL/heads/mlp_head.py
# https://raw.githubusercontent.com/abdulhaim/LMRL-Gym/main/LLM_RL/heads/linear_head.py
# https://raw.githubusercontent.com/abdulhaim/LMRL-Gym/main/LLM_RL/algorithms/ilql/gpt2/interface.py
#
# Abdulhai et al. 2023 "LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning
# with Language Models" -- the LLM_RL library used by the benchmark trains a GPT-2
# language-model backbone jointly with twin Q-value head MLPs for offline RL
# (ILQL/CQL/MC-returns) over dialogue/text actions. The upstream implementation
# (`LLM_RL/algorithms/value_rl_base/gpt2/generation.py::GPT2ValueRLGeneration.__call__`)
# composes:
#   1. a frozen/reference policy LM ("pi_beta", optional) producing base LM logits,
#   2. a "value_base" GPT-2 LM whose final hidden states feed
#   3. one or two independent Q-head MLPs ("q_head", `LLM_RL/heads/mlp_head.py::MLPHead`
#      -- Dense(hidden) -> ReLU -> Dense(output), applied per token position) that are
#      *twin* networks (q1/q2, following CQL/SAC-style double-Q), min-pooled when both
#      are present,
#   4. combined as `logits = pi_beta_logits + beta * q_logits` (or just `beta * q_logits`
#      when no reference policy is used) -- this fused base-LM+Q-head architecture (not
#      merely a usage of an unmodified GPT-2) is the actual contribution being captured.
#
# This is a FAITHFUL PORT (not a vendor) because the real code is JAX/Flax
# (`flax.linen.Module`, `jax.numpy`, `optax`, `jax.sharding`/`pjit` for TPU pod
# sharding) and cannot run in the torch-only base environment; JAX/Flax are not
# installed and are not being added. Every mechanism above is transcribed faithfully
# into torch:
#   - `value_base` / `pi_beta` -> real `transformers.GPT2LMHeadModel` (unmodified,
#     `output_hidden_states=True` mirrors the Flax `output_hidden_states=True` call),
#   - `MLPHead` -> `QHeadMLP` below, a literal line-for-line torch re-expression of
#     `LLM_RL/heads/mlp_head.py::MLPHead.__call__` (`dense1 -> relu -> dense2`, same
#     two-layer shape contract; `use_bias` kept as the real config default `True`),
#   - twin Q1/Q2 heads + `jnp.minimum(q1_logits, q2_logits)` min-pooling -> `torch.min`,
#   - the `pi_beta_logits + beta * q_logits` combination rule (both branches, with and
#     without a reference policy) -> preserved verbatim as `+`/`*` arithmetic,
#   - JAX-only concerns genuinely irrelevant to the traced architecture (device mesh /
#     `PartitionSpec` sharding, `TrainState`/`optax` optimizer wiring, checkpoint I/O,
#     `dropout_rng` PRNG threading, `ModelLoadMode` config-loading plumbing) are dropped;
#     they configure *how* the JAX version is sharded/trained/loaded, not what tensors
#     the forward pass computes.

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class QHeadMLP(nn.Module):
    """Faithful torch port of LLM_RL/heads/mlp_head.py::MLPHead.

    Upstream (Flax) forward:
        x = self.dense1(x); x = nn.relu(x); x = self.dense2(x); return x
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, use_bias: bool = True):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, output_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


class GPT2ValueRLGeneration(nn.Module):
    """Faithful torch port of
    LLM_RL/algorithms/value_rl_base/gpt2/generation.py::GPT2ValueRLGeneration.

    Fuses a GPT-2 LM backbone with twin Q-value MLP heads for offline RL
    (ILQL / CQL / MC-returns dialogue policies), as used throughout the LMRL-Gym
    LLM_RL training/eval scripts (train_ilql_gpt2.py, train_ppo_gpt2.py, etc.).
    """

    def __init__(
        self,
        base_config: GPT2Config,
        vocab_size: int,
        q_hidden_dim: int,
        beta: float = 1.0,
        use_pi_beta: bool = True,
        use_twin_q: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.pi_beta = GPT2LMHeadModel(base_config) if use_pi_beta else None
        self.value_base = GPT2LMHeadModel(base_config)
        hidden_size = base_config.n_embd
        self.q1_head = QHeadMLP(hidden_size, q_hidden_dim, vocab_size)
        self.q2_head = QHeadMLP(hidden_size, q_hidden_dim, vocab_size) if use_twin_q else None

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pi_beta_logits = None
        if self.pi_beta is not None:
            pi_beta_out = self.pi_beta(input_ids, attention_mask=attention_mask)
            pi_beta_logits = pi_beta_out.logits

        value_base_out = self.value_base(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        base_hidden_states = value_base_out.hidden_states[-1]

        q1_logits = self.q1_head(base_hidden_states)
        if self.q2_head is not None:
            q2_logits = self.q2_head(base_hidden_states)
            q_logits = torch.minimum(q1_logits, q2_logits)
        else:
            q_logits = q1_logits

        if pi_beta_logits is not None:
            logits = pi_beta_logits + self.beta * q_logits
        else:
            logits = self.beta * q_logits
        return logits


def build_lmrl_gym():
    base_config = GPT2Config(
        vocab_size=200,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    return GPT2ValueRLGeneration(
        base_config=base_config,
        vocab_size=200,
        q_hidden_dim=32,
        beta=1.0,
        use_pi_beta=True,
        use_twin_q=True,
    )


def example_input_lmrl_gym():
    batch = 2
    seq_len = 8
    input_ids = torch.randint(0, 200, (batch, seq_len))
    return (input_ids,)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("LMRL-Gym GPT2-ValueRL", "build_lmrl_gym", "example_input_lmrl_gym", 2023, "ported"),
]

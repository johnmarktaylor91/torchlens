# FAITHFUL PORT of czhang99/Capsule-NLU @ master (original framework: TensorFlow 1.x)
"""Capsule-NLU: dynamic-routing capsule network for joint intent detection
and slot filling.

Paper: Zhang, Li, Zhou & Yu, "Joint Slot Filling and Intent Detection via
Capsule Neural Networks" (ACL 2019).

The official implementation is TensorFlow 1.x (``tf.contrib.rnn``,
``tf.get_variable`` variable scopes, ``tf.while_loop`` dynamic routing) and
cannot run in the base torch environment. This module transcribes the
architecture faithfully from the real source files:

  - ``capsule_masked.py``: ``shared_routing_uhat`` (dense + tanh projection
    of BiLSTM states into per-capsule "uhat" votes) and
    ``masked_routing_iter`` (length-masked dynamic routing-by-agreement with
    the paper's Eq. 1 squash nonlinearity, plus the ``w_rr`` re-routing bias
    term added to the routing logits when re-routing).
  - ``create_model.py``: the ``build_model`` wiring -- a bidirectional
    stacked-LSTM sentence encoder feeds a **slot capsule** layer whose
    per-timestep routing logits become the slot-tag logits; the slot
    capsule output is itself routed into an **intent capsule** layer; the
    argmax-selected intent capsule vector is then fed back as ``caps_ihat``
    for a *second*, re-routed pass through the slot capsule layer (the
    "rerouting by intent" cross-task feedback that is Capsule-NLU's core
    contribution over vanilla capsule routing).

Every routing/squash/re-routing equation below mirrors the TF source
one-for-one; only the RNN cell primitives (``BasicLSTMCell`` ->
``nn.LSTM``) and variable-scope bookkeeping are translated to their PyTorch
equivalents.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-9


def _squash(s: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Capsule squash nonlinearity (Eq. 1): ``_squash`` in capsule_masked.py."""

    squared_norm = (s * s).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm) / torch.sqrt(squared_norm + _EPSILON)
    return scale * s


class SharedRoutingUhat(nn.Module):
    """Ports ``shared_routing_uhat``: dense+tanh projection into vote tensors."""

    def __init__(self, in_dim: int, out_caps_num: int, out_caps_dim: int) -> None:
        super().__init__()
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.dense = nn.Linear(in_dim, out_caps_num * out_caps_dim)

    def forward(self, caps: torch.Tensor) -> torch.Tensor:
        # caps: (b_sz, caps_num, in_dim) -> caps_uhat: (b_sz, caps_num, out_caps_num, out_caps_dim)
        b_sz, tstp, _ = caps.shape
        caps_uhat = torch.tanh(self.dense(caps))
        return caps_uhat.view(b_sz, tstp, self.out_caps_num, self.out_caps_dim)


def masked_routing_iter(
    caps_uhat: torch.Tensor,
    seq_len: torch.Tensor,
    iter_num: int,
    caps_ihat: torch.Tensor | None = None,
    w_rr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ports ``masked_routing_iter``: length-masked dynamic routing-by-agreement.

    Parameters
    ----------
    caps_uhat:
        Vote tensor, shape ``(b_sz, tstp, out_caps_num, out_caps_dim)``.
    seq_len:
        Valid sequence length per example, shape ``(b_sz,)``.
    iter_num:
        Number of routing iterations.
    caps_ihat:
        Optional re-routing signal (the selected intent capsule), shape
        ``(b_sz, 1, out_caps_dim_in, 1)`` broadcastable against ``w_rr``.
    w_rr:
        Optional re-routing weight, shape ``(1, 1, wrr_dim0, wrr_dim1)``.

    Returns
    -------
    tuple
        ``(V, S, C, B_logits)`` matching the TF source's return signature.
    """

    assert iter_num > 0
    b_sz, tstp, out_caps_num, _ = caps_uhat.shape
    seq_len = torch.where(seq_len == 0, torch.ones_like(seq_len), seq_len)
    positions = torch.arange(tstp, device=caps_uhat.device).unsqueeze(0)
    mask = positions < seq_len.unsqueeze(1)  # (b_sz, tstp)
    floatmask = mask.to(caps_uhat.dtype).unsqueeze(-1)  # (b_sz, tstp, 1)

    B = torch.zeros(b_sz, tstp, out_caps_num, dtype=caps_uhat.dtype, device=caps_uhat.device)
    C_list = []
    V = S = B_logits = None
    for _ in range(iter_num):
        B_logits = B
        C = F.softmax(B, dim=2)  # (b_sz, tstp, out_caps_num)
        C = (C * floatmask).unsqueeze(-1)  # (b_sz, tstp, out_caps_num, 1)
        weighted_uhat = C * caps_uhat  # (b_sz, tstp, out_caps_num, out_caps_dim)
        C_list.append(C)
        S = weighted_uhat.sum(dim=1)  # (b_sz, out_caps_num, out_caps_dim)
        V = _squash(S, dim=2)  # (b_sz, out_caps_num, out_caps_dim)
        V_exp = V.unsqueeze(1)  # (b_sz, 1, out_caps_num, out_caps_dim)
        if caps_ihat is None:
            B = (caps_uhat * V_exp).sum(dim=-1) + B
        else:
            w_rr_tiled = w_rr.expand(caps_uhat.shape[0], caps_uhat.shape[1], -1, -1)
            caps_ihat_tiled = caps_ihat.expand(-1, caps_uhat.shape[1], -1, -1)
            rerouting_term = torch.matmul(
                torch.matmul(caps_uhat, w_rr_tiled), caps_ihat_tiled
            ).squeeze(-1)
            B = (caps_uhat * V_exp).sum(dim=-1) + 0.1 * rerouting_term + B

    V_ret = V
    S_ret = S
    C_ret = torch.stack(C_list, dim=0).squeeze(-1)  # (iter_num, b_sz, tstp, out_caps_num)
    return V_ret, S_ret, C_ret, B_logits


class Capsule(nn.Module):
    """Ports the ``Capsule`` layer class in capsule_masked.py."""

    def __init__(
        self,
        in_dim: int,
        out_caps_num: int,
        out_caps_dim: int,
        iter_num: int = 3,
        wrr_dim: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.iter_num = iter_num
        self.uhat = SharedRoutingUhat(in_dim, out_caps_num, out_caps_dim)
        self.w_rr = nn.Parameter(torch.randn(1, 1, wrr_dim[0], wrr_dim[1]) * 0.02)

    def forward(
        self,
        in_caps: torch.Tensor,
        seq_len: torch.Tensor,
        caps_ihat: torch.Tensor | None = None,
        re_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        caps_uhat = self.uhat(in_caps)
        if not re_routing:
            V, _S, C, B = masked_routing_iter(
                caps_uhat, seq_len, self.iter_num, caps_ihat, w_rr=None
            )
        else:
            V, _S, C, B = masked_routing_iter(
                caps_uhat, seq_len, self.iter_num, caps_ihat, w_rr=self.w_rr
            )
        return V, C, B


class CapsuleNLU(nn.Module):
    """Faithful port of ``create_model.build_model``.

    Bidirectional stacked-LSTM encoder -> slot capsule layer (routing
    logits reshaped into slot-tag logits) -> intent capsule layer -> argmax
    intent re-routes the slot capsule layer for a second pass.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        embed_dim: int = 16,
        layer_size: int = 12,
        slot_size: int = 10,
        intent_size: int = 6,
        intent_dim: int = 8,
        iter_slot: int = 2,
        iter_intent: int = 2,
    ) -> None:
        super().__init__()
        self.slot_size = slot_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, layer_size, batch_first=True, bidirectional=True)

        # BiLSTM output width feeds the slot capsule as "in_caps".
        enc_dim = layer_size * 2
        self.slot_capsule = Capsule(
            enc_dim, slot_size, layer_size, iter_num=iter_slot, wrr_dim=(layer_size, intent_dim)
        )
        self.intent_capsule = Capsule(layer_size, intent_size, intent_dim, iter_num=iter_intent)

    def forward(
        self, input_ids: torch.Tensor, seq_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.embedding(input_ids)
        H, _ = self.encoder(inputs)  # (b_sz, tstp, layer_size * 2)

        slot_capsule, _routing_weight, routing_logits = self.slot_capsule(
            H, seq_len, re_routing=False
        )
        slot_p = routing_logits.reshape(-1, self.slot_size)

        intent_capsule, _intent_routing_weight, _ = self.intent_capsule(
            slot_capsule, seq_len.new_full(seq_len.shape, self.slot_size)
        )

        pred_intent_index = torch.argmax(torch.norm(intent_capsule, dim=-1), dim=-1)
        pred_intent_onehot = F.one_hot(pred_intent_index, intent_capsule.shape[1]).to(
            intent_capsule.dtype
        )
        pred_intent_onehot = pred_intent_onehot.unsqueeze(-1).expand(
            -1, -1, intent_capsule.shape[2]
        )
        intent_capsule_max = (intent_capsule * pred_intent_onehot).sum(dim=1)
        caps_ihat = intent_capsule_max.unsqueeze(1).unsqueeze(3)

        _slot_capsule_new, _routing_weight_new, routing_logits_new = self.slot_capsule(
            H, seq_len, caps_ihat=caps_ihat, re_routing=True
        )
        slot_p_new = routing_logits_new.reshape(-1, self.slot_size)

        return slot_p_new, intent_capsule


def build_capsule_nlu() -> nn.Module:
    """Build a small Capsule-NLU joint intent/slot network.

    Returns
    -------
    nn.Module
        Random-initialized ``CapsuleNLU``.
    """

    return CapsuleNLU()


def example_input_capsule_nlu() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a token-id batch and per-example sequence length.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(token_ids, seq_len)`` with shapes ``(2, 8)`` and ``(2,)``.
    """

    x = torch.randint(0, 64, (2, 8))
    seq_len = torch.tensor([8, 8])
    return x, seq_len


MENAGERIE_ENTRIES = [
    ("Capsule-NLU", "build_capsule_nlu", "example_input_capsule_nlu", "2019", "nlp_slu"),
]

MENAGERIE_ZOO = "ported-pytorch"

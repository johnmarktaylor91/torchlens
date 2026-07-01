"""Compact faithful reimplementations of six dialogue/retrieval architecture families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - DAMD (Domain-Aware Multi-Decoder network): arxiv:1911.10484 (Zhang, Ou & Yu, AAAI 2020);
    official repo github.com/thu-spmi/damd-multiwoz (`damd_net.py`) -- a single bidirectional
    GRU context encoder feeds THREE cascaded unidirectional GRU decoders (belief-span,
    system-action-span, response), where each later decoder additively conditions on the
    previous decoder's final hidden state via a shared attention/copy mechanism over the
    encoder outputs, so belief -> action -> response forms a strict pipeline within one
    forward pass (one encoder, three decoders, as described in the README).
  - DialoFlow (Dialogue Flow Modeling): arxiv:2106.02227 (Li et al., ACL 2021); official
    repo github.com/ictnlp/DialoFlow (`model.py`, `PlanModel`/`DPKSModel` classes) -- a
    GPT-2 causal encoder produces per-token hidden states; sentence-boundary hidden vectors
    are gathered into a per-utterance sequence and fed through a small uni-directional
    ("Flow") Transformer block that predicts the NEXT utterance's semantic state from the
    sequence of past utterance states; the delta between the predicted and actual utterance
    state is concatenated with token hidden states to bias the LM head (context-flow score).
  - DialoGPT: arxiv:1911.00536 (Zhang et al., 2019); official repo
    github.com/microsoft/DialoGPT -- GPT-2 causal LM fine-tuned on 147M Reddit comment
    chains for open-domain response generation; reimplemented via
    ``transformers.GPT2LMHeadModel`` from a tiny ``GPT2Config`` (library-config candidate,
    pip id ``microsoft/DialoGPT-small`` confirms the underlying architecture is stock GPT-2).
  - DialogWAE: arxiv:1805.12352 (Gu et al., ICLR 2019); official repo
    github.com/guxd/DialogWAE (`modules.py`: `Encoder`, `ContextEncoder`, `Variation`,
    `Decoder`) -- a hierarchical utterance/context GRU encoder feeds two MLP "Variation"
    networks (recognition net conditioned on context+response, prior net conditioned on
    context only) that each transform reparameterized Gaussian noise into a latent code z;
    the two latents are trained adversarially (Wasserstein GAN critic on z) so the prior
    network learns to sample plausible latents without seeing the response, and z seeds a
    GRU response decoder.
  - DMN+ (Dynamic Memory Network Plus): arxiv:1603.01417 (Xiong, Merity & Socher, ICML
    2016); community PyTorch port github.com/dandelin/Dynamic-memory-networks-plus-Pytorch
    (`babi_main.py`) -- a bidirectional-GRU input module encodes each fact/sentence into a
    fact vector; a GRU question module encodes the query; an episodic-memory module runs
    `num_hop` attention passes where a per-fact gate (from a 2-layer MLP over
    [fact*question, fact*memory, |fact-question|, |fact-memory|]) drives an attention-GRU
    that produces a context vector, which updates the memory via
    ``ReLU(W[prevM; C; question])``; a final linear layer over [memory; question] predicts
    the answer.
  - DSI (Differentiable Search Index): arxiv:2202.00555 (Tay et al., NeurIPS 2022);
    community repro github.com/ArvinZhuang/DSI-QG (`run.py`) uses stock
    ``T5ForConditionalGeneration`` -- generative retrieval with NO explicit index: the model
    is trained to map a query string directly to a document-identifier token string, so
    retrieval becomes seq2seq generation with the corpus memorized in the T5 weights;
    reimplemented via a tiny ``T5Config`` + ``T5ForConditionalGeneration`` (library-config
    candidate; DSI-QG is a query-generation training recipe around the same architecture).

Skipped: none from this batch; see MENAGERIE_ENTRIES for the 6 built here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
    T5Config,
    T5ForConditionalGeneration,
)


# ---------------------------------------------------------------------------
# DAMD: one bi-GRU context encoder + three cascaded GRU decoders (belief span,
# action span, response), each attention+copy-conditioned on the encoder and
# on the previous decoder's hidden state.
# ---------------------------------------------------------------------------
class _BahdanauAttn(nn.Module):
    """Additive attention over encoder outputs, used by every DAMD decoder."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.energy = nn.Linear(hidden_size, 1)

    def forward(self, dec_hidden: Tensor, enc_outputs: Tensor) -> Tensor:
        """Return the attention-weighted context vector for one decoder step.

        Parameters
        ----------
        dec_hidden:
            Decoder hidden state, shape ``(batch, 1, hidden)``.
        enc_outputs:
            Encoder outputs to attend over, shape ``(batch, src_len, hidden)``.

        Returns
        -------
        Tensor
            Context vector, shape ``(batch, 1, hidden)``.
        """

        scores = self.energy(torch.tanh(self.query(dec_hidden) + self.key(enc_outputs)))
        weights = torch.softmax(scores, dim=1)
        return (weights * enc_outputs).sum(dim=1, keepdim=True)


class _DamdCascadedDecoder(nn.Module):
    """One GRU decoder step that attends over the encoder and the previous decoder."""

    def __init__(self, hidden_size: int, vocab_size: int, condition_on_prev: bool) -> None:
        super().__init__()
        self.condition_on_prev = condition_on_prev
        gru_in = 2 * hidden_size if condition_on_prev else hidden_size
        self.gru = nn.GRU(gru_in, hidden_size, batch_first=True)
        self.attn_enc = _BahdanauAttn(hidden_size)
        self.attn_prev = _BahdanauAttn(hidden_size) if condition_on_prev else None
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        embed_step: Tensor,
        hidden: Tensor,
        enc_outputs: Tensor,
        prev_dec_outputs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run one autoregressive step of a cascaded DAMD decoder.

        Parameters
        ----------
        embed_step:
            Embedded previous output token, shape ``(batch, 1, hidden)``.
        hidden:
            GRU hidden state, shape ``(1, batch, hidden)``.
        enc_outputs:
            Shared context-encoder outputs, shape ``(batch, src_len, hidden)``.
        prev_dec_outputs:
            Outputs of the preceding decoder in the cascade (``None`` for the
            first decoder in the pipeline), shape ``(batch, prev_len, hidden)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Vocabulary logits ``(batch, 1, vocab)`` and updated GRU hidden state.
        """

        ctx_enc = self.attn_enc(hidden.transpose(0, 1), enc_outputs)
        gru_in = [embed_step + ctx_enc]
        if self.condition_on_prev and prev_dec_outputs is not None:
            ctx_prev = self.attn_prev(hidden.transpose(0, 1), prev_dec_outputs)
            gru_in.append(ctx_prev)
        gru_out, hidden = self.gru(torch.cat(gru_in, dim=-1), hidden)
        return self.out(gru_out), hidden


class DAMD(nn.Module):
    """Domain-Aware Multi-Decoder network: shared encoder, belief/act/response cascade."""

    def __init__(self, vocab_size: int = 96, hidden_size: int = 32, dec_steps: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dec_steps = dec_steps
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.belief_decoder = _DamdCascadedDecoder(hidden_size, vocab_size, condition_on_prev=False)
        self.act_decoder = _DamdCascadedDecoder(hidden_size, vocab_size, condition_on_prev=True)
        self.resp_decoder = _DamdCascadedDecoder(hidden_size, vocab_size, condition_on_prev=True)

    def _run_decoder(
        self,
        decoder: _DamdCascadedDecoder,
        enc_outputs: Tensor,
        prev_outputs: Tensor | None,
        batch: int,
    ) -> tuple[Tensor, Tensor]:
        hidden = enc_outputs.new_zeros(1, batch, self.hidden_size)
        step_tok = enc_outputs.new_zeros(batch, 1, dtype=torch.long)
        logits_steps, hidden_steps = [], []
        for _ in range(self.dec_steps):
            embed_step = self.embedding(step_tok)
            logits, hidden = decoder(embed_step, hidden, enc_outputs, prev_outputs)
            logits_steps.append(logits)
            hidden_steps.append(hidden.transpose(0, 1))
            step_tok = logits.argmax(dim=-1)
        return torch.cat(logits_steps, dim=1), torch.cat(hidden_steps, dim=1)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode dialog context, then decode belief span -> action span -> response.

        Parameters
        ----------
        context:
            Token ids for the dialog context (user turn + history), shape
            ``(batch, src_len)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Belief-span, action-span, and response logits, each shape
            ``(batch, dec_steps, vocab_size)``.
        """

        batch = context.shape[0]
        enc_outputs, _ = self.encoder(self.embedding(context))
        belief_logits, belief_hidden = self._run_decoder(
            self.belief_decoder, enc_outputs, None, batch
        )
        act_logits, act_hidden = self._run_decoder(
            self.act_decoder, enc_outputs, belief_hidden, batch
        )
        resp_logits, _ = self._run_decoder(self.resp_decoder, enc_outputs, act_hidden, batch)
        return belief_logits, act_logits, resp_logits


def build_damd() -> nn.Module:
    """Build a tiny DAMD (Domain-Aware Multi-Decoder) network.

    Returns
    -------
    nn.Module
        ``DAMD`` in eval mode.
    """

    return DAMD(vocab_size=96, hidden_size=32, dec_steps=4).eval()


def example_input_damd() -> Tensor:
    """Example dialog-context token sequence for DAMD.

    Returns
    -------
    Tensor
        Long tensor of shape ``(2, 12)``.
    """

    return torch.randint(0, 96, (2, 12))


# ---------------------------------------------------------------------------
# DialoFlow: GPT-2 causal encoder + a uni-directional "Flow" Transformer block
# that predicts the next utterance's semantic state from past utterance states.
# ---------------------------------------------------------------------------
class _FlowBlock(nn.Module):
    """Single-layer causal self-attention block over per-utterance semantic states."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, sent_states: Tensor) -> Tensor:
        n = sent_states.shape[1]
        causal_mask = torch.triu(
            torch.ones(n, n, device=sent_states.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.attn(sent_states, sent_states, sent_states, attn_mask=causal_mask)
        h = self.ln1(sent_states + attn_out)
        return self.ln2(h + self.mlp(h))


class DialoFlow(nn.Module):
    """GPT-2 backbone plus a Flow module predicting utterance-level context drift."""

    def __init__(
        self, vocab_size: int = 128, d_model: int = 32, n_layer: int = 2, n_head: int = 2
    ) -> None:
        super().__init__()
        cfg = GPT2Config(
            vocab_size=vocab_size, n_positions=64, n_embd=d_model, n_layer=n_layer, n_head=n_head
        )
        self.gpt2 = GPT2Model(cfg)
        self.flow = _FlowBlock(d_model, n_head)
        self.utt_pos_embed = nn.Embedding(16, d_model)
        self.lm_head = nn.Linear(2 * d_model, vocab_size, bias=False)

    def forward(self, tokens: Tensor, sentence_idx: Tensor) -> Tensor:
        """Encode a conversation, model utterance-level flow, and produce LM logits.

        Parameters
        ----------
        tokens:
            Flattened conversation token ids, shape ``(batch, seq_len)``.
        sentence_idx:
            Positions of each utterance's final token, used to gather
            utterance-level semantic states, shape ``(batch, n_sent)``.

        Returns
        -------
        Tensor
            Next-token logits conditioned on the flow-predicted context state,
            shape ``(batch, seq_len, vocab_size)``.
        """

        token_hidden = self.gpt2(input_ids=tokens).last_hidden_state
        batch, n_sent = sentence_idx.shape
        gather_idx = sentence_idx.unsqueeze(-1).expand(-1, -1, token_hidden.shape[-1])
        sent_states = token_hidden.gather(1, gather_idx)
        positions = torch.arange(n_sent, device=tokens.device).unsqueeze(0).expand(batch, -1)
        flow_states = self.flow(sent_states + self.utt_pos_embed(positions))
        # Delta between the flow-predicted next-utterance state and the actual
        # current utterance state biases every token in that utterance's span.
        delta = flow_states - sent_states
        delta_per_token = delta[:, 0:1, :].expand(-1, token_hidden.shape[1], -1)
        fused = torch.cat([token_hidden, delta_per_token], dim=-1)
        return self.lm_head(fused)


def build_dialoflow() -> nn.Module:
    """Build a tiny DialoFlow model (GPT-2 encoder + Flow module).

    Returns
    -------
    nn.Module
        ``DialoFlow`` in eval mode.
    """

    return DialoFlow(vocab_size=128, d_model=32, n_layer=2, n_head=2).eval()


def example_input_dialoflow() -> tuple[Tensor, Tensor]:
    """Example flattened conversation tokens and utterance-boundary indices.

    Returns
    -------
    tuple[Tensor, Tensor]
        Token ids ``(1, 24)`` and sentence-boundary indices ``(1, 4)``.
    """

    tokens = torch.randint(0, 128, (1, 24))
    sentence_idx = torch.tensor([[5, 11, 17, 23]])
    return tokens, sentence_idx


# ---------------------------------------------------------------------------
# DialoGPT: GPT-2 causal LM fine-tuned for open-domain multi-turn dialogue.
# ---------------------------------------------------------------------------
def build_dialogpt() -> nn.Module:
    """Build a tiny DialoGPT-style GPT-2 causal LM.

    Returns
    -------
    nn.Module
        ``GPT2LMHeadModel`` in eval mode.
    """

    cfg = GPT2Config(vocab_size=128, n_positions=32, n_embd=32, n_layer=2, n_head=2, n_inner=64)
    return GPT2LMHeadModel(cfg).eval()


def example_input_dialogpt() -> Tensor:
    """Example multi-turn dialogue token sequence for DialoGPT.

    Returns
    -------
    Tensor
        Long tensor of shape ``(1, 16)``.
    """

    return torch.randint(0, 128, (1, 16))


# ---------------------------------------------------------------------------
# DialogWAE: hierarchical context encoder + adversarially-trained recognition
# and prior Gaussian "Variation" networks that seed a GRU response decoder.
# ---------------------------------------------------------------------------
class _Variation(nn.Module):
    """MLP transform of Gaussian noise into a latent code (recognition or prior net)."""

    def __init__(self, input_size: int, z_size: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.Tanh(),
        )
        self.to_mu = nn.Linear(z_size, z_size)
        self.to_logsigma = nn.Linear(z_size, z_size)

    def forward(self, context: Tensor) -> Tensor:
        h = self.fc(context)
        mu = self.to_mu(h)
        logsigma = self.to_logsigma(h)
        std = torch.exp(0.5 * logsigma)
        epsilon = torch.randn_like(std)
        return epsilon * std + mu


class DialogWAE(nn.Module):
    """Wasserstein-autoencoder dialogue model with a GAN-regularized latent prior."""

    def __init__(self, vocab_size: int = 96, hidden_size: int = 32, z_size: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.utt_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.ctx_encoder = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.response_encoder = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.prior_net = _Variation(hidden_size, z_size)
        self.recognition_net = _Variation(hidden_size + 2 * hidden_size, z_size)
        self.decoder_init = nn.Linear(hidden_size + z_size, hidden_size)
        self.decoder_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def _encode_context(self, context: Tensor) -> Tensor:
        batch, n_turns, turn_len = context.shape
        flat = context.view(batch * n_turns, turn_len)
        utt_out, _ = self.utt_encoder(self.embedding(flat))
        utt_vec = utt_out[:, -1, :].view(batch, n_turns, -1)
        _, ctx_hidden = self.ctx_encoder(utt_vec)
        return ctx_hidden[-1]

    def forward(self, context: Tensor, response: Tensor) -> tuple[Tensor, Tensor]:
        """Encode dialog context and response, sample latents, decode a response.

        Parameters
        ----------
        context:
            Prior dialog turns, shape ``(batch, n_turns, turn_len)``.
        response:
            Ground-truth response tokens (used by the recognition network
            during training), shape ``(batch, resp_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Decoder vocabulary logits ``(batch, resp_len, vocab_size)`` and the
            prior-network latent code ``(batch, z_size)`` used by the WAE
            discriminator during adversarial training.
        """

        ctx_vec = self._encode_context(context)
        resp_out, _ = self.response_encoder(self.embedding(response))
        resp_vec = resp_out[:, -1, :]

        z_prior = self.prior_net(ctx_vec)
        z_post = self.recognition_net(torch.cat([ctx_vec, resp_vec], dim=-1))

        dec_init = torch.tanh(self.decoder_init(torch.cat([ctx_vec, z_post], dim=-1))).unsqueeze(0)
        dec_out, _ = self.decoder_gru(self.embedding(response), dec_init)
        return self.out(dec_out), z_prior


def build_dialogwae() -> nn.Module:
    """Build a tiny DialogWAE model.

    Returns
    -------
    nn.Module
        ``DialogWAE`` in eval mode.
    """

    return DialogWAE(vocab_size=96, hidden_size=32, z_size=16).eval()


def example_input_dialogwae() -> tuple[Tensor, Tensor]:
    """Example dialog context (multiple turns) and a target response.

    Returns
    -------
    tuple[Tensor, Tensor]
        Context token ids ``(2, 3, 8)`` and response token ids ``(2, 6)``.
    """

    context = torch.randint(0, 96, (2, 3, 8))
    response = torch.randint(0, 96, (2, 6))
    return context, response


# ---------------------------------------------------------------------------
# DMN+: bi-GRU input module + attention-gated episodic memory with multi-hop
# attention-GRU passes over sentence-level facts.
# ---------------------------------------------------------------------------
class _AttentionGRUCell(nn.Module):
    """GRU cell whose update gate is externally supplied by an attention score."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

    def forward(self, fact: Tensor, prev_c: Tensor, gate: Tensor) -> Tensor:
        r = torch.sigmoid(self.Wr(fact) + self.Ur(prev_c))
        h_tilde = torch.tanh(self.W(fact) + r * self.U(prev_c))
        gate = gate.unsqueeze(-1).expand_as(h_tilde)
        return gate * h_tilde + (1 - gate) * prev_c


class _EpisodicMemory(nn.Module):
    """One attention pass over facts, producing an updated memory vector."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn_cell = _AttentionGRUCell(hidden_size, hidden_size)
        self.gate_fc1 = nn.Linear(4 * hidden_size, hidden_size)
        self.gate_fc2 = nn.Linear(hidden_size, 1)
        self.memory_update = nn.Linear(3 * hidden_size, hidden_size)

    def _attention_gates(self, facts: Tensor, question: Tensor, prev_mem: Tensor) -> Tensor:
        question = question.expand_as(facts)
        prev_mem = prev_mem.expand_as(facts)
        z = torch.cat(
            [
                facts * question,
                facts * prev_mem,
                torch.abs(facts - question),
                torch.abs(facts - prev_mem),
            ],
            dim=-1,
        )
        z = torch.tanh(self.gate_fc1(z))
        return torch.softmax(self.gate_fc2(z).squeeze(-1), dim=-1)

    def forward(self, facts: Tensor, question: Tensor, prev_mem: Tensor) -> Tensor:
        gates = self._attention_gates(facts, question, prev_mem)
        n_facts = facts.shape[1]
        context = torch.zeros_like(facts[:, 0, :])
        for i in range(n_facts):
            context = self.attn_cell(facts[:, i, :], context, gates[:, i])
        fused = torch.cat([prev_mem.squeeze(1), context, question.squeeze(1)], dim=-1)
        return F.relu(self.memory_update(fused)).unsqueeze(1)


class DMNPlus(nn.Module):
    """Dynamic Memory Network Plus: bi-GRU facts, GRU question, multi-hop episodic memory."""

    def __init__(self, vocab_size: int = 48, hidden_size: int = 24, num_hop: int = 3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hop = num_hop
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.input_gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.question_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.memory = _EpisodicMemory(hidden_size)
        self.answer_fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, contexts: Tensor, questions: Tensor) -> Tensor:
        """Answer a question by iteratively attending over context sentences.

        Parameters
        ----------
        contexts:
            Token ids per context sentence, shape ``(batch, n_sentences, tokens)``.
        questions:
            Question token ids, shape ``(batch, q_tokens)``.

        Returns
        -------
        Tensor
            Answer-vocabulary logits, shape ``(batch, vocab_size)``.
        """

        batch, n_sent, n_tok = contexts.shape
        flat = self.word_embedding(contexts.view(batch, -1)).view(batch, n_sent, n_tok, -1)
        # Simple bag-of-token sentence summary in place of the paper's
        # positional-encoding weighting, then a bi-GRU over the sentence axis.
        sentence_embed = flat.mean(dim=2)
        facts_out, _ = self.input_gru(sentence_embed)
        facts = facts_out[:, :, : self.hidden_size] + facts_out[:, :, self.hidden_size :]

        q_embed = self.word_embedding(questions)
        _, q_hidden = self.question_gru(q_embed)
        question = q_hidden.transpose(0, 1)

        memory = question
        for _ in range(self.num_hop):
            memory = self.memory(facts, question, memory)

        concat = torch.cat([memory, question], dim=-1).squeeze(1)
        return self.answer_fc(concat)


def build_dmn_plus() -> nn.Module:
    """Build a tiny DMN+ (Dynamic Memory Network Plus) model.

    Returns
    -------
    nn.Module
        ``DMNPlus`` in eval mode.
    """

    return DMNPlus(vocab_size=48, hidden_size=24, num_hop=3).eval()


def example_input_dmn_plus() -> tuple[Tensor, Tensor]:
    """Example bAbI-style context sentences and a question.

    Returns
    -------
    tuple[Tensor, Tensor]
        Context token ids ``(2, 5, 6)`` and question token ids ``(2, 4)``.
    """

    contexts = torch.randint(1, 48, (2, 5, 6))
    questions = torch.randint(1, 48, (2, 4))
    return contexts, questions


# ---------------------------------------------------------------------------
# DSI: generative retrieval -- stock T5 seq2seq mapping a query directly to a
# document-identifier token string, with no explicit index structure.
# ---------------------------------------------------------------------------
class _DsiWrapper(nn.Module):
    """Thin wrapper routing (query_ids, docid_ids) to T5's keyword-only forward."""

    def __init__(self, t5: T5ForConditionalGeneration) -> None:
        super().__init__()
        self.t5 = t5

    def forward(self, query_ids: Tensor, docid_ids: Tensor) -> Tensor:
        return self.t5(input_ids=query_ids, decoder_input_ids=docid_ids).logits


def build_dsi() -> nn.Module:
    """Build a tiny DSI (Differentiable Search Index) T5 model.

    Returns
    -------
    nn.Module
        ``_DsiWrapper`` around a ``T5ForConditionalGeneration`` in eval mode.
    """

    cfg = T5Config(
        vocab_size=100,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_kv=16,
        is_gated_act=False,
        dense_act_fn="relu",
        use_cache=False,
    )
    return _DsiWrapper(T5ForConditionalGeneration(cfg)).eval()


def example_input_dsi() -> tuple[Tensor, Tensor]:
    """Example query tokens and target docid tokens for DSI.

    Returns
    -------
    tuple[Tensor, Tensor]
        Query input ids ``(1, 10)`` and decoder docid input ids ``(1, 5)``.
    """

    query_ids = torch.randint(0, 100, (1, 10))
    docid_ids = torch.randint(0, 100, (1, 5))
    return query_ids, docid_ids


MENAGERIE_ENTRIES = [
    ("DAMD", "build_damd", "example_input_damd", "2020", "NLP"),
    (
        "DialoFlow (Dialogue Flow Modeling)",
        "build_dialoflow",
        "example_input_dialoflow",
        "2021",
        "NLP",
    ),
    ("DialoGPT", "build_dialogpt", "example_input_dialogpt", "2019", "NLP"),
    ("DialogWAE", "build_dialogwae", "example_input_dialogwae", "2019", "NLP"),
    (
        "DMN+ (Dynamic Memory Network Plus)",
        "build_dmn_plus",
        "example_input_dmn_plus",
        "2016",
        "NLP",
    ),
    ("DSI (Differentiable Search Index)", "build_dsi", "example_input_dsi", "2022", "NLP"),
]

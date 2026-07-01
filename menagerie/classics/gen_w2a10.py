"""Compact faithful reimplementations of six dialogue/style-transfer architecture families.

Sources checked (paper + official source; reimplemented compactly from scratch, no
clone/pip-install except the already-installed ``transformers``/``torch_geometric``
libraries used for the GPT-2 backbone and the relational graph conv):
  - UBAR: github.com/TonyNemo/UBAR-MultiWOZ (Yang, Li & Quan, AAAI 2021),
    arxiv:2012.03539; official repo (``model.py``, ``train.py``, class
    ``GPT2LMHeadModel`` fine-tuning). Distinctive mechanism is SESSION-LEVEL,
    fully end-to-end modeling: instead of turn-level (context -> single-target)
    training, UBAR concatenates the ENTIRE dialog session into one flat sequence
    -- every turn's user utterance, belief state, DB result, system act, and
    system response, for ALL turns -- and fine-tunes a single GPT-2 causal LM
    directly on that flat session-long token stream with a plain next-token LM
    loss (no auxiliary heads, no turn-level windowing). At inference the model
    also conditions on its OWN previously generated belief/act/response spans
    within the same session (fully autoregressive over the whole session), which
    is what makes it "fully end-to-end" vs. turn-level TOD baselines that see
    oracle context. Reimplemented directly with the installed ``transformers``
    ``GPT2LMHeadModel`` over a tiny config, fed one flat multi-turn session
    sequence (the paper's exact training-time input format).
  - UniCRS: github.com/RUCAIBox/UniCRS (Wang, Zhou, Wen & Zhao, KDD 2022);
    official repo (``src/model_prompt.py``, class ``KGPrompt``). Distinctive
    mechanism is KNOWLEDGE-ENHANCED PROMPT LEARNING over a FROZEN PLM backbone:
    a relational graph conv net (``RGCNConv``) encodes a small item/entity
    knowledge graph into entity embeddings; token embeddings (from a frozen
    text encoder) and entity embeddings are semantically FUSED via a learned
    cross-attention (entity-to-token attention weights, softmax-normalized,
    used to pool token embeddings into each entity's slot, or vice versa) that
    produces one shared "prompt" sequence; the fused prompt is further mixed
    with task-specific learnable soft-prefix tokens (separate prefixes for the
    recommendation vs. conversation subtask) and reshaped into a per-layer,
    per-block KV-prefix tensor injected into every transformer block of the
    (frozen) backbone -- i.e. ONLY the KG encoder + fusion + prefix-projection
    modules are trained; the backbone LM itself is never fine-tuned. Reimplemented
    with the same RGCNConv entity encoder -> token/entity two-way projection ->
    cross-attention fusion -> task-prefix concatenation -> per-layer prompt
    reshape pipeline exactly mirroring ``KGPrompt.forward``.
  - VHCR: github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-
    Conversation-Modeling (Park, Cho & Kim, NAACL 2018 Oral); official repo
    (``model/models.py``, class ``VHCR``). Distinctive mechanism is a
    TWO-LEVEL hierarchical VAE over multi-turn conversations: (1) a
    conversation-level GLOBAL latent ``z_conv`` sampled ONCE per whole
    conversation from a bidirectional context-inference RNN run over all
    utterance encodings (standard-Gaussian prior, amortized posterior); (2) a
    sequence of utterance-level LOCAL latents ``z_sent`` sampled once per turn
    from a CONTEXT-DEPENDENT prior (conditioned on the running context-RNN
    state AND ``z_conv``, not a fixed Gaussian) and an amortized posterior
    (conditioned additionally on that turn's true encoder output); the sampled
    ``z_conv`` is broadcast back into the context RNN's per-turn input (via a
    learned ``z_conv2context`` projection) so every turn's context update is
    conditioned on the single global latent that ties the whole conversation
    together (the "hierarchical latent structure" that gives VHCR its name,
    vs. flat per-utterance VAEs like VHRED); the paper's utterance-DROP
    regularization (randomly replacing whole encoded utterances with a learned
    ``unk_sent`` vector before the context RNN) is also reimplemented as a
    fixed, trace-static random mask applied at construction-consistent shape.
    Reimplemented with a fixed number of turns and a fixed sequence length
    (trace-friendly) preserving the exact conv-level/sent-level two-stage
    prior/posterior/sampling structure of the official ``VHCR`` class.
  - Wizard of Wikipedia: github.com/facebookresearch/ParlAI, official impl in
    ``projects/wizard_of_wikipedia/generator/modules.py`` (Dinan, Roller,
    Shuster, Fan, Auli & Weston, ICLR 2019), arxiv:1811.01241; classes
    ``ContextKnowledgeEncoder``/``ContextKnowledgeDecoder``/
    ``universal_sentence_embedding``. Distinctive mechanism is TWO-STAGE
    knowledge-grounded generation: a shared Transformer encoder embeds both the
    dialogue context AND every candidate knowledge sentence; each is pooled
    into a fixed-size vector via UNIVERSAL SENTENCE EMBEDDING (masked sum over
    time, divided by ``sqrt(length)`` -- NOT mean pooling nor a [CLS] token);
    a dot-product "chosen-sentence" attention (``ck_attn``) between the
    context's pooled vector and each candidate knowledge sentence's pooled
    vector scores which knowledge sentence to select (trained end-to-end
    against a chosen-sentence label, but at test time argmax-selected); the
    argmax/label-selected knowledge sentence's FULL token-level encoding
    (not just its pooled vector) is concatenated with the full context
    encoding and fed to a standard Transformer decoder that attends over the
    concatenated (knowledge, context) memory. Reimplemented with the same
    shared-encoder -> universal-sentence-embedding pooling -> dot-product
    knowledge-selection attention -> concatenated-memory Transformer-decoder
    pipeline, using ``torch.nn.TransformerEncoder``/``TransformerDecoder``
    in place of ParlAI's custom transformer.
  - ZSDG: github.com/snakeztc/NeuralDialog-ZSDG (Zhao & Eskenazi, SIGDIAL 2018
    Best Paper), aclanthology.org/W18-5001; official repo
    (``zsdg/models/models.py``, class ``ZeroShotHRED``). Distinctive mechanism
    is cross-domain ZERO-SHOT dialog generation via shared LATENT ACTIONS and
    ACTION MATCHING: a hierarchical encoder (per-utterance RNN encoder feeding
    a context-level RNN encoder, domain-agnostic and shared across every
    training domain) produces a context vector that a "policy" network maps to
    a predicted latent-action embedding; SEPARATELY, the true system response
    is independently encoded by the SAME utterance encoder into a "true action"
    embedding; the ACTION-MATCHING loss (an added L2 term between the
    predicted latent action and the true-response-derived action embedding,
    on top of the standard decoder NLL) forces the shared latent-action space
    to be consistent across domains, which is what lets the decoder generalize
    to an unseen (zero-shot) domain at test time by simply plugging in that
    domain's dialog-act attention context instead of full dialog history.
    Reimplemented with the same utterance-encoder -> context-encoder ->
    policy-latent-action head, a parallel true-action encoding of the target
    response for the L2 action-matching loss, and an attention decoder over
    the flattened per-word context encodings (mirroring
    ``ZeroShotHRED.forward``'s context-branch, the branch exercised whenever
    dialogue history is available).
  - AdaConv (Adaptive Convolutions for structure-aware style transfer):
    official impl is TensorFlow (taki0112/AdaConv-Tensorflow); unofficial but
    faithful PyTorch port at github.com/RElbers/ada-conv-pytorch (Chandran,
    Zoss, Gotardo, Gross & Bradley, CVPR 2021), openaccess.thecvf.com/content/
    CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_
    Transfer_CVPR_2021_paper.pdf; classes ``AdaConv2d``/``KernelPredictor``/
    ``GlobalStyleEncoder``/``AdaConvDecoder`` in ``lib/adaconv/*.py``.
    Distinctive mechanism is PER-INSTANCE, PER-LAYER PREDICTED CONVOLUTION
    KERNELS (not just per-channel affine AdaIN scale/shift): a small VGG-like
    style encoder extracts a global style feature map; a ``KernelPredictor``
    turns that style map into THREE style-conditioned tensors per decoder
    layer -- a depthwise "spatial" kernel, a grouped "pointwise" kernel, and a
    bias -- via a small conv head (spatial kernel via a same-size conv,
    pointwise/bias via global-average-pool + 1x1 conv); ``AdaConv2d`` then
    applies INSTANCE NORM to the content feature map followed by the
    style-predicted depthwise conv then the style-predicted grouped pointwise
    conv, looping over the batch dimension because plain ``F.conv2d`` does not
    support per-sample (batched) filter weights -- the paper's core departure
    from AdaIN-style feature-statistics transfer toward directly modulating
    the convolution operator itself with style-derived structure. Reimplemented
    with the same style-encoder -> KernelPredictor -> per-sample-looped
    AdaConv2d (instance-norm -> depthwise -> pointwise) decoder stack, using a
    small random-init conv stack in place of a pretrained VGG encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from torch_geometric.nn import RGCNConv

    _HAS_PYG = True
except ImportError:  # pragma: no cover - base env always has torch_geometric
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# UBAR: single GPT-2 causal LM fine-tuned SESSION-LEVEL -- one flat sequence
# concatenating every turn's (user utterance, belief state, DB result, system
# act, system response) across the WHOLE dialog session, fully end-to-end.
# ---------------------------------------------------------------------------
class Ubar(nn.Module):
    """UBAR (Yang et al., AAAI 2021): session-level end-to-end GPT-2 for TOD.

    Parameters
    ----------
    vocab_size : int
        Shared token vocabulary size (covers utterance/belief/DB/act/response
        sub-vocabularies, as in the paper's single flat GPT-2 vocabulary).
    n_embd : int
        GPT-2 hidden size.
    n_layer : int
        Number of GPT-2 decoder blocks.
    n_head : int
        Number of attention heads.
    n_positions : int
        Maximum session sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 180,
        n_embd: int = 32,
        n_layer: int = 2,
        n_head: int = 2,
        n_positions: int = 128,
    ) -> None:
        super().__init__()
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        # UBAR's entire model IS a plain GPT2LMHeadModel: no auxiliary heads,
        # no turn-level windowing -- the "fully end-to-end" trick is purely in
        # how the SESSION is flattened into one training sequence (every
        # turn's user utterance + belief state + DB result + system act +
        # system response, for all turns, concatenated).
        self.gpt2 = GPT2LMHeadModel(cfg)

    def forward(self, session_ids: Tensor) -> Tensor:
        """Run UBAR's session-level causal LM over one flattened dialog session.

        Parameters
        ----------
        session_ids : Tensor
            ``(batch, session_len)`` long token ids for the flat
            (u_1, b_1, db_1, a_1, r_1, u_2, b_2, ..., u_T, b_T, db_T, a_T, r_T)
            session sequence.

        Returns
        -------
        Tensor
            ``(batch, session_len, vocab_size)`` next-token logits.
        """
        return self.gpt2(input_ids=session_ids).logits


def build_ubar() -> nn.Module:
    """Build a small UBAR session-level end-to-end TOD model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return Ubar().eval()


def example_input_ubar() -> Tensor:
    """Example flattened multi-turn session token ids for UBAR.

    Returns
    -------
    Tensor
        ``(batch, session_len)`` long token ids.
    """
    return torch.randint(0, 180, (2, 60))


# ---------------------------------------------------------------------------
# UniCRS: RGCN entity encoder -> token/entity semantic-fusion cross-attention
# -> task-specific soft-prefix concatenation -> per-layer KV-prefix ("prompt")
# reshape, injected into a FROZEN backbone LM (only the prompt modules train).
# ---------------------------------------------------------------------------
class KgPrompt(nn.Module):
    """UniCRS ``KGPrompt`` (Wang et al., KDD 2022): knowledge-enhanced prompt learning.

    Parameters
    ----------
    hidden_size : int
        Backbone LM hidden size (also the fused-prompt embedding width).
    token_hidden_size : int
        Width of the (frozen) text-encoder token embeddings being fused.
    n_head : int
        Backbone attention heads per layer (for the per-layer prefix reshape).
    n_layer : int
        Number of backbone transformer layers the prefix must supply.
    n_block : int
        Number of KV blocks per layer (2 = key + value).
    n_entity : int
        Number of entities in the small item/knowledge graph.
    num_relations : int
        Number of relation types in the knowledge graph.
    num_bases : int
        RGCN basis-decomposition rank.
    n_prefix_conv : int
        Number of learnable conversation-task soft-prefix tokens.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        token_hidden_size: int = 24,
        n_head: int = 4,
        n_layer: int = 2,
        n_block: int = 2,
        n_entity: int = 30,
        num_relations: int = 4,
        num_bases: int = 3,
        n_prefix_conv: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        if _HAS_PYG:
            self.kg_encoder = RGCNConv(
                entity_hidden_size,
                entity_hidden_size,
                num_relations=num_relations,
                num_bases=num_bases,
            )
        else:  # pragma: no cover - defensive fallback, base env always has PyG
            self.kg_encoder = nn.Linear(entity_hidden_size, entity_hidden_size)
        self.node_embeds = nn.Parameter(torch.randn(n_entity, entity_hidden_size) * 0.02)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        # Semantic-fusion cross-attention between fused entity and token
        # representations -- the "knowledge-enhanced" fusion mechanism.
        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        self.conv_prefix_embeds = nn.Parameter(torch.randn(n_prefix_conv, hidden_size) * 0.02)
        self.conv_prefix_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def _entity_embeds(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        if _HAS_PYG:
            entity_embeds = (
                self.kg_encoder(self.node_embeds, edge_index, edge_type) + self.node_embeds
            )
        else:  # pragma: no cover
            entity_embeds = self.kg_encoder(self.node_embeds) + self.node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        return self.entity_proj2(entity_embeds)

    def forward(
        self, entity_ids: Tensor, token_embeds: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> Tensor:
        """Fuse KG entities with text tokens and build a per-layer prefix prompt.

        Parameters
        ----------
        entity_ids : Tensor
            ``(batch, entity_len)`` long ids into the small entity graph.
        token_embeds : Tensor
            ``(batch, token_len, token_hidden_size)`` frozen text-encoder
            token embeddings (e.g. from a frozen DialoGPT/RoBERTa backbone).
        edge_index : Tensor
            ``(2, n_edges)`` long knowledge-graph edge index.
        edge_type : Tensor
            ``(n_edges,)`` long knowledge-graph edge relation types.

        Returns
        -------
        Tensor
            ``(n_layer, n_block, batch, n_head, prompt_len, head_dim)``
            per-layer KV-prefix prompt tensor to inject into every block of
            the frozen backbone LM.
        """
        batch_size = entity_ids.size(0)
        entity_embeds = self._entity_embeds(edge_index, edge_type)
        entity_embeds = entity_embeds[entity_ids]  # (batch, entity_len, hidden)

        token_embeds = self.token_proj1(token_embeds) + token_embeds
        token_embeds = self.token_proj2(token_embeds)  # (batch, token_len, hidden)

        attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2, 1)
        attn_weights = attn_weights / self.hidden_size
        entity_weights = F.softmax(attn_weights, dim=2)
        # Fuse entity representations into the token stream via the learned
        # cross-attention weights -- the "semantic fusion" step.
        prompt_embeds = entity_weights @ entity_embeds + token_embeds
        prompt_len = token_embeds.size(1)

        # Prepend task-specific learnable soft-prefix tokens.
        prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)
        return prompt_embeds


def build_unicrs() -> nn.Module:
    """Build a small UniCRS knowledge-enhanced prompt-learning module.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return KgPrompt().eval()


def example_input_unicrs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example entity ids, token embeddings, and KG edges for UniCRS.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(entity_ids, token_embeds, edge_index, edge_type)``.
    """
    batch, entity_len, token_len = 2, 5, 8
    entity_ids = torch.randint(0, 30, (batch, entity_len))
    token_embeds = torch.randn(batch, token_len, 24)
    edge_index = torch.randint(0, 30, (2, 40))
    edge_type = torch.randint(0, 4, (40,))
    return entity_ids, token_embeds, edge_index, edge_type


# ---------------------------------------------------------------------------
# VHCR: two-level hierarchical VAE -- ONE conversation-level global latent
# z_conv (standard-Gaussian prior) broadcast back into a context RNN that
# drives a sequence of per-turn LOCAL latents z_sent (context-DEPENDENT prior,
# conditioned on z_conv), with utterance-drop regularization.
# ---------------------------------------------------------------------------
class Vhcr(nn.Module):
    """VHCR (Park et al., NAACL 2018): hierarchical latent conversation VAE.

    Parameters
    ----------
    vocab_size : int
        Word-token vocabulary size.
    embed_size : int
        Word embedding dimensionality.
    encoder_hidden : int
        Per-utterance BiGRU encoder hidden size (per direction).
    context_size : int
        Context-RNN hidden size.
    z_conv_size : int
        Conversation-level (global) latent dimensionality.
    z_sent_size : int
        Utterance-level (local) latent dimensionality.
    decoder_hidden : int
        Decoder GRU hidden size.
    n_turns : int
        Fixed (trace-static) number of utterances per conversation.
    drop_prob : float
        Utterance-drop probability (replaces an encoded utterance with a
        learned ``unk_sent`` vector before the context RNN, as regularization).
    """

    def __init__(
        self,
        vocab_size: int = 90,
        embed_size: int = 24,
        encoder_hidden: int = 24,
        context_size: int = 32,
        z_conv_size: int = 16,
        z_sent_size: int = 16,
        decoder_hidden: int = 32,
        n_turns: int = 4,
        drop_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_turns = n_turns
        self.z_conv_size = z_conv_size
        self.z_sent_size = z_sent_size
        self.drop_prob = drop_prob

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.utt_encoder = nn.GRU(embed_size, encoder_hidden, batch_first=True, bidirectional=True)
        utt_repr_size = 2 * encoder_hidden

        self.unk_sent = nn.Parameter(torch.randn(utt_repr_size) * 0.02)

        # Context-encoder RNN: per-turn input is (utterance repr, z_conv),
        # the mechanism that ties every turn's context update to the single
        # conversation-level global latent.
        self.context_rnn = nn.GRU(utt_repr_size + z_conv_size, context_size, batch_first=True)
        self.z_conv2context = nn.Linear(z_conv_size, context_size)

        # Bidirectional context-INFERENCE RNN (posterior side only, no z_conv
        # input) used to amortize the conversation-level posterior q(z_conv|.).
        self.context_inference = nn.GRU(
            utt_repr_size, context_size, batch_first=True, bidirectional=True
        )
        self.conv_posterior_h = nn.Sequential(nn.Linear(2 * context_size, context_size), nn.Tanh())
        self.conv_posterior_mu = nn.Linear(context_size, z_conv_size)
        self.conv_posterior_var = nn.Linear(context_size, z_conv_size)

        # Context-DEPENDENT sentence prior p(z_sent | context, z_conv) -- NOT
        # a fixed standard Gaussian, the paper's key departure from flat VAEs.
        self.sent_prior_h = nn.Sequential(
            nn.Linear(context_size + z_conv_size, context_size), nn.Tanh()
        )
        self.sent_prior_mu = nn.Linear(context_size, z_sent_size)
        self.sent_prior_var = nn.Linear(context_size, z_sent_size)

        self.sent_posterior_h = nn.Sequential(
            nn.Linear(context_size + utt_repr_size + z_conv_size, context_size), nn.Tanh()
        )
        self.sent_posterior_mu = nn.Linear(context_size, z_sent_size)
        self.sent_posterior_var = nn.Linear(context_size, z_sent_size)

        self.context2decoder = nn.Linear(context_size + z_sent_size + z_conv_size, decoder_hidden)
        self.decoder_rnn = nn.GRU(embed_size, decoder_hidden, batch_first=True)
        self.output_proj = nn.Linear(decoder_hidden, vocab_size)
        self.softplus = nn.Softplus()

    def forward(self, utterances: Tensor, targets: Tensor, drop_mask: Tensor) -> Tensor:
        """Run VHCR's two-level hierarchical VAE over one fixed-length conversation.

        Parameters
        ----------
        utterances : Tensor
            ``(batch, n_turns, seq_len)`` long word-token ids for each turn.
        targets : Tensor
            ``(batch, n_turns, seq_len)`` long decoder-input token ids (e.g.
            the same utterances, teacher-forced) for reconstructing each turn.
        drop_mask : Tensor
            ``(batch, n_turns)`` bool mask; ``True`` entries have their
            encoded utterance replaced with the learned ``unk_sent`` vector
            (utterance-drop regularization) before the context RNN.

        Returns
        -------
        Tensor
            ``(batch, n_turns, seq_len, vocab_size)`` per-turn reconstruction
            logits.
        """
        bsz, n_turns, seq_len = utterances.shape
        embedded = self.embedding(utterances.reshape(bsz * n_turns, seq_len))
        _, h_n = self.utt_encoder(embedded)
        utt_repr = h_n.transpose(0, 1).reshape(bsz, n_turns, -1)  # (batch, n_turns, utt_repr_size)

        # Utterance-drop regularization: swap dropped turns for a shared
        # learned placeholder before they reach the context encoder.
        unk = self.unk_sent.view(1, 1, -1).expand(bsz, n_turns, -1)
        utt_repr_dropped = torch.where(drop_mask.unsqueeze(-1), unk, utt_repr)

        # --- Conversation-level (global) latent z_conv ---
        _, h_inf = self.context_inference(utt_repr_dropped)
        h_inf = h_inf.transpose(0, 1).reshape(bsz, -1)
        h_post = self.conv_posterior_h(h_inf)
        conv_mu = self.conv_posterior_mu(h_post)
        conv_var = self.softplus(self.conv_posterior_var(h_post))
        z_conv = conv_mu + conv_var.sqrt() * torch.randn_like(conv_mu)

        # --- Sequential per-turn context + local latent z_sent ---
        z_conv_seq = z_conv.unsqueeze(1).expand(-1, n_turns, -1)
        context_in = torch.cat([utt_repr_dropped, z_conv_seq], dim=-1)
        context_outs, _ = self.context_rnn(context_in, self.z_conv2context(z_conv).unsqueeze(0))

        # Roll context one step (turn t's prior/posterior use the context
        # state BEFORE turn t, matching the paper's causal per-turn ordering).
        context_prev = torch.cat(
            [torch.zeros_like(context_outs[:, :1]), context_outs[:, :-1]], dim=1
        )

        prior_h = self.sent_prior_h(torch.cat([context_prev, z_conv_seq], dim=-1))
        prior_mu = self.sent_prior_mu(prior_h)
        prior_var = self.softplus(self.sent_prior_var(prior_h))

        post_h = self.sent_posterior_h(torch.cat([context_prev, utt_repr, z_conv_seq], dim=-1))
        post_mu = self.sent_posterior_mu(post_h)
        post_var = self.softplus(self.sent_posterior_var(post_h))
        z_sent = post_mu + post_var.sqrt() * torch.randn_like(post_mu)
        _ = prior_mu, prior_var  # (used for KL(q(z_sent)||p(z_sent)) at train time)

        dec_init = self.context2decoder(torch.cat([context_prev, z_sent, z_conv_seq], dim=-1))
        dec_init = dec_init.reshape(1, bsz * n_turns, -1)

        target_embedded = self.embedding(targets.reshape(bsz * n_turns, seq_len))
        dec_outs, _ = self.decoder_rnn(target_embedded, dec_init)
        logits = self.output_proj(dec_outs).reshape(bsz, n_turns, seq_len, -1)
        return logits


def build_vhcr() -> nn.Module:
    """Build a small VHCR hierarchical-latent conversation VAE.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return Vhcr().eval()


def example_input_vhcr() -> tuple[Tensor, Tensor, Tensor]:
    """Example fixed-shape conversation batch for VHCR.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(utterances, targets, drop_mask)``.
    """
    batch, n_turns, seq_len = 2, 4, 8
    utterances = torch.randint(0, 90, (batch, n_turns, seq_len))
    targets = torch.randint(0, 90, (batch, n_turns, seq_len))
    drop_mask = torch.zeros(batch, n_turns, dtype=torch.bool)
    drop_mask[:, 1] = True
    return utterances, targets, drop_mask


# ---------------------------------------------------------------------------
# Wizard of Wikipedia: shared Transformer encoder -> universal-sentence-
# embedding pooling of context AND each candidate knowledge sentence -> dot-
# product knowledge-SELECTION attention -> chosen sentence's full token
# encoding concatenated with context, fed to a Transformer decoder.
# ---------------------------------------------------------------------------
def _universal_sentence_embedding(hidden: Tensor) -> Tensor:
    """Sum-then-``sqrt(len)``-normalize pooling (no padding mask needed: fixed len)."""
    return hidden.sum(dim=1) / (hidden.size(1) ** 0.5)


class WizardOfWikipedia(nn.Module):
    """Wizard of Wikipedia (Dinan et al., ICLR 2019): knowledge-grounded ContextKnowledge Transformer.

    Parameters
    ----------
    vocab_size : int
        Shared token vocabulary size.
    hidden_size : int
        Transformer hidden size.
    n_heads : int
        Number of attention heads.
    n_enc_layers : int
        Number of shared Transformer encoder layers.
    n_dec_layers : int
        Number of Transformer decoder layers.
    n_know : int
        Number of candidate knowledge sentences per example.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 32,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        n_know: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_know = n_know
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=4 * hidden_size, batch_first=True
        )
        # ONE shared Transformer encoder embeds both the dialogue context and
        # every candidate knowledge sentence (weight sharing across both).
        self.shared_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=4 * hidden_size, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, context_ids: Tensor, know_ids: Tensor, decoder_input_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run Wizard of Wikipedia: select a knowledge sentence, then decode conditioned on it.

        Parameters
        ----------
        context_ids : Tensor
            ``(batch, ctx_len)`` long dialogue-context token ids.
        know_ids : Tensor
            ``(batch, n_know, know_len)`` long candidate knowledge-sentence
            token ids.
        decoder_input_ids : Tensor
            ``(batch, tgt_len)`` long teacher-forced response token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(response_logits, ck_attn)``: next-token logits
            ``(batch, tgt_len, vocab_size)`` and knowledge-selection attention
            scores ``(batch, n_know)``.
        """
        bsz, n_know, know_len = know_ids.shape

        context_encoded = self.shared_encoder(self.embedding(context_ids))
        know_flat = know_ids.reshape(bsz * n_know, know_len)
        know_encoded = self.shared_encoder(self.embedding(know_flat))

        # Universal Sentence Embedding pooling (sum / sqrt(len)) of context
        # and each candidate knowledge sentence -- the paper's specific
        # pooling choice, distinct from mean-pooling or a [CLS] token.
        context_use = _universal_sentence_embedding(context_encoded) / (self.hidden_size**0.5)
        know_use = _universal_sentence_embedding(know_encoded).reshape(bsz, n_know, -1)
        know_use = know_use / (self.hidden_size**0.5)

        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)  # (batch, n_know)
        cs_ids = ck_attn.argmax(dim=1)  # knowledge-SELECTION: pick the top-scoring sentence

        know_encoded = know_encoded.reshape(bsz, n_know, know_len, -1)
        cs_encoded = know_encoded[torch.arange(bsz), cs_ids]  # (batch, know_len, hidden)

        # Concatenate the CHOSEN sentence's full token encoding with the full
        # context encoding -- the decoder's memory is (knowledge, context).
        full_enc = torch.cat([cs_encoded, context_encoded], dim=1)

        tgt = self.embedding(decoder_input_ids)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        dec_out = self.decoder(tgt, full_enc, tgt_mask=causal_mask)
        return self.output_proj(dec_out), ck_attn


def build_wizard_of_wikipedia() -> nn.Module:
    """Build a small Wizard of Wikipedia knowledge-grounded generator.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return WizardOfWikipedia().eval()


def example_input_wizard_of_wikipedia() -> tuple[Tensor, Tensor, Tensor]:
    """Example context/knowledge/decoder-input token ids for Wizard of Wikipedia.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(context_ids, know_ids, decoder_input_ids)``.
    """
    batch, ctx_len, n_know, know_len, tgt_len = 2, 12, 4, 10, 8
    context_ids = torch.randint(0, 100, (batch, ctx_len))
    know_ids = torch.randint(0, 100, (batch, n_know, know_len))
    decoder_input_ids = torch.randint(0, 100, (batch, tgt_len))
    return context_ids, know_ids, decoder_input_ids


# ---------------------------------------------------------------------------
# ZSDG: shared hierarchical (utterance -> context) encoder feeds a "policy"
# latent-action head; the true response is independently encoded into a "true
# action" for an L2 ACTION-MATCHING term; attention decoder over flattened
# per-word context encodings supports cross-domain zero-shot generalization.
# ---------------------------------------------------------------------------
class ZeroShotHred(nn.Module):
    """ZSDG ``ZeroShotHRED`` (Zhao & Eskenazi, SIGDIAL 2018): cross-domain latent-action HRED.

    Parameters
    ----------
    vocab_size : int
        Shared word-token vocabulary size (domain-agnostic).
    embed_size : int
        Word embedding dimensionality.
    utt_hidden : int
        Per-utterance RNN encoder hidden size.
    ctx_hidden : int
        Context-level RNN encoder hidden size (also the latent-action size).
    dec_hidden : int
        Decoder RNN hidden size.
    n_ctx_turns : int
        Fixed (trace-static) number of context utterances per example.
    """

    def __init__(
        self,
        vocab_size: int = 95,
        embed_size: int = 24,
        utt_hidden: int = 32,
        ctx_hidden: int = 32,
        dec_hidden: int = 32,
        n_ctx_turns: int = 3,
    ) -> None:
        super().__init__()
        self.n_ctx_turns = n_ctx_turns
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Shared, domain-agnostic per-utterance encoder used for BOTH context
        # turns and the (separately-encoded) true response -- the weight
        # sharing that lets latent actions transfer zero-shot across domains.
        self.utt_encoder = nn.GRU(embed_size, utt_hidden, batch_first=True)
        self.ctx_encoder = nn.GRU(utt_hidden, ctx_hidden, batch_first=True)

        # Policy head maps the context summary to a predicted latent action.
        self.policy = nn.Linear(ctx_hidden, ctx_hidden)
        self.connector = nn.Linear(ctx_hidden, dec_hidden)

        self.attn_query = nn.Linear(dec_hidden, ctx_hidden)
        self.decoder = nn.GRU(embed_size + ctx_hidden, dec_hidden, batch_first=True)
        self.output_proj = nn.Linear(dec_hidden, vocab_size)

    def forward(
        self, ctx_utts: Tensor, response_ids: Tensor, decoder_input_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run ZSDG: encode context + true response, decode with attention.

        Parameters
        ----------
        ctx_utts : Tensor
            ``(batch, n_ctx_turns, utt_len)`` long context-utterance token ids.
        response_ids : Tensor
            ``(batch, resp_len)`` long true target-response token ids
            (independently encoded for the action-matching loss).
        decoder_input_ids : Tensor
            ``(batch, tgt_len)`` long teacher-forced decoder-input token ids.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(response_logits, predicted_action, true_action)``: decoder
            logits ``(batch, tgt_len, vocab_size)`` and the two latent-action
            embeddings ``(batch, ctx_hidden)`` whose L2 distance is the
            action-matching training signal.
        """
        bsz, n_turns, utt_len = ctx_utts.shape
        utt_embedded = self.embedding(ctx_utts.reshape(bsz * n_turns, utt_len))
        utt_outs, utt_h = self.utt_encoder(utt_embedded)
        utt_summary = utt_h[-1].reshape(bsz, n_turns, -1)

        ctx_outs, ctx_h = self.ctx_encoder(utt_summary)
        predicted_action = self.policy(ctx_h[-1])  # (batch, ctx_hidden)

        # Independently encode the TRUE response with the SAME shared
        # utterance encoder to get the "true action" embedding.
        resp_embedded = self.embedding(response_ids)
        _, resp_h = self.utt_encoder(resp_embedded)
        true_action = resp_h[-1]  # (batch, utt_hidden); utt_hidden == ctx_hidden by construction

        dec_init = self.connector(predicted_action).unsqueeze(0)

        # Attention memory: per-word context encodings flattened across turns
        # and words (ctx_outs broadcast + word-level utt_outs, as in the
        # official `attn_inputs = ctx_outs + utt_outs` construction).
        ctx_outs_rep = (
            ctx_outs.unsqueeze(2).expand(-1, -1, utt_len, -1).reshape(bsz, n_turns * utt_len, -1)
        )
        utt_outs_flat = utt_outs.reshape(bsz, n_turns * utt_len, -1)
        attn_mem = ctx_outs_rep + utt_outs_flat

        tgt_embedded = self.embedding(decoder_input_ids)
        dec_hidden = dec_init
        outputs = []
        for t in range(tgt_embedded.size(1)):
            query = self.attn_query(dec_hidden[-1]).unsqueeze(1)  # (batch, 1, ctx_hidden)
            scores = torch.bmm(query, attn_mem.transpose(1, 2))
            attn = F.softmax(scores, dim=-1)
            context_vec = torch.bmm(attn, attn_mem).squeeze(1)
            step_in = torch.cat([tgt_embedded[:, t], context_vec], dim=-1).unsqueeze(1)
            dec_out, dec_hidden = self.decoder(step_in, dec_hidden)
            outputs.append(dec_out.squeeze(1))
        logits = self.output_proj(torch.stack(outputs, dim=1))
        return logits, predicted_action, true_action


def build_zsdg() -> nn.Module:
    """Build a small ZSDG cross-domain latent-action HRED model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return ZeroShotHred().eval()


def example_input_zsdg() -> tuple[Tensor, Tensor, Tensor]:
    """Example context/response/decoder-input token ids for ZSDG.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(ctx_utts, response_ids, decoder_input_ids)``.
    """
    batch, n_turns, utt_len, resp_len, tgt_len = 2, 3, 6, 7, 6
    ctx_utts = torch.randint(0, 95, (batch, n_turns, utt_len))
    response_ids = torch.randint(0, 95, (batch, resp_len))
    decoder_input_ids = torch.randint(0, 95, (batch, tgt_len))
    return ctx_utts, response_ids, decoder_input_ids


# ---------------------------------------------------------------------------
# AdaConv: style encoder -> KernelPredictor produces PER-INSTANCE depthwise
# ("spatial") + grouped pointwise conv kernels + bias from the style
# embedding; AdaConv2d applies instance-norm then those style-predicted
# kernels (looped per sample -- F.conv2d has no batched-filter support).
# ---------------------------------------------------------------------------
class KernelPredictor(nn.Module):
    """Predicts per-instance depthwise/pointwise conv kernels + bias from a style map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int,
        style_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size

        self.spatial = nn.Conv2d(
            style_channels,
            in_channels * out_channels // n_groups,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels, out_channels * out_channels // n_groups, kernel_size=1),
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels, out_channels, kernel_size=1),
        )

    def forward(self, w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = w.size(0)
        w_spatial = self.spatial(w).reshape(
            n,
            self.out_channels,
            self.in_channels // self.n_groups,
            self.kernel_size,
            self.kernel_size,
        )
        w_pointwise = self.pointwise(w).reshape(
            n, self.out_channels, self.out_channels // self.n_groups, 1, 1
        )
        bias = self.bias(w).reshape(n, self.out_channels)
        return w_spatial, w_pointwise, bias


class AdaConv2d(nn.Module):
    """Instance-norm then style-predicted depthwise + pointwise conv, per sample."""

    def __init__(
        self, in_channels: int, out_channels: int, n_groups: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.final_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x: Tensor, w_spatial: Tensor, w_pointwise: Tensor, bias: Tensor) -> Tensor:
        x = F.instance_norm(x)
        ys = []
        for i in range(x.size(0)):
            ys.append(self._forward_single(x[i : i + 1], w_spatial[i], w_pointwise[i], bias[i]))
        y = torch.cat(ys, dim=0)
        return self.final_conv(y)

    def _forward_single(
        self, x: Tensor, w_spatial: Tensor, w_pointwise: Tensor, bias: Tensor
    ) -> Tensor:
        pad = (w_spatial.size(-1) - 1) // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x


class GlobalStyleEncoder(nn.Module):
    """Downscales a style feature map and projects it into a spatial style tensor W."""

    def __init__(self, in_channels: int, style_channels: int, style_kernel: int) -> None:
        super().__init__()
        self.style_channels = style_channels
        self.style_kernel = style_kernel
        self.downscale = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )
        # Fixed small feature map (8x8 style-encoder output in this compact
        # build) -> flatten -> project to the spatial style tensor W used by
        # every KernelPredictor in the decoder.
        self.fc = nn.Linear(in_channels * 4 * 4, style_channels * style_kernel * style_kernel)

    def forward(self, x: Tensor) -> Tensor:
        y = self.downscale(x)
        y = y.reshape(y.size(0), -1)
        w = self.fc(y)
        return w.reshape(w.size(0), self.style_channels, self.style_kernel, self.style_kernel)


class AdaConvStyleTransfer(nn.Module):
    """AdaConv (Chandran et al., CVPR 2021): predicted-kernel adaptive convolution style transfer.

    Parameters
    ----------
    content_channels : int
        Channel width of the (small, random-init in this build) content/style
        feature encoder's output, standing in for a pretrained VGG encoder.
    style_channels : int
        Channel width of the style descriptor fed to every KernelPredictor.
    kernel_size : int
        Spatial size of the predicted adaptive-convolution kernels.
    """

    def __init__(
        self, content_channels: int = 32, style_channels: int = 32, kernel_size: int = 3
    ) -> None:
        super().__init__()
        # Small random-init stand-in for the pretrained VGG encoder (frozen
        # in the official implementation); captures the "encode content and
        # style with a shared feature extractor" step compactly.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, content_channels, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.style_encoder = GlobalStyleEncoder(content_channels, style_channels, kernel_size)

        group_div = 2
        n_groups = content_channels // group_div
        self.kernel_predictor = KernelPredictor(
            content_channels,
            content_channels,
            n_groups=n_groups,
            style_channels=style_channels,
            kernel_size=kernel_size,
        )
        self.ada_conv = AdaConv2d(
            content_channels, content_channels, n_groups=n_groups, kernel_size=kernel_size
        )
        self.to_rgb = nn.Conv2d(content_channels, 3, 3, padding=1, padding_mode="reflect")

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        """Stylize a content image using style-predicted per-instance conv kernels.

        Parameters
        ----------
        content : Tensor
            ``(batch, 3, H, W)`` content image.
        style : Tensor
            ``(batch, 3, H, W)`` style image.

        Returns
        -------
        Tensor
            ``(batch, 3, H, W)`` stylized output image.
        """
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        w_style = self.style_encoder(style_feat)

        w_spatial, w_pointwise, bias = self.kernel_predictor(w_style)
        stylized = self.ada_conv(content_feat, w_spatial, w_pointwise, bias)
        stylized = F.relu(stylized)
        stylized = F.interpolate(stylized, size=content.shape[-2:], mode="nearest")
        return self.to_rgb(stylized)


def build_adaconv() -> nn.Module:
    """Build a small AdaConv predicted-kernel style-transfer model.

    Returns
    -------
    nn.Module
        Model in eval mode.
    """
    return AdaConvStyleTransfer().eval()


def example_input_adaconv() -> tuple[Tensor, Tensor]:
    """Example content/style image pair for AdaConv.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(content, style)`` each ``(batch, 3, 32, 32)``.
    """
    content = torch.randn(2, 3, 32, 32)
    style = torch.randn(2, 3, 32, 32)
    return content, style


MENAGERIE_ENTRIES = [
    ("UBAR", "build_ubar", "example_input_ubar", "2021", "NLP"),
    (
        "UniCRS (Unified Conversational Recommender System)",
        "build_unicrs",
        "example_input_unicrs",
        "2022",
        "NLP",
    ),
    ("VHCR", "build_vhcr", "example_input_vhcr", "2018", "NLP"),
    (
        "Wizard of Wikipedia",
        "build_wizard_of_wikipedia",
        "example_input_wizard_of_wikipedia",
        "2019",
        "NLP",
    ),
    ("ZSDG (Zero-Shot Dialogue Generation)", "build_zsdg", "example_input_zsdg", "2018", "NLP"),
    ("AdaConv style transfer", "build_adaconv", "example_input_adaconv", "2021", "VIS"),
]

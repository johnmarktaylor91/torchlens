"""Compact faithful reimplementations of five dialogue/retrieval architecture families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch):
  - UniMSE (COSMIC-Transformer style unified MSA/ERC): arxiv:2211.11256; official repo
    github.com/LeMei/UniMSE (`src/model.py`, `src/modules/*`) -- unified T5 backbone with
    modality-specific RNN encoders for visual/acoustic streams, fused into the T5
    encoder-decoder via adapter cross-attention, trained with contrastive alignment
    between modalities and between sentiment/emotion samples.
  - CR-Walker: arxiv:2010.10333; official repo github.com/truthless11/CR-Walker
    (`model/CR_walker.py`, `model/graph_walker.py`, `model/intent_selector.py`,
    `model/graph_embedder.py`) -- dialog RNN/BERT utterance encoder + 3-way intent
    classifier that dispatches a tree-structured multi-hop walk over an R-GCN-embedded
    knowledge graph, with bilinear attention scoring candidate KG nodes at each hop.
  - KBRD (CRSLab KBRD-style policy net): arxiv:1901.04085-adjacent KBRD paper (EMNLP
    2019, Chen et al.); CRSLab official repo github.com/RUCAIBox/CRSLab
    (`crslab/model/crs/kbrd/kbrd.py`) -- R-GCN knowledge-graph entity encoder with
    self-attention user pooling, feeding BOTH a bilinear recommendation head over KG
    entities AND a Transformer encoder-decoder dialogue generator whose decoder logits
    are additively biased by the projected KG user embedding.
  - CVAE dialogue model (kgCVAE, Zhao et al. ACL 2017): arxiv:1703.10960; PyTorch port
    github.com/ruotianluo/NeuralDialog-CVAE-pytorch (`models/cvae.py`) -- hierarchical
    utterance/context RNN encoder feeding a recognition network (posterior q(z|x,c)) and
    a prior network (p(z|c)); Gaussian reparameterization sample z is concatenated with
    context and fed to (a) a bag-of-words auxiliary decoder and (b) an RNN response
    decoder whose initial state is a projection of [context, z].
  - Cross-Encoder / BERT Ranker (Nogueira & Cho 2019, MS-MARCO/TREC): arxiv:1901.04085;
    official repo github.com/nyu-dl/dl4marco-bert -- BERT joint query-document encoder
    with the [CLS] pooled representation fed to a binary relevance classifier head
    (single joint encoding of the concatenated query+document pair, not a dual encoder).

Skipped: cand_00027 "CVAE-CO (Conditional VAE for Diverse Dialogue)" -- verified dedup,
not a distinct architecture. Its repo_url and arxiv id are byte-identical to cand_00026
("CVAE dialogue model"), both pointing to github.com/ruotianluo/NeuralDialog-CVAE-pytorch
and arxiv:1703.10960 (Zhao et al., ACL 2017). Fetched the live repo tree and README: the
repo ships exactly one model (`models/cvae.py`, titled "Knowledge-Guided CVAE"); there is
no second "CO" variant file, class, or flag anywhere in the source. The paper itself names
only two models -- the base CVAE and its "novel variant integrated with linguistic prior
knowledge" (kgCVAE) -- neither called "CVAE-CO". A web search for the exact string
"CVAE-CO" returns no matching model in the dialogue-CVAE literature (only unrelated
variants: SepaCVAE, A-CVAE, dual-view CVAE). The build queue's own notes column already
flags this as "confidence med, CVAE-CO specific variant not independently verified" and
"likely the same Zhao et al. 2017 model". Building it would mean re-emitting the identical
class under a second label with no architectural delta -- see MENAGERIE_ENTRIES for the 5
built here.

cand_00024 ("Cross-Encoder (MS-MARCO, TREC)") is architecturally identical to cand_00010
("BERT Ranker") per the build queue's own POTENTIAL_DEDUP annotation (same repo, same
oneline); built here under its own catalog name per queue assignment, sharing the
CrossEncoder class with cand_00010's canonical name-space left free for a separate builder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import RGCNConv
from transformers import T5Config, T5ForConditionalGeneration


# ---------------------------------------------------------------------------
# UniMSE: unified T5 backbone with modality-specific RNN encoders fused via
# cross-attention adapters, for joint multimodal sentiment analysis + ERC.
# ---------------------------------------------------------------------------
class ModalityRNNEncoder(nn.Module):
    """Per-modality (visual/acoustic) sequence encoder producing a pooled vector."""

    def __init__(self, in_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        seq, _ = self.rnn(x)
        return self.proj(seq[:, -1])


class CrossModalAdapter(nn.Module):
    """Cross-attention fusion of a modality vector into T5 hidden states (adapter)."""

    def __init__(self, d_model: int, modality_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(modality_dim, d_model)
        self.value = nn.Linear(modality_dim, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, hidden: Tensor, modality_vec: Tensor) -> Tensor:
        q = self.query(hidden)
        k = self.key(modality_vec).unsqueeze(1)
        v = self.value(modality_vec).unsqueeze(1)
        attn = torch.softmax((q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1)
        fused = attn @ v
        return hidden + self.out(fused)


class UniMSE(nn.Module):
    """Compact unified T5 encoder-decoder MSA/ERC model with adapter fusion + contrastive alignment.

    UniMSE's namesake mechanism is (a) a genuine T5 ENCODER-DECODER backbone
    (not an encoder-only stack) that unifies multiple tasks/modalities, and
    (b) a contrastive alignment objective across modalities/utterances whose
    projection must actually flow into the model's output. Both are wired in
    here: ``self.t5.encoder``/``self.t5.decoder``/``self.t5.lm_head`` provide
    the real encoder-decoder path (decoder input synthesized via T5's
    standard ``_shift_right`` on the same token ids, so the public forward
    signature stays a simple ``(tokens, visual, acoustic)``), and
    ``contrastive_proj`` projects BOTH the fused encoder pooled state and the
    decoder pooled state into a shared space whose dot-product similarity is
    returned alongside the label logits -- not computed and discarded.
    """

    def __init__(
        self,
        vocab_size: int = 96,
        d_model: int = 32,
        visual_dim: int = 8,
        acoustic_dim: int = 8,
        n_labels: int = 7,
    ) -> None:
        """Build token embedding, dual modality encoders, tiny T5 backbone, adapters."""

        super().__init__()
        self.visual_enc = ModalityRNNEncoder(visual_dim, d_model)
        self.acoustic_enc = ModalityRNNEncoder(acoustic_dim, d_model)

        t5_cfg = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=4 * d_model,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=2,
            d_kv=d_model // 2,
            decoder_start_token_id=0,
            pad_token_id=0,
            is_gated_act=False,
            dense_act_fn="relu",
            use_cache=False,
        )
        self.t5 = T5ForConditionalGeneration(t5_cfg)

        self.visual_adapter = CrossModalAdapter(d_model, d_model)
        self.acoustic_adapter = CrossModalAdapter(d_model, d_model)
        self.contrastive_proj = nn.Linear(d_model, d_model)
        self.label_head = nn.Linear(d_model, n_labels)

    def forward(self, tokens: Tensor, visual: Tensor, acoustic: Tensor) -> tuple[Tensor, Tensor]:
        """Encode-decode via T5, fuse visual/acoustic adapters, and return (label_logits, contrastive_sim).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``label_logits`` ``(batch, n_labels)`` from the fused encoder
            state, and ``contrastive_sim`` ``(batch,)`` -- the per-example
            dot-product similarity between the contrastive projections of
            the (adapter-fused) encoder pooled state and the decoder pooled
            state, i.e. the contrastive-alignment objective's actual output.
        """

        # T5 encoder over the text stream (shares the T5 token embedding table).
        enc_hidden = self.t5.encoder(input_ids=tokens).last_hidden_state

        visual_vec = self.visual_enc(visual)
        acoustic_vec = self.acoustic_enc(acoustic)
        enc_hidden = self.visual_adapter(enc_hidden, visual_vec)
        enc_hidden = self.acoustic_adapter(enc_hidden, acoustic_vec)

        decoder_input_ids = self.t5._shift_right(tokens)
        dec_hidden = self.t5.decoder(
            input_ids=decoder_input_ids, encoder_hidden_states=enc_hidden
        ).last_hidden_state

        enc_pooled = enc_hidden.mean(dim=1)
        dec_pooled = dec_hidden.mean(dim=1)

        # Contrastive alignment: project both pooled views through the SAME
        # head and score their similarity -- this output is returned, not
        # discarded, unlike the prior dead-code side-effect call.
        enc_contrast = self.contrastive_proj(enc_pooled)
        dec_contrast = self.contrastive_proj(dec_pooled)
        contrastive_sim = (enc_contrast * dec_contrast).sum(dim=-1)

        label_logits = self.label_head(enc_pooled)
        return label_logits, contrastive_sim


def build_unimse() -> nn.Module:
    """Build the compact UniMSE unified T5 encoder-decoder multimodal sentiment/emotion model."""

    return UniMSE().eval()


def example_input_unimse() -> tuple[Tensor, Tensor, Tensor]:
    """Return (token ids, visual sequence, acoustic sequence)."""

    tokens = torch.randint(0, 96, (2, 12), dtype=torch.long)
    visual = torch.randn(2, 6, 8)
    acoustic = torch.randn(2, 6, 8)
    return tokens, visual, acoustic


# ---------------------------------------------------------------------------
# CR-Walker: intent-dispatched tree-structured multi-hop walk over an
# R-GCN-embedded knowledge graph for conversational recommendation.
# ---------------------------------------------------------------------------
class BilinearAttention(nn.Module):
    """Query/key bilinear attention producing a softmax distribution over nodes."""

    def __init__(self, query_size: int, key_size: int, hidden: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(query_size, hidden, bias=False)
        self.w_k = nn.Linear(key_size, hidden, bias=False)
        self.w_v = nn.Linear(hidden, 1, bias=False)

    def forward(self, query: Tensor, keys: Tensor) -> Tensor:
        """query: (B, query_size); keys: (N, key_size) -> attn weights (B, N)."""

        q = self.w_q(query).unsqueeze(1)
        k = self.w_k(keys).unsqueeze(0)
        scores = self.w_v(torch.tanh(q + k)).squeeze(-1)
        return torch.softmax(scores, dim=-1)


class CRWalker(nn.Module):
    """Compact CR-Walker: intent selector + genuine sequential tree-walk over a KG.

    The distinctive CR-Walker mechanism is a multi-hop TREE walk: at hop 0 the
    dialogue-state query attends over the root node's children; the SELECTED
    child (a soft, differentiable combination weighted by hop-0's attention)
    then becomes part of hop-1's query, which attends over ITS children; and so
    on down a fixed shallow tree. Each hop's candidate set and query genuinely
    depend on the previous hop's selection -- unlike a set of independent
    attention passes over one static candidate set, severing that dependency
    (feeding every hop the same static query) would change the output.
    """

    def __init__(
        self,
        vocab_size: int = 96,
        utter_hidden: int = 24,
        n_nodes: int = 13,
        n_relations: int = 6,
        graph_dim: int = 24,
        n_intents: int = 3,
        branching: int = 3,
        depth: int = 2,
    ) -> None:
        """Build utterance RNN, intent head, R-GCN embedder, and per-hop tree walkers.

        The KG is a small fixed tree of ``n_nodes = 1 + branching + branching**2``
        nodes (root at index 0, ``depth`` levels of ``branching``-way fan-out),
        stored as a ``(n_nodes, branching)`` children-index buffer (root's
        children are indices 1..branching, etc.) so hop t's candidate set is
        literally "the children of the node selected at hop t-1".
        """

        super().__init__()
        self.depth = depth
        self.branching = branching
        self.n_nodes = n_nodes
        self.token_embed = nn.Embedding(vocab_size, utter_hidden)
        self.utter_rnn = nn.GRU(utter_hidden, utter_hidden, batch_first=True)
        self.intent_head = nn.Sequential(
            nn.Linear(utter_hidden, 2 * utter_hidden),
            nn.ReLU(),
            nn.Linear(2 * utter_hidden, n_intents),
        )
        self.node_init = nn.Parameter(torch.randn(n_nodes, graph_dim) * 0.02)
        self.graph_gcn = RGCNConv(graph_dim, graph_dim, n_relations, num_bases=3)

        # Fixed parent->children index tree: level-order layout, node 0 is the
        # root, node i's children start at ``i * branching + 1``. Registered as
        # a buffer (not learned) since the walk's TOPOLOGY is fixed; only the
        # per-hop selection over it is learned/adaptive.
        child_index = torch.zeros(n_nodes, branching, dtype=torch.long)
        for node in range(n_nodes):
            for b in range(branching):
                child = node * branching + 1 + b
                child_index[node, b] = child if child < n_nodes else node
        self.register_buffer("child_index", child_index)

        query_size = utter_hidden + n_intents + graph_dim
        self.hop_attn = nn.ModuleList(
            [BilinearAttention(query_size, graph_dim, graph_dim) for _ in range(depth)]
        )
        # Projects [prior query, selected-node embedding] back down to a
        # query-sized vector so it can seed the NEXT hop's query -- this is
        # the hop-to-hop data path that makes the walk genuinely sequential.
        self.hop_update = nn.ModuleList(
            [nn.Linear(query_size + graph_dim, query_size) for _ in range(depth)]
        )

    def forward(self, tokens: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Encode dialog -> select intent -> sequential multi-hop tree walk over the KG."""

        embedded = self.token_embed(tokens)
        _, h_n = self.utter_rnn(embedded)
        utter_vec = h_n[-1]
        intent_logits = self.intent_head(utter_vec)
        intent_soft = torch.softmax(intent_logits, dim=-1)

        graph_embed = torch.relu(self.graph_gcn(self.node_init, edge_index, edge_type))
        batch = tokens.shape[0]

        node_scores = torch.zeros(batch, self.n_nodes)
        # Hop 0's query is purely the dialogue state; later hops' queries are
        # built from the PREVIOUS hop's selected-node embedding below.
        query = torch.cat(
            [utter_vec, intent_soft, graph_embed.new_zeros(batch, graph_embed.size(-1))], dim=-1
        )
        current_node = torch.zeros(batch, dtype=torch.long)  # start the walk at the root

        for hop in range(self.depth):
            # Candidate set for this hop = children of the node selected at
            # the PREVIOUS hop (root's children for hop 0) -- a real
            # gather/index_select keyed on the running walk state, not a
            # static set shared across hops.
            candidate_idx = self.child_index[current_node]  # (batch, branching)
            candidates = graph_embed[candidate_idx]  # (batch, branching, graph_dim)

            q_proj = self.hop_attn[hop].w_q(query).unsqueeze(1)
            k_proj = self.hop_attn[hop].w_k(candidates)
            hop_scores_local = self.hop_attn[hop].w_v(torch.tanh(q_proj + k_proj)).squeeze(-1)
            hop_weights = torch.softmax(hop_scores_local, dim=-1)  # (batch, branching)

            # Scatter this hop's per-candidate weights into the full node-score
            # vector so the returned distribution reflects every hop's walk.
            node_scores = node_scores.scatter_add(1, candidate_idx, hop_weights)

            # Soft (differentiable) selection: a weighted combination of the
            # candidate embeddings, weighted by THIS hop's attention -- the
            # walk's actual "descend to the selected child" step.
            selected_embed = torch.einsum("bn,bnd->bd", hop_weights, candidates)
            # The next hop's query is built from [this query, selected node] --
            # genuine hop-to-hop dependency: sever this and every hop would
            # see the same static query, which is exactly the bug being fixed.
            query = self.hop_update[hop](torch.cat([query, selected_embed], dim=-1))
            # Advance the walk pointer to the (soft) argmax child so the next
            # hop's candidate set is that child's children.
            current_node = candidate_idx.gather(
                1, hop_weights.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)

        return node_scores


def build_cr_walker() -> nn.Module:
    """Build the compact CR-Walker tree-structured KG-reasoning recommender."""

    return CRWalker().eval()


def example_input_cr_walker() -> tuple[Tensor, Tensor, Tensor]:
    """Return (token ids, KG edge_index, KG edge_type) for a tiny synthetic tree-shaped graph."""

    tokens = torch.randint(0, 96, (2, 8), dtype=torch.long)
    n_nodes, n_relations = 13, 6
    # Build edges along the same fixed tree used inside the model (root=0,
    # branching=3, depth=2) plus a handful of extra random edges so the R-GCN
    # embedder sees a slightly richer local neighborhood than the bare tree.
    tree_src, tree_dst = [], []
    for node in range(n_nodes):
        for b in range(3):
            child = node * 3 + 1 + b
            if child < n_nodes:
                tree_src.append(node)
                tree_dst.append(child)
    extra_src = torch.randint(0, n_nodes, (10,))
    extra_dst = torch.randint(0, n_nodes, (10,))
    src = torch.cat([torch.tensor(tree_src, dtype=torch.long), extra_src])
    dst = torch.cat([torch.tensor(tree_dst, dtype=torch.long), extra_dst])
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, n_relations, (src.shape[0],))
    return tokens, edge_index, edge_type


# ---------------------------------------------------------------------------
# KBRD (CRSLab KBRD-style policy net): R-GCN knowledge-graph entity encoder
# with self-attention user pooling, feeding a KG-biased Transformer dialogue
# decoder AND a bilinear recommendation head over KG entities.
# ---------------------------------------------------------------------------
class SelfAttentionPool(nn.Module):
    """Self-attention pooling of a set of KG entity embeddings into a user vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, entity_embeds: Tensor) -> Tensor:
        """entity_embeds: (B, K, dim) -> pooled user embedding (B, dim)."""

        scores = self.v(torch.tanh(self.w(entity_embeds))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * entity_embeds).sum(dim=1)


class KBRD(nn.Module):
    """Compact KBRD: R-GCN KG encoder driving both recommendation and KG-biased chat."""

    def __init__(
        self,
        vocab_size: int = 96,
        token_dim: int = 24,
        n_entity: int = 50,
        n_relation: int = 6,
        kg_dim: int = 24,
        n_heads: int = 4,
    ) -> None:
        """Build the KG R-GCN encoder, user pooling, and Transformer dialogue stack."""

        super().__init__()
        self.n_entity = n_entity
        self.kg_encoder = RGCNConv(kg_dim, kg_dim, n_relation, num_bases=3)
        self.kg_attn = SelfAttentionPool(kg_dim)
        self.rec_bias = nn.Linear(kg_dim, n_entity)

        self.token_embed = nn.Embedding(vocab_size, token_dim)
        enc_layer = nn.TransformerEncoderLayer(token_dim, n_heads, 4 * token_dim, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(token_dim, n_heads, 4 * token_dim, batch_first=True)
        self.dialog_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.user_proj_1 = nn.Linear(kg_dim, 2 * kg_dim)
        self.user_proj_2 = nn.Linear(2 * kg_dim, vocab_size)
        self.node_init = nn.Parameter(torch.randn(n_entity, kg_dim) * 0.02)

    def forward(
        self,
        context_tokens: Tensor,
        response_tokens: Tensor,
        mentioned_entities: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (recommendation scores, KG-biased response token logits)."""

        kg_embed = torch.relu(self.kg_encoder(self.node_init, edge_index, edge_type))
        user_entities = kg_embed[mentioned_entities]
        user_embed = self.kg_attn(user_entities)
        rec_scores = F.linear(user_embed, kg_embed, self.rec_bias.bias)

        enc_hidden = self.dialog_encoder(self.token_embed(context_tokens))
        dec_hidden = self.decoder(self.token_embed(response_tokens), enc_hidden)
        token_logits = F.linear(dec_hidden, self.token_embed.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embed))).unsqueeze(1)
        conv_logits = token_logits + user_logits
        return rec_scores, conv_logits


def build_kbrd() -> nn.Module:
    """Build the compact KBRD KG-biased conversational recommender."""

    return KBRD().eval()


def example_input_kbrd() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (context tokens, response tokens, mentioned entities, edge_index, edge_type)."""

    context_tokens = torch.randint(0, 96, (2, 10), dtype=torch.long)
    response_tokens = torch.randint(0, 96, (2, 6), dtype=torch.long)
    mentioned_entities = torch.randint(0, 50, (2, 3), dtype=torch.long)
    n_edges, n_relations = 70, 6
    src = torch.randint(0, 50, (n_edges,))
    dst = torch.randint(0, 50, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, n_relations, (n_edges,))
    return context_tokens, response_tokens, mentioned_entities, edge_index, edge_type


# ---------------------------------------------------------------------------
# CVAE dialogue model (kgCVAE, Zhao et al. 2017): hierarchical RNN encoder,
# recognition + prior networks, Gaussian reparameterization, BOW auxiliary
# loss, and a decoder conditioned on [context, latent].
# ---------------------------------------------------------------------------
class CVAEDialogue(nn.Module):
    """Compact hierarchical CVAE dialogue model with recognition/prior networks."""

    def __init__(
        self,
        vocab_size: int = 96,
        embed_dim: int = 24,
        utter_hidden: int = 24,
        ctx_hidden: int = 24,
        latent_size: int = 16,
        meta_size: int = 4,
    ) -> None:
        """Build utterance/context encoders, recognition+prior nets, BOW head, decoder."""

        super().__init__()
        self.latent_size = latent_size
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.utter_rnn = nn.GRU(embed_dim, utter_hidden, batch_first=True, bidirectional=True)
        self.ctx_rnn = nn.GRU(2 * utter_hidden, ctx_hidden, batch_first=True)
        self.meta_embed = nn.Linear(meta_size, ctx_hidden)

        cond_size = ctx_hidden + ctx_hidden
        recog_input_size = cond_size + 2 * utter_hidden  # condition + response utterance
        self.recog_net = nn.Linear(recog_input_size, latent_size * 2)
        self.prior_net = nn.Sequential(
            nn.Linear(cond_size, max(latent_size * 2, 32)),
            nn.Tanh(),
            nn.Linear(max(latent_size * 2, 32), latent_size * 2),
        )

        gen_input_size = cond_size + latent_size
        self.bow_project = nn.Sequential(
            nn.Linear(gen_input_size, 2 * embed_dim),
            nn.Tanh(),
            nn.Linear(2 * embed_dim, vocab_size),
        )
        self.dec_init_net = nn.Linear(gen_input_size, ctx_hidden)
        self.decoder_rnn = nn.GRU(embed_dim, ctx_hidden, batch_first=True)
        self.output_head = nn.Linear(ctx_hidden, vocab_size)

    @staticmethod
    def _sample_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self,
        context_tokens: Tensor,
        response_tokens: Tensor,
        meta: Tensor,
        decoder_input: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode context+response, sample latent, decode; return (logits, bow, mu, logvar)."""

        batch, n_turns, seq_len = context_tokens.shape
        flat_tokens = context_tokens.reshape(batch * n_turns, seq_len)
        _, utter_h = self.utter_rnn(self.token_embed(flat_tokens))
        utter_vec = utter_h.transpose(0, 1).reshape(batch * n_turns, -1)
        utter_seq = utter_vec.reshape(batch, n_turns, -1)
        _, ctx_h = self.ctx_rnn(utter_seq)
        ctx_vec = ctx_h[-1]
        meta_vec = self.meta_embed(meta)
        cond_embedding = torch.cat([ctx_vec, meta_vec], dim=-1)

        _, resp_h = self.utter_rnn(self.token_embed(response_tokens))
        resp_vec = resp_h.transpose(0, 1).reshape(batch, -1)
        recog_mulogvar = self.recog_net(torch.cat([cond_embedding, resp_vec], dim=-1))
        recog_mu, recog_logvar = recog_mulogvar.chunk(2, dim=-1)

        prior_mulogvar = self.prior_net(cond_embedding)
        prior_mu, prior_logvar = prior_mulogvar.chunk(2, dim=-1)

        latent = self._sample_gaussian(recog_mu, recog_logvar)
        gen_inputs = torch.cat([cond_embedding, latent], dim=-1)

        bow_logits = self.bow_project(gen_inputs)

        dec_init = self.dec_init_net(gen_inputs).unsqueeze(0)
        dec_hidden, _ = self.decoder_rnn(self.token_embed(decoder_input), dec_init)
        response_logits = self.output_head(dec_hidden)
        return response_logits, bow_logits, recog_mu, prior_mu


def build_cvae_dialogue() -> nn.Module:
    """Build the compact kgCVAE hierarchical dialogue model."""

    return CVAEDialogue().eval()


def example_input_cvae_dialogue() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (context turns, response tokens, meta features, decoder input tokens)."""

    context_tokens = torch.randint(0, 96, (2, 3, 8), dtype=torch.long)
    response_tokens = torch.randint(0, 96, (2, 8), dtype=torch.long)
    meta = torch.randn(2, 4)
    decoder_input = torch.randint(0, 96, (2, 6), dtype=torch.long)
    return context_tokens, response_tokens, meta, decoder_input


# ---------------------------------------------------------------------------
# Cross-Encoder / BERT Ranker (Nogueira & Cho 2019): joint query-document BERT
# encoding of the concatenated pair, [CLS]-pooled binary relevance classifier.
# ---------------------------------------------------------------------------
class CrossEncoderRanker(nn.Module):
    """Compact joint-encoding cross-encoder for MS-MARCO / TREC passage reranking."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 64,
    ) -> None:
        """Build token+position embeddings, a BERT-style encoder, and a [CLS] classifier."""

        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.segment_embed = nn.Embedding(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.relevance_head = nn.Linear(d_model, 2)

    def forward(self, pair_tokens: Tensor, segment_ids: Tensor) -> Tensor:
        """pair_tokens/segment_ids: (B, L) joint [CLS] query [SEP] doc [SEP] sequence."""

        positions = torch.arange(pair_tokens.shape[1], device=pair_tokens.device)
        embedded = (
            self.token_embed(pair_tokens)
            + self.pos_embed(positions).unsqueeze(0)
            + self.segment_embed(segment_ids)
        )
        hidden = self.encoder(self.layer_norm(embedded))
        cls_repr = hidden[:, 0]
        return self.relevance_head(cls_repr)


def build_cross_encoder_ranker() -> nn.Module:
    """Build the compact BERT cross-encoder relevance ranker."""

    return CrossEncoderRanker().eval()


def example_input_cross_encoder_ranker() -> tuple[Tensor, Tensor]:
    """Return (joint query+document token ids, segment ids)."""

    pair_tokens = torch.randint(0, 256, (2, 24), dtype=torch.long)
    segment_ids = torch.cat(
        [torch.zeros(2, 10, dtype=torch.long), torch.ones(2, 14, dtype=torch.long)], dim=1
    )
    return pair_tokens, segment_ids


MENAGERIE_ENTRIES = [
    ("COSMIC-Transformer (UniMSE, UniERC)", "build_unimse", "example_input_unimse", "2022", "NLP"),
    (
        "CR-Walker (Commonsense-Guided Recommender-Dialogue)",
        "build_cr_walker",
        "example_input_cr_walker",
        "2021",
        "NLP",
    ),
    (
        "CRSLab KBRD-style policy nets",
        "build_kbrd",
        "example_input_kbrd",
        "2019",
        "NLP",
    ),
    (
        "CVAE dialogue model",
        "build_cvae_dialogue",
        "example_input_cvae_dialogue",
        "2017",
        "NLP",
    ),
    (
        "Cross-Encoder (MS-MARCO, TREC)",
        "build_cross_encoder_ranker",
        "example_input_cross_encoder_ranker",
        "2019",
        "NLP",
    ),
]

# SOURCE: vendored from prometheusXN/D-LADAN @ main
#
# Real files (Bert_Ladan/ subtree -- the Transformer/BERT variant of D-LADAN):
#   Bert_Ladan/Dladan_component.py   (dLadan_full -- the novel D-LADAN architecture)
#   Bert_Ladan/AttenRNN.py           (AttentionOriContext, ContextAttention)
#   Bert_Ladan/GraphDistillOperators.py  (GraphDistillOperator, GraphDistillOperatorWithEdgeWeight)
#   Bert_Ladan/TransformerLayer.py   (BertLayer, TransformerFeatureWithLabel)
#   Bert_Ladan/common_utils.py       (dynamic_partition)
#
# `dLadan_full` is D-LADAN's core novel contribution: a BERT fact/law encoder plus a
# *dual* graph-distillation stack over the legal-charge-confusion graph -- a "prior"
# distiller (GraphDistillOperator, static law-relation adjacency) and a "posterior"
# distiller (GraphDistillOperatorWithEdgeWeight, dynamically re-weighted via a learned
# soft/Gumbel-softmax adjacency), each feeding a separate cross-attention "re-encoding"
# of the fact representation via a custom BertLayer-based re-encoder plus a label-aware
# Transformer matching module. This is genuinely new architecture (distillation
# operators, dual prior/posterior GNN branches, group-context pooling) well beyond a
# stock BertModel, so it is vendored, not constructed from a base-lib class.
#
# All five real files above are reproduced verbatim below (module bodies unchanged;
# only `sys.path.append('..')` / cross-file imports collapsed into this single file,
# and the unused `ipdb` debug import from the original package `__init__` chain
# dropped since it is never invoked by `dLadan_full.forward`). The top-level
# `DLADAN_Bert_full` loss-wrapper class (which additionally needs a full accusation-
# charge relation graph + memory-momentum decoders for training) is intentionally not
# vendored here; `dLadan_full` alone is the traceable feature-extraction architecture
# and is what is registered/traced below, called via its own real, unmodified
# `forward()`.

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BertConfig as _HFBertConfig
from transformers.models.bert import BertModel as _HFBertModel

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# Bert_Ladan/common_utils.py (dynamic_partition only; the rest of the file is
# training-loss / metric helpers unrelated to the traced architecture)
# ============================================================================


def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
    assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
    assert data.shape[0] == partitions.shape[0], "Partitions requires the same size as data"

    if num_partitions is None:
        num_partitions = max(torch.unique(partitions))

    return [data[partitions == i] for i in range(num_partitions)]


# ============================================================================
# Bert_Ladan/AttenRNN.py (AttentionOriContext + ContextAttention only; RNN-based
# classes in the real file are alternate encoders not used by dLadan_full)
# ============================================================================


class ContextAttention(nn.Module):
    def __init__(self, config, input_dim, with_value_fn=True):
        super(ContextAttention, self).__init__()
        self.hidden_dim = config.net.hidden_size
        self.fk = nn.Linear(input_dim, self.hidden_dim)
        if with_value_fn:
            self.fv = nn.Linear(input_dim, self.hidden_dim)
        self.fq = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activate = nn.GELU()
        self.scale = self.hidden_dim ** (-0.5)
        self.dropout = nn.Dropout(config.rnn.dropout)
        self.with_value_fn = with_value_fn

    def forward(self, feature: Tensor, contexts: Tensor, masks: Tensor):
        # hidden: B * M * H, contexts: B * H
        ratio = torch.matmul(self.fk(feature), self.fq(contexts).unsqueeze(-1))
        # ratio: B * M * 1
        ratio = ratio * self.scale
        if self.with_value_fn:
            value = self.fv(feature)
        else:
            value = feature
        max_ratio = ratio.squeeze()  # [B, M]
        max_ratio = max_ratio * masks + (masks - 1) * 1e9
        attention_score = F.softmax(max_ratio, dim=-1).unsqueeze(-1)  # [B, M, 1]
        attention_score = self.dropout(attention_score)
        result = torch.sum(attention_score * value, dim=-2)  # [B, H]
        return result, attention_score


class AttentionOriContext(nn.Module):
    def __init__(self, config, input_dim, with_value_fn=True):
        super(AttentionOriContext, self).__init__()
        self.hidden_dim = config.net.hidden_size
        self.fk = nn.Linear(input_dim, self.hidden_dim)
        if with_value_fn:
            self.fv = nn.Linear(input_dim, self.hidden_dim)
        self.context = nn.Parameter(torch.rand(1, self.hidden_dim))
        self.scale = self.hidden_dim ** (-0.5)
        self.dropout = nn.Dropout(config.rnn.dropout)
        self.with_value_fn = with_value_fn

    def forward(self, feature: Tensor, masks: Tensor):
        # feature: B * M * H, contexts: B * H
        key = self.fk(feature)
        if self.with_value_fn:
            value = self.fv(feature)
        else:
            value = feature
        ratio = torch.sum(key * self.context.unsqueeze(dim=1), dim=-1) * self.scale
        max_ratio = ratio * masks + (masks - 1) * 1e9
        attention_score = F.softmax(max_ratio, dim=-1).unsqueeze(-1)  # [B, M, 1]
        attention_score = self.dropout(attention_score)
        result = torch.sum(attention_score * value, dim=1)  # [B, H]

        return result, attention_score


# ============================================================================
# Bert_Ladan/GraphDistillOperators.py (verbatim)
# ============================================================================


def softmax_with_mask(logits, masks=None, dim=-1):
    if masks is not None:
        logits = logits + (1 - masks) * (-1e32)
    score = torch.softmax(logits, dim=dim)
    return score


class GraphDistillOperator(nn.Module):  # [node_num, 768] -> [node_num, 512]
    def __init__(self, config, input_dim, activation=True, withAgg=False):
        super(GraphDistillOperator, self).__init__()
        self.withAgg = withAgg
        self.activation = activation
        self.activation_fuc = nn.Tanh()
        self.dropout = nn.Dropout(p=config.GraphDistill.dropout)

        self.input_dim = input_dim  # 768
        self.out_dim = config.net.hidden_size  # 512

        self.distill_dence = nn.Linear(self.input_dim * 2, self.out_dim)
        self.distill_out_dence = nn.Linear(self.input_dim, self.out_dim)

        if self.withAgg:
            self.aggregate_dense = nn.Linear(self.input_dim, self.out_dim)
            self.aggregate_out_dense = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor):
        # features [node_num, feature_dim]
        adj_matrix = adj_matrix.float()
        node_num, feature_dim = features.shape
        head_features = features.unsqueeze(dim=1).repeat(1, node_num, 1)
        tail_features = features.unsqueeze(dim=0).repeat(node_num, 1, 1)
        neight_features = torch.cat([head_features, tail_features], dim=-1)
        # [node_num, node_num, feature_dim]

        neight_features_sum = torch.sum(adj_matrix.unsqueeze(dim=-1) * neight_features, dim=1)
        neigh_mask = torch.max(adj_matrix, dim=-1, keepdim=True).values

        neigh_num = adj_matrix.sum(dim=-1, keepdims=True) + (1 - neigh_mask) * 1
        neight_features_ave = neight_features_sum / neigh_num

        neigh_features = self.distill_dence(neight_features_ave)
        feature_updated: torch.Tensor = self.distill_out_dence(features) - neigh_features
        feature_updated = feature_updated.reshape([node_num, self.out_dim])
        if self.activation:
            feature_updated = self.activation_fuc(feature_updated)
        feature_updated = self.dropout(feature_updated)

        if self.withAgg:
            neighbor_features_aggregate = self.aggregate_dense(adj_matrix @ features)
            feature_aggregate: torch.Tensor = (
                self.aggregate_out_dense(features) + neighbor_features_aggregate
            )
            feature_aggregate = feature_aggregate.reshape([node_num, self.out_dim])
            feature_aggregate = self.activation(feature_aggregate)
            feature_aggregate = self.dropout(feature_aggregate)
            return feature_updated, feature_aggregate
        else:
            return feature_updated, feature_updated


class GraphDistillOperatorWithEdgeWeight(nn.Module):  # [node_num, ]
    def __init__(self, config, input_dim, activation=True, withAgg=False):
        super(GraphDistillOperatorWithEdgeWeight, self).__init__()
        self.withAgg = withAgg
        self.activation = activation
        self.activation_fuc = nn.Tanh()
        self.dropout = nn.Dropout(p=config.GraphDistill.dropout)

        self.input_dim = input_dim  # 256
        self.out_dim = config.net.hidden_size  # 256

        self.distill_dence = nn.Linear(self.input_dim * 2, self.out_dim)
        self.distill_out_dence = nn.Linear(self.input_dim, self.out_dim)

        if self.withAgg:
            self.aggregate_dense = nn.Linear(self.input_dim, self.out_dim)
            self.aggregate_out_dense = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, features: torch.Tensor, key_features: torch.Tensor, adj_matrix: torch.Tensor):
        # features [node_num, feature_dim]
        adj_matrix = adj_matrix.float()
        node_num, feature_dim = features.shape
        head_features = features.unsqueeze(dim=1).repeat(1, node_num, 1)
        tail_features = features.unsqueeze(dim=0).repeat(node_num, 1, 1)
        neigh_features_distill = torch.cat([head_features, tail_features], dim=-1)
        # [node_num, node_num, feature_dim]

        self_loop_mask = 1.0 - torch.eye(n=node_num, dtype=torch.float)
        self_loop_mask = self_loop_mask.to(adj_matrix.device)
        adj_matrix_soft = softmax_with_mask(
            adj_matrix * 5.0, masks=self_loop_mask, dim=-1
        )  # Gumbal Softmax
        neight_features_norm = torch.sum(
            adj_matrix_soft.unsqueeze(dim=-1) * neigh_features_distill, dim=1
        )

        neigh_features = self.distill_dence(neight_features_norm)
        feature_updated: torch.Tensor = self.distill_out_dence(features) - neigh_features
        feature_updated = feature_updated.reshape([node_num, self.out_dim])

        if self.activation:
            feature_updated = self.activation_fuc(feature_updated)
        feature_updated = self.dropout(feature_updated)

        if self.withAgg:
            neighbor_features_aggregate = self.aggregate_dense(adj_matrix @ key_features)
            feature_aggregate: torch.Tensor = (
                self.aggregate_out_dense(key_features) + neighbor_features_aggregate
            )
            feature_aggregate = feature_aggregate.reshape([node_num, self.out_dim])
            feature_aggregate = self.activation(feature_aggregate)
            feature_aggregate = self.dropout(feature_aggregate)
            return feature_updated, feature_aggregate
        else:
            return feature_updated, feature_updated


# ============================================================================
# Bert_Ladan/TransformerLayer.py (BertLayer stack + TransformerFeatureWithLabel,
# verbatim; this is D-LADAN's OWN from-scratch mini-BERT-layer reimplementation,
# used as the fact/law "re-encoder", separate from the real pretrained self.PLM_model)
# ============================================================================

hidden_size = 768


class MHAttention(nn.Module):
    def __init__(self, multihead=False):
        super(MHAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.num_attention_heads = 12
        if multihead:
            self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 12*64=768
        else:
            self.attention_head_size = hidden_size
        self.all_head_size = hidden_size
        self.multihead = multihead

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, toks, masks: torch.Tensor):
        # mask [batch_size, sentence_len]
        masks = masks.float()
        mask = torch.bmm(masks.unsqueeze(dim=-1), masks.unsqueeze(dim=1))
        mask = (1.0 - mask) * -1e31
        if self.multihead:
            mixed_query_layer = self.query(toks)
            mixed_key_layer = self.key(toks)
            mixed_value_layer = self.value(toks)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
        else:
            query_layer = self.query(toks)
            key_layer = self.key(toks)
            value_layer = self.value(toks)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size**0.5)
        attention_scores = attention_scores + mask.unsqueeze(dim=1).repeat(
            1, self.num_attention_heads, 1, 1
        )
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if self.multihead:
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.contiguous()
        return context_layer


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertCrossOutput(nn.Module):
    def __init__(self):
        super(BertCrossOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TInterAttention(nn.Module):
    def __init__(self, m=True):
        super(TInterAttention, self).__init__()
        self.cross = MHAttention(multihead=m)
        self.output = BertCrossOutput()

    def forward(self, toks, masks: torch.Tensor):
        cross_output = self.cross(toks, masks)  # compute the neighbor aggregation
        attention_output = self.output(cross_output, toks)
        attention_output = attention_output * masks.unsqueeze(dim=-1)
        return attention_output


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / (2.0**0.5)))


class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        intermediate_size = 3072
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        intermediate_size = 3072
        hidden_dropout_prob = 0.1
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, m=True):
        super(BertLayer, self).__init__()
        self.attention = TInterAttention(m=m)
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, toks, masks):
        attention_output = self.attention(toks, masks)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TransformerFeatureWithLabel(nn.Module):
    def __init__(self, feature_dim, nhead, dropout):
        super().__init__()
        self.nhead = nhead
        self.self_atten = nn.MultiheadAttention(feature_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, Transformer_input, mask: torch.Tensor = None, **kwargs):
        Transformer_input = Transformer_input.transpose(0, 1)  # [seq_len, batch_size, feature_dim]
        mask = mask.unsqueeze(dim=-1) @ mask.unsqueeze(dim=1)
        mask = mask.repeat(self.nhead, 1, 1)  # [batch_size*head_num, sentence_num, sentence_num]
        Transformer_out: torch.Tensor = self.self_atten(
            Transformer_input, Transformer_input, Transformer_input, attn_mask=mask
        )[0]
        Transformer_out = Transformer_out.transpose(0, 1)

        Transformer_out = Transformer_input.transpose(0, 1) + self.dropout1(Transformer_out)
        Transformer_out = self.norm1(Transformer_out)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(Transformer_out))))
        Transformer_out = Transformer_out + self.dropout2(src2)
        Transformer_out = self.norm2(Transformer_out)

        return Transformer_out


# ============================================================================
# Bert_Ladan/Dladan_component.py :: dLadan_full (verbatim, modulo the BertConfig
# `.from_pretrained`/`BertModel.from_pretrained` calls being swapped for a tiny
# randomly-initialized in-process BertModel -- same real `transformers.BertModel`
# class, just not downloaded from the hub -- exactly as the `build_*` factory does
# for every other vendored HF-based menagerie entry)
# ============================================================================


class dLadan_full(nn.Module):
    """
    Define the Transformer version of D-LADAN.
    """

    def __init__(self, config, group_num, accu_relation=1, **kwargs):
        super(dLadan_full, self).__init__(**kwargs)

        self.config = config
        self.law_sentence_len = config.train.law_sentence_len
        self.fact_sentence_len = 512
        self.use_mean = config.train.use_mean_pooling
        self.group_num = group_num
        self.num_distill_layers = self.config.net.num_distill_layers
        self.accu_relation = accu_relation
        self.bert_config = config.train.plm_config
        self.PLM_model = _HFBertModel(self.bert_config)

        # 'define_encoder_base'
        self.sentence_encoder = BertLayer(m=True)
        self.sentence_attention = AttentionOriContext(config=config, input_dim=config.net.bert_size)

        # 'define_distiller_prior'
        self.graph_distillers_prior = []
        graph_input_piror = config.net.hidden_size
        for i in range(self.num_distill_layers):
            distill_layer = GraphDistillOperator(config, input_dim=graph_input_piror)
            self.graph_distillers_prior.append(distill_layer)
            graph_input_piror = distill_layer.out_dim
        self.graph_distillers_prior = nn.ModuleList(self.graph_distillers_prior)

        # 'define_distiller_posterior'
        self.graph_distillers_posterior = []
        self.hidden_size = config.net.hidden_size
        graph_input_posterior = config.net.hidden_size
        for i in range(self.num_distill_layers):
            distill_layer = GraphDistillOperatorWithEdgeWeight(
                config, input_dim=graph_input_posterior
            )  # with edge weight
            self.graph_distillers_posterior.append(distill_layer)
            graph_input_posterior = distill_layer.out_dim
        self.graph_distillers_posterior = nn.ModuleList(self.graph_distillers_posterior)

        # 'context_generator_prior'
        self.group_chosen_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.group_chosen = nn.Linear(self.hidden_size, self.group_num)
        self.context_s_prior = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 'define_encoder_prior'
        self.sentence_encoder_prior = BertLayer(m=True)
        self.sentence_attention_prior = ContextAttention(
            config=config, input_dim=config.net.bert_size
        )

        # define_encoder_posterior'
        self.sentence_encoder_posterior = BertLayer(m=True)
        self.sentence_attention_posterior = ContextAttention(
            config=config, input_dim=config.net.bert_size
        )

        # context_generator_posterior
        self.Transformer_posterior = TransformerFeatureWithLabel(
            feature_dim=self.hidden_size, nhead=4, dropout=config.GraphDistill.dropout
        )
        self.matching_law_posterior = nn.Linear(self.hidden_size, self.hidden_size)
        self.matching_fact_posterior = nn.Linear(config.net.bert_size * 2, self.hidden_size)
        self.context_s_posterior = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.GraphDistill.dropout)

        if self.accu_relation is not None:
            # build posterior of charge
            self.graph_distillers_posterior_accu = []
            graph_input_posterior = config.net.hidden_size
            for i in range(self.num_distill_layers):
                distill_layer = GraphDistillOperatorWithEdgeWeight(
                    config, input_dim=graph_input_posterior
                )
                self.graph_distillers_posterior_accu.append(distill_layer)
                graph_input_posterior = distill_layer.out_dim
            self.graph_distillers_posterior_accu = nn.ModuleList(
                self.graph_distillers_posterior_accu
            )

            # define_encoder_posterior_accu
            self.sentence_encoder_posterior_A = BertLayer(m=True)
            self.sentence_attention_posterior_A = ContextAttention(
                config=config, input_dim=config.net.bert_size
            )

            # context_generator_posterior_accu
            self.Transformer_posterior_A = TransformerFeatureWithLabel(
                feature_dim=self.hidden_size, nhead=4, dropout=config.GraphDistill.dropout
            )
            self.matching_law_posterior_A = nn.Linear(self.hidden_size, self.hidden_size)
            self.matching_fact_posterior_A = nn.Linear(config.net.bert_size * 2, self.hidden_size)
            self.context_s_posterior_A = nn.Linear(self.hidden_size, self.hidden_size)

        self.posterior_mask = nn.Parameter(torch.zeros([1, self.hidden_size]), requires_grad=False)
        self.posterior_maskF = nn.Parameter(torch.zeros([1, self.hidden_size]), requires_grad=False)

    def distill_pooling(self, features, group_index):
        node_num, feature_dim = features.shape
        features_grouped = dynamic_partition(features, group_index, num_partitions=self.group_num)

        group_contexts = []
        for i in range(self.group_num):
            u = torch.max(features_grouped[i], 0).values  # law_representation[i]: [n, law_size]
            u_2 = torch.min(features_grouped[i], 0).values
            group_contexts.append(torch.cat([u, u_2], dim=-1))
        group_contexts = torch.reshape(torch.cat(group_contexts, 0), (-1, 2 * feature_dim))
        return group_contexts

    def re_encoding_fact(self, name, inputs, dense_funcs, context_funcs, encoder_funcs, masks):
        fact_base, key_list, context_list, fact_rep_sentences, fact_sentence_level = inputs
        law_Dense, fact_Dense = dense_funcs
        group_chosen, Transformer, context_generation_s = context_funcs
        sentence_re_encoder, sentence_reattention = encoder_funcs
        sentence_mask, sentence_mask_1, real_mask = masks

        if name == "Posterior":
            batch_size, word_num, feature_dim = fact_sentence_level.shape
            key_num, feature_dim = key_list.shape
            matching_law_prior: torch.Tensor = law_Dense(key_list)  # part of source input
            matching_fact_prior = fact_Dense(fact_sentence_level)
            label_mask = torch.ones([batch_size, key_num], dtype=torch.float).to(
                sentence_mask_1.device
            )
            matching_mask = torch.cat([sentence_mask_1, label_mask], axis=-1)
            matching_mask = torch.reshape(matching_mask, shape=(-1, word_num + key_num))

            label_input = matching_law_prior.unsqueeze(dim=0).repeat(batch_size, 1, 1)

            Transformer_input = torch.cat([matching_fact_prior, label_input], dim=1)
            Transformer_out = Transformer(Transformer_input, mask=matching_mask)

            CLS_output = Transformer_out[:, 0, :].unsqueeze(dim=1)
            law_output: torch.Tensor = Transformer_out[:, word_num:, :]

            scale = feature_dim ** (-0.5)
            group_pred_scores: torch.Tensor = (
                torch.bmm(CLS_output, law_output.transpose(1, 2)) * scale
            )
            group_pred_scores = group_pred_scores.squeeze(dim=1)
            group_pred = torch.softmax(group_pred_scores, dim=-1)
            re_context = group_pred @ context_list
        else:
            group_pred_scores = group_chosen(self.group_chosen_hidden(fact_base))
            group_pred = torch.softmax(group_pred_scores, dim=-1)
            re_context = group_pred @ context_list

        context_sentence = torch.reshape(
            context_generation_s(re_context), shape=(-1, self.config.net.hidden_size)
        )
        re_fact_sentence_level = sentence_re_encoder(fact_rep_sentences, sentence_mask)
        fact_prior, score_sentence = sentence_reattention(
            re_fact_sentence_level, context_sentence, masks=real_mask
        )

        return fact_prior, re_fact_sentence_level, group_pred_scores, score_sentence

    @staticmethod
    def get_real_mask(masks: torch.Tensor):
        new_mask = masks.clone().to(masks.device)
        new_mask[:, 0] = 0
        sum_mask = masks.sum(dim=-1)  # [batch_size]
        one_hot_matrix = F.one_hot(sum_mask - 1, masks.shape[-1]).to(masks.device)
        return new_mask - one_hot_matrix

    def forward(
        self,
        inputs,
        law_information=None,
        warming_up=False,
        fact_attention_mask=None,
        sentence_mask=None,
        accu_information=None,
        time_information=None,
        law_attention_mask=None,
    ):
        fact_inputs_ids, fact_token_type_ids = inputs
        (
            law_input_ids,
            law_token_type_ids,
            adj_matrix_law,
            group_indexes,
            law_inputs_posterior,
            adj_matrix_posterior,
        ) = law_information
        real_mask_law = self.get_real_mask(law_attention_mask)
        real_mask_fact = self.get_real_mask(fact_attention_mask)
        # law_encoding_base: [law_num, sentence_len] --> [law_num, feature_dim]
        word_embedding_law = self.PLM_model(
            input_ids=law_input_ids,
            attention_mask=law_attention_mask,
            token_type_ids=law_token_type_ids,
        )[0]
        law_word_level = self.sentence_encoder(word_embedding_law, masks=law_attention_mask)
        law_base, _ = self.sentence_attention(law_word_level, masks=real_mask_law)

        # GeneratePriorGroupInformation
        distilled_law_prior = law_base
        for i in range(self.num_distill_layers):
            distilled_law_prior, aggregate_law_prior = self.graph_distillers_prior[i](
                features=distilled_law_prior, adj_matrix=adj_matrix_law
            )
        context_list_prior = self.distill_pooling(
            features=distilled_law_prior, group_index=group_indexes
        )
        # 'EncodingLaw'
        distilled_law_posterior, aggregate_law_posterior = (
            law_inputs_posterior,
            law_inputs_posterior,
        )
        for i in range(self.num_distill_layers):
            distilled_law_posterior, aggregate_law_posterior = self.graph_distillers_posterior[i](
                features=distilled_law_posterior,
                key_features=aggregate_law_posterior,
                adj_matrix=adj_matrix_posterior,
            )

        if warming_up:
            distilled_law_posterior *= self.posterior_mask
            aggregate_law_posterior *= self.posterior_mask
            distilled_law_posterior = distilled_law_posterior.detach()
            aggregate_law_posterior = aggregate_law_posterior.detach()

        context_list_posterior = distilled_law_posterior

        # EncodeFact
        word_embedding_fact = self.PLM_model(
            fact_inputs_ids, fact_attention_mask, fact_token_type_ids
        )[0]
        fact_word_level = self.sentence_encoder(word_embedding_fact, sentence_mask)
        fact_base, score_s_base = self.sentence_attention(fact_word_level, masks=real_mask_fact)

        # PriorEncodingFact
        fact_prior, fact_prior_word, group_pred_prior, score_s_prior = self.re_encoding_fact(
            inputs=[fact_base, None, context_list_prior, word_embedding_fact, None],
            dense_funcs=[None, None],
            context_funcs=[self.group_chosen, None, self.context_s_prior],
            encoder_funcs=[self.sentence_encoder_prior, self.sentence_attention_prior],
            masks=[sentence_mask, None, real_mask_fact],
            name="Prior",
        )

        # PosteriorEncodingFact_Law
        fact_sentence = torch.cat([fact_word_level, fact_prior_word], axis=-1)
        new_sentence_mask = sentence_mask
        fact_posterior, fact_posterior_word, group_pred_posterior, score_s_posterior = (
            self.re_encoding_fact(
                inputs=[
                    None,
                    law_inputs_posterior,
                    context_list_posterior,
                    word_embedding_fact,
                    fact_sentence,
                ],
                dense_funcs=[self.matching_law_posterior, self.matching_fact_posterior],
                context_funcs=[None, self.Transformer_posterior, self.context_s_posterior],
                encoder_funcs=[self.sentence_encoder_posterior, self.sentence_attention_posterior],
                masks=[sentence_mask, new_sentence_mask, real_mask_fact],
                name="Posterior",
            )
        )

        if warming_up:
            fact_posterior: torch.Tensor = fact_posterior * self.posterior_maskF
            fact_posterior = fact_posterior.detach()

        if self.accu_relation is not None:
            accu_inputs_posterior, accu_adj_matrix_posterior = accu_information
            distilled_accu_posterior, aggregate_accu_posterior = (
                accu_inputs_posterior,
                accu_inputs_posterior,
            )
            for i in range(self.num_distill_layers):
                distilled_accu_posterior, aggregate_accu_posterior = (
                    self.graph_distillers_posterior_accu[i](
                        features=distilled_accu_posterior,
                        key_features=aggregate_accu_posterior,
                        adj_matrix=accu_adj_matrix_posterior,
                    )
                )
            if warming_up:
                distilled_accu_posterior: torch.Tensor = (
                    distilled_accu_posterior * self.posterior_mask
                )
                aggregate_accu_posterior: torch.Tensor = (
                    aggregate_accu_posterior * self.posterior_mask
                )
                distilled_accu_posterior = distilled_accu_posterior.detach()
                aggregate_accu_posterior = aggregate_accu_posterior.detach()

            context_list_posterior_A = self.dropout(distilled_accu_posterior)

            # 'PosteriorEncodingFact_Accu'
            fact_sentence = torch.cat([fact_word_level, fact_prior_word], axis=-1)
            new_sentence_mask = sentence_mask
            (
                fact_posterior_A,
                fact_posterior_sentence_A,
                group_pred_posterior_A,
                score_s_posterior_A,
            ) = self.re_encoding_fact(
                inputs=[
                    None,
                    accu_inputs_posterior,
                    context_list_posterior_A,
                    word_embedding_fact,
                    fact_sentence,
                ],
                dense_funcs=[self.matching_law_posterior_A, self.matching_fact_posterior_A],
                context_funcs=[None, self.Transformer_posterior_A, self.context_s_posterior_A],
                encoder_funcs=[
                    self.sentence_encoder_posterior_A,
                    self.sentence_attention_posterior_A,
                ],
                masks=[sentence_mask, new_sentence_mask, real_mask_fact],
                name="Posterior",
            )
            if warming_up:
                fact_posterior_A: torch.Tensor = fact_posterior_A * self.posterior_maskF
                fact_posterior_A = fact_posterior_A.detach()
                group_pred_posterior = group_pred_posterior.detach()
                group_pred_posterior_A = group_pred_posterior_A.detach()

            fact_rep = torch.cat([fact_base, fact_prior, fact_posterior, fact_posterior_A], axis=-1)
            law_rep = law_base

            return [
                fact_rep,
                law_rep,
                group_pred_prior,
                group_pred_posterior,
                group_pred_posterior_A,
                score_s_base,
                score_s_prior,
                score_s_posterior,
                score_s_posterior_A,
            ]

        fact_rep = torch.cat([fact_base, fact_prior, fact_posterior], axis=-1)
        law_rep = law_base
        return [
            fact_rep,
            law_rep,
            group_pred_prior,
            group_pred_posterior,
            score_s_base,
            score_s_prior,
            score_s_posterior,
        ]


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def _make_dladan_config(hidden=768, bert_size=768, num_distill_layers=2, dropout=0.0):
    """Tiny SimpleNamespace mirroring the real repo's nested `config.train.X` /
    `config.net.X` / `config.GraphDistill.X` / `config.rnn.X` attribute-access config
    object (the real repo builds this from a .config INI file via a custom Config
    class; only the attribute surface actually touched by dLadan_full.__init__/
    forward is reproduced here). `hidden`/`bert_size` are pinned to 768 because
    TransformerLayer.py's real `BertLayer`/`MHAttention` re-encoder hardcodes a
    module-level `hidden_size = 768` constant (not driven by config in the real
    repo) -- keeping it faithful means matching that fixed width rather than
    shrinking it, even though the BERT encoder itself is otherwise tiny."""
    plm_config = _HFBertConfig(
        vocab_size=200,
        hidden_size=bert_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return SimpleNamespace(
        train=SimpleNamespace(
            law_sentence_len=16,
            fact_sentence_len=16,
            use_mean_pooling=True,
            plm_config=plm_config,
        ),
        net=SimpleNamespace(
            num_distill_layers=num_distill_layers,
            hidden_size=hidden,
            bert_size=bert_size,
        ),
        GraphDistill=SimpleNamespace(dropout=dropout),
        rnn=SimpleNamespace(dropout=dropout),
    )


def build_dladan():
    config = _make_dladan_config()
    model = dLadan_full(config=config, group_num=3, accu_relation=1)
    model.eval()
    return model


def example_input_dladan():
    torch.manual_seed(0)
    hidden = 768  # must match _make_dladan_config()'s net.hidden_size
    law_num = 6
    accu_num = 5
    seq_len = 16
    batch = 1

    fact_input_ids = torch.randint(1, 200, (batch, seq_len))
    fact_token_type_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    fact_attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    sentence_mask = torch.ones(batch, seq_len, dtype=torch.long)

    law_input_ids = torch.randint(1, 200, (law_num, seq_len))
    law_token_type_ids = torch.zeros(law_num, seq_len, dtype=torch.long)
    law_attention_mask = torch.ones(law_num, seq_len, dtype=torch.long)

    adj_matrix_law = (torch.rand(law_num, law_num) > 0.5).float()
    # every group index (0..group_num-1) must appear at least once, or
    # dLadan_full.distill_pooling's torch.max over an empty group errors out
    group_indexes = torch.tensor([0, 1, 2, 0, 1, 2][:law_num])
    law_inputs_posterior = torch.randn(law_num, hidden)
    adj_matrix_posterior = (torch.rand(law_num, law_num) > 0.5).float()

    accu_inputs_posterior = torch.randn(accu_num, hidden)
    accu_adj_matrix_posterior = (torch.rand(accu_num, accu_num) > 0.5).float()

    law_information = (
        law_input_ids,
        law_token_type_ids,
        adj_matrix_law,
        group_indexes,
        law_inputs_posterior,
        adj_matrix_posterior,
    )
    accu_information = (accu_inputs_posterior, accu_adj_matrix_posterior)

    return (
        (fact_input_ids, fact_token_type_ids),
        law_information,
        False,
        fact_attention_mask,
        sentence_mask,
        accu_information,
        None,
        law_attention_mask,
    )


MENAGERIE_ENTRIES = [
    ("D-LADAN", build_dladan, example_input_dladan, 2023, "vendored-pytorch"),
]

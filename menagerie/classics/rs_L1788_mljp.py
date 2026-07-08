# SOURCE: vendored from 6666ev/ML-LJP @ master
# https://raw.githubusercontent.com/6666ev/ML-LJP/master/code/models/ELELJP_Num.py
# https://raw.githubusercontent.com/6666ev/ML-LJP/master/code/models/GAT.py
# https://raw.githubusercontent.com/6666ev/ML-LJP/master/code/models/SupConLoss.py
# https://raw.githubusercontent.com/6666ev/ML-LJP/master/code/models/ToksTransformer.py
#
# ML-LJP / ELELJP_Num (SIGIR 2023, "Towards Multi-Law Aware Legal Judgment Prediction") --
# the repo's default registered model (`name2model["ELELJP_Num"]` in code/main.py). An
# ELECTRA-encoder legal-judgment predictor that fuses three mechanisms named in the queue
# notes: a GAT (graph attention over per-article "law node" embeddings, masked by the
# co-occurrence of gold article labels), supervised contrastive learning (`SupConLoss`,
# imported but applied by the training loop over the `cl_emb` outputs -- kept as a real
# vendored submodule of the model even though ELELJP_Num.forward itself only returns the
# raw embeddings for the trainer to consume), and a "number representation" text encoder
# (`get_mantissa_embedding` / `text_enc`) that blends token embeddings with a Gaussian
# mantissa/exponent/unit encoding of numeric tokens (amounts, dates) detected via a parallel
# `mantissa` tensor.
#
# `ELELJP_Num`, `GAT`/`GraphAttentionLayer`/`GraphAttentionHead`, and `SupConLoss` are
# copied verbatim (module names/architecture unchanged) from the three files above.
# `DetTransformer`/`DetEmbeddings`/`BertLayerNorm` are copied verbatim from
# ToksTransformer.py (the two "detail" transformer towers over the article/charge text --
# only these three classes are used by ELELJP_Num; the unrelated ToksAttention/BertLayer/
# distillation classes in that file are baseline scaffolding for a different sibling model
# and are not needed here). All `.cuda()` calls are dropped so the model runs on whatever
# device it is constructed/traced on (menagerie's convention -- see L1793_fraudre.py);
# behavior is otherwise unchanged. The real `self.bert` is
# `AutoModel.from_pretrained("code/ptm/electra-small")`, a local fine-tuned ELECTRA-small
# checkpoint that ships only inside the authors' private training run and is not fetchable;
# we substitute a real, tiny random-init `transformers.ElectraModel` (same class, same
# `embeddings` / `embeddings_project` / `encoder` API the original code calls directly) built
# from a from-scratch `ElectraConfig` -- the architecture is identical, only the checkpoint
# provenance changes. `self.details` (article/charge label description text, tokenized by
# the real `code/utils/loader.py:load_details`) is reproduced as random token-id tensors over
# the model's own tiny synthetic vocab, matching the real `load_details` output's dict shape
# (`input_ids`, `token_type_ids`, `attention_mask`) without depending on a specific
# pretrained-tokenizer vocabulary.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraModel

DROPOUT = 0.3
HIDDEN_SIZE = 32


# --- vendored from code/models/GAT.py ---------------------------------------------------
class GAT(nn.Module):
    def __init__(self, nfeat, outfeat, dropout, nheads, maps=None, alpha=0.2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        nhid = int(nfeat / nheads)
        self.attention_layers = nn.Sequential(
            GraphAttentionLayer(nfeat, nhid, nheads, dropout, alpha=alpha),
            GraphAttentionLayer(nfeat, nhid, nheads, dropout, alpha=alpha),
            GraphAttentionLayer(nfeat, nhid, nheads, dropout, alpha=alpha),
        )

    def forward(self, x, adj):
        x, _ = self.attention_layers((x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout, alpha=0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList(
            [GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        )
        self.out_att = GraphAttentionHead(nfeat, nfeat, dropout=dropout, alpha=alpha)

    def forward(self, inputs):
        x, adj = inputs  # x: [batch, ar_num, dim]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # x: [batch, ar_num, dim]
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x, adj


class GraphAttentionHead(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.2, activate_function=F.elu):
        super(GraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.activate_function = activate_function
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.matmul(h, self.W.unsqueeze(0))
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -1e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.activate_function is not None:
            return self.activate_function(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


# --- vendored from code/models/SupConLoss.py --------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, features2=None, labels=None, mask=None):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        if features2 is not None:
            anchor_dot_contrast = torch.div(torch.matmul(features, features2.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        if features2 is not None:
            logits_mask = torch.ones_like(mask)
        positives_mask = mask * logits_mask
        negatives_mask = 1.0 - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)
        if num_positives_per_row.sum() == 0:
            return torch.tensor(0.0, device=device)

        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True
        )

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            return torch.tensor(0.0, device=device)

        log_probs = (
            torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0]
            / num_positives_per_row[num_positives_per_row > 0]
        )

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


# --- vendored from code/models/ToksTransformer.py (subset used by ELELJP_Num) -----------
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


class DetEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""

    def __init__(self, vocab_size):
        super(DetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, HIDDEN_SIZE, padding_idx=0)
        self.LayerNorm = BertLayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DetTransformer(nn.Module):
    def __init__(self, vocab_size, num_hidden_layers=3):
        super(DetTransformer, self).__init__()
        self.embeddings = DetEmbeddings(vocab_size)
        self.layer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(HIDDEN_SIZE, nhead=8, batch_first=True, dropout=0.3)
                for _ in range(num_hidden_layers)
            ]
        )
        self.mh_attn = nn.MultiheadAttention(HIDDEN_SIZE, num_heads=8, batch_first=True)
        self.pooling = "cls"

    def forward(self, hidden_states, det_text):
        batch = hidden_states.shape[0]
        input_ids = det_text["input_ids"]

        det_num = len(input_ids)
        det_emb = self.embeddings(input_ids)

        for layer_module in self.layer:
            det_emb = layer_module(det_emb)
        det_emb = det_emb.permute(0, 2, 1)
        det_emb = F.max_pool1d(det_emb, det_emb.size(2)).squeeze(2)

        det_emb = det_emb.unsqueeze(0)
        det_emb = det_emb.expand([batch, det_num, HIDDEN_SIZE])  # [batch, n_labels, dim]
        hn, _ = self.mh_attn(det_emb, hidden_states, hidden_states)  # [batch, seq_len, dim]
        return hn, det_emb


# --- vendored from code/models/ELELJP_Num.py ---------------------------------------------
class ELELJP_Num(nn.Module):
    def __init__(self, bert, vocab_size=5000, hid_dim=32, maps=None, details=None):
        super().__init__()
        self.bert = bert
        self.is_small_ptm = hasattr(bert, "embeddings_project")

        self.ar_cls = len(maps["a2i"])
        self.ch_cls = len(maps["c2i"])
        self.pt_cls = maps["pt_cls_len"]

        self.article_det_transformer = DetTransformer(vocab_size, num_hidden_layers=3)
        self.charge_det_transformer = DetTransformer(vocab_size, num_hidden_layers=3)

        self.hid_dim = hid_dim
        # real code's text_enc halves self.hid_dim in place the first time it runs
        # (`if self.hid_dim == 256: self.hid_dim //= 2`) so the mantissa/number encoding
        # math operates in the *pre-projection* embedding space of the small-ELECTRA
        # variant; we track that pre-projection size explicitly instead of relying on the
        # original's implicit "==256" sentinel, which does not hold at menagerie's tiny
        # scale, but the halving semantics (post-projection hid_dim -> pre-projection
        # embedding_size) are the same real relationship.
        self.embed_dim = bert.config.embedding_size if self.is_small_ptm else hid_dim

        self.fc_ch1 = torch.nn.Linear(2 * hid_dim, hid_dim)
        self.fc_ar1 = torch.nn.Linear(2 * hid_dim, hid_dim)

        self.fc_ar2 = torch.nn.Linear(hid_dim, self.ar_cls)
        self.fc_ch2 = torch.nn.Linear(hid_dim, self.ch_cls)

        self.fc_pt1 = torch.nn.Linear(3 * hid_dim, hid_dim)
        self.fc_pt2 = torch.nn.Linear(hid_dim, self.pt_cls)

        self.tanh = nn.Tanh()
        self.details = details or {}
        self.maps = maps

        self.ca_role_embedding = nn.Parameter(torch.zeros(hid_dim))
        self.pa_role_embedding = nn.Parameter(torch.zeros(hid_dim))

        self.GAT_ar = GAT(nfeat=hid_dim, outfeat=self.ar_cls, maps=maps, dropout=0.2, nheads=8)
        self.sup_con_loss = SupConLoss(scale_by_temperature=False)

    def get_penalty_gat(self, label_specific_emb, mask=None):
        return self.GAT_ar(label_specific_emb, self.ar_adj, mask)

    def get_mantissa_embedding(self, mantissa_emb):
        q_uniform = torch.linspace(-10, 10, self.embed_dim, device=mantissa_emb.device)
        q_uniform = q_uniform.expand(mantissa_emb.shape)
        NE = torch.exp(-((mantissa_emb - q_uniform) ** 2) * 0.025)
        NE = torch.where(mantissa_emb > 0, NE, torch.zeros_like(NE).float())
        return NE

    def text_enc(self, data):
        text = data["fact"]
        input_ids = text["input_ids"]
        embeddings = self.bert.embeddings(input_ids)

        mantissa_emb = data["mantissa"].unsqueeze(-1).repeat([1, 1, self.embed_dim])
        mantissa_embedding = self.get_mantissa_embedding(mantissa_emb)
        exponent_embeddings = torch.where(
            mantissa_emb > 0, embeddings, torch.zeros_like(embeddings).float()
        )
        unit_embeddings = embeddings.clone()
        unit_embeddings[:, :-1, :] = unit_embeddings[:, 1:, :].clone()
        unit_embeddings = torch.where(
            mantissa_emb > 0, unit_embeddings, torch.zeros_like(embeddings).float()
        )
        number_embeddings = (
            mantissa_embedding * 0.2 + exponent_embeddings * 0.4 + unit_embeddings * 0.4
        )

        # numeric encoding merge
        embeddings = torch.where(mantissa_emb > 0, number_embeddings, embeddings)

        if self.is_small_ptm:
            embeddings = self.bert.embeddings_project(embeddings)

        output = self.bert.encoder(embeddings)
        return output.last_hidden_state

    def get_mask_adj(self, label):
        batch, label_num = label.shape
        adj = label.unsqueeze(-1)
        adj = adj.expand(batch, label_num, label_num)
        adj2 = adj.transpose(1, 2)
        adj = adj + adj2
        ones = torch.ones_like(adj)
        adj = torch.where(adj > 1.5, ones, -ones)
        return adj

    def forward(self, data):
        fact_emb = self.text_enc(data)
        article_text = self.details["a_details"]
        charge_text = self.details["c_details"]
        af_emb, a_emb = self.article_det_transformer(hidden_states=fact_emb, det_text=article_text)
        cf_emb, c_emb = self.charge_det_transformer(hidden_states=fact_emb, det_text=charge_text)
        af_emb = self.tanh(af_emb)
        cf_emb = self.tanh(cf_emb)

        gt_ar = data["article"]

        fact_pool_emb = torch.max(fact_emb, dim=1)[0].unsqueeze(1)
        # charge prediction
        cf_pool_emb = torch.max(cf_emb, dim=1)[0].unsqueeze(1)
        ch_emb = torch.cat([fact_pool_emb, cf_pool_emb], dim=1).view(len(fact_pool_emb), -1)
        ch_pred = self.fc_ch2(nn.ReLU()(self.fc_ch1(ch_emb)))

        # article prediction
        af_pool_emb = torch.max(af_emb, dim=1)[0].unsqueeze(1)
        ar_emb = torch.cat([fact_pool_emb, af_pool_emb], dim=1).view(len(fact_pool_emb), -1)
        ar_pred = self.fc_ar2(nn.ReLU()(self.fc_ar1(ar_emb)))

        # GAT over article-conditioned label embeddings
        ar_adj = self.get_mask_adj(gt_ar)
        af_emb_tilde = self.GAT_ar(af_emb, ar_adj)
        af_pool_emb_tilde = torch.max(af_emb_tilde, dim=1)[0].unsqueeze(1)
        pt_emb = torch.cat([fact_pool_emb, af_pool_emb_tilde, cf_pool_emb], dim=1).view(
            len(fact_pool_emb), -1
        )

        pt_pred = self.fc_pt2(nn.ReLU()(self.fc_pt1(pt_emb)))

        return {
            "article": ar_pred,
            "charge": ch_pred,
            "penalty": pt_pred,
            "cl_emb": {
                "af_emb": af_emb,
                "a_emb": a_emb,
                "cf_emb": cf_emb,
                "c_emb": c_emb,
            },
        }


def _make_details(vocab_size, n_labels, max_len=8):
    """Mirrors the shape/keys of the real `utils/loader.py:load_details` output (a
    tokenizer call over the article/charge label description texts) without depending on
    a specific pretrained tokenizer's vocab -- ELELJP_Num's own `vocab_size` argument
    (`self.tokenizer.vocab_size` in the real Trainer) already sizes the det-transformer
    embedding table, so the detail token ids just need to be valid indices into that
    same vocab."""
    input_ids = torch.randint(1, vocab_size, (n_labels, max_len))
    return {
        "input_ids": input_ids,
        "token_type_ids": torch.zeros(n_labels, max_len, dtype=torch.long),
        "attention_mask": torch.ones(n_labels, max_len, dtype=torch.long),
    }


def build_mljp():
    vocab_size = 200
    n_article = 6
    n_charge = 5
    n_penalty = 4

    config = ElectraConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        embedding_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    bert = ElectraModel(config)

    maps = {
        "a2i": {str(i): i for i in range(n_article)},
        "c2i": {str(i): i for i in range(n_charge)},
        "pt_cls_len": n_penalty,
    }
    details = {
        "a_details": _make_details(vocab_size, n_article),
        "c_details": _make_details(vocab_size, n_charge),
    }

    model = ELELJP_Num(bert=bert, vocab_size=vocab_size, hid_dim=32, maps=maps, details=details)
    model.eval()
    return model


def example_input_mljp():
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    vocab_size = 200
    n_article = 6

    fact = {
        "input_ids": torch.randint(1, vocab_size, (batch, seq_len)),
        "token_type_ids": torch.zeros(batch, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
    }
    mantissa = torch.zeros(batch, seq_len)
    mantissa[:, 3] = 1.0  # mark one numeric token per sample, like the real digit mask

    article = torch.randint(0, 2, (batch, n_article)).float()

    data = {
        "fact": fact,
        "mantissa": mantissa,
        "article": article,
    }
    return (data,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ML-LJP", "build_mljp", "example_input_mljp", 2023, "vendored"),
]

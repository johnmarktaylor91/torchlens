# SOURCE: vendored from GanjinZero/ICD-MSMN @ master
# https://raw.githubusercontent.com/GanjinZero/ICD-MSMN/master/model/word_encoder.py
# https://raw.githubusercontent.com/GanjinZero/ICD-MSMN/master/model/combiner.py
# https://raw.githubusercontent.com/GanjinZero/ICD-MSMN/master/model/text_encoder.py
# https://raw.githubusercontent.com/GanjinZero/ICD-MSMN/master/model/mlp.py
# https://raw.githubusercontent.com/GanjinZero/ICD-MSMN/master/model/label_encoder.py
# https://raw.githubusercontent.com/GanjinZero/ICD-MSMN/master/model/decoder.py
# https://raw.githubusercontent.com/GanjinZero/ICD-MSMN/master/model/icd_model.py
#
# "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic ICD Coding"
# (Yuan, Tan, Huang; ACL 2022). IcdModel = word embedding -> LSTM combiner -> multi-head
# label-attention (LAATv2) decoder that queries the text hidden states against ICD-code
# synonym embeddings ("label_feats" -- produced by re-running the same text encoder over the
# code-description/synonym text and pooling). Classes are copied verbatim from the real repo
# with only these minimal, non-architectural trims:
#   - Combiner: only Naive_Combiner + LSTM_Combiner are kept (the default `--combiner lstm`
#     config path used by every README recipe); Reformer_Combiner is dropped because it
#     requires the external `reformer_pytorch` + `axial_positional_embedding` packages, which
#     are not part of the model this repo ships as its trained checkpoints (mimic3/mimic3-50
#     README commands all pass `--combiner lstm`).
#   - Decoder._code_emb_init (label-embedding warm-start from a UMLS code-embedding pickle)
#     is dropped: it is a training-time initialization convenience gated behind
#     `code_embedding_path` truthiness (`if not self.code_embedding_path: return`) that reads
#     an external pickle file; the decoder module (LAAT/MultiLabelMultiHeadLAAT/V2) and its
#     forward pass are architecturally untouched, we simply never pass a
#     code_embedding_path so the guard short-circuits.
#   - IcdModel.configure_optimizers (training-loop optimizer/scheduler construction, not part
#     of the forward architecture) is dropped as out of scope for a traced nn.Module.
#   - Word_Encoder's pretrained-embedding branch (`if self.word_embedding_path:`) calls the
#     repo's `data_util.load_embeddings` helper (reads a word2vec file from disk); since our
#     build always passes `word_embedding_path=None` this branch is dead code, so
#     `load_embeddings` is imported lazily inside the branch instead of vendoring
#     `data_util.py` wholesale.
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from opt_einsum import contract

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from model/word_encoder.py ---
class Word_Encoder(nn.Module):
    def __init__(self, word_config={}):
        super(Word_Encoder, self).__init__()
        self.word_config = word_config
        self.word_count = self.word_config["count"]
        self.word_dim = self.word_config["dim"]
        self.word_padding_idx = self.word_config["padding_idx"]
        self.word_dropout_prob = self.word_config["dropout"]
        self.word_embedding_path = self.word_config["word_embedding_path"]

        self.word_embedding = nn.Embedding(
            self.word_count, self.word_dim, padding_idx=self.word_padding_idx
        )
        if self.word_embedding_path:
            from data_util import load_embeddings  # noqa: PLC0415 (dead branch in this build; see header)

            W = torch.Tensor(load_embeddings(self.word_embedding_path))
            assert self.word_count == W.shape[0]
            assert self.word_dim == W.shape[1]
            self.word_embedding.weight.data = W.clone()
            if self.word_config["frz"]:
                self.word_embedding.weight.requires_grad = False
        self.word_dropout = nn.Dropout(self.word_dropout_prob)

    def forward(self, input_word):
        input_word_embed = self.word_embedding(input_word)
        input_word_embed = self.word_dropout(input_word_embed)
        return input_word_embed


# --- vendored from model/combiner.py (Reformer_Combiner dropped; see header) ---
class Combiner(nn.Module):
    def __init__(self, combine_config={}):
        super(Combiner, self).__init__()
        self.combine_config = combine_config
        self.input_dim = self.combine_config["input_dim"]
        self.output_dim = self.combine_config["dim"]

    def lstm_forward(self, x, lengths, lstm):
        np_lengths = lengths.cpu().numpy()
        np_lengths[np_lengths == 0] = 1
        x_pack = pack_padded_sequence(x, np_lengths, batch_first=True, enforce_sorted=False)
        h_pack, _ = lstm(x_pack)
        h, _ = pad_packed_sequence(h_pack, batch_first=True)
        return h

    def combine_forward(self, embeds, word_mask):
        raise NotImplementedError

    def forward(self, word_hidden, word_mask):
        embeds = self.combine_forward(word_hidden, word_mask)
        if hasattr(self, "reduce_linear"):
            embeds = self.reduce_linear(embeds)
        return embeds


class Naive_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(Naive_Combiner, self).__init__(combine_config)
        if self.input_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.input_dim, self.output_dim)

    def combine_forward(self, embeds, word_mask):
        return embeds


class LSTM_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(LSTM_Combiner, self).__init__(combine_config)
        self.rnn_dim = self.combine_config["rnn_dim"]
        self.num_layers = self.combine_config["num_layers"]
        self.combine_lstm_dropout = self.combine_config["lstm_dropout"]
        self.combine_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_dim // 2,
            self.num_layers,
            bidirectional=True,
            dropout=self.combine_lstm_dropout,
            batch_first=True,
        )
        if self.rnn_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.rnn_dim, self.output_dim)

    def combine_forward(self, embeds, word_mask):
        word_lengths = torch.sum(word_mask, dim=1)
        h = self.lstm_forward(embeds, word_lengths, self.combine_lstm)
        return h


def create_combiner(combine_config):
    if combine_config["model"] == "naive":
        combiner = Naive_Combiner(combine_config)
    if combine_config["model"] == "lstm":
        combiner = LSTM_Combiner(combine_config)
    return combiner


# --- vendored from model/text_encoder.py ---
class TextEncoder(nn.Module):
    def __init__(self, word_config={}, combine_config={}):
        super(TextEncoder, self).__init__()
        self.word_config = word_config
        self.combine_config = combine_config

        self.input_dim = 0
        self.word_encoder = Word_Encoder(self.word_config)
        self.input_dim += self.word_config["dim"]
        self.combine_config["input_dim"] = self.input_dim
        self.combiner = create_combiner(self.combine_config)

    def forward(self, input_word, word_mask):
        word_hidden = self.word_encoder(input_word)
        hidden = self.combiner(word_hidden, word_mask)
        return hidden


# --- vendored from model/mlp.py ---
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.num_layers - 1))
        if act == "relu":
            self.act_fn = F.relu
        elif act == "gelu":
            self.act_fn = F.gelu

    def forward(self, x):
        if not hasattr(self, "act_fn"):
            self.act_fn = F.relu
        for i, layer in enumerate(self.layers):
            x = self.act_fn(layer(x)) if i < self.num_layers - 1 else layer(x)
            if hasattr(self, "dropouts") and i < self.num_layers - 1:
                x = self.dropouts[i](x)
        return x


# --- vendored from model/label_encoder.py ---
class LabelEncoder(nn.Module):
    def __init__(self, label_config={}):
        super(LabelEncoder, self).__init__()
        self.label_config = label_config
        self.pooling = self.label_config["pooling"]
        if label_config["num_layers"] > 0:
            self.mlp = MLP(
                input_dim=label_config["input_dim"],
                hidden_dim=label_config["input_dim"],
                output_dim=label_config["input_dim"],
                num_layers=label_config["num_layers"],
                dropout=label_config["dropout"],
            )

    def forward(self, input_feat, word_mask=None):
        # input_feat: Label * Seq * Hidden
        if not hasattr(self, "pooling") or self.pooling == "max":
            input_feat = input_feat.max(dim=1)[0]  # Label * Hidden
        elif self.pooling == "mean":
            input_feat = input_feat.mean(dim=1)
        elif self.pooling == "last":
            assert word_mask is not None
            word_length = word_mask.sum(-1)
            word_length[word_length == 0] = 1
            idx = (word_length - 1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, input_feat.shape[-1])
            input_feat = torch.gather(input_feat, 1, idx).squeeze(1)

        if hasattr(self, "mlp"):
            input_feat = self.mlp(input_feat)
        return input_feat


# --- vendored from model/decoder.py (_code_emb_init warm-start dropped; see header) ---
class Decoder(nn.Module):
    def __init__(self, decoder_config={}):
        super(Decoder, self).__init__()
        self.decoder_config = decoder_config
        self.input_dim = self.decoder_config["input_dim"]
        self.attention_dim = self.decoder_config["attention_dim"]
        self.code_embedding_path = self.decoder_config["code_embedding_path"]
        self.ind2c = self.decoder_config["ind2c"]
        self.ind2mc = self.decoder_config["ind2mc"]

        self.final = nn.Linear(self.input_dim, len(self.ind2c))
        self.ignore_mask = False

        self.est_cls = self.decoder_config["est_cls"]
        if self.est_cls > 0:
            self.w_linear = MLP(
                self.attention_dim, self.attention_dim, self.attention_dim, self.est_cls
            )
            self.b_linear = MLP(self.attention_dim, self.attention_dim, 1, self.est_cls)
        elif self.est_cls == -1:
            self.w_linear = nn.Identity()
            self.b_linear = self.zero_first_dim

    def zero_first_dim(self, x):
        return torch.zeros_like(x)[:, 0]

    def forward(self, h, word_mask, label_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat)

        if hasattr(self, "w_linear"):
            w = self.w_linear(label_feat)  # label * hidden
            b = self.b_linear(label_feat)  # label * 1
            logits = self.get_logits(m, w, b)
        else:
            logits = self.get_logits(m)
        return logits

    def get_logits(self, m, w=None, b=None):
        if w is None:
            logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            logits = contract("blh,lh->bl", m, w) + b.squeeze(-1)
        return logits

    def get_label_queried_features(self, h, word_mask, label_feat):
        raise NotImplementedError


class LAAT(Decoder):
    def __init__(self, decoder_config={}):
        super(LAAT, self).__init__(decoder_config)
        self.W = nn.Linear(self.input_dim, self.attention_dim)
        self.U = nn.Linear(self.attention_dim, len(self.ind2mc))

        self.xavier = self.decoder_config["xavier"]

        if self.xavier:
            nn.init.xavier_uniform_(self.W.weight)

    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None and not self.ignore_mask:
            l = word_mask.shape[-1]  # noqa: E741 (verbatim var name from real repo)
            h = h[:, 0:l]
        z = torch.tanh(self.W(h))
        if label_feat is None:
            label_feat = self.U.weight
        score = label_feat.matmul(z.transpose(1, 2))
        if word_mask is not None and not self.ignore_mask:
            word_mask = word_mask.bool()
            score = score.masked_fill(
                mask=~word_mask[:, 0 : score.shape[-1]].unsqueeze(1).expand_as(score),
                value=float("-1e6"),
            )
        alpha = F.softmax(score, dim=2)
        m = alpha.matmul(h)  # Batch * Label * Hidden
        return m


class MultiLabelMultiHeadLAAT(LAAT):
    def __init__(self, decoder_config={}):
        super(MultiLabelMultiHeadLAAT, self).__init__(decoder_config)
        self.attention_head = self.decoder_config["attention_head"]
        self.rep_dropout = nn.Dropout(decoder_config["rep_dropout"])

        self.head_pooling = decoder_config["head_pooling"]
        if self.head_pooling == "concat":
            assert self.attention_dim % self.attention_head == 0
            self.reduce = nn.Linear(self.attention_dim, self.attention_dim // self.attention_head)
        if decoder_config.get("att_dropout") > 0.0:
            self.att_dropout_rate = decoder_config["att_dropout"]
            self.att_dropout = nn.Dropout(self.att_dropout_rate)

    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None:
            if not hasattr(self, "ignore_mask") or not self.ignore_mask:
                l = word_mask.shape[-1]  # noqa: E741 (verbatim var name from real repo)
                h = h[:, 0:l]
        z = torch.tanh(self.W(h))  # batch_size * seq_length * att_dim
        batch_size, seq_length, att_dim = z.size()
        if label_feat is None:
            label_feat = self.U.weight
        label_count = label_feat.size(0) // self.attention_head
        u_reshape = label_feat.reshape(label_count, self.attention_head, att_dim)
        score = contract("abd,ecd->aebc", z, u_reshape)
        # batch_size, label_count, seq_length, att_head

        if word_mask is not None:
            if not hasattr(self, "ignore_mask") or not self.ignore_mask:
                word_mask = word_mask.bool()
                score = score.masked_fill(
                    mask=~word_mask[:, 0 : score.shape[-2]]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand_as(score),
                    value=float("-1e6"),
                )
        alpha = F.softmax(
            score, dim=2
        )  # softmax on seq_length # batch_size, label_count, seq_length, att_head
        if hasattr(self, "att_dropout"):
            alpha = self.att_dropout(alpha)
            if self.training:
                alpha_sum = torch.clamp(alpha.sum(dim=2, keepdim=True), 1e-5)
                alpha = alpha / alpha_sum
        m = contract("abd,aebc->aedc", h, alpha)
        # h: batch * seq * hidden
        # a: batch * label * seq * head
        if not hasattr(self, "head_pooling") or self.head_pooling == "max":
            m = m.max(-1)[0]
        elif self.head_pooling == "concat":
            m = self.reduce(m.permute(0, 1, 3, 2))  # batch * label * hidden // head * head
            m = m.reshape(batch_size, -1, att_dim)

        m = self.rep_dropout(m)
        return m

    def transform_label_feats(self, label_feat):
        if not hasattr(self, "head_pooling") or self.head_pooling == "max":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = label_feat.reshape(label_count, self.attention_head, -1).max(1)[0]
        elif self.head_pooling == "concat":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = self.reduce(label_feat)  # (label * head) * (hidden // head)
            label_feat = label_feat.reshape(
                label_count, self.attention_head, -1
            )  # label * head * hidden
            label_feat = label_feat.reshape(label_count, -1)
        return label_feat

    def forward(self, h, word_mask, label_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat)

        if hasattr(self, "w_linear"):
            label_feat = self.transform_label_feats(label_feat)
            w = self.w_linear(label_feat)  # label * hidden
            b = self.b_linear(label_feat)  # label * 1
            logits = self.get_logits(m, w, b)
        else:
            logits = self.get_logits(m)
        return logits


ACT2FN = {"tanh": torch.tanh, "relu": torch.relu}


class MultiLabelMultiHeadLAATV2(MultiLabelMultiHeadLAAT):
    def __init__(self, decoder_config={}):
        super(MultiLabelMultiHeadLAATV2, self).__init__(decoder_config)
        self.act_fn_name = decoder_config["act_fn_name"]
        self.act_fn = ACT2FN[self.act_fn_name]
        self.u_reduce = nn.Linear(self.attention_dim, self.attention_dim // self.attention_head)

    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None:
            if not hasattr(self, "ignore_mask") or not self.ignore_mask:
                l = word_mask.shape[-1]  # noqa: E741 (verbatim var name from real repo)
                h = h[:, 0:l]
        z = self.act_fn(self.W(h))  # batch_size * seq_length * att_dim
        batch_size, seq_length, att_dim = z.size()
        z_reshape = z.reshape(
            batch_size, seq_length, self.attention_head, att_dim // self.attention_head
        )
        # batch_size, seq_length, att_head, sub_dim
        if label_feat is None:
            label_feat = self.U.weight
        label_count = label_feat.size(0) // self.attention_head
        u_reshape = self.u_reduce(label_feat.reshape(label_count, self.attention_head, att_dim))
        score = contract("abcd,ecd->aebc", z_reshape, u_reshape)
        if word_mask is not None:
            if not hasattr(self, "ignore_mask") or not self.ignore_mask:
                word_mask = word_mask.bool()
                score = score.masked_fill(
                    mask=~word_mask[:, 0 : score.shape[-2]]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand_as(score),
                    value=float("-1e6"),
                )
        alpha = F.softmax(
            score, dim=2
        )  # softmax on seq_length # batch_size, label_count, seq_length, att_head
        if hasattr(self, "att_dropout"):
            alpha = self.att_dropout(alpha)
            if self.training:
                alpha *= 1 - self.att_dropout_rate

        m = contract("abd,aebc->aedc", h, alpha)

        if not hasattr(self, "head_pooling") or self.head_pooling == "max":
            m = m.max(-1)[0]
        elif self.head_pooling == "concat":
            m = self.reduce(m.permute(0, 1, 3, 2))  # batch * label * hidden // head * head
            m = m.reshape(batch_size, -1, att_dim)

        m = self.rep_dropout(m)
        return m


def create_decoder(decoder_config):
    if decoder_config["model"] == "LAAT":
        decoder = LAAT(decoder_config)
    if decoder_config["model"] == "MultiLabelMultiHeadLAAT":
        decoder = MultiLabelMultiHeadLAAT(decoder_config)
    if decoder_config["model"] == "MultiLabelMultiHeadLAATV2":
        decoder = MultiLabelMultiHeadLAATV2(decoder_config)
    return decoder


# --- vendored from model/icd_model.py (configure_optimizers dropped; see header) ---
class IcdModel(nn.Module):
    def __init__(
        self,
        word_config={},
        combine_config={},
        decoder_config={},
        label_config={},
        loss_config={},
        args=None,
    ):
        super().__init__()
        self.encoder = TextEncoder(word_config, combine_config)
        self.decoder = create_decoder(decoder_config)
        self.label_encoder = LabelEncoder(label_config)
        self.loss_config = loss_config
        self.args = args

    def calculate_text_hidden(self, input_word, word_mask):
        hidden = self.encoder(input_word, word_mask)
        return hidden

    def calculate_label_hidden(self):
        label_hidden = self.calculate_text_hidden(self.c_input_word, self.c_word_mask)
        self.label_feats = self.label_encoder(label_hidden, self.c_word_mask)

    def forward(self, input_word, word_mask):
        hidden = self.calculate_text_hidden(input_word, word_mask)

        label_hidden = self.calculate_text_hidden(self.c_input_word, self.c_word_mask)
        label_feats = self.label_encoder(label_hidden, self.c_word_mask)
        c_logits = self.decoder(hidden, word_mask, label_feats)
        return c_logits


def build_msmn():
    torch.manual_seed(0)
    vocab_size = 64
    word_dim = 32
    rnn_dim = 32
    attention_dim = 32
    attention_head = 4
    n_codes = 10

    word_config = {
        "count": vocab_size,
        "dim": word_dim,
        "padding_idx": 0,
        "dropout": 0.0,
        "word_embedding_path": None,
        "frz": False,
    }
    combine_config = {
        "model": "lstm",
        "dim": rnn_dim,
        "rnn_dim": rnn_dim,
        "num_layers": 1,
        "lstm_dropout": 0.0,
    }
    ind2c = {i: str(i) for i in range(n_codes)}
    ind2mc = {i: str(i) for i in range(n_codes * attention_head)}
    decoder_config = {
        "model": "MultiLabelMultiHeadLAATV2",
        "input_dim": rnn_dim,
        "attention_dim": attention_dim,
        "code_embedding_path": None,
        "ind2c": ind2c,
        "ind2mc": ind2mc,
        "est_cls": 0,
        "xavier": True,
        "attention_head": attention_head,
        "rep_dropout": 0.0,
        "head_pooling": "max",
        "att_dropout": 0.0,
        "act_fn_name": "tanh",
    }
    label_config = {
        "pooling": "max",
        "num_layers": 0,
        "input_dim": rnn_dim,
        "dropout": 0.0,
    }
    loss_config = {"name": "ce"}

    model = IcdModel(
        word_config, combine_config, decoder_config, label_config, loss_config, args=None
    )

    # code-synonym/description text (label side); normally produced from a tokenized
    # ICD-code-description dataset -- here a small random stand-in with the same shape
    # contract (n_codes * attention_head rows, one "synonym description" per multi-code row).
    torch.manual_seed(1)
    c_seq_len = 6
    n_mc = n_codes * attention_head
    model.c_input_word = torch.randint(1, vocab_size, (n_mc, c_seq_len))
    model.c_word_mask = torch.ones(n_mc, c_seq_len)
    model.calculate_label_hidden()
    return model


def example_input_msmn():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 12
    vocab_size = 64
    input_word = torch.randint(1, vocab_size, (batch_size, seq_len))
    word_mask = torch.ones(batch_size, seq_len)
    return (input_word, word_mask)


MENAGERIE_ENTRIES = [
    ("MSMN", "build_msmn", "example_input_msmn", 2022, "vendored"),
]

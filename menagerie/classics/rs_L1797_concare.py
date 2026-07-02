# SOURCE: vendored from choczhang/ConCare @ master (official AAAI-2020 repo; SingleAttention
# class in concare-notebook.ipynb cell 9 is byte-identical to this file) + Neronjust2017/concare
# @ 00db32fbca5990c770d0675f5e3001305bad0ea1 (concare.py) for the remaining model classes.
"""ConCare: Personalized Clinical Feature Embedding via Capturing the Healthcare Context
(Ma et al., AAAI 2020). Per-feature GRU embeds each of the ``input_dim`` irregular clinical
time series independently; a time-aware ``SingleAttention`` collapses each feature's GRU
sequence to a single feature vector; those feature vectors (plus a demographics-derived
"virtual" token) are passed through a Transformer-style cross-feature ``MultiHeadedAttention``
+ position-wise FFN stack (with a DeCov regularization loss returned alongside); a final
QKV attention pools the resulting per-feature contexts into a single hidden vector, which a
linear+sigmoid head maps to a mortality-risk score.

The official ``choczhang/ConCare`` repo's public notebook (``concare-notebook.ipynb``)
defines ``SingleAttention`` (cell 9) but the notebook as released never defines the
downstream ``ConCare``/``FinalAttentionQKV``/``MultiHeadedAttention`` classes it later
calls (``model = ...`` is missing) -- the release is incomplete for the modeling half.
``Neronjust2017/concare`` hosts the complete ``concare.py`` with the identical
``SingleAttention`` implementation (byte-for-byte the same as the official notebook cell)
plus the rest of the architecture the official repo's public code omits. Vendored verbatim
from that file's model classes; only the training script (data readers, optimizer loop,
``model = ConCare(...).to(device)`` instantiation call, CLI-driven ``device`` global) is
dropped in favor of ``build_concare()``/``example_input_concare()`` below, and ``device`` is
threaded through as an explicit module-level CPU constant instead of the original script's
global. No architectural change.
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

MENAGERIE_ZOO = "vendored-pytorch"

device = torch.device("cpu")


class SingleAttention(nn.Module):
    def __init__(
        self,
        attention_input_dim,
        attention_hidden_dim,
        attention_type="add",
        demographic_dim=12,
        time_aware=False,
        use_demographic=False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.use_demographic = use_demographic
        self.demographic_dim = demographic_dim
        self.time_aware = time_aware

        self.attn = None

        if attention_type == "add":
            if self.time_aware == True:  # noqa: E712 (kept verbatim from real repo)
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wd = nn.Parameter(torch.randn(demographic_dim, attention_hidden_dim))
            self.bh = nn.Parameter(
                torch.zeros(
                    attention_hidden_dim,
                )
            )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "mul":
            self.Wa = nn.Parameter(torch.randn(attention_input_dim, attention_input_dim))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "concat":
            if self.time_aware == True:  # noqa: E712 (kept verbatim from real repo)
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim + 1, attention_hidden_dim)
                )
            else:
                self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "new":
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))

            self.rate = nn.Parameter(torch.ones(1))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError("Wrong attention type.")

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, demo=None):
        batch_size, time_step, input_dim = input.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(device)
        )  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1)  # b t 1

        if self.attention_type == "add":  # B*T*I  @ H*I
            q = torch.matmul(input[:, -1, :], self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:  # noqa: E712 (kept verbatim from real repo)
                k = torch.matmul(input, self.Wx)  # b t h
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(input, self.Wx)  # b t h
            if self.use_demographic == True:  # noqa: E712 (kept verbatim from real repo)
                d = torch.matmul(demo, self.Wd)  # B*H
                d = torch.reshape(d, (batch_size, 1, self.attention_hidden_dim))  # b 1 h
            h = q + k + self.bh  # b t h
            if self.time_aware == True:  # noqa: E712 (kept verbatim from real repo)
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == "mul":
            e = torch.matmul(input[:, -1, :], self.Wa)  # b i
            e = torch.matmul(e.unsqueeze(1), input.permute(0, 2, 1)).squeeze() + self.ba  # b t
        elif self.attention_type == "concat":
            q = input[:, -1, :].unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = input
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware == True:  # noqa: E712 (kept verbatim from real repo)
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "new":
            q = torch.matmul(input[:, -1, :], self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(input, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze()  # b t
            denominator = self.rate * torch.log(
                2.71828 + (1 - self.sigmoid(dot_product)) * (b_time_decays.squeeze())
            )
            e = self.tanh(dot_product / denominator)  # b * t# b * t

        a = self.softmax(e)  # B*T
        self.attn = a
        v = torch.matmul(a.unsqueeze(1), input).squeeze()  # B*I

        return v, a


class FinalAttentionQKV(nn.Module):
    def __init__(
        self, attention_input_dim, attention_hidden_dim, attention_type="add", dropout=None
    ):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.b_out = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size, time_step, input_dim = (
            input.size()
        )  # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:, -1, :])  # b h
        input_k = self.W_k(input)  # b t h
        input_v = self.W_v(input)  # b t h

        if self.attention_type == "add":  # B*T*I  @ H*I
            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "mul":
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1))  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):  # new added
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):  # new added / not use anymore
    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # b h t d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # b h t t
    if mask is not None:  # 1 1 t t
        scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
    p_attn = F.softmax(scores, dim=-1)  # b h t t
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        feature_dim = query.size(-1)  # i+1

        # d_model => h * d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # b num_head d_input d_k  # noqa: E741 (kept verbatim from real repo)

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # b num_head d_input d_v (d_k)

        x = (
            x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        )  # batch_size * d_input * hidden_dim

        # DeCov
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # I+1 H B
        Covs = cov(DeCov_contexts[0, :, :])
        DeCov_loss = 0.5 * (torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2)
        for i in range(feature_dim - 1 + 1):
            Covs = cov(DeCov_contexts[i + 1, :, :])
            DeCov_loss += 0.5 * (torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2)

        return self.final_linear(x), DeCov_loss


def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class ConCare(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, d_model, MHD_num_head, d_ff, output_dim, keep_prob=0.5
    ):
        super(ConCare, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # d_model
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        # layers
        self.PositionalEncoding = PositionalEncoding(self.d_model, dropout=0, max_len=400)

        self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first=True), self.input_dim)
        self.LastStepAttentions = clones(
            SingleAttention(
                self.hidden_dim,
                8,
                attention_type="concat",
                demographic_dim=12,
                time_aware=True,
                use_demographic=False,
            ),
            self.input_dim,
        )

        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim, self.hidden_dim, attention_type="mul", dropout=1 - self.keep_prob
        )

        self.MultiHeadedAttention = MultiHeadedAttention(
            self.MHD_num_head, self.d_model, dropout=1 - self.keep_prob
        )
        self.SublayerConnection = SublayerConnection(self.d_model, dropout=1 - self.keep_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)

        self.demo_proj_main = nn.Linear(12, self.hidden_dim)
        self.demo_proj = nn.Linear(12, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, demo_input):
        # input shape [batch_size, timestep, feature_dim]
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)  # b hidden_dim

        batch_size = input.size(0)
        feature_dim = input.size(2)
        assert feature_dim == self.input_dim  # input Tensor : 256 * 48 * 76
        assert self.d_model % self.MHD_num_head == 0

        GRU_embeded_input = self.GRUs[0](
            input[:, :, 0].unsqueeze(-1),
            Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device),
        )[0]  # b t h
        Attention_embeded_input, self.gru_atten = self.LastStepAttentions[0](GRU_embeded_input)
        Attention_embeded_input = Attention_embeded_input.unsqueeze(1)  # b 1 h
        self.gru_atten = self.gru_atten.unsqueeze(1)  # b 1 h
        for i in range(feature_dim - 1):
            embeded_input = self.GRUs[i + 1](
                input[:, :, i + 1].unsqueeze(-1),
                Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device),
            )[0]  # b 1 h
            embeded_input, atten = self.LastStepAttentions[i + 1](embeded_input)
            embeded_input = embeded_input.unsqueeze(1)  # b 1 h
            atten = atten.unsqueeze(1)  # b 1 h

            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, embeded_input), 1
            )  # b i h
            self.gru_atten = torch.cat((self.gru_atten, atten), 1)  # b i h

        Attention_embeded_input = torch.cat((Attention_embeded_input, demo_main), 1)  # b i+1 h
        posi_input = self.dropout(Attention_embeded_input)  # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(
            posi_input,
            lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None),
        )  # # batch_size * d_input * hidden_dim

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(
            contexts, lambda x: self.PositionwiseFeedForward(contexts)
        )[0]  # # batch_size * d_input * hidden_dim

        weighted_contexts = self.FinalAttentionQKV(contexts)[0]
        output = self.output(weighted_contexts)  # b 1
        output = self.sigmoid(output)

        return output, DeCov_loss


def build_concare():
    # menagerie-sized: real repo uses input_dim=76, hidden_dim=32, d_model=32, MHD_num_head=4, d_ff=256.
    # MultiHeadedAttention's DeCov loop (real code, line 469-471 of the source) indexes the
    # (input_dim+1)-length DeCov_contexts using range(feature_dim) where feature_dim=d_model,
    # so this vendored copy must keep d_model <= input_dim+1 exactly as the real hyperparameters do.
    return ConCare(input_dim=8, hidden_dim=8, d_model=8, MHD_num_head=2, d_ff=16, output_dim=1)


def example_input_concare():
    batch_size, time_step, input_dim = 2, 5, 8
    x = torch.randn(batch_size, time_step, input_dim)
    demo = torch.randn(batch_size, 12)
    return (x, demo)


MENAGERIE_ENTRIES = [
    ("ConCare", "build_concare", "example_input_concare", 2020, "SOURCE_AVAILABLE"),
]

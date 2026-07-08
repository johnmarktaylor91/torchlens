# SOURCE: vendored from yuelinan/NeurJudge @ main
# https://raw.githubusercontent.com/yuelinan/NeurJudge/main/neurjudge/model.py
# https://raw.githubusercontent.com/yuelinan/NeurJudge/main/neurjudge/utils.py
#
# Yue, Liu, Duan, Wu, Wang, Yao, Wu, Wang 2021 (SIGIR 2021) "NeurJudge: A Circumstance-aware
# Neural Framework for Legal Judgment Prediction" -- jointly predicts a case's charge,
# applicable law article, and prison term from the case-fact text, using a "circumstance-
# aware" pipeline: (1) a GRU document encoder over the fact text and over label
# descriptions (charge/article text); (2) a "Graph Decomposition Operation" (GDO) that
# refines label (charge/article) embeddings against a label-similarity graph via iterative
# vector-rejection (removing the component shared with confusable sibling labels); (3)
# label-to-fact "Code-Wise Attention" that pools the fact encoding against the refined
# label embeddings for both the raw and GDO-refined label sets; (4) "Fact Separation" via
# `Mask_Attention` + vector rejection, which splits the fact representation into a
# "circumstance relevant to the predicted charge/article" (similar) vs. "circumstance still
# relevant to the next stage" (dissimilar) component, cascading charge -> article -> term
# prediction so each downstream head only sees the fact residual not already explained by
# the upstream verdict.
#
# `Mask_Attention`, `Code_Wise_Attention`, and `NeurJudge` (the `nn.GRU`-based "Version of
# NeurJudge with nn.GRU" variant, the non-commented-out class in the real `model.py`) are
# copied verbatim -- only the `__init__` file-loading of the real Chinese legal-taxonomy
# JSONs (`charge_tong.json`/`art_tong.json`/`id2charge.json`/`id2article.json`, from
# `neurjudge/data/`) is replaced with tiny synthetic label-similarity graphs of the same
# shape, and the real `Data_Process.process_law` (`neurjudge/utils.py`, itself pure JSON/
# string data lookup and tensor packing -- no architecture) is replaced with a tiny
# `_TinyProcess` stand-in built from the same synthetic label descriptions. No computation
# in `NeurJudge.forward`, `graph_decomposition_operation`, or `fact_separation` is changed.

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (kept: real file imports it)


class Mask_Attention(nn.Module):
    def __init__(self):
        super(Mask_Attention, self).__init__()

    def forward(self, query, context):
        attention = torch.bmm(context, query.transpose(1, 2))
        mask = attention.new(attention.size()).zero_()
        mask[:, :, :] = -float("inf")
        attention_mask = torch.where(attention == 0, mask, attention)
        attention_mask = torch.nn.functional.softmax(attention_mask, dim=-1)
        mask_zero = attention.new(attention.size()).zero_()
        final_attention = torch.where(attention_mask != attention_mask, mask_zero, attention_mask)
        context_vec = torch.bmm(final_attention, query)
        return context_vec


class Code_Wise_Attention(nn.Module):
    def __init__(self):
        super(Code_Wise_Attention, self).__init__()

    def forward(self, query, context):
        S = torch.bmm(context, query.transpose(1, 2))
        attention = torch.nn.functional.softmax(torch.max(S, 2)[0], dim=-1)
        context_vec = torch.bmm(attention.unsqueeze(1), context)
        return context_vec


class _TinyProcess:
    """Stand-in for the real `Data_Process` (neurjudge/utils.py): pure JSON/string data
    lookup + tensor packing, not architecture. `process_law` mirrors the real method's
    contract (label name -> parsed word-id tensor + length) against a tiny synthetic
    word vocabulary/label-description table instead of the real Chinese legal taxonomy."""

    def __init__(self, word2id, charge2detail, law):
        self.word2id = word2id
        self.charge2detail = charge2detail
        self.law = law

    def _seq2tensor(self, sents, max_len):
        sent_len_max = min(max(len(s) for s in sents), max_len)
        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()
        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max:
                    break
                sent_tensor[s_id][w_id] = self.word2id.get(word, self.word2id["UNK"])
        return sent_tensor, sent_len

    def process_law(self, label_names, types="charge"):
        if types == "charge":
            labels = [self.charge2detail[name] for name in label_names]
            return self._seq2tensor(labels, max_len=10)
        labels = [self.law[name] for name in label_names]
        return self._seq2tensor(labels, max_len=10)


# Version of NeurJudge with nn.GRU
class NeurJudge(nn.Module):
    def __init__(
        self, embedding, charge_tong, art_tong, id2charge, id2article, data_size=16, hidden_dim=12
    ):
        super(NeurJudge, self).__init__()
        self.charge_tong = charge_tong
        self.art_tong = art_tong
        self.id2charge = id2charge
        self.data_size = data_size
        self.hidden_dim = hidden_dim

        n_vocab = embedding.size(0)
        self.embs = nn.Embedding(n_vocab, data_size)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad = False

        self.encoder = nn.GRU(self.data_size, self.hidden_dim, batch_first=True, bidirectional=True)
        self.code_wise = Code_Wise_Attention()
        self.mask_attention = Mask_Attention()

        self.encoder_term = nn.GRU(
            self.hidden_dim * 6, self.hidden_dim * 3, batch_first=True, bidirectional=True
        )
        self.encoder_article = nn.GRU(
            self.hidden_dim * 8, self.hidden_dim * 4, batch_first=True, bidirectional=True
        )

        self.id2article = id2article
        self.mask_attention_article = Mask_Attention()

        num_charge = len(charge_tong)
        num_article = len(art_tong)
        self.encoder_charge = nn.GRU(
            self.data_size, self.hidden_dim, batch_first=True, bidirectional=True
        )
        self.charge_pred = nn.Linear(self.hidden_dim * 6, num_charge)
        self.article_pred = nn.Linear(self.hidden_dim * 8, num_article)
        self.time_pred = nn.Linear(self.hidden_dim * 6, 11)

    def graph_decomposition_operation(
        self, _label, label_sim, id2label, label2id, num_label, layers=2
    ):
        for i in range(layers):
            new_label_tong = []
            for index in range(num_label):
                Li = _label[index]
                Lj_list = []
                if len(label_sim[id2label[str(index)]]) == 0:
                    new_label_tong.append(Li.unsqueeze(0))
                else:
                    for sim_label in label_sim[id2label[str(index)]]:
                        Lj = _label[int(label2id[str(sim_label)])]
                        x1 = Li * Lj
                        x1 = torch.sum(x1, -1)
                        x2 = Lj * Lj
                        x2 = torch.sum(x2, -1)
                        x2 = x2 + 1e-10
                        xx = x1 / x2
                        Lj = xx.unsqueeze(-1) * Lj
                        Lj_list.append(Lj)
                    Lj_list = torch.stack(Lj_list, 0).squeeze(1)
                    Lj_list = torch.mean(Lj_list, 0).unsqueeze(0)
                    new_label_tong.append(Li - Lj_list)
            new_label_tong = torch.stack(new_label_tong, 0).squeeze(1)
            _label = new_label_tong
        return _label

    def fact_separation(
        self, process, verdict_names, device, embs, encoder, circumstance, mask_attention, types
    ):
        verdict, verdict_len = process.process_law(verdict_names, types)
        verdict = verdict.to(device)
        verdict_len = verdict_len.to(device)
        verdict = embs(verdict)
        verdict_hidden, _ = encoder(verdict)
        # Fact Separation
        scenario = mask_attention(verdict_hidden, circumstance)
        # vector rejection
        x3 = circumstance * scenario
        x3 = torch.sum(x3, 2)
        x4 = scenario * scenario
        x4 = torch.sum(x4, 2)
        x4 = x4 + 1e-10
        xx = x3 / x4
        # similar vectors
        similar = xx.unsqueeze(-1) * scenario
        # dissimilar vectors
        dissimilar = circumstance - similar
        return similar, dissimilar

    def forward(
        self,
        charge,
        charge_sent_len,
        article,
        article_sent_len,
        charge_tong2id,
        id2charge_tong,
        art2id,
        id2art,
        documents,
        sent_lent,
        process,
        device,
    ):
        # deal the semantics of labels (i.e., charges and articles)
        charge = self.embs(charge)
        article = self.embs(article)
        charge, _ = self.encoder_charge(charge)
        article, _ = self.encoder_charge(article)
        _charge = charge.mean(1)
        _article = article.mean(1)
        # the original charge and article features
        ori_a = charge.mean(1)
        ori_b = article.mean(1)
        # GDO for the charge graph
        new_charge = self.graph_decomposition_operation(
            _label=_charge,
            label_sim=self.charge_tong,
            id2label=id2charge_tong,
            label2id=charge_tong2id,
            num_label=len(self.charge_tong),
            layers=2,
        )

        # GDO for the article graph
        new_article = self.graph_decomposition_operation(
            _label=_article,
            label_sim=self.art_tong,
            id2label=id2art,
            label2id=art2id,
            num_label=len(self.art_tong),
            layers=2,
        )

        # deal the case fact
        doc = self.embs(documents)
        d_hidden, _ = self.encoder(doc)

        # L2F attention for charges
        new_charge_repeat = new_charge.unsqueeze(0).repeat(d_hidden.size(0), 1, 1)
        d_hidden_charge = self.code_wise(new_charge_repeat, d_hidden)
        d_hidden_charge = d_hidden_charge.repeat(1, doc.size(1), 1)

        a_repeat = ori_a.unsqueeze(0).repeat(d_hidden.size(0), 1, 1)
        d_a = self.code_wise(a_repeat, d_hidden)
        d_a = d_a.repeat(1, doc.size(1), 1)

        # the charge prediction
        fact_charge = torch.cat([d_hidden, d_hidden_charge, d_a], -1)
        fact_charge_hidden = fact_charge
        df = fact_charge_hidden.mean(1)
        charge_out = self.charge_pred(df)

        charge_pred = charge_out.cpu().argmax(dim=1).numpy()
        charge_names = [self.id2charge[str(i)] for i in charge_pred]
        # Fact Separation for verdicts
        adc_vector, sec_vector = self.fact_separation(
            process=process,
            verdict_names=charge_names,
            device=device,
            embs=self.embs,
            encoder=self.encoder,
            circumstance=d_hidden,
            mask_attention=self.mask_attention,
            types="charge",
        )

        # L2F attention for articles
        new_article_repeat = new_article.unsqueeze(0).repeat(d_hidden.size(0), 1, 1)
        d_hidden_article = self.code_wise(new_article_repeat, d_hidden)
        d_hidden_article = d_hidden_article.repeat(1, doc.size(1), 1)

        b_repeat = ori_b.unsqueeze(0).repeat(d_hidden.size(0), 1, 1)
        d_b = self.code_wise(b_repeat, d_hidden)
        d_b = d_b.repeat(1, doc.size(1), 1)
        # the article prediction
        fact_article = torch.cat([d_hidden, d_hidden_article, adc_vector, d_b], -1)
        fact_legal_article_hidden, _ = self.encoder_article(fact_article)

        fact_article_hidden = fact_legal_article_hidden.mean(1)
        article_out = self.article_pred(fact_article_hidden)

        article_pred = article_out.cpu().argmax(dim=1).numpy()
        article_names = [self.id2article[str(i)] for i in article_pred]

        # Fact Separation for sentencing
        ssc_vector, dsc_vector = self.fact_separation(
            process=process,
            verdict_names=article_names,
            device=device,
            embs=self.embs,
            encoder=self.encoder,
            circumstance=sec_vector,
            mask_attention=self.mask_attention,
            types="article",
        )

        # the term of penalty prediction
        term_message = torch.cat([d_hidden, ssc_vector, dsc_vector], -1)
        term_message, _ = self.encoder_term(term_message)

        fact_legal_time_hidden = term_message.mean(1)
        time_out = self.time_pred(fact_legal_time_hidden)

        return charge_out, article_out, time_out


class NeurJudgeTraceAdapter(nn.Module):
    """Thin positional-arg adapter: `NeurJudge.forward` (vendored verbatim above) needs
    Python dict/graph objects (`charge_tong2id`, `process`, ...) that are not tensors, so
    a single-example-input tracer cannot pass them as ordinary tensor args. This wrapper
    closes over the (non-tensor) label-graph bookkeeping built at construction time and
    exposes only the two real tensor inputs (`documents`, `sent_lent`) as its forward
    signature; it performs no architecture computation itself -- every op below is a call
    into the unmodified `NeurJudge.forward`.
    """

    def __init__(
        self,
        neurjudge,
        charge,
        charge_sent_len,
        article,
        article_sent_len,
        charge_tong2id,
        id2charge_tong,
        art2id,
        id2art,
        process,
    ):
        super().__init__()
        self.neurjudge = neurjudge
        self.register_buffer("charge", charge)
        self.register_buffer("charge_sent_len", charge_sent_len)
        self.register_buffer("article", article)
        self.register_buffer("article_sent_len", article_sent_len)
        self.charge_tong2id = charge_tong2id
        self.id2charge_tong = id2charge_tong
        self.art2id = art2id
        self.id2art = id2art
        self.process = process

    def forward(self, documents, sent_lent):
        return self.neurjudge(
            self.charge,
            self.charge_sent_len,
            self.article,
            self.article_sent_len,
            self.charge_tong2id,
            self.id2charge_tong,
            self.art2id,
            self.id2art,
            documents,
            sent_lent,
            self.process,
            documents.device,
        )


def build_neurjudge():
    torch.manual_seed(0)
    vocab_size = 64
    word2id = {"UNK": 0}
    for i in range(1, vocab_size):
        word2id[f"w{i}"] = i

    # tiny synthetic charge/article label-similarity graphs (shape mirrors the real
    # charge_tong.json / art_tong.json: label -> [confusable sibling labels])
    charge_names = [f"charge_{i}" for i in range(4)]
    article_names = [f"article_{i}" for i in range(3)]
    charge_tong = {
        c: [charge_names[(i + 1) % len(charge_names)]] for i, c in enumerate(charge_names)
    }
    art_tong = {a: [] for a in article_names}
    id2charge_tong = {str(i): c for i, c in enumerate(charge_names)}
    charge_tong2id = {c: str(i) for i, c in enumerate(charge_names)}
    id2art = {str(i): a for i, a in enumerate(article_names)}
    art2id = {a: str(i) for i, a in enumerate(article_names)}
    id2charge = {str(i): c for i, c in enumerate(charge_names)}
    id2article = {str(i): a for i, a in enumerate(article_names)}

    charge2detail = {c: [f"w{(i % (vocab_size - 1)) + 1}" for i in range(5)] for c in charge_names}
    law = {a: [f"w{(i % (vocab_size - 1)) + 1}" for i in range(5)] for a in article_names}
    process = _TinyProcess(word2id, charge2detail, law)

    embedding = torch.randn(vocab_size, 16)
    neurjudge = NeurJudge(
        embedding, charge_tong, art_tong, id2charge, id2article, data_size=16, hidden_dim=12
    )
    neurjudge.eval()

    seq_len = 5
    charge_ids, _ = process._seq2tensor([charge2detail[c] for c in charge_names], max_len=seq_len)
    charge_len = torch.tensor([seq_len] * len(charge_names))
    article_ids, _ = process._seq2tensor([law[a] for a in article_names], max_len=seq_len)
    article_len = torch.tensor([seq_len] * len(article_names))

    return NeurJudgeTraceAdapter(
        neurjudge,
        charge_ids,
        charge_len,
        article_ids,
        article_len,
        charge_tong2id,
        id2charge_tong,
        art2id,
        id2art,
        process,
    )


def example_input_neurjudge():
    torch.manual_seed(1)
    batch, doc_len = 2, 6
    documents = torch.randint(1, 64, (batch, doc_len), dtype=torch.long)
    sent_lent = torch.tensor([doc_len] * batch)
    return (documents, sent_lent)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("NeurJudge", "build_neurjudge", "example_input_neurjudge", 2021, "vendored"),
]

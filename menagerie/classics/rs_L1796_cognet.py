# SOURCE: vendored from BarryRun/COGNet @ master (src/COGNet_model.py, src/layers.py)
#
# COGNet (Wu et al., "Conditional Generation Net for Medication Recommendation",
# WWW 2022). Copy-or-predict medication recommendation over a patient's visit
# history: a per-visit GCN over EHR/DDI drug-interaction graphs, per-visit
# TransformerEncoder passes over diagnoses/procedures/prior-medications, a custom
# cross-visit attention score, and a custom Transformer-style decoder
# (MedTransformerDecoder) that fuses a generate-mode softmax with a copy-mode
# attention-over-history mechanism (pointer-generator style) to recommend the next
# visit's medication set. All real classes below (COGNet, GCN, MaskLinear,
# MedTransformerDecoder, PositionEmbedding, GraphConvolution, SelfAttend) are
# transcribed verbatim from the official repo (only the `data.processing` import,
# which is dead code never called in COGNet_model.py, and Chinese comments are
# elided/kept as-is); only base torch/numpy are used, so no faithful-port step was
# needed -- this is a straight vendor of the real source.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---- from src/layers.py (verbatim) ----------------------------------------


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(nn.Linear(embedding_size, 32), nn.Tanh())

        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates + seq_masks
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output


# ---- from src/COGNet_model.py (verbatim) -----------------------------------


class COGNet(nn.Module):
    def __init__(
        self,
        voc_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        emb_dim=64,
        device=torch.device("cpu:0"),
    ):
        super(COGNet, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.nhead = 2
        self.SOS_TOKEN = voc_size[2]  # start of sentence
        self.END_TOKEN = voc_size[2] + 1
        self.MED_PAD_TOKEN = voc_size[2] + 2
        self.DIAG_PAD_TOKEN = voc_size[0] + 2
        self.PROC_PAD_TOKEN = voc_size[1] + 2

        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        # dig_num * emb_dim
        self.diag_embedding = nn.Sequential(
            nn.Embedding(voc_size[0] + 3, emb_dim, self.DIAG_PAD_TOKEN), nn.Dropout(0.3)
        )

        # proc_num * emb_dim
        self.proc_embedding = nn.Sequential(
            nn.Embedding(voc_size[1] + 3, emb_dim, self.PROC_PAD_TOKEN), nn.Dropout(0.3)
        )

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 3, emb_dim, self.MED_PAD_TOKEN), nn.Dropout(0.3)
        )

        self.medication_encoder = nn.TransformerEncoderLayer(
            emb_dim, self.nhead, batch_first=True, dropout=0.2
        )
        self.diagnoses_encoder = nn.TransformerEncoderLayer(
            emb_dim, self.nhead, batch_first=True, dropout=0.2
        )
        self.procedure_encoder = nn.TransformerEncoderLayer(
            emb_dim, self.nhead, batch_first=True, dropout=0.2
        )

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.gcn = GCN(
            voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device
        )
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.diag_self_attend = SelfAttend(emb_dim)
        self.proc_self_attend = SelfAttend(emb_dim)

        self.decoder = MedTransformerDecoder(
            emb_dim, self.nhead, dim_feedforward=emb_dim * 2, dropout=0.2, layer_norm_eps=1e-5
        )

        self.dec_gru = nn.GRU(emb_dim * 3, emb_dim, batch_first=True)

        self.diag_attn = nn.Linear(emb_dim * 2, 1)
        self.proc_attn = nn.Linear(emb_dim * 2, 1)
        self.W_diag_attn = nn.Linear(emb_dim, emb_dim)
        self.W_proc_attn = nn.Linear(emb_dim, emb_dim)
        self.W_diff_attn = nn.Linear(emb_dim, emb_dim)
        self.W_diff_proc_attn = nn.Linear(emb_dim, emb_dim)

        self.Ws = nn.Linear(emb_dim * 2, emb_dim)  # only used at initial stage
        self.Wo = nn.Linear(emb_dim, voc_size[2] + 2)  # generate mode
        self.Wc = nn.Linear(emb_dim, emb_dim)  # copy mode

        self.W_dec = nn.Linear(emb_dim, emb_dim)
        self.W_stay = nn.Linear(emb_dim, emb_dim)
        self.W_proc_dec = nn.Linear(emb_dim, emb_dim)
        self.W_proc_stay = nn.Linear(emb_dim, emb_dim)

        self.W_z = nn.Linear(emb_dim, 1)

        self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        self.bipartite_transform = nn.Sequential(nn.Linear(emb_dim, ddi_mask_H.shape[1]))
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], voc_size[2], False)

    def encode(
        self,
        diseases,
        procedures,
        medications,
        d_mask_matrix,
        p_mask_matrix,
        m_mask_matrix,
        seq_length,
        dec_disease,
        stay_disease,
        dec_disease_mask,
        stay_disease_mask,
        dec_proc,
        stay_proc,
        dec_proc_mask,
        stay_proc_mask,
        max_len=20,
    ):
        device = self.device
        batch_size, max_visit_num, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]

        input_disease_embdding = self.diag_embedding(diseases).view(
            batch_size * max_visit_num, max_diag_num, self.emb_dim
        )
        input_proc_embedding = self.proc_embedding(procedures).view(
            batch_size * max_visit_num, max_proc_num, self.emb_dim
        )
        d_enc_mask_matrix = (
            d_mask_matrix.view(batch_size * max_visit_num, max_diag_num)
            .unsqueeze(dim=1)
            .unsqueeze(dim=1)
            .repeat(1, self.nhead, max_diag_num, 1)
        )
        d_enc_mask_matrix = d_enc_mask_matrix.view(
            batch_size * max_visit_num * self.nhead, max_diag_num, max_diag_num
        )
        p_enc_mask_matrix = (
            p_mask_matrix.view(batch_size * max_visit_num, max_proc_num)
            .unsqueeze(dim=1)
            .unsqueeze(dim=1)
            .repeat(1, self.nhead, max_proc_num, 1)
        )
        p_enc_mask_matrix = p_enc_mask_matrix.view(
            batch_size * max_visit_num * self.nhead, max_proc_num, max_proc_num
        )
        input_disease_embdding = self.diagnoses_encoder(
            input_disease_embdding, src_mask=d_enc_mask_matrix
        ).view(batch_size, max_visit_num, max_diag_num, self.emb_dim)
        input_proc_embedding = self.procedure_encoder(
            input_proc_embedding, src_mask=p_enc_mask_matrix
        ).view(batch_size, max_visit_num, max_proc_num, self.emb_dim)

        visit_diag_embedding = self.diag_self_attend(
            input_disease_embdding.view(batch_size * max_visit_num, max_diag_num, -1),
            d_mask_matrix.view(batch_size * max_visit_num, -1),
        )
        visit_proc_embedding = self.proc_self_attend(
            input_proc_embedding.view(batch_size * max_visit_num, max_proc_num, -1),
            p_mask_matrix.view(batch_size * max_visit_num, -1),
        )
        visit_diag_embedding = visit_diag_embedding.view(batch_size, max_visit_num, -1)
        visit_proc_embedding = visit_proc_embedding.view(batch_size, max_visit_num, -1)

        cross_visit_scores = self.calc_cross_visit_scores(
            visit_diag_embedding, visit_proc_embedding
        )

        last_seq_medication = torch.full((batch_size, 1, max_med_num), 0).to(device)
        last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1)
        last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device)
        last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :-1, :]], dim=1)
        last_seq_medication_emb = self.med_embedding(last_seq_medication)
        last_m_enc_mask = (
            last_m_mask.view(batch_size * max_visit_num, max_med_num)
            .unsqueeze(dim=1)
            .unsqueeze(dim=1)
            .repeat(1, self.nhead, max_med_num, 1)
        )
        last_m_enc_mask = last_m_enc_mask.view(
            batch_size * max_visit_num * self.nhead, max_med_num, max_med_num
        )
        encoded_medication = self.medication_encoder(
            last_seq_medication_emb.view(batch_size * max_visit_num, max_med_num, self.emb_dim),
            src_mask=last_m_enc_mask,
        )
        encoded_medication = encoded_medication.view(
            batch_size, max_visit_num, max_med_num, self.emb_dim
        )

        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter
        drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        return (
            input_disease_embdding,
            input_proc_embedding,
            encoded_medication,
            cross_visit_scores,
            last_seq_medication,
            last_m_mask,
            drug_memory,
        )

    def decode(
        self,
        input_medications,
        input_disease_embedding,
        input_proc_embedding,
        last_medication_embedding,
        last_medications,
        cross_visit_scores,
        d_mask_matrix,
        p_mask_matrix,
        m_mask_matrix,
        last_m_mask,
        drug_memory,
    ):
        batch_size = input_medications.size(0)
        max_visit_num = input_medications.size(1)
        max_med_num = input_medications.size(2)
        max_diag_num = input_disease_embedding.size(2)
        max_proc_num = input_proc_embedding.size(2)

        input_medication_embs = self.med_embedding(input_medications).view(
            batch_size * max_visit_num, max_med_num, -1
        )
        input_medication_memory = drug_memory[input_medications].view(
            batch_size * max_visit_num, max_med_num, -1
        )

        m_self_mask = m_mask_matrix

        last_m_enc_mask = (
            m_self_mask.view(batch_size * max_visit_num, max_med_num)
            .unsqueeze(dim=1)
            .unsqueeze(dim=1)
            .repeat(1, self.nhead, max_med_num, 1)
        )
        medication_self_mask = last_m_enc_mask.view(
            batch_size * max_visit_num * self.nhead, max_med_num, max_med_num
        )
        m2d_mask_matrix = (
            d_mask_matrix.view(batch_size * max_visit_num, max_diag_num)
            .unsqueeze(dim=1)
            .unsqueeze(dim=1)
            .repeat(1, self.nhead, max_med_num, 1)
        )
        m2d_mask_matrix = m2d_mask_matrix.view(
            batch_size * max_visit_num * self.nhead, max_med_num, max_diag_num
        )
        m2p_mask_matrix = (
            p_mask_matrix.view(batch_size * max_visit_num, max_proc_num)
            .unsqueeze(dim=1)
            .unsqueeze(dim=1)
            .repeat(1, self.nhead, max_med_num, 1)
        )
        m2p_mask_matrix = m2p_mask_matrix.view(
            batch_size * max_visit_num * self.nhead, max_med_num, max_proc_num
        )

        dec_hidden = self.decoder(
            input_medication_embedding=input_medication_embs,
            input_medication_memory=input_medication_memory,
            input_disease_embdding=input_disease_embedding.view(
                batch_size * max_visit_num, max_diag_num, -1
            ),
            input_proc_embedding=input_proc_embedding.view(
                batch_size * max_visit_num, max_proc_num, -1
            ),
            input_medication_self_mask=medication_self_mask,
            d_mask=m2d_mask_matrix,
            p_mask=m2p_mask_matrix,
        )

        score_g = self.Wo(dec_hidden)
        score_g = score_g.view(batch_size, max_visit_num, max_med_num, -1)
        prob_g = F.softmax(score_g, dim=-1)
        score_c = self.copy_med(
            dec_hidden.view(batch_size, max_visit_num, max_med_num, -1),
            last_medication_embedding,
            last_m_mask,
            cross_visit_scores,
        )
        prob_c_to_g = (
            torch.zeros_like(prob_g)
            .to(self.device)
            .view(batch_size, max_visit_num * max_med_num, -1)
        )

        copy_source = last_medications.view(batch_size, 1, -1).repeat(
            1, max_visit_num * max_med_num, 1
        )
        prob_c_to_g.scatter_add_(2, copy_source, score_c)
        prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)

        generate_prob = F.sigmoid(self.W_z(dec_hidden)).view(
            batch_size, max_visit_num, max_med_num, 1
        )
        prob = prob_g * generate_prob + prob_c_to_g * (1.0 - generate_prob)
        prob[:, 0, :, :] = prob_g[:, 0, :, :]

        return torch.log(prob)

    def forward(
        self,
        diseases,
        procedures,
        medications,
        d_mask_matrix,
        p_mask_matrix,
        m_mask_matrix,
        seq_length,
        dec_disease,
        stay_disease,
        dec_disease_mask,
        stay_disease_mask,
        dec_proc,
        stay_proc,
        dec_proc_mask,
        stay_proc_mask,
        max_len=20,
    ):
        device = self.device
        batch_size, max_seq_length, max_med_num = medications.size()

        (
            input_disease_embdding,
            input_proc_embedding,
            encoded_medication,
            cross_visit_scores,
            last_seq_medication,
            last_m_mask,
            drug_memory,
        ) = self.encode(
            diseases,
            procedures,
            medications,
            d_mask_matrix,
            p_mask_matrix,
            m_mask_matrix,
            seq_length,
            dec_disease,
            stay_disease,
            dec_disease_mask,
            stay_disease_mask,
            dec_proc,
            stay_proc,
            dec_proc_mask,
            stay_proc_mask,
            max_len=20,
        )

        input_medication = torch.full((batch_size, max_seq_length, 1), self.SOS_TOKEN).to(device)
        input_medication = torch.cat([input_medication, medications], dim=2)

        m_sos_mask = torch.zeros((batch_size, max_seq_length, 1), device=self.device).float()
        m_mask_matrix = torch.cat([m_sos_mask, m_mask_matrix], dim=-1)

        output_logits = self.decode(
            input_medication,
            input_disease_embdding,
            input_proc_embedding,
            encoded_medication,
            last_seq_medication,
            cross_visit_scores,
            d_mask_matrix,
            p_mask_matrix,
            m_mask_matrix,
            last_m_mask,
            drug_memory,
        )

        return output_logits

    def calc_cross_visit_scores(self, visit_diag_embedding, visit_proc_embedding):
        max_visit_num = visit_diag_embedding.size(1)
        batch_size = visit_diag_embedding.size(0)

        mask = (
            torch.triu(torch.ones((max_visit_num, max_visit_num), device=self.device)) == 1
        ).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)

        padding = torch.zeros((batch_size, 1, self.emb_dim), device=self.device).float()
        diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]], dim=1)
        proc_keys = torch.cat([padding, visit_proc_embedding[:, :-1, :]], dim=1)

        diag_scores = torch.matmul(visit_diag_embedding, diag_keys.transpose(-2, -1)) / math.sqrt(
            visit_diag_embedding.size(-1)
        )
        proc_scores = torch.matmul(visit_proc_embedding, proc_keys.transpose(-2, -1)) / math.sqrt(
            visit_proc_embedding.size(-1)
        )
        scores = F.softmax(diag_scores + proc_scores + mask, dim=-1)

        return scores

    def copy_med(self, decode_input_hiddens, last_medications, last_m_mask, cross_visit_scores):
        max_visit_num = decode_input_hiddens.size(1)
        input_med_num = decode_input_hiddens.size(2)
        max_med_num = last_medications.size(2)
        copy_query = self.Wc(decode_input_hiddens).view(
            -1, max_visit_num * input_med_num, self.emb_dim
        )
        attn_scores = torch.matmul(
            copy_query,
            last_medications.view(-1, max_visit_num * max_med_num, self.emb_dim).transpose(-2, -1),
        ) / math.sqrt(self.emb_dim)
        med_mask = last_m_mask.view(-1, 1, max_visit_num * max_med_num).repeat(
            1, max_visit_num * input_med_num, 1
        )
        attn_scores = F.softmax(attn_scores + med_mask, dim=-1)

        visit_scores = cross_visit_scores.repeat(1, 1, input_med_num).view(
            -1, max_visit_num * input_med_num, max_visit_num
        )

        visit_scores = (
            visit_scores.unsqueeze(-1)
            .repeat(1, 1, 1, max_med_num)
            .view(-1, max_visit_num * input_med_num, max_visit_num * max_med_num)
        )

        scores = torch.mul(attn_scores, visit_scores).clamp(min=1e-9)
        row_scores = scores.sum(dim=-1, keepdim=True)
        scores = scores / row_scores

        return scores


class MedTransformerDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5
    ) -> None:
        super(MedTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.m2p_multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead

    def forward(
        self,
        input_medication_embedding,
        input_medication_memory,
        input_disease_embdding,
        input_proc_embedding,
        input_medication_self_mask,
        d_mask,
        p_mask,
    ):
        input_len = input_medication_embedding.size(0)
        tgt_len = input_medication_embedding.size(1)

        subsequent_mask = self.generate_square_subsequent_mask(
            tgt_len, input_len * self.nhead, input_disease_embdding.device
        )
        self_attn_mask = subsequent_mask + input_medication_self_mask

        x = input_medication_embedding + input_medication_memory

        x = self.norm1(x + self._sa_block(x, self_attn_mask))
        x = self.norm2(
            x
            + self._m2d_mha_block(x, input_disease_embdding, d_mask)
            + self._m2p_mha_block(x, input_proc_embedding, p_mask)
        )
        x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def generate_square_subsequent_mask(self, sz: int, batch_size: int, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask


class PositionEmbedding(nn.Module):
    """
    We assume that the sequence length is less than 512.
    """

    def __init__(self, emb_size, max_length=512):
        super(PositionEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding_layer = nn.Embedding(max_length, emb_size)

    def forward(self, batch_size, seq_length, device):
        assert seq_length <= self.max_length
        ids = torch.arange(0, seq_length).long().to(torch.device(device))
        ids = ids.unsqueeze(0).repeat(batch_size, 1)
        emb = self.embedding_layer(ids)
        return emb


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device("cpu:0")):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


# ---- staging build/example helpers -----------------------------------------


def build_cognet() -> COGNet:
    torch.manual_seed(0)
    voc_size = (10, 8, 12)  # (diag_vocab, proc_vocab, med_vocab)
    med_vocab = voc_size[2]
    ehr_adj = np.random.RandomState(0).randint(0, 2, size=(med_vocab, med_vocab)).astype("float32")
    ehr_adj = np.maximum(ehr_adj, ehr_adj.T)
    np.fill_diagonal(ehr_adj, 0)
    ddi_adj = np.random.RandomState(1).randint(0, 2, size=(med_vocab, med_vocab)).astype("float32")
    ddi_adj = np.maximum(ddi_adj, ddi_adj.T)
    np.fill_diagonal(ddi_adj, 0)
    ddi_mask_H = np.random.RandomState(2).randint(0, 2, size=(med_vocab, 6)).astype("float32")

    model = COGNet(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=16, device=torch.device("cpu"))
    model.eval()
    return model


def example_input_cognet():
    torch.manual_seed(0)
    voc_size = (10, 8, 12)
    diag_vocab, proc_vocab, med_vocab = voc_size
    batch_size, max_visit_num = 1, 2
    d_max_num, p_max_num, m_max_num = 3, 3, 3

    diseases = torch.randint(0, diag_vocab, (batch_size, max_visit_num, d_max_num))
    procedures = torch.randint(0, proc_vocab, (batch_size, max_visit_num, p_max_num))
    medications = torch.randint(0, med_vocab, (batch_size, max_visit_num, m_max_num))

    seq_length = torch.tensor([max_visit_num] * batch_size)

    d_mask_matrix = torch.zeros((batch_size, max_visit_num, d_max_num))
    p_mask_matrix = torch.zeros((batch_size, max_visit_num, p_max_num))
    m_mask_matrix = torch.zeros((batch_size, max_visit_num, m_max_num))

    dec_disease = torch.zeros((batch_size, max_visit_num, d_max_num), dtype=torch.long)
    stay_disease = torch.zeros((batch_size, max_visit_num, d_max_num), dtype=torch.long)
    dec_disease_mask = torch.zeros((batch_size, max_visit_num, d_max_num))
    stay_disease_mask = torch.zeros((batch_size, max_visit_num, d_max_num))

    dec_proc = torch.zeros((batch_size, max_visit_num, p_max_num), dtype=torch.long)
    stay_proc = torch.zeros((batch_size, max_visit_num, p_max_num), dtype=torch.long)
    dec_proc_mask = torch.zeros((batch_size, max_visit_num, p_max_num))
    stay_proc_mask = torch.zeros((batch_size, max_visit_num, p_max_num))

    return (
        diseases,
        procedures,
        medications,
        d_mask_matrix,
        p_mask_matrix,
        m_mask_matrix,
        seq_length,
        dec_disease,
        stay_disease,
        dec_disease_mask,
        stay_disease_mask,
        dec_proc,
        stay_proc,
        dec_proc_mask,
        stay_proc_mask,
    )


MENAGERIE_ENTRIES = [
    ("COGNet", "build_cognet", "example_input_cognet", 2022, "vendored-pytorch"),
]

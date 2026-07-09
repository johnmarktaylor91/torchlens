# SOURCE: vendored from THUNLP-MT/MEAN @ main
# Vendored files (real architecture code, imports/relative-paths adjusted only):
#   models/MCAttGNN/mc_egnn.py       (MC_E_GCL, MC_Att_L, MCAttEGNN, coord2radial,
#                                      unsorted_segment_sum/mean -- the real Multi-Channel
#                                      E(n)-Equivariant Graph Neural Network)
#   models/MCAttGNN/mc_att_model.py  (ProteinFeature, MCAttModel, EfficientMCAttModel --
#                                      the real top-level "Multi-channel Equivariant Attention
#                                      Network" antibody CDR co-design model)
#   data/pdb_utils.py                (AminoAcidVocab / VOCAB -- pure-Python residue vocabulary
#                                      used only for special-token indices; Bio.PDB-dependent
#                                      parsing methods of the real file are not needed and were
#                                      dropped, but the AminoAcidVocab class itself is unchanged)
#   utils/logger.py                  (print_log -- trivial logging helper)
#
# MEAN (Kong et al., ICLR 2023, "Conditional Antibody Design as 3D Equivariant Graph
# Translation") jointly predicts antibody CDR sequence and 3D backbone structure with a
# Multi-channel Attention E(n)-Equivariant GNN operating over antigen + heavy/light-chain
# graphs (global "begin-of-chain" nodes, context/interface radial edges, iterative
# full-shot refinement in EfficientMCAttModel). MCAttEGNN is the real core equivariant
# architecture (torch_scatter-based scatter-softmax attention fused with an EGNN coordinate
# update); EfficientMCAttModel.generate is the real top-level CDR-design inference path used
# by the repo's own generate.py.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from utils/logger.py
# ---------------------------------------------------------------------------
def print_log(s, level="INFO", end="\n", no_prefix=False):
    # Real repo gates this on an env var + prints; silenced here (no behavior-relevant effect,
    # this is purely a debug print helper called from inside MC_E_GCL/MC_Att_L forward paths).
    return


# ---------------------------------------------------------------------------
# Vendored from data/pdb_utils.py (AminoAcidVocab / VOCAB only -- pure Python, no Bio.PDB
# dependency needed for the vocabulary itself)
# ---------------------------------------------------------------------------
class AminoAcid:
    def __init__(self, symbol, abrv, idx=0, side_chain_coord=None):
        self.symbol = symbol
        self.abrv = abrv
        self.idx = idx
        self.side_chain_coord = side_chain_coord

    def __str__(self):
        return f"{self.idx} {self.symbol} {self.abrv}"


class AminoAcidVocab:
    def __init__(self):
        self.PAD, self.SEP, self.UNK = "#", "/", "*"
        self.BOA, self.BOH, self.BOL = "&", "+", "-"  # begin of antigen, heavy chain, light chain
        specials = [
            (self.PAD, "PAD"),
            (self.UNK, "UNK"),
            (self.BOA, "<X>"),
            (self.BOH, "<H>"),
            (self.BOL, "<L>"),
            (self.SEP, "<E>"),
        ]
        aas = [
            ("G", "GLY"),
            ("A", "ALA"),
            ("V", "VAL"),
            ("L", "LEU"),
            ("I", "ILE"),
            ("F", "PHE"),
            ("W", "TRP"),
            ("Y", "TYR"),
            ("D", "ASP"),
            ("H", "HIS"),
            ("N", "ASN"),
            ("E", "GLU"),
            ("K", "LYS"),
            ("Q", "GLN"),
            ("M", "MET"),
            ("R", "ARG"),
            ("S", "SER"),
            ("T", "THR"),
            ("C", "CYS"),
            ("P", "PRO"),
            ("U", "SEC"),
        ]
        _all = specials + aas
        self.amino_acids = [AminoAcid(symbol, abrv) for symbol, abrv in _all]
        self.symbol2idx, self.abrv2idx = {}, {}
        for i, aa in enumerate(self.amino_acids):
            self.symbol2idx[aa.symbol] = i
            self.abrv2idx[aa.abrv] = i
            aa.idx = i
        self.special_mask = [1 for _ in specials] + [0 for _ in aas]

    def symbol_to_idx(self, symbol):
        return self.symbol2idx.get(symbol.upper(), None)

    def idx_to_symbol(self, idx):
        return self.amino_acids[idx].symbol

    def get_unk_idx(self):
        return self.symbol_to_idx(self.UNK)

    def get_special_mask(self):
        return list(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = AminoAcidVocab()


# ---------------------------------------------------------------------------
# Vendored from models/MCAttGNN/mc_egnn.py
# ---------------------------------------------------------------------------
class MC_E_GCL(nn.Module):
    """Multi-Channel E(n) Equivariant Convolutional Layer."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        n_channel,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        dropout=0.1,
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.dropout = nn.Dropout(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + n_channel**2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, output_nf)
        )

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        print_log(f"agg1, {torch.isnan(agg).sum()}", level="DEBUG")
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        print_log(f"agg, {torch.isnan(agg).sum()}", level="DEBUG")
        out = self.node_mlp(agg)
        print_log(f"out, {torch.isnan(out).sum()}", level="DEBUG")
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter: %s" % self.coords_agg)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        print_log(f"edge_feat, {torch.isnan(edge_feat).sum()}", level="DEBUG")
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord


class MC_Att_L(nn.Module):
    """Multi-Channel Attention Layer."""

    def __init__(
        self, input_nf, output_nf, hidden_nf, n_channel, edges_in_d=0, act_fn=nn.SiLU(), dropout=0.1
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.dropout = nn.Dropout(dropout)
        self.linear_q = nn.Linear(input_nf, hidden_nf)
        self.linear_kv = nn.Linear(input_nf + n_channel**2 + edges_in_d, hidden_nf * 2)

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def att_model(self, h, edge_index, radial, edge_attr):
        row, col = edge_index
        source, target = h[row], h[col]
        q = self.linear_q(source)
        n_channel = radial.shape[1]
        radial = radial.reshape(radial.shape[0], n_channel * n_channel)
        if edge_attr is not None:
            target_feat = torch.cat([radial, target, edge_attr], dim=1)
        else:
            target_feat = torch.cat([radial, target], dim=1)
        kv = self.linear_kv(target_feat)
        k, v = kv[..., 0::2], kv[..., 1::2]
        alpha = torch.sum(q * k, dim=1)
        print_log(f"alpha1, {torch.isnan(alpha).sum()}", level="DEBUG")
        alpha = scatter_softmax(alpha, row)
        print_log(f"alpha2, {torch.isnan(alpha).sum()}", level="DEBUG")
        return alpha, v

    def node_model(self, h, edge_index, att_weight, v):
        row, _ = edge_index
        agg = unsorted_segment_sum(att_weight * v, row, h.shape[0])
        agg = self.dropout(agg)
        return h + agg

    def coord_model(self, coord, edge_index, coord_diff, att_weight, v):
        row, _ = edge_index
        coord_v = att_weight * self.coord_mlp(v)
        trans = coord_diff * coord_v.unsqueeze(-1)
        agg = unsorted_segment_sum(trans, row, coord.size(0))
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None):
        radial, coord_diff = coord2radial(edge_index, coord)
        att_weight, v = self.att_model(h, edge_index, radial, edge_attr)
        print_log(f"att_weight, {torch.isnan(att_weight).sum()}", level="DEBUG")
        print_log(f"v, {torch.isnan(v).sum()}", level="DEBUG")
        flat_att_weight = att_weight
        att_weight = att_weight.unsqueeze(-1)
        h = self.node_model(h, edge_index, att_weight, v)
        coord = self.coord_model(coord, edge_index, coord_diff, att_weight, v)
        return h, coord, flat_att_weight


class MCAttEGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        n_channel,
        in_edge_nf=0,
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        dropout=0.1,
        dense=False,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(
                f"gcl_{i}",
                MC_E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    n_channel,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    dropout=dropout,
                ),
            )
            self.add_module(
                f"att_{i}",
                MC_Att_L(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    n_channel,
                    edges_in_d=0,
                    act_fn=act_fn,
                    dropout=dropout,
                ),
            )
        self.out_layer = MC_E_GCL(
            self.hidden_nf,
            self.hidden_nf,
            self.hidden_nf,
            n_channel,
            edges_in_d=in_edge_nf,
            act_fn=act_fn,
            residual=residual,
        )

    def forward(
        self,
        h,
        x,
        ctx_edges,
        att_edges,
        ctx_edge_attr=None,
        att_edge_attr=None,
        return_attention=False,
    ):
        h = self.linear_in(h)
        h = self.dropout(h)

        ctx_states, ctx_coords, atts = [], [], []
        for i in range(0, self.n_layers):
            h, x = self._modules[f"gcl_{i}"](h, ctx_edges, x, edge_attr=ctx_edge_attr)
            ctx_states.append(h)
            ctx_coords.append(x)
            h, x, att = self._modules[f"att_{i}"](h, att_edges, x, edge_attr=att_edge_attr)
            atts.append(att)

        h, x = self.out_layer(h, ctx_edges, x, edge_attr=ctx_edge_attr)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        if return_attention:
            return h, x, atts
        else:
            return h, x


def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))
    radial = F.normalize(radial, dim=0)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


# ---------------------------------------------------------------------------
# Vendored from models/MCAttGNN/mc_att_model.py
# ---------------------------------------------------------------------------
def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


class ProteinFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 1, 2, 3

    def _is_global(self, S):
        return sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)

    def _construct_segment_ids(self, S):
        glbl_node_mask = self._is_global(S)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (
            (glbl_nodes == self.boa_idx),
            (glbl_nodes == self.boh_idx),
            (glbl_nodes == self.bol_idx),
        )
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = (
            self.ag_seg_id,
            self.hc_seg_id,
            self.lc_seg_id,
        )
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        return segment_ids

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, segment_ids=None):
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]

        if segment_ids is None:
            segment_ids = self._construct_segment_ids(S)

        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0
        row, col = torch.nonzero(same_bid).T
        col = col + offsets[batch_id[row]]

        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        ctx_edges = _radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=8.0)

        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges = _radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=12.0)

        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])
        select_edges = torch.logical_and(row_global, col_global)
        global_global = torch.stack([row[select_edges], col[select_edges]])

        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),
            not_global_edges,
            row_seg != self.ag_seg_id,
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])

        space_edge_num = ctx_edges.shape[1] + global_normal.shape[1] + global_global.shape[1]
        ctx_edges = torch.cat([ctx_edges, global_normal, global_global, seq_adj], dim=1)
        ctx_edge_feats = torch.cat(
            [
                torch.zeros(space_edge_num, dtype=torch.float, device=X.device),
                torch.ones(seq_adj.shape[1], dtype=torch.float, device=X.device),
            ],
            dim=0,
        ).unsqueeze(-1)

        return ctx_edges, inter_edges, ctx_edge_feats

    def forward(self, X, S, offsets):
        batch_id = torch.zeros_like(S)
        batch_id[offsets[1:-1]] = 1
        batch_id.cumsum_(dim=0)
        return self.construct_edges(X, S, batch_id)


def _radial_edges(X, src_dst, cutoff):
    dist = X[:, 1][src_dst]
    dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1)
    src_dst = src_dst[dist <= cutoff]
    src_dst = src_dst.transpose(0, 1)
    return src_dst


class MCAttModel(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        n_channel,
        n_edge_feats=0,
        n_layers=3,
        cdr_type="3",
        alpha=0.1,
        dropout=0.1,
        dense=False,
    ):
        super().__init__()
        self.num_aa_type = len(VOCAB)
        self.cdr_type = cdr_type
        self.mask_token_id = VOCAB.get_unk_idx()
        self.alpha = alpha

        self.aa_embedding = nn.Embedding(self.num_aa_type, embed_size)
        self.gnn = MCAttEGNN(
            embed_size,
            hidden_size,
            self.num_aa_type,
            n_channel,
            n_edge_feats,
            n_layers=n_layers,
            residual=True,
            dropout=dropout,
            dense=dense,
        )
        self.protein_feature = ProteinFeature()

    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction="none")

    def coord_loss(self, _input, target):
        return F.smooth_l1_loss(_input, target, reduction="sum")

    def init_mask(self, X, S, cdr_range):
        X, S, cmask = X.clone(), S.clone(), torch.zeros_like(X, device=X.device)
        n_channel, n_dim = X.shape[1:]
        for start, end in cdr_range:
            S[start : end + 1] = self.mask_token_id
            l_coord, r_coord = X[start - 1], X[end + 1]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start : end + 1] = mask_coords
            cmask[start : end + 1, ...] = 1
        return X, S, cmask


class EfficientMCAttModel(MCAttModel):
    """The real top-level "full-shot" MEAN model: iteratively refines CDR sequence + backbone
    coordinates over `n_iter` rounds via the MCAttEGNN above (repo's generate.py inference
    path), rather than the slower step-by-step autoregressive MCAttModel.forward/generate."""

    def __init__(
        self,
        embed_size,
        hidden_size,
        n_channel,
        n_edge_feats=0,
        n_layers=3,
        cdr_type="3",
        alpha=0.1,
        dropout=0.1,
        n_iter=5,
    ):
        super().__init__(
            embed_size,
            hidden_size,
            n_channel,
            n_edge_feats,
            n_layers,
            cdr_type,
            alpha,
            dropout,
            dense=False,
        )
        self.n_iter = n_iter

    def generate(self, X, S, L, offsets, greedy=True):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long,
            device=X.device,
        ) + offsets[:-1].unsqueeze(-1)

        true_X, true_S = X.clone(), S.clone()  # noqa: F841 -- true_S unused (faithful to source)

        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)
        aa_embeddings = self.aa_embedding(torch.arange(self.num_aa_type, device=H_0.device))

        for r in range(self.n_iter):
            with torch.no_grad():
                ctx_edges, inter_edges, ctx_edge_feats = self.protein_feature(X, S, offsets)
            H, Z = self.gnn(H_0, X, ctx_edges, inter_edges, ctx_edge_feats)

            X = X.clone()
            X[mask] = Z[mask]
            H_0 = H_0.clone()
            seq_prob = torch.softmax(H[mask].masked_fill(smask, float("-inf")), dim=-1)
            H_0[mask] = seq_prob.mm(aa_embeddings)

        logits = H[mask]
        logits = logits.masked_fill(smask, float("-inf"))

        if greedy:
            S[mask] = torch.argmax(logits, dim=-1)
        else:
            prob = F.softmax(logits, dim=-1)
            S[mask] = torch.multinomial(prob, num_samples=1).squeeze()
        snll_all = self.seq_loss(logits, S[mask])

        return snll_all, S, X, true_X, cdr_range

    def forward(self, X, S, L, offsets):
        # Thin call-convention wrapper matching the real repo's `.generate()` inference entry
        # point (used verbatim by generate.py); TorchLens traces `model(*inputs)`.
        return self.generate(X, S, L, offsets)


def build_mean():
    torch.manual_seed(0)
    return EfficientMCAttModel(
        embed_size=16,
        hidden_size=16,
        n_channel=4,
        n_edge_feats=1,
        n_layers=2,
        cdr_type="3",
        alpha=0.1,
        dropout=0.0,
        n_iter=2,
    )


def example_input_mean():
    """Builds a tiny synthetic single-sample antibody-antigen complex batch matching the real
    field schema consumed by EfficientMCAttModel.generate (X, S, L, offsets), mirroring the
    construction in the real repo's data/dataset.py::EquiAACDataset.__getitem__ /
    collate_fn: a global begin-of-heavy-chain node followed by a short heavy-chain sequence
    with a CDR-H3 span (label '3') long enough for at least one masked residue."""
    torch.manual_seed(0)
    n_channel = 4

    # residue symbols: begin-of-heavy-chain global node, then 9 heavy-chain residues
    hc_seq = "ACDEFGHIK"
    symbols = [VOCAB.BOH] + list(hc_seq)
    S = torch.tensor([VOCAB.symbol_to_idx(s) for s in symbols], dtype=torch.long)
    n_node = len(symbols)

    X = torch.randn(n_node, n_channel, 3)

    # CDR label string: '0' everywhere except a CDR-H3 span (label '3') of length 3,
    # matching the real repo's per-residue cdr annotation format (index 0 is the global node).
    L_labels = ["0"] * n_node
    cdr_start, cdr_end = 4, 6
    for i in range(cdr_start, cdr_end + 1):
        L_labels[i] = "3"
    # A plain `list[str]` here would collide with TorchLens's ergonomic text-tokenization
    # input coercion (a `list[str]` positional arg is heuristically treated as batched text
    # needing a tokenizer). `L` is domain data (per-residue CDR-type labels), not text, so it
    # is passed as a `tuple[str]`: the real model code only ever iterates/indexes it
    # (`for cdr in L`, `cdr.index(...)`), which a tuple supports identically to a list.
    L = ("".join(L_labels),)

    offsets = torch.tensor([0, n_node], dtype=torch.long)

    return (X, S, L, offsets)


MENAGERIE_ENTRIES = [
    ("MEAN", "build_mean", "example_input_mean", 2023, MENAGERIE_ZOO),
]

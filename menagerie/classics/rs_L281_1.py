# SOURCE: vendored from decisionforce/mmTransformer @ be25d26118d2dfdac72b1d1e0cf6cbf14f7f4a0b
# Files: lib/models/mmTransformer.py, lib/models/TF_version/stacked_transformer.py,
#        lib/models/TF_utils.py (verbatim real architecture; only import paths flattened
#        into this single staging module).
"""mmTransformer: stacked transformer for multimodal motion prediction (CVPR 2021).

Real repo: https://github.com/decisionforce/mmTransformer
"""

import copy
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# lib/models/TF_utils.py (verbatim)
# ---------------------------------------------------------------------------
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed

    def forward(self, src, tgt, src_mask, tgt_mask, query_pos=None):
        """
        Take in and process masked src and target sequences.
        """
        output = self.encode(src, src_mask)
        return self.decode(output, src_mask, tgt, tgt_mask, query_pos)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, query_pos=None):
        return self.decoder(tgt, memory, src_mask, tgt_mask, query_pos)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, x_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, n, return_intermediate=False):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.size)
        self.return_intermediate = return_intermediate

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):
        intermediate = []

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(x))

        if self.norm is not None:
            x = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # TODO How to fusion the feature
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, x, memory, src_mask, tgt_mask, query_pos=None):
        """
        Follow Figure 1 (right) for connections.
        """
        m = memory
        q = k = self.with_pos_embed(x, query_pos)
        x = self.sublayer[0](x, lambda x: self.self_attn(q, k, x, tgt_mask))
        x = self.with_pos_embed(x, query_pos)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=True), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if len(query.shape) > 3:
            batch_dim = len(query.shape) - 2
            batch = query.shape[:batch_dim]
            mask_dim = batch_dim
        else:
            batch = (query.shape[0],)
            mask_dim = 1
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(dim=mask_dim)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(*batch, -1, self.h, self.d_k).transpose(-3, -2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(-3, -2).contiguous().view(*batch, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)

    # Q,K,V: [bs,h,num,dim]
    # scores: [bs,h,num1,num2]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask: [bs,1,1,num2] => dimension expansion

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model, bias=True)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[-2]].requires_grad_(False)
        return self.dropout(x)


# for 626
class GeneratorWithParallelHeads626(nn.Module):
    def __init__(self, d_model, out_size, dropout, reg_h_dim=128, dis_h_dim=128, cls_h_dim=128):
        super(GeneratorWithParallelHeads626, self).__init__()
        self.reg_mlp = nn.Sequential(
            nn.Linear(d_model, reg_h_dim * 2, bias=True),
            nn.LayerNorm(reg_h_dim * 2),
            nn.ReLU(),
            nn.Linear(reg_h_dim * 2, reg_h_dim, bias=True),
            nn.Linear(reg_h_dim, out_size, bias=True),
        )
        self.dis_emb = nn.Linear(2, dis_h_dim, bias=True)
        self.cls_FFN = PointerwiseFeedforward(d_model, 2 * d_model, dropout=dropout)
        self.classification_layer = nn.Sequential(
            nn.Linear(d_model, cls_h_dim), nn.Linear(cls_h_dim, 1, bias=True)
        )
        self.cls_opt = nn.Softmax(dim=-1)

    def forward(self, x):
        pred = self.reg_mlp(x)
        pred = pred.view(*pred.shape[0:3], -1, 2).cumsum(dim=-2)
        # return pred
        cls_h = self.cls_FFN(x)
        cls_h = self.classification_layer(cls_h).squeeze(dim=-1)
        conf = self.cls_opt(cls_h)
        return pred, conf


def split_dim(x: torch.Tensor, split_shape: tuple, dim: int):
    if dim < 0:
        dim = len(x.shape) + dim
    return x.reshape(*x.shape[:dim], *split_shape, *x.shape[dim + 1 :])


# ---------------------------------------------------------------------------
# lib/models/TF_version/stacked_transformer.py (verbatim)
# ---------------------------------------------------------------------------
class STF(nn.Module):
    def __init__(self, cfg):
        super(STF, self).__init__()
        "Helper: Construct a model from hyperparameters."

        # Hyperparameters from cfg
        hist_inp_size = cfg["in_channels"]
        lane_inp_size = cfg["enc_dim"]
        num_queries = cfg["queries"]
        dec_out_size = cfg["out_channels"]
        # Hyperparameters predefined
        N = 2
        N_lane = 2
        N_social = 2
        d_model = 128
        d_ff = 256
        pos_dim = 64
        dist_dim = 128
        h = 2
        dropout = 0
        #

        self.aux_loss = cfg["aux_task"]
        c = copy.deepcopy
        dropout_atten = dropout
        # dropout_atten = 0.1
        attn = MultiHeadAttention(h, d_model, dropout=dropout_atten)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.hist_tf = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(hist_inp_size, d_model), c(position)),
        )
        self.lane_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_lane)
        self.lane_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_lane)
        self.lane_emb = LinearEmbedding(lane_inp_size, d_model)

        self.pos_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim, bias=True),
        )
        self.dist_emb = nn.Sequential(
            nn.Linear(num_queries * d_model, dist_dim, bias=True),
            nn.LayerNorm(dist_dim),
            nn.ReLU(),
            nn.Linear(dist_dim, dist_dim, bias=True),
        )

        self.fusion1 = nn.Sequential(
            nn.Linear(d_model + pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(dist_dim + pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.social_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_social)
        self.social_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_social)

        # self.g = Generator(d_model*2, dec_out_size)
        self.prediction_header = GeneratorWithParallelHeads626(d_model * 2, dec_out_size, dropout)
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.query_embed.weight.requires_grad == False  # noqa: E712 (verbatim no-op from original repo)
        nn.init.orthogonal_(self.query_embed.weight)

    # input: [inp, dec_inp, src_att, trg_att]

    def forward(self, traj, pos, social_num, social_mask, lane_enc, lane_mask):
        """
        Args:
            traj: [batch size, max_agent_num, 19, 4]
            pos: [batch size, max_agent_num, 2]
            social_num: float = max_agent_num
            social_mask: [batch size, 1, max_agent_num]
            lane_enc: [batch size, max_lane_num, 64]
            lane_mask: [batch size, 1, max_lane_num]

        Returns:
            outputs_coord: [batch size, max_agent_num, num_query, 30, 2]
            outputs_class: [batch size, max_agent_num, num_query]
        """

        self.query_batches = self.query_embed.weight.view(
            1, 1, *self.query_embed.weight.shape
        ).repeat(*traj.shape[:2], 1, 1)

        # Trajectory transfomer
        hist_out = self.hist_tf(traj, self.query_batches, None, None)
        pos = self.pos_emb(pos)
        hist_out = torch.cat(
            [pos.unsqueeze(dim=2).repeat(1, 1, self.num_queries, 1), hist_out], dim=-1
        )
        hist_out = self.fusion1(hist_out)

        # Lane encoder
        lane_mem = self.lane_enc(self.lane_emb(lane_enc), lane_mask)
        lane_mem = lane_mem.unsqueeze(1).repeat(1, social_num, 1, 1)
        lane_mask = lane_mask.unsqueeze(1).repeat(1, social_num, 1, 1)

        # Lane decoder
        lane_out = self.lane_dec(hist_out, lane_mem, lane_mask, None)

        # Fuse position information
        dist = lane_out.view(*traj.shape[0:2], -1)
        dist = self.dist_emb(dist)

        # Social layer
        social_inp = self.fusion2(torch.cat([pos, dist], -1))
        social_mem = self.social_enc(social_inp, social_mask)
        social_out = social_mem.unsqueeze(dim=2).repeat(1, 1, self.num_queries, 1)
        out = torch.cat([social_out, lane_out], -1)

        # Prediction head
        outputs_coord, outputs_class = self.prediction_header(out)

        return outputs_coord, outputs_class


# ---------------------------------------------------------------------------
# lib/models/mmTransformer.py (verbatim)
# ---------------------------------------------------------------------------
class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f"lmlp_{i}", MLP(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, lane):
        """
            Extract lane_feature from vectorized lane representation

        Args:
            lane: [batch size, max_lane_num, 9, 7] (vectorized representation)

        Returns:
            x_max: [batch size, max_lane_num, 64]
        """
        x = lane
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit), nn.LayerNorm(hidden_unit), nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class mmTrans(nn.Module):
    def __init__(self, stacked_transformer, cfg):
        super(mmTrans, self).__init__()
        # stacked transformer class
        self.stacked_transformer = stacked_transformer(cfg)

        lane_channels = cfg["lane_channels"]
        self.hist_feature_size = cfg["in_channels"]

        self.polyline_vec_shape = 2 * cfg["subgraph_width"]
        self.subgraph = LaneNet(lane_channels, cfg["subgraph_width"], cfg["num_subgraph_layres"])

        self.FUTURE_LEN = cfg["future_num_frames"]
        self.OBS_LEN = cfg["history_num_frames"] - 1
        self.lane_length = cfg["lane_length"]

    def preprocess_traj(self, traj):
        """
        Generate the trajectory mask for all agents (including target agent)

        Args:
            traj: [batch, max_agent_num, obs_len, 4]

        Returns:
            social mask: [batch, 1, max_agent_num]

        """
        # social mask
        social_valid_len = self.traj_valid_len
        social_mask = torch.zeros((self.B, 1, int(self.max_agent_num))).to(traj.device)
        for i in range(self.B):
            social_mask[i, 0, : social_valid_len[i]] = 1

        return social_mask

    def preprocess_lane(self, lane):
        """
            preprocess lane segments using LaneNet

        Args:
            lane: [batch size, max_lane_num, 10, 5]

        Returns:
            lane_feature: [batch size, max_lane_num, 64 (feature_dim)]
            lane_mask: [batch size, 1, max_lane_num]

        """

        # transform lane to vector
        lane_v = torch.cat(
            [lane[:, :, :-1, :2], lane[:, :, 1:, :2], lane[:, :, 1:, 2:]], dim=-1
        )  # bxnlinex9x7

        # lane mask
        lane_valid_len = self.lane_valid_len
        lane_mask = torch.zeros((self.B, 1, int(self.max_lane_num))).to(lane_v.device)
        for i in range(lane_valid_len.shape[0]):
            lane_mask[i, 0, : lane_valid_len[i]] = 1

        # use vector like structure process lane
        lane_feature = self.subgraph(lane_v)  # [batch size, max_lane_num, 64]

        return lane_feature, lane_mask

    def forward(self, data: dict):
        """
        Args:
            data (Data):
                HIST: [batch size, max_agent_num, 19, 4]
                POS: [batch size, max_agent_num, 2]
                LANE: [batch size, max_lane_num, 10, 5]
                VALID_LEN: [batch size, 2] (number of valid agents & valid lanes)

        Note:
            max_lane_num/max_agent_num indicates maximum number of agents/lanes after padding in a single batch
        """
        # initialized
        self.B = data["HISTORY"].shape[0]

        self.traj_valid_len = data["VALID_LEN"][:, 0]
        self.max_agent_num = torch.max(self.traj_valid_len)

        self.lane_valid_len = data["VALID_LEN"][:, 1]
        self.max_lane_num = torch.max(self.lane_valid_len)

        # preprocess
        pos = data["POS"]
        trajs = data["HISTORY"]
        social_mask = self.preprocess_traj(data["HISTORY"])
        lane_enc, lane_mask = self.preprocess_lane(data["LANE"])

        out = self.stacked_transformer(
            trajs, pos, self.max_agent_num, social_mask, lane_enc, lane_mask
        )

        return out


# ---------------------------------------------------------------------------
# staging harness (tiny random-init construction, matching the real repo's
# config/demo.py hyperparameter shapes but scaled down for a fast trace)
# ---------------------------------------------------------------------------
_MMT_CFG = {
    "in_channels": 4,  # [x, y, heading, valid] per history frame
    "enc_dim": 64,  # lane encoder output width (2 * subgraph_width)
    "queries": 6,  # number of trajectory query modes
    "queries_dim": 128,
    "out_channels": 60,  # 30 future frames * 2 (x,y) -> matches GeneratorWithParallelHeads626
    "aux_task": False,
    "lane_channels": 7,  # vectorized lane feature dim (9x7 -> per point)
    "subgraph_width": 32,
    "num_subgraph_layres": 3,
    "future_num_frames": 30,
    "history_num_frames": 20,
    "lane_length": 10,
}


def build_mmtransformer():
    return mmTrans(STF, _MMT_CFG)


def example_input_mmtransformer():
    batch = 2
    max_agent_num = 5
    max_lane_num = 4
    obs_len = 19
    data = {
        "HISTORY": torch.randn(batch, max_agent_num, obs_len, 4),
        "POS": torch.randn(batch, max_agent_num, 2),
        "LANE": torch.randn(batch, max_lane_num, 10, 5),
        "VALID_LEN": torch.tensor([[max_agent_num, max_lane_num]] * batch, dtype=torch.long),
    }
    return (data,)


MENAGERIE_ENTRIES = [
    ("mmTransformer", build_mmtransformer, example_input_mmtransformer, 2021, "vendored-pytorch"),
]

# SOURCE: vendored from Graylab/GeoDock @ main
# Files: geodock/model/modules/iterative_transformer.py, geodock/model/modules/graph_module.py,
#        geodock/model/modules/structure_module.py, geodock/utils/coords6d.py,
#        geodock/utils/transforms.py (quaternion_to_matrix only)
# GeoDock: end-to-end differentiable protein-protein docking model. IterativeTransformer is
# the real trainable architecture inside geodock.model.GeoDock.GeoDock (a pytorch_lightning
# LightningModule that just embeds two protein ESM feature streams and calls this network);
# we trace IterativeTransformer directly (its GraphModule + StructureModule/IPA stack) rather
# than the LightningModule wrapper, since pytorch_lightning is training-loop scaffolding
# around the same nn.Module graph, not part of the architecture. Vendored verbatim aside from
# merging the three files into one module (relative imports flattened).
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum

"""============================================================================================="""
""" utils/transforms.py (quaternion_to_matrix only) """
"""============================================================================================="""


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


"""============================================================================================="""
""" utils/coords6d.py """
"""============================================================================================="""


def calc_dist(a_coords, b_coords):
    assert a_coords.shape == b_coords.shape
    mat_shape = list(a_coords.shape)
    mat_shape.insert(-1, mat_shape[-2])

    a_coords = a_coords.unsqueeze(-3).expand(mat_shape)
    b_coords = b_coords.unsqueeze(-2).expand(mat_shape)

    dist_mat = (a_coords - b_coords).norm(dim=-1)

    return dist_mat


def calc_dihedral(a_coords, b_coords, c_coords, d_coords, convert_to_degree=True):
    b1 = a_coords - b_coords
    b2 = b_coords - c_coords
    b3 = c_coords - d_coords

    n1 = torch.cross(b1, b2)
    n1 = torch.div(n1, n1.norm(dim=-1, keepdim=True))
    n2 = torch.cross(b2, b3)
    n2 = torch.div(n2, n2.norm(dim=-1, keepdim=True))
    m1 = torch.cross(n1, torch.div(b2, b2.norm(dim=-1, keepdim=True)))

    dihedral = torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    if convert_to_degree:
        dihedral = dihedral * 180 / math.pi

    return dihedral


def calc_planar(a_coords, b_coords, c_coords, convert_to_degree=True):
    v1 = a_coords - b_coords
    v2 = c_coords - b_coords

    a = (v1 * v2).sum(-1)
    b = v1.norm(dim=-1) * v2.norm(dim=-1)

    planar = torch.acos(a / b)

    if convert_to_degree:
        planar = planar * 180 / math.pi

    return planar


# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, use_Cb=False):
    n = xyz.shape[0]

    # three anchor atoms
    N = xyz[..., 0, :]
    Ca = xyz[..., 1, :]
    C = xyz[..., 2, :]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    if use_Cb:
        dist = calc_dist(Cb, Cb)
    else:
        dist = calc_dist(Ca, Ca)

    omega = calc_dihedral(
        repeat(Ca, "r i -> r c i", c=n),
        repeat(Cb, "r i -> r c i", c=n),
        repeat(Cb, "c i -> r c i", r=n),
        repeat(Ca, "c i -> r c i", r=n),
    )

    theta = calc_dihedral(
        repeat(N, "r i -> r c i", c=n),
        repeat(Ca, "r i -> r c i", c=n),
        repeat(Cb, "r i -> r c i", c=n),
        repeat(Cb, "c i -> r c i", r=n),
    )
    phi = calc_planar(
        repeat(Ca, "r i -> r c i", c=n),
        repeat(Cb, "r i -> r c i", c=n),
        repeat(Cb, "c i -> r c i", r=n),
    )

    return dist, omega, theta, phi


"""============================================================================================="""
""" model/modules/graph_module.py """
"""============================================================================================="""


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class NodeAttentionWithEdgeBias(nn.Module):
    """
    hidden_dim = 32
    """

    def __init__(self, *, node_dim, edge_dim, head_dim=32, heads=8):
        super().__init__()
        hidden_dim = head_dim * heads
        self.heads = heads
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

        self.q = nn.Linear(node_dim, hidden_dim, bias=False)
        self.k = nn.Linear(node_dim, hidden_dim, bias=False)
        self.v = nn.Linear(node_dim, hidden_dim, bias=False)
        self.b = nn.Linear(edge_dim, heads, bias=False)
        self.g = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.Sigmoid())
        self.attn_logits_scale = head_dim**-0.5
        self.to_out = nn.Linear(hidden_dim, node_dim)

    def forward(self, node, edge):
        h = self.heads
        # layer norm
        x = self.node_norm(node)
        y = self.edge_norm(edge)
        # get queries, keys, values, gated, pairwise
        q, k, v, g, b = self.q(x), self.k(x), self.v(x), self.g(x), self.b(y)
        # split out heads
        q, k, v, g = map(lambda t: rearrange(t, "b i (h d) -> (b h) i d", h=h), (q, k, v, g))
        b = rearrange(b, "b i j h -> (b h) i j", h=h)
        # derive attn logits
        attn_logits = einsum("b i d, b j d -> b i j", q, k) * self.attn_logits_scale + b
        # attention
        attn = attn_logits.softmax(dim=-1)
        # aggregate values
        results = einsum("b i j, b j d -> b i d", attn, v) * g
        # merge back heads
        results = rearrange(results, "(b h) i d -> b i (h d)", h=h)
        return self.to_out(results)


class Transition(nn.Module):
    """
    hidden_dim = 2 * node_dim
    """

    def __init__(self, *, dim, n=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_act = nn.Linear(dim, dim * n)
        self.act = nn.ReLU()
        self.to_out = nn.Linear(dim * n, dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.to_act(x)
        x = self.act(x)
        x = self.to_out(x)
        return x


class NodeUpdate(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.attention = NodeAttentionWithEdgeBias(node_dim=node_dim, edge_dim=edge_dim)
        self.transition = Transition(dim=node_dim)

    def forward(self, node, edge):
        x = node + self.attention(node, edge)
        x = x + self.transition(x)
        return x


class TriangleMultiplicativeModule(nn.Module):
    """
    hidden_dim = edge_dim
    """

    def __init__(self, *, dim, hidden_dim=None, mix="ingoing"):
        super().__init__()
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"
        if exists(mask):
            mask = rearrange(mask, "b i j -> b i j ()")

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class EdgeUpdate(nn.Module):
    """
    dropout = 0.1
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=dim, mix="outgoing")
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=dim, mix="ingoing")
        self.transition = Transition(dim=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.triangle_multiply_outgoing(x, mask=mask)) + x
        x = self.dropout(self.triangle_multiply_ingoing(x, mask=mask)) + x
        x = self.transition(x) + x
        return x


class NodeToEdge(nn.Module):
    """
    hidden_dim = 2 * node_dim
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = default(hidden_dim, node_dim)
        self.layernorm = nn.LayerNorm(node_dim)
        self.proj = nn.Linear(node_dim, hidden_dim * 2)
        self.o_proj = nn.Linear(2 * hidden_dim, edge_dim)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, node):
        assert len(node.shape) == 3

        s = self.layernorm(node)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


class GraphBlock(nn.Module):
    def __init__(self, *, node_dim, edge_dim):
        super().__init__()
        self.node_update = NodeUpdate(node_dim=node_dim, edge_dim=edge_dim)
        self.node_to_edge = NodeToEdge(node_dim=node_dim, edge_dim=edge_dim)
        self.edge_update = EdgeUpdate(dim=edge_dim)

    def forward(self, node, edge):
        node = self.node_update(node, edge)
        edge = edge + self.node_to_edge(node)
        edge = self.edge_update(edge)
        return node, edge


class GraphModule(nn.Module):
    def __init__(self, *, node_dim, edge_dim, depth):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(GraphBlock(node_dim=node_dim, edge_dim=edge_dim))

    def forward(self, node, edge):
        for block in self.blocks:
            node, edge = block(node, edge)
        return node, edge


"""============================================================================================="""
""" model/modules/structure_module.py """
"""============================================================================================="""


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        heads=8,
        scalar_key_dim=16,
        scalar_value_dim=16,
        point_key_dim=4,
        point_value_dim=4,
        eps=1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads

        # Number of attentions
        num_attn_logits = 3

        # QKV projection for scalar attention
        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5
        self.to_scalar_q = nn.Linear(node_dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_k = nn.Linear(node_dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_v = nn.Linear(node_dim, scalar_value_dim * heads, bias=False)

        # QKV projection for point attention
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.0)) - 1.0)
        self.point_weights = nn.Parameter(point_weight_init_value)
        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5
        self.to_point_q = nn.Linear(node_dim, point_key_dim * heads * 3, bias=False)
        self.to_point_k = nn.Linear(node_dim, point_key_dim * heads * 3, bias=False)
        self.to_point_v = nn.Linear(node_dim, point_value_dim * heads * 3, bias=False)

        # Pairwise representation projection to attention bias
        self.pairwise_attn_logits_scale = num_attn_logits**-0.5

        self.to_pairwise_attn_bias = nn.Sequential(
            nn.Linear(edge_dim, heads), Rearrange("b ... h -> (b h) ...")
        )

        # Combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)
        self.to_out = nn.Linear(
            heads * (scalar_value_dim + edge_dim + point_value_dim * (3 + 1)), node_dim
        )

    def forward(self, node, edge, mask=None, *, rotations, translations):
        x, b, h, eps = node, node.size(0), self.heads, self.eps

        # Get queries, keys, values for scalar and point (coordinate-aware) attention pathways
        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)
        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)

        # Split out heads
        q_scalar, k_scalar, v_scalar = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q_scalar, k_scalar, v_scalar)
        )
        q_point, k_point, v_point = map(
            lambda t: rearrange(t, "b n (h d c) -> (b h) n d c", h=h, c=3),
            (q_point, k_point, v_point),
        )
        rotations = repeat(rotations, "b n r1 r2 -> (b h) n r1 r2", h=h)
        translations = repeat(translations, "b n c -> (b h) n () c", h=h)

        # Rotate qkv points into global frame
        q_point = einsum("b n d c, b n r c -> b n d r", q_point, rotations) + translations
        k_point = einsum("b n d c, b n r c -> b n d r", k_point, rotations) + translations
        v_point = einsum("b n d c, b n r c -> b n d r", v_point, rotations) + translations

        # Derive attn logits for scalar and pairwise
        attn_logits_scalar = (
            einsum("b i d, b j d -> b i j", q_scalar, k_scalar) * self.scalar_attn_logits_scale
        )

        attn_logits_pairwise = self.to_pairwise_attn_bias(edge) * self.pairwise_attn_logits_scale

        # Derive attn logits for point attention
        point_qk_diff = rearrange(q_point, "b i d c -> b i () d c") - rearrange(
            k_point, "b j d c -> b () j d c"
        )
        point_dist = (point_qk_diff**2.0).sum(dim=-2)
        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, "h -> (b h) () () ()", b=b)
        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale).sum(
            dim=-1
        )

        # Add look ahead mask for point attention
        if mask is not None:
            mask = repeat(mask, "b w l -> (h b) w l", h=h)
            attn_logits_points += mask * -1e9

        # Combine attn logits
        attn_logits = attn_logits_scalar + attn_logits_points
        attn_logits = attn_logits + attn_logits_pairwise

        # Attention
        attn = attn_logits.softmax(dim=-1)

        # Aggregate values
        results_scalar = einsum("b i j, b j d -> b i d", attn, v_scalar)
        attn_with_heads = rearrange(attn, "(b h) i j -> b h i j", h=h)
        results_pairwise = einsum("b h i j, b i j d -> b h i d", attn_with_heads, edge)

        # Aggregate point values
        results_points = einsum("b i j, b j d c -> b i d c", attn, v_point)

        # Rotate aggregated point values back into local frame
        results_points = einsum(
            "b n d r, b n c r -> b n d c",
            results_points - translations,
            rotations.transpose(-1, -2),
        )
        results_points_norm = torch.sqrt(
            sum(map(torch.square, results_points.unbind(dim=-1))) + eps
        )

        # Merge back heads
        results_scalar = rearrange(results_scalar, "(b h) n d -> b n (h d)", h=h)
        results_points = rearrange(results_points, "(b h) n d c -> b n (h d c)", h=h)
        results_points_norm = rearrange(results_points_norm, "(b h) n d -> b n (h d)", h=h)

        results = (results_scalar, results_points, results_points_norm)

        results_pairwise = rearrange(results_pairwise, "b h n d -> b n (h d)", h=h)
        results = (*results, results_pairwise)

        # Concat results and project out
        results = torch.cat(results, dim=-1)
        return self.to_out(results)


class StructureTransition(nn.Module):
    """
    hidden_dim = node_dim
    """

    def __init__(self, *, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x


class IPABlock(nn.Module):
    """
    dropout = 0.1
    """

    def __init__(self, *, node_dim, edge_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.attn_norm = nn.LayerNorm(node_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn = InvariantPointAttention(node_dim=node_dim, edge_dim=edge_dim, **kwargs)

        self.ff_norm = nn.LayerNorm(node_dim)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff = StructureTransition(dim=node_dim)

    def forward(self, node, edge, **kwargs):
        x = self.attn(node, edge, **kwargs) + node
        x = self.attn_dropout(x)
        x = self.attn_norm(x)

        x = self.ff(x) + x
        x = self.ff_dropout(x)
        x = self.ff_norm(x)
        return x


class StructureModule(nn.Module):
    def __init__(self, *, node_dim, edge_dim, depth):
        super().__init__()
        self.depth = depth
        self.IPA_block = IPABlock(node_dim=node_dim, edge_dim=edge_dim)

        self.to_update = nn.Linear(node_dim, 6)

    def forward(self, node, edge, mask=None, rotations=None, translations=None):
        device = node.device
        b, n, *_ = node.shape

        # If no initial rotations passed in, start from identity
        if not exists(rotations):
            rotations = torch.eye(3, device=device)  # initial rotations
            rotations = repeat(rotations, "i j -> b n i j", b=b, n=n)

        # If no initial translations passed in, start from identity
        if not exists(translations):
            translations = torch.zeros((b, n, 3), device=device)

        for i in range(self.depth):
            # update nodes and backbone frames
            node = self.IPA_block(
                node, edge, mask=mask, rotations=rotations, translations=translations
            )

            quaternions_update, translations_update = self.to_update(node).split([3, 3], dim=-1)
            quaternions_update = F.pad(quaternions_update, (1, 0), value=1.0)
            quaternions_update = quaternions_update / torch.linalg.norm(
                quaternions_update, dim=-1, keepdim=True
            )
            rotations_update = quaternion_to_matrix(quaternions_update)
            translations = (
                einsum("b n i j, b n j -> b n i", rotations, translations_update) + translations
            )
            rotations = einsum("b n i j, b n j k -> b n i k", rotations, rotations_update)

        return node, rotations, translations


"""============================================================================================="""
""" model/modules/iterative_transformer.py """
"""============================================================================================="""


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, c_in, c_hidden, no_bins):
        super().__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.layer_norm = nn.LayerNorm(self.c_in)
        self.linear_1 = nn.Linear(self.c_in, self.c_hidden)
        self.linear_2 = nn.Linear(self.c_hidden, self.c_hidden)
        self.linear_3 = nn.Linear(self.c_hidden, self.no_bins)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_z = c_z
        self.no_bins = no_bins
        self.linear = nn.Linear(self.c_z, self.no_bins)

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)

        return logits


class IterativeTransformer(nn.Module):
    def __init__(self, *, node_dim, edge_dim, gm_depth, sm_depth, num_iter):
        super().__init__()
        self.num_iter = num_iter
        self.recycle_bins = 16
        self.graph_module = GraphModule(node_dim=node_dim, edge_dim=edge_dim, depth=gm_depth)
        self.structure_module = StructureModule(
            node_dim=node_dim, edge_dim=edge_dim, depth=sm_depth
        )
        self.recycled_node_norm = nn.LayerNorm(node_dim)
        self.recycled_edge_norm = nn.LayerNorm(edge_dim)
        self.recycled_disto = nn.Linear(self.recycle_bins * 4, edge_dim)
        self.plddt = PerResidueLDDTCaPredictor(c_in=node_dim, c_hidden=128, no_bins=50)
        self.disto = DistogramHead(c_z=edge_dim, no_bins=64)

    def forward(self, node, edge, mask=None, rotat=None, trans=None):
        device = node.device
        is_grad_enabled = torch.is_grad_enabled()

        # sample the number iterations (eval path used at trace time: fixed at self.num_iter)
        if self.training:
            num_iter = random.randint(1, self.num_iter)
        else:
            num_iter = self.num_iter

        # get initial graph features
        node_0 = node
        edge_0 = edge
        recycled_node = torch.zeros_like(node)
        recycled_edge = torch.zeros_like(edge)
        recycled_bins = torch.zeros(
            (*edge.shape[:-1], self.recycle_bins * 4), device=device, dtype=torch.float
        )

        for i in range(num_iter):
            is_final_iter = i == (num_iter - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                # recycling
                recycled_node = self.recycled_node_norm(recycled_node.detach())
                recycled_edge = self.recycled_edge_norm(recycled_edge.detach())
                recycled_edge = recycled_edge + self.recycled_disto(recycled_bins.detach())

                # graph module
                node, edge = self.graph_module(node_0 + recycled_node, edge_0 + recycled_edge)

                # structure module
                feats, rotat, trans = self.structure_module(node, edge, mask=mask)

                coords = self.get_coords(rotat, trans)

                recycled_node = node
                recycled_edge = edge

                # orientogram
                recycled_bins = self.orientogram(coords, self.recycle_bins)

        lddt_logits = self.plddt(feats)
        dist_logits = self.disto(edge)

        return lddt_logits, dist_logits, coords, rotat, trans

    def orientogram(self, coords, num_bins):
        dist, omega, theta, phi = get_coords6d(coords.squeeze(0), use_Cb=True)

        mask = dist < 22.0

        num_bins = 16
        dist_bin = self.get_bins(dist, 2.0, 22.0, num_bins)
        omega_bin = self.get_bins(omega, -180.0, 180.0, num_bins)
        theta_bin = self.get_bins(theta, -180.0, 180.0, num_bins)
        phi_bin = self.get_bins(phi, -180.0, 180.0, num_bins)

        def mask_mat(mat):
            mat[~mask] = num_bins - 1
            mat.fill_diagonal_(num_bins - 1)
            return mat

        omega_bin = mask_mat(omega_bin)
        theta_bin = mask_mat(theta_bin)
        phi_bin = mask_mat(phi_bin)

        # to onehot
        dist = F.one_hot(dist_bin, num_classes=num_bins).float()
        omega = F.one_hot(omega_bin, num_classes=num_bins).float()
        theta = F.one_hot(theta_bin, num_classes=num_bins).float()
        phi = F.one_hot(phi_bin, num_classes=num_bins).float()

        return torch.cat([dist, omega, theta, phi], dim=-1).unsqueeze(0)

    def get_coords(self, rotat, trans):
        # get batch size, total_len
        batch_size, total_len, *_ = rotat.shape

        # initialize residue frames
        input_coords = torch.tensor(
            [[-0.527, 1.359, 0.0], [0.0, 0.0, 0.0], [1.525, 0.0, 0.0]], device=rotat.device
        )

        input_coords = repeat(input_coords, "i j -> b n i j", b=batch_size, n=total_len)

        # update coordinates according to predicted rotations and translations
        coords = torch.einsum("b n a j, b n i j -> b n a i", input_coords, rotat) + trans.unsqueeze(
            -2
        )
        return coords

    def get_bins(self, x, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(min_bin, max_bin, num_bins - 1, device=x.device)
        bins = torch.sum(x.unsqueeze(-1) > boundaries, dim=-1)  # [..., L, L]

        return bins


MENAGERIE_ZOO = "vendored-pytorch"


def build_geodock():
    torch.manual_seed(0)
    return IterativeTransformer(node_dim=32, edge_dim=16, gm_depth=1, sm_depth=1, num_iter=1)


def example_input_geodock():
    torch.manual_seed(0)
    batch, n_res = 1, 10
    node = torch.randn(batch, n_res, 32)
    edge = torch.randn(batch, n_res, n_res, 16)
    return (node, edge)


MENAGERIE_ENTRIES = [
    ("GeoDock", "build_geodock", "example_input_geodock", 2023, "SOURCE_AVAILABLE"),
]

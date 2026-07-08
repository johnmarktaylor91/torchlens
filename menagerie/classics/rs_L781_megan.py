# SOURCE: vendored from molecule-one/megan @ master
# Files: src/model/megan.py, src/model/megan_modules/encoder.py,
#        src/model/megan_modules/decoder.py, src/model/graph/gat.py
#
# MEGAN (Sacha et al., J. Chem. Inf. Model. 2021): "Molecule Edit Graph Attention Network" --
# a stateful graph-attention sequence-of-graph-edits model for one-step retrosynthesis. A
# custom multi-head Graph Attention Network layer (`MultiHeadGraphConvLayer`, edge-feature-
# aware attention over a dense atom adjacency tensor) is stacked into an encoder
# (`MeganEncoder`) and decoder (`MeganDecoder`) that jointly predict per-atom and per-bond
# "graph edit" action distributions at each generation step. Verbatim from the real repo
# except: (1) the `@gin.configurable(...)` decorators (gin-config hyperparameter injection,
# not an architectural change -- forward behavior is identical without them) are stripped
# since `gin-config` is not a base dependency, (2) the two feature-key constant lists
# (`ORDERED_ATOM_OH_KEYS`, `ORDERED_BOND_OH_KEYS`) are inlined from `src/feat/__init__.py`
# instead of importing the full `src.feat` package (which pulls in unrelated dataset/JSON
# I/O machinery), (3) the CUDA-only `torch.cuda.FloatTensor` branch of `to_one_hot` is
# dropped in favor of the CPU branch (device selection only, same op).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.autograd import Variable

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# inlined from src/feat/__init__.py (molecule-one/megan)
ORDERED_ATOM_OH_KEYS = [
    "is_supernode",
    "atomic_num",
    "formal_charge",
    "chiral_tag",
    "num_explicit_hs",
    "is_aromatic",
    "is_edited",
    "is_reactant",
]
ORDERED_BOND_OH_KEYS = ["bond_type", "bond_stereo", "is_edited"]


def to_one_hot(x, dims: int):
    # suggested in https://github.com/molecule-one/megan/issues/8
    one_hot = torch.FloatTensor(*x.shape, dims).zero_().to(device)

    x = torch.unsqueeze(x, -1)
    target = one_hot.scatter_(x.dim() - 1, x.data, 1)

    target = Variable(target)
    return target


default_atom_features = (
    "is_supernode",
    "atomic_num",
    "formal_charge",
    "chiral_tag",
    "num_explicit_hs",
    "is_aromatic",
    "is_edited",
)
default_bond_features = "bond_type", "bond_stereo", "is_edited"


# --- src/model/graph/gat.py ---
class MultiHeadGraphConvLayer(nn.Module):
    def __init__(
        self,
        bond_dim: int,
        input_dim: int,
        output_dim: int,
        residual: bool,
        att_heads: int = 8,
        att_dim: int = 64,
    ):
        """
        :param att_dim: dimensionality of narrowed nodes representation for the attention
        :param att_heads: number of attention heads
        """
        super(MultiHeadGraphConvLayer, self).__init__()
        self.n_att = att_heads
        self.att_dim = att_dim

        self.atoms_att = nn.Linear(input_dim, att_dim)
        self.final_att = nn.Linear(att_dim * 2 + bond_dim, att_heads)

        self.conv_layers = []

        if output_dim % att_heads != 0:
            raise ValueError(
                f"Output dimension ({output_dim} "
                f"must be a multiple of number of attention heads ({att_heads}"
            )

        for i in range(att_heads):
            conv = nn.Linear(input_dim, int(output_dim / att_heads))
            self.conv_layers.append(conv)
            setattr(self, f"graph_conv_{i + 1}", conv)

        self.residual = residual

    def forward(self, x, adj, mask, soft_mask, apply_activation: bool = True):
        x_att = torch.relu(self.atoms_att(x))
        x_att_shape = adj.shape[:-1] + (x_att.shape[-1],)
        x_rows = torch.unsqueeze(x_att, 1).expand(x_att_shape)
        x_cols = torch.unsqueeze(x_att, 2).expand(x_att_shape)

        x_att = torch.cat([x_rows, x_cols, adj], dim=-1)
        x_att = self.final_att(x_att)

        head_outs = []

        for i, conv in enumerate(self.conv_layers):
            att = x_att[:, :, :, i]
            att = torch.softmax(att + soft_mask, dim=-1) * mask
            out = torch.bmm(att, x)
            out = conv(out)
            head_outs.append(out)

        out = torch.cat(head_outs, dim=-1)

        if self.residual:
            out = x + out

        if apply_activation:
            out = torch.relu(out)

        return out


# --- src/model/megan_modules/encoder.py ---
class MeganEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bond_emb_dim: int,
        n_encoder_conv: int = 4,
        enc_residual: bool = True,
        enc_dropout: float = 0.0,
    ):
        super(MeganEncoder, self).__init__()
        self.n_conv = n_encoder_conv
        self.residual = enc_residual
        self.dropout = nn.Dropout(enc_dropout) if enc_dropout > 0 else lambda x: x

        self.conv_layers = []
        for i in range(self.n_conv):
            conv = MultiHeadGraphConvLayer(
                bond_dim=bond_emb_dim, input_dim=hidden_dim, output_dim=hidden_dim, residual=False
            )
            self.conv_layers.append(conv)
            setattr(self, f"MultiHeadGraphConv_{i + 1}", conv)

    def forward(self, x: dict) -> dict:
        atom_feats = x["node_features"]
        prev_atom_feats = atom_feats

        for i, conv in enumerate(self.conv_layers):
            residual = self.residual and i % 2 == 1
            atom_feats = conv(
                atom_feats,
                x["adj"],
                x["conv_mask"],
                x["conv_soft_mask"],
                apply_activation=not residual,
            )
            atom_feats = self.dropout(atom_feats)
            if residual:
                atom_feats = torch.relu(atom_feats + prev_atom_feats)
                prev_atom_feats = atom_feats
        x["node_features"] = atom_feats
        return x


# --- src/model/megan_modules/decoder.py ---
def softmax(values, base, dim):
    exp = torch.exp(base * values)
    return exp / torch.sum(exp, dim=dim).unsqueeze(-1)


class MeganDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bond_emb_dim,
        n_atom_actions: int,
        n_bond_actions: int,
        n_fc: int = 3,
        n_decoder_conv: int = 4,
        dec_residual: bool = True,
        bond_atom_dim: int = 32,
        atom_fc_hidden_dim: int = 256,
        bond_fc_hidden_dim: int = 256,
        dec_dropout: float = 0.0,
        dec_hidden_dim: int = 0,
        dec_att_heads: int = 0,
    ):
        super(MeganDecoder, self).__init__()
        if dec_hidden_dim == 0:
            dec_hidden_dim = hidden_dim

        self.n_actions = n_atom_actions
        self.n_bond_actions = n_bond_actions
        self.hidden_dim = hidden_dim
        self.n_fc = n_fc
        self.n_conv = n_decoder_conv
        self.residual = dec_residual
        self.atom_fc_hidden_dim = atom_fc_hidden_dim
        self.bond_fc_hidden_dim = bond_fc_hidden_dim
        self.bond_atom_dim = bond_atom_dim
        self.dropout = nn.Dropout(dec_dropout) if dec_dropout > 0 else lambda x: x

        self.fc_atom_layers = []
        self.fc_bond_layers = []

        self.conv_layers = []
        for i in range(self.n_conv):
            input_dim = hidden_dim if i == 0 else dec_hidden_dim
            output_dim = hidden_dim if i == self.n_conv - 1 else dec_hidden_dim
            if dec_att_heads == 0:
                conv = MultiHeadGraphConvLayer(
                    bond_dim=bond_emb_dim,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    residual=False,
                )
            else:
                conv = MultiHeadGraphConvLayer(
                    bond_dim=bond_emb_dim,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    residual=False,
                    att_heads=dec_att_heads,
                )

            self.conv_layers.append(conv)
            setattr(self, f"MultiHeadGraphConv_{i + 1}", conv)

        for i in range(n_fc):
            in_dim = hidden_dim if i == 0 else atom_fc_hidden_dim
            out_dim = atom_fc_hidden_dim if i < n_fc - 1 else n_atom_actions

            atom_fc = nn.Linear(in_dim, out_dim)
            setattr(self, f"fc_atom_{i + 1}", atom_fc)
            self.fc_atom_layers.append(atom_fc)

        self.fc_atom_bond = nn.Linear(hidden_dim, bond_atom_dim)

        for i in range(n_fc):
            in_dim = bond_atom_dim + bond_emb_dim if i == 0 else bond_fc_hidden_dim
            out_dim = bond_fc_hidden_dim if i < n_fc - 1 else n_bond_actions

            bond_fc = nn.Linear(in_dim, out_dim)
            setattr(self, f"fc_bond_{i + 1}", bond_fc)
            self.fc_bond_layers.append(bond_fc)

    def _forward_atom_features(self, atom_feats):
        for layer in self.fc_atom_layers[:-1]:
            atom_feats = torch.relu(layer(atom_feats))
            atom_feats = self.dropout(atom_feats)
        return atom_feats

    def _forward_bond_features(self, atom_feats, adj):
        atom_feats = torch.relu(self.fc_atom_bond(atom_feats))

        x_exp_shape = adj.shape[:-1] + (atom_feats.shape[-1],)
        x_rows = torch.unsqueeze(atom_feats, 1).expand(x_exp_shape)
        x_cols = torch.unsqueeze(atom_feats, 2).expand(x_exp_shape)

        x_sum = x_rows + x_cols
        bond_actions_feat = torch.cat([x_sum, adj], dim=-1)
        for bond_layer in self.fc_bond_layers[:-1]:
            bond_actions_feat = torch.relu(bond_layer(bond_actions_feat))
            bond_actions_feat = self.dropout(bond_actions_feat)

        return bond_actions_feat

    def forward_embedding(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_feats = x["node_features"]
        prev_atom_feats = atom_feats

        for i, conv in enumerate(self.conv_layers):
            residual = self.residual and i % 2 == 1
            atom_feats = conv(
                atom_feats,
                x["adj"],
                x["conv_mask"],
                x["conv_soft_mask"],
                apply_activation=not residual,
            )
            atom_feats = self.dropout(atom_feats)
            if residual:
                atom_feats = torch.relu(atom_feats + prev_atom_feats)
                prev_atom_feats = atom_feats

        atom_feats = atom_feats * x["node_mask"].expand(*atom_feats.shape)
        node_state = atom_feats

        # calculate final features for atom and bond actions
        atom_actions_feat = self._forward_atom_features(atom_feats)
        bond_actions_feat = self._forward_bond_features(atom_feats, x["adj"])

        return node_state, atom_actions_feat, bond_actions_feat

    def forward(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_state, atom_actions_feat, bond_actions_feat = self.forward_embedding(x)

        atom_actions_feat = self.fc_atom_layers[-1](atom_actions_feat)
        bond_actions_feat = self.fc_bond_layers[-1](bond_actions_feat)

        max_feat_shape = max(atom_actions_feat.shape[-1], bond_actions_feat.shape[-1])

        atom_actions_exp = torch.zeros(
            atom_actions_feat.shape[:-1] + (max_feat_shape,), dtype=torch.float, device=device
        )
        atom_actions_exp[:, :, : atom_actions_feat.shape[-1]] = atom_actions_feat

        bond_actions_exp = torch.zeros(
            bond_actions_feat.shape[:-1] + (max_feat_shape,), dtype=torch.float, device=device
        )
        bond_actions_exp[:, :, :, : bond_actions_feat.shape[-1]] = bond_actions_feat

        bond_mask = x["node_adj_mask"].contiguous()
        bond_mask_exp = torch.zeros(
            bond_mask.shape[:-1] + (max_feat_shape,), dtype=torch.float, device=device
        )
        bond_mask = bond_mask * x["bond_action_mask"]
        bond_mask_exp[:, :, :, : bond_actions_feat.shape[-1]] = bond_mask

        atom_mask = x["node_mask"].expand(*atom_actions_feat.shape)
        atom_mask = atom_mask * x["atom_action_mask"]
        atom_mask_exp = torch.zeros(
            atom_mask.shape[:-1] + (max_feat_shape,), dtype=torch.float, device=device
        )
        atom_mask_exp[:, :, : atom_actions_feat.shape[-1]] = atom_mask

        atom_mask_exp = atom_mask_exp.unsqueeze(1)
        atom_actions_exp = atom_actions_exp.unsqueeze(1)

        all_actions = torch.cat([bond_actions_exp, atom_actions_exp], dim=1)
        all_actions_mask = torch.cat([bond_mask_exp, atom_mask_exp], dim=1)
        all_actions = torch.reshape(all_actions, (all_actions.shape[0], -1))
        all_actions_mask = torch.reshape(all_actions_mask, (all_actions_mask.shape[0], -1))

        soft_mask = (1.0 - all_actions_mask) * -1e9
        base = x.get("base", 1.0)

        if "sigmoid" in x and x["sigmoid"]:
            all_actions = torch.sigmoid(all_actions) * all_actions_mask
        else:
            if base == 1.0:
                all_actions = torch.softmax(all_actions + soft_mask, dim=-1) * all_actions_mask
            else:
                all_actions = softmax(all_actions + soft_mask, base=base, dim=-1) * all_actions_mask

        return node_state, all_actions, all_actions_mask


# --- src/model/megan.py ---
class Megan(nn.Module):
    def __init__(
        self,
        n_atom_actions: int,
        n_bond_actions: int,
        prop2oh: dict,
        bond_emb_dim: int = 8,
        hidden_dim: int = 512,
        stateful: bool = True,
        atom_feature_keys: Tuple[str] = default_atom_features,
        bond_feature_keys: Tuple[str] = default_bond_features,
        reaction_type_given: bool = False,
        n_reaction_types: int = 10,
        reaction_type_emb_dim: int = 16,
    ):
        super(Megan, self).__init__()
        self.prop2oh = prop2oh
        self.n_actions = n_atom_actions
        self.n_bond_actions = n_bond_actions
        self.bond_emb_dim = bond_emb_dim
        self.hidden_dim = hidden_dim
        self.stateful = stateful
        self.reaction_type_given = reaction_type_given

        total_atom_oh_len = sum(len(self.prop2oh["atom"][key]) + 1 for key in atom_feature_keys)
        total_bond_oh_len = sum(len(self.prop2oh["bond"][key]) + 1 for key in bond_feature_keys)

        self.numbered_atom_oh_keys = [
            (ORDERED_ATOM_OH_KEYS.index(key), key) for key in atom_feature_keys
        ]
        self.numbered_bond_oh_keys = [
            (ORDERED_BOND_OH_KEYS.index(key), key) for key in bond_feature_keys
        ]

        if reaction_type_given:
            assert reaction_type_emb_dim < hidden_dim
            assert reaction_type_emb_dim < bond_emb_dim
            self.reaction_type_embedding = nn.Embedding(n_reaction_types, reaction_type_emb_dim)
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim - reaction_type_emb_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, bond_emb_dim - reaction_type_emb_dim)
        else:
            self.reaction_type_embedding = None
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, bond_emb_dim)

        self.encoder = MeganEncoder(hidden_dim=hidden_dim, bond_emb_dim=bond_emb_dim)
        self.decoder = MeganDecoder(
            hidden_dim=hidden_dim,
            bond_emb_dim=bond_emb_dim,
            n_atom_actions=n_atom_actions,
            n_bond_actions=n_bond_actions,
        )

    def _preprocess(self, x: dict) -> dict:
        oh_atom_feats = []
        for i, key in self.numbered_atom_oh_keys:
            oh_feat = to_one_hot(
                x["node_features"][:, :, i], dims=len(self.prop2oh["atom"][key]) + 1
            )
            oh_atom_feats.append(oh_feat)
        atom_feats = torch.cat(oh_atom_feats, dim=-1)

        x["node_features"] = atom_feats

        # "node_adj_mask" has shape fo the adjacency matrix and puts 1 in every place where a bond is possible
        node_adj_mask = x["node_mask"].unsqueeze(-1)
        node_adj_mask = node_adj_mask.expand(*node_adj_mask.shape)
        node_adj_mask = node_adj_mask * node_adj_mask.permute(0, 2, 1, 3).contiguous()
        x["node_adj_mask"] = node_adj_mask

        conv_mask = x["adj_mask"].float().squeeze(-1)

        # two different masks are needed to mask for softmax activation
        conv_soft_mask = (-conv_mask + 1.0) * -1e9
        x["conv_mask"] = conv_mask
        x["conv_soft_mask"] = conv_soft_mask

        oh_bond_feats = []
        for i, key in self.numbered_bond_oh_keys:
            oh_feat = to_one_hot(x["adj"][:, :, :, i], dims=len(self.prop2oh["bond"][key]) + 1)
            oh_bond_feats.append(oh_feat)
        adj = torch.cat(oh_bond_feats, dim=-1)
        x["adj"] = adj

        x["node_features"] = self.atom_embedding(x["node_features"])
        x["adj"] = self.bond_embedding(x["adj"])

        if self.reaction_type_given:
            reaction_type = x["reaction_type"]
            r_type_emb = self.reaction_type_embedding(reaction_type)
            r_type_emb = r_type_emb.unsqueeze(1).expand(-1, x["node_features"].shape[1], -1)

            x["node_features"] = torch.cat((x["node_features"], r_type_emb), dim=-1)

            r_type_emb = r_type_emb.unsqueeze(2).expand(-1, -1, x["node_features"].shape[1], -1)
            x["adj"] = torch.cat((x["adj"], r_type_emb), dim=-1)

        return x

    def forward(self, x: dict) -> dict:
        batch_size, n_steps, n_nodes = x["adj"].shape[0], x["adj"].shape[1], x["adj"].shape[2]  # noqa: F841 -- unused in real source too (verbatim)
        outputs = []
        output_masks = []
        state_dict = None

        for step_i in range(n_steps):
            step_batch = dict((k, v[:, step_i]) for k, v in x.items() if v.dim() > 1)

            step_results = self.forward_step(step_batch, state_dict=state_dict)
            state_dict = {
                "state": step_results["state"],
            }

            outputs.append(step_results["output"])
            output_masks.append(step_results["output_mask"])

        outputs = torch.stack(outputs, dim=1)
        output_masks = torch.stack(output_masks, dim=1)

        result = {"output": outputs, "output_mask": output_masks}

        return result

    def forward_step(
        self, step_batch: dict, state_dict=Optional[dict], first_step: Optional[List[int]] = None
    ) -> dict:
        batch_size, n_nodes = step_batch["adj"].shape[0], step_batch["adj"].shape[2]

        step_batch = self._preprocess(step_batch)

        # run encoder only on the first step of generation
        if self.stateful:
            if state_dict is None:  # first generation step
                step_batch = self.encoder(step_batch)
            else:  # generation step > 1
                state = state_dict["state"]
                if state.shape[1] != n_nodes:
                    min_n_nodes = min(state.shape[1], n_nodes)
                    new_state = torch.zeros((batch_size, n_nodes, self.hidden_dim), device=device)
                    new_state[:, :min_n_nodes] = state[:, :min_n_nodes]
                    state = new_state

                # merge embeddings of nodes with their "state" (features taken from previous decoder)
                merged_node_features = torch.max(step_batch["node_features"], state)

                # this means there can be some samples for which this is the first step
                if first_step:
                    encoded_step_batch = self.encoder(step_batch)

                    # for samples for which this is not the first step, ignore encoder results
                    for i, first in enumerate(first_step):
                        if first:
                            step_batch["node_features"][i] = encoded_step_batch["node_features"][i]
                        else:
                            step_batch["node_features"][i] = merged_node_features[i]
                else:
                    step_batch["node_features"] = merged_node_features

            state, output, mask = self.decoder(step_batch)

            result = {"state": state, "output": output, "output_mask": mask}
        else:
            step_batch = self.encoder(step_batch)
            _, output, mask = self.decoder(step_batch)
            result = {"output": output, "output_mask": mask}
        return result


def build_megan():
    # tiny prop2oh vocab: atom/bond one-hot feature-value spaces used by default_atom_features
    # / default_bond_features (see ORDERED_ATOM_OH_KEYS / ORDERED_BOND_OH_KEYS above)
    prop2oh = {
        "atom": {
            "is_supernode": {0: 0, 1: 1},
            "atomic_num": {i: i for i in range(10)},
            "formal_charge": {i: i for i in range(5)},
            "chiral_tag": {i: i for i in range(4)},
            "num_explicit_hs": {i: i for i in range(5)},
            "is_aromatic": {0: 0, 1: 1},
            "is_edited": {0: 0, 1: 1},
        },
        "bond": {
            "bond_type": {i: i for i in range(5)},
            "bond_stereo": {i: i for i in range(4)},
            "is_edited": {0: 0, 1: 1},
        },
    }
    # `to_one_hot` above builds its scatter buffer on the module-level `device` constant
    # (the real repo's own single-accelerator assumption -- see header note); keep the
    # model's own parameters on that same device so CPU/CUDA machines are both consistent.
    # stateful=True (the repo default): `Megan.forward` unconditionally reads
    # `step_results['state']` after each `forward_step` call, a key only present on the
    # stateful decoder branch -- stateful=False is not a runnable configuration upstream.
    return Megan(
        n_atom_actions=8,
        n_bond_actions=6,
        prop2oh=prop2oh,
        bond_emb_dim=8,
        hidden_dim=32,
        stateful=True,
    ).to(device)


def example_input_megan():
    # Megan.forward iterates over a leading "generation step" axis (x['adj'].shape[1]),
    # slicing x[:, step_i] out of every tensor value each iteration -- so every field here
    # carries an explicit n_steps dimension in position 1, matching the real batched input.
    bsz, n_steps, n_atom = 2, 1, 6
    n_atom_feat = 7  # len(default_atom_features)
    n_bond_feat = 3  # len(default_bond_features)
    # the real featurizer (src/feat/megan_graph.py) builds node/edge feature arrays with
    # `.astype(int)` -- node_features/adj are integer category codes, not floats, since
    # `to_one_hot` uses them directly as a torch.scatter_ index.
    x = {
        "node_features": torch.randint(
            0, 3, (bsz, n_steps, n_atom, n_atom_feat), dtype=torch.int64, device=device
        ),
        "adj": torch.randint(
            0, 3, (bsz, n_steps, n_atom, n_atom, n_bond_feat), dtype=torch.int64, device=device
        ),
        "node_mask": torch.ones(bsz, n_steps, n_atom, 1, device=device),
        "adj_mask": torch.ones(bsz, n_steps, n_atom, n_atom, 1, device=device),
        "atom_action_mask": torch.ones(bsz, n_steps, n_atom, 8, device=device),
        "bond_action_mask": torch.ones(bsz, n_steps, n_atom, n_atom, 6, device=device),
    }
    return (x,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MEGAN", "build_megan", "example_input_megan", 2021, "vendored"),
]

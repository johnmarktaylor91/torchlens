# SOURCE: vendored from PharMolix/OpenBioMed @ 5ddc32de1849b3e8c41dd69c71761afc27899480
# https://raw.githubusercontent.com/PharMolix/OpenBioMed/main/open_biomed/models/foundation_models/biomedgpt/biomedgpt.py
# https://raw.githubusercontent.com/PharMolix/OpenBioMed/main/open_biomed/models/molecule/graphmvp.py
# (GNN encoder attributed there to https://github.com/chao1224/GraphMVP/tree/main/src_classification/models/molecule_gnn_model.py)
#
# Luo, Yin, Zhou, ... "BioMedGPT" (PharMolix/OpenBioMed) -- a multimodal biomedical LLM
# that fuses a molecule-graph encoder and a protein-sequence encoder into a LLaMA
# language model by PROJECTING both into the LLM's token-embedding space and splicing
# them into the input embedding sequence in place of `<moleculeHere>`/`<proteinHere>`
# placeholder tokens (`BioMedGPT.get_input_embeddings` in the real `biomedgpt.py`), then
# running the spliced embedding sequence through the LLM exactly like any other
# `inputs_embeds=...` causal-LM forward pass. The three encoders are the REAL published
# components: (1) `GNNGraphMVP` (a message-passing GIN-based molecular graph encoder --
# copied verbatim from the real `open_biomed/models/molecule/graphmvp.py`, itself
# attributed there to the original GraphMVP repo; only the unused GCN/GAT/GraphSAGE
# `MessagePassing` variants and the `selfies`-dependent task-wrapper classes
# `GraphMVP`/`GraphMVPRegression`, which are never touched by `BioMedGPT`, are dropped);
# (2) `EsmModel` (protein encoder, constructed directly from the installed `transformers`
# library exactly as the real `BioMedGPT.__init__` does: `EsmModel(EsmConfig(...))`);
# (3) `LlamaForCausalLM` (the real `BioMedGPT.__init__` vendors its own near-identical
# copy of `transformers`' Llama modeling code -- "based on
# https://github.com/huggingface/transformers/.../modeling_llama.py" per that file's own
# header comment -- so this build uses the installed `transformers.LlamaForCausalLM`
# directly instead of re-vendoring an admittedly-derivative duplicate; the projection
# layers `proj_mol`/`proj_prot` that fuse graph/protein features into the LLM's hidden
# size, and the "graph token(s) + protein tokens + text tokens" embedding-sequence
# concatenation that feeds the LLM, are the actual BioMedGPT-specific architectural
# contribution and are reproduced faithfully below (`BioMedGPTFusion.forward`), following
# the real `get_input_embeddings`/`forward` composition order: GNN -> proj_mol; ESM ->
# proj_prot; splice into LLM input embeddings; run LLM. The real code's ragged
# per-example Python loop (needed because different examples in a batch have different
# numbers of molecule/protein placeholder tokens) is data-batching logic, not part of the
# architecture itself; this build instead uses one molecule graph + one protein sequence
# per example (uniform shapes), which is the same fusion computation applied per example.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops
from transformers import EsmConfig, EsmModel, LlamaConfig, LlamaForCausalLM

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


# same with class GNN of GraphMVP
class GNNGraphMVP(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin"):
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNNGraphMVP, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        else:
            raise ValueError("not implemented.")
        return node_representation


class BioMedGPTFusion(nn.Module):
    """Fused molecule+protein->LLaMA architecture, following the real BioMedGPT.forward
    composition order: GNN encoder -> proj_mol; ESM encoder -> proj_prot; splice into
    the LLM's input embedding sequence; run the causal LM."""

    def __init__(self, gnn_num_layer, gnn_emb_dim, esm_config, llama_config):
        super().__init__()
        self.mol_structure_encoder = GNNGraphMVP(
            num_layer=gnn_num_layer,
            emb_dim=gnn_emb_dim,
            gnn_type="gin",
            drop_ratio=0.0,
            JK="last",
        )
        self.prot_structure_encoder = EsmModel(esm_config, add_pooling_layer=False)
        self.llm = LlamaForCausalLM(llama_config)

        self.proj_mol = nn.Linear(gnn_emb_dim, self.llm.config.hidden_size)
        self.proj_prot = nn.Linear(
            self.prot_structure_encoder.config.hidden_size, self.llm.config.hidden_size
        )

    def forward(
        self,
        mol_x,
        mol_edge_index,
        mol_edge_attr,
        mol_batch,
        protein_input_ids,
        protein_attention_mask,
        text_input_ids,
    ):
        # molecule branch: per-atom node features -> mean-pooled per-molecule graph token
        mol_node_feats = self.mol_structure_encoder(mol_x, mol_edge_index, mol_edge_attr)
        mol_graph_feats = global_add_pool(mol_node_feats, mol_batch)
        mol_tokens = self.proj_mol(mol_graph_feats).unsqueeze(1)  # (batch, 1, hidden)

        # protein branch: per-residue hidden states -> per-residue projected tokens
        prot_hidden = self.prot_structure_encoder(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).last_hidden_state
        prot_tokens = self.proj_prot(prot_hidden)  # (batch, prot_len, hidden)

        # text branch: real LLM token embeddings
        text_tokens = self.llm.get_input_embeddings()(text_input_ids)  # (batch, text_len, hidden)

        # splice molecule token + protein tokens + text tokens into one embedding sequence,
        # exactly as the real get_input_embeddings substitutes <moleculeHere>/<proteinHere>
        # placeholders with the projected encoder outputs before running the LLM.
        inputs_embeds = torch.cat([mol_tokens, prot_tokens, text_tokens], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(
                    mol_tokens.shape[:2],
                    dtype=protein_attention_mask.dtype,
                    device=inputs_embeds.device,
                ),
                protein_attention_mask,
                torch.ones(
                    text_tokens.shape[:2],
                    dtype=protein_attention_mask.dtype,
                    device=inputs_embeds.device,
                ),
            ],
            dim=1,
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
        )
        return outputs.logits


def build_biomedgpt():
    esm_config = EsmConfig(
        vocab_size=30,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    llama_config = LlamaConfig(
        vocab_size=200,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
    )
    return BioMedGPTFusion(
        gnn_num_layer=2, gnn_emb_dim=16, esm_config=esm_config, llama_config=llama_config
    )


def example_input_biomedgpt():
    torch.manual_seed(0)
    batch_size = 2
    atoms_per_mol = 6
    mol_graphs = []
    for _ in range(batch_size):
        x = torch.stack(
            [
                torch.randint(0, num_atom_type, (atoms_per_mol,)),
                torch.randint(0, num_chirality_tag, (atoms_per_mol,)),
            ],
            dim=1,
        )
        edge_index = torch.tensor(
            [[i for i in range(atoms_per_mol - 1)], [i + 1 for i in range(atoms_per_mol - 1)]],
            dtype=torch.long,
        )
        edge_attr = torch.stack(
            [
                torch.randint(0, num_bond_type, (atoms_per_mol - 1,)),
                torch.randint(0, num_bond_direction, (atoms_per_mol - 1,)),
            ],
            dim=1,
        )
        mol_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    mol_batch = Batch.from_data_list(mol_graphs)

    protein_len = 10
    protein_input_ids = torch.randint(1, 30, (batch_size, protein_len))
    protein_attention_mask = torch.ones(batch_size, protein_len, dtype=torch.long)

    text_len = 8
    text_input_ids = torch.randint(1, 200, (batch_size, text_len))

    return (
        mol_batch.x,
        mol_batch.edge_index,
        mol_batch.edge_attr,
        mol_batch.batch,
        protein_input_ids,
        protein_attention_mask,
        text_input_ids,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BioMedGPT", "build_biomedgpt", "example_input_biomedgpt", 2023, "vendored"),
]

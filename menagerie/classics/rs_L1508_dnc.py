# SOURCE: vendored from https://github.com/ixaxaar/pytorch-dnc @ master
# (dnc/util.py + dnc/memory.py + dnc/dnc.py, merged into one file; imports adjusted to be
# self-contained (relative imports -> local names) and the SAM/SDNC/SparseMemory/
# SparseTemporalMemory branches -- which require the optional `faiss` dependency -- are
# dropped since the default DNC configuration (`share_memory_between_layers=True`,
# non-sparse) never touches them; the `Memory` module is vendored verbatim and is the only
# memory backend actually exercised here. No architectural changes.)
"""Differentiable Neural Computer (Graves et al., 2016, Nature). A recurrent controller
(LSTM/GRU/RNN) augmented with an external, differentiable, content- and temporal-
addressable read/write memory matrix (content-based lookup, dynamic memory allocation,
and temporal link tracking for sequential reads). This staging module vendors the actual
`DNC` controller class and its `Memory` module from the reference `ixaxaar/pytorch-dnc`
package (used as the RL-agent controller in "Differentiable Neural Computer RL agent"
projects such as `ixaxaar/pytorch-dnc`-based Atari/bAbI/copy-task agents); the RL training
loop wrapped around it is agent-specific glue code, not part of the DNC architecture."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

MemoryHiddenState = dict


# --- from dnc/util.py (verbatim, only what Memory/DNC actually use) ---
δ = 1e-6


def cuda(x, requires_grad=False, device=None):
    if device is None:
        return x.float().requires_grad_(requires_grad)
    else:
        return x.float().to(device).requires_grad_(requires_grad)


def θ(a, b, norm_by=2):  # noqa: N802
    """Batchwise cosine similarity between two tensors (b*m*w, b*r*w) -> (b*r*m)."""
    dot = torch.bmm(a, b.transpose(1, 2))
    a_norm = torch.norm(a, p=norm_by, dim=2).unsqueeze(2)
    b_norm = torch.norm(b, p=norm_by, dim=2).unsqueeze(1)
    cos = dot / (a_norm * b_norm + δ)
    return cos.transpose(1, 2).contiguous()


def σ(input, axis=1):  # noqa: N802
    return F.softmax(input, dim=axis)


# --- from dnc/memory.py (verbatim) ---
class Memory(nn.Module):
    """Memory module: content- and temporal-addressable differentiable read/write memory."""

    def __init__(
        self,
        input_size,
        nr_cells=512,
        cell_size=32,
        read_heads=4,
        independent_linears=True,
        device=None,
    ):
        super().__init__()

        self.nr_cells = nr_cells
        self.cell_size = cell_size
        self.read_heads = read_heads
        self.input_size = input_size
        self.independent_linears = independent_linears
        self.device = device

        if self.independent_linears:
            self.read_keys_transform = nn.Linear(self.input_size, self.cell_size * self.read_heads)
            self.read_strengths_transform = nn.Linear(self.input_size, self.read_heads)
            self.write_key_transform = nn.Linear(self.input_size, self.cell_size)
            self.write_strength_transform = nn.Linear(self.input_size, 1)
            self.erase_vector_transform = nn.Linear(self.input_size, self.cell_size)
            self.write_vector_transform = nn.Linear(self.input_size, self.cell_size)
            self.free_gates_transform = nn.Linear(self.input_size, self.read_heads)
            self.allocation_gate_transform = nn.Linear(self.input_size, 1)
            self.write_gate_transform = nn.Linear(self.input_size, 1)
            self.read_modes_transform = nn.Linear(self.input_size, 3 * self.read_heads)

            torch.nn.init.kaiming_uniform_(self.read_keys_transform.weight)
            torch.nn.init.kaiming_uniform_(self.read_strengths_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_key_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_strength_transform.weight)
            torch.nn.init.kaiming_uniform_(self.erase_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_vector_transform.weight)
            torch.nn.init.kaiming_uniform_(self.free_gates_transform.weight)
            torch.nn.init.kaiming_uniform_(self.allocation_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.write_gate_transform.weight)
            torch.nn.init.kaiming_uniform_(self.read_modes_transform.weight)
        else:
            self.interface_size = (
                (self.cell_size * self.read_heads)
                + (3 * self.cell_size)
                + (5 * self.read_heads)
                + 3
            )
            self.interface_weights = nn.Linear(self.input_size, self.interface_size)
            torch.nn.init.kaiming_uniform_(self.interface_weights.weight)

        self.I = cuda(1 - torch.eye(self.nr_cells).unsqueeze(0), device=self.device)
        if self.device is not None and self.device.type == "cuda":
            self.to(self.device)

    def new(self, batch_size=1):
        return {
            "memory": cuda(
                torch.zeros(batch_size, self.nr_cells, self.cell_size), device=self.device
            ),
            "link_matrix": cuda(
                torch.zeros(batch_size, 1, self.nr_cells, self.nr_cells), device=self.device
            ),
            "precedence": cuda(torch.zeros(batch_size, 1, self.nr_cells), device=self.device),
            "read_weights": cuda(
                torch.zeros(batch_size, self.read_heads, self.nr_cells), device=self.device
            ),
            "write_weights": cuda(torch.zeros(batch_size, 1, self.nr_cells), device=self.device),
            "usage_vector": cuda(torch.zeros(batch_size, self.nr_cells), device=self.device),
        }

    def clone(self, hidden):
        cloned = {}
        for vector in [
            "memory",
            "link_matrix",
            "precedence",
            "read_weights",
            "write_weights",
            "usage_vector",
        ]:
            cloned[vector] = hidden[vector].clone()
        return cloned

    def erase(self, hidden):
        hidden["memory"].data.zero_()
        hidden["link_matrix"].data.zero_()
        hidden["precedence"].data.zero_()
        hidden["read_weights"].data.zero_()
        hidden["write_weights"].data.zero_()
        hidden["usage_vector"].data.zero_()
        return hidden

    def reset(self, batch_size=1, hidden=None, erase=True):
        if hidden is None:
            return self.new(batch_size)
        else:
            hidden = self.clone(hidden)
            if erase:
                hidden = self.erase(hidden)
        return hidden

    def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
        usage = usage + (1 - usage) * (1 - torch.prod(1 - write_weights, 1))
        retention_vector = torch.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
        return usage * retention_vector

    def allocate(self, usage, write_gate):
        usage = δ + (1 - δ) * usage
        batch_size = usage.size(0)
        sorted_usage, φ = torch.topk(usage, self.nr_cells, dim=1, largest=False)

        v = torch.ones((batch_size, 1), device=usage.device)
        cat_sorted_usage = torch.cat((v, sorted_usage), 1)
        prod_sorted_usage = torch.cumprod(cat_sorted_usage, 1)[:, :-1]

        sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

        _, φ_rev = torch.topk(φ, k=self.nr_cells, dim=1, largest=False)
        allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())

        return allocation_weights.unsqueeze(1), usage

    def write_weighting(
        self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate
    ):
        ag = allocation_gate.unsqueeze(-1)
        wg = write_gate.unsqueeze(-1)
        return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)

    def get_link_matrix(self, link_matrix, write_weights, precedence):
        precedence = precedence.unsqueeze(2)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)

        prev_scale = 1 - write_weights_i - write_weights_j
        new_link_matrix = write_weights_i * precedence

        link_matrix = prev_scale * link_matrix + new_link_matrix
        return self.I.expand_as(link_matrix) * link_matrix

    def update_precedence(self, precedence, write_weights):
        return (1 - torch.sum(write_weights, 2, keepdim=True)) * precedence + write_weights

    def write(
        self,
        write_key,
        write_vector,
        erase_vector,
        free_gates,
        read_strengths,
        write_strength,
        write_gate,
        allocation_gate,
        hidden,
    ):
        hidden["usage_vector"] = self.get_usage_vector(
            hidden["usage_vector"], free_gates, hidden["read_weights"], hidden["write_weights"]
        )

        write_content_weights = self.content_weightings(hidden["memory"], write_key, write_strength)

        alloc, _ = self.allocate(hidden["usage_vector"], allocation_gate * write_gate)

        hidden["write_weights"] = self.write_weighting(
            hidden["memory"], write_content_weights, alloc, write_gate, allocation_gate
        )

        weighted_resets = hidden["write_weights"].unsqueeze(3) * erase_vector.unsqueeze(2)
        reset_gate = torch.prod(1 - weighted_resets, 1)
        hidden["memory"] = hidden["memory"] * reset_gate

        hidden["memory"] = hidden["memory"] + torch.bmm(
            hidden["write_weights"].transpose(1, 2), write_vector
        )

        hidden["link_matrix"] = self.get_link_matrix(
            hidden["link_matrix"], hidden["write_weights"], hidden["precedence"]
        )
        hidden["precedence"] = self.update_precedence(hidden["precedence"], hidden["write_weights"])

        return hidden

    def content_weightings(self, memory, keys, strengths):
        d = θ(memory, keys)
        return σ(d * strengths.unsqueeze(2), 2)

    def directional_weightings(self, link_matrix, read_weights):
        rw = read_weights.unsqueeze(1)
        f = torch.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
        b = torch.matmul(rw, link_matrix)
        return f.transpose(1, 2), b.transpose(1, 2)

    def read_weightings(self, memory, content_weights, link_matrix, read_modes, read_weights):
        forward_weight, backward_weight = self.directional_weightings(link_matrix, read_weights)

        content_mode = read_modes[:, :, 2].contiguous().unsqueeze(2) * content_weights
        backward_mode = torch.sum(
            read_modes[:, :, 0:1].contiguous().unsqueeze(3) * backward_weight, 2
        )
        forward_mode = torch.sum(
            read_modes[:, :, 1:2].contiguous().unsqueeze(3) * forward_weight, 2
        )

        return backward_mode + content_mode + forward_mode

    def read_vectors(self, memory, read_weights):
        return torch.bmm(read_weights, memory)

    def read(self, read_keys, read_strengths, read_modes, hidden):
        content_weights = self.content_weightings(hidden["memory"], read_keys, read_strengths)

        hidden["read_weights"] = self.read_weightings(
            hidden["memory"],
            content_weights,
            hidden["link_matrix"],
            read_modes,
            hidden["read_weights"],
        )
        read_vectors = self.read_vectors(hidden["memory"], hidden["read_weights"])
        return read_vectors, hidden

    def forward(self, ξ, hidden):  # noqa: N803
        m = self.nr_cells  # noqa: F841 (kept for parity with upstream, unused there too)
        w = self.cell_size
        r = self.read_heads
        b = ξ.size()[0]

        if self.independent_linears:
            read_keys = torch.tanh(self.read_keys_transform(ξ).view(b, r, w))
            read_strengths = F.softplus(self.read_strengths_transform(ξ).view(b, r))
            write_key = torch.tanh(self.write_key_transform(ξ).view(b, 1, w))
            write_strength = F.softplus(self.write_strength_transform(ξ).view(b, 1))
            erase_vector = torch.sigmoid(self.erase_vector_transform(ξ).view(b, 1, w))
            write_vector = torch.tanh(self.write_vector_transform(ξ).view(b, 1, w))
            free_gates = torch.sigmoid(self.free_gates_transform(ξ).view(b, r))
            allocation_gate = torch.sigmoid(self.allocation_gate_transform(ξ).view(b, 1))
            write_gate = torch.sigmoid(self.write_gate_transform(ξ).view(b, 1))
            read_modes = σ(self.read_modes_transform(ξ).view(b, r, 3), -1)
        else:
            ξ = self.interface_weights(ξ)
            read_keys = torch.tanh(ξ[:, : r * w].contiguous().view(b, r, w))
            read_strengths = F.softplus(ξ[:, r * w : r * w + r].contiguous().view(b, r))
            write_key = torch.tanh(ξ[:, r * w + r : r * w + r + w].contiguous().view(b, 1, w))
            write_strength = F.softplus(ξ[:, r * w + r + w].contiguous().view(b, 1))
            erase_vector = torch.sigmoid(
                ξ[:, r * w + r + w + 1 : r * w + r + 2 * w + 1].contiguous().view(b, 1, w)
            )
            write_vector = torch.tanh(
                ξ[:, r * w + r + 2 * w + 1 : r * w + r + 3 * w + 1].contiguous().view(b, 1, w)
            )
            free_gates = torch.sigmoid(
                ξ[:, r * w + r + 3 * w + 1 : r * w + 2 * r + 3 * w + 1].contiguous().view(b, r)
            )
            allocation_gate = torch.sigmoid(
                ξ[:, r * w + 2 * r + 3 * w + 1].contiguous().unsqueeze(1).view(b, 1)
            )
            write_gate = (
                torch.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 2].contiguous()).unsqueeze(1).view(b, 1)
            )
            read_modes = σ(
                ξ[:, r * w + 2 * r + 3 * w + 3 : r * w + 5 * r + 3 * w + 3]
                .contiguous()
                .view(b, r, 3),
                -1,
            )

        hidden = self.write(
            write_key,
            write_vector,
            erase_vector,
            free_gates,
            read_strengths,
            write_strength,
            write_gate,
            allocation_gate,
            hidden,
        )
        return self.read(read_keys, read_strengths, read_modes, hidden)


# --- from dnc/dnc.py (verbatim; SAM/SDNC/SparseMemory/SparseTemporalMemory branches
# dropped -- unreachable at default share_memory_between_layers=True, non-sparse config) ---
class DNC(nn.Module):
    """Differentiable neural computer."""

    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type="lstm",
        num_layers=1,
        num_hidden_layers=2,
        bias=True,
        batch_first=True,
        dropout=0,
        nr_cells=5,
        read_heads=2,
        cell_size=10,
        nonlinearity="tanh",
        independent_linears=False,
        share_memory_between_layers=True,
        debug=False,
        clip=20,
        device=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.nr_cells = nr_cells
        self.read_heads = read_heads
        self.cell_size = cell_size
        self.nonlinearity = nonlinearity
        self.independent_linears = independent_linears
        self.share_memory_between_layers = share_memory_between_layers
        self.debug = debug
        self.clip = clip
        self.device = device

        self.w = self.cell_size
        self.r = self.read_heads

        self.read_vectors_size = self.read_heads * self.cell_size
        self.output_size = self.hidden_size

        self.nn_input_size = self.input_size + self.read_vectors_size
        self.nn_output_size = self.output_size + self.read_vectors_size

        self.rnns = []
        self.memories = []

        for layer in range(self.num_layers):
            if self.rnn_type.lower() == "rnn":
                self.rnns.append(
                    nn.RNN(
                        (self.nn_input_size if layer == 0 else self.nn_output_size),
                        self.output_size,
                        bias=self.bias,
                        nonlinearity=self.nonlinearity,
                        batch_first=True,
                        dropout=self.dropout,
                        num_layers=self.num_hidden_layers,
                    )
                )
            elif self.rnn_type.lower() == "gru":
                self.rnns.append(
                    nn.GRU(
                        (self.nn_input_size if layer == 0 else self.nn_output_size),
                        self.output_size,
                        bias=self.bias,
                        batch_first=True,
                        dropout=self.dropout,
                        num_layers=self.num_hidden_layers,
                    )
                )
            elif self.rnn_type.lower() == "lstm":
                self.rnns.append(
                    nn.LSTM(
                        (self.nn_input_size if layer == 0 else self.nn_output_size),
                        self.output_size,
                        bias=self.bias,
                        batch_first=True,
                        dropout=self.dropout,
                        num_layers=self.num_hidden_layers,
                    )
                )
            setattr(self, self.rnn_type.lower() + "_layer_" + str(layer), self.rnns[layer])

            if not self.share_memory_between_layers:
                self.memories.append(
                    Memory(
                        input_size=self.output_size,
                        nr_cells=self.nr_cells,
                        cell_size=self.w,
                        read_heads=self.r,
                        device=self.device,
                        independent_linears=self.independent_linears,
                    )
                )
                setattr(self, "rnn_layer_memory_" + str(layer), self.memories[layer])

        if self.share_memory_between_layers:
            self.memories.append(
                Memory(
                    input_size=self.output_size,
                    nr_cells=self.nr_cells,
                    cell_size=self.w,
                    read_heads=self.r,
                    device=self.device,
                    independent_linears=self.independent_linears,
                )
            )
            setattr(self, "rnn_layer_memory_shared", self.memories[0])

        self.output = nn.Linear(self.nn_output_size, self.input_size)
        torch.nn.init.kaiming_uniform_(self.output.weight)

        if self.device is not None and self.device.type == "cuda":
            self.to(self.device)

    def _init_hidden(self, hx, batch_size, reset_experience):
        if hx is not None:
            chx, mhx, last_read = hx
        else:
            chx, mhx, last_read = None, None, None

        if chx is None:
            h = cuda(
                torch.zeros(self.num_hidden_layers, batch_size, self.output_size),
                device=self.device,
            )
            torch.nn.init.xavier_uniform_(h)
            chx = [(h, h) if self.rnn_type.lower() == "lstm" else h for _ in range(self.num_layers)]

        if last_read is None:
            last_read = cuda(torch.zeros(batch_size, self.w * self.r), device=self.device)

        if mhx is None:
            if self.share_memory_between_layers:
                mhx = [self.memories[0].reset(batch_size, erase=reset_experience)]
            else:
                mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
        else:
            if self.share_memory_between_layers:
                if len(mhx) == 0 or mhx[0] is None:
                    mhx = [self.memories[0].reset(batch_size, erase=reset_experience)]
                else:
                    mhx = [self.memories[0].reset(batch_size, mhx[0], erase=reset_experience)]
            else:
                if len(mhx) == 0:
                    mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
                else:
                    new_mhx = []
                    for i, m in enumerate(self.memories):
                        if i < len(mhx) and mhx[i] is not None:
                            new_mhx.append(m.reset(batch_size, mhx[i], erase=reset_experience))
                        else:
                            new_mhx.append(m.reset(batch_size, erase=reset_experience))
                    mhx = new_mhx

        return chx, mhx, last_read

    def _debug(self, mhx, debug_obj):
        if not self.debug:
            return None

        if not debug_obj:
            debug_obj = {
                "memory": [],
                "link_matrix": [],
                "precedence": [],
                "read_weights": [],
                "write_weights": [],
                "usage_vector": [],
            }

        debug_obj["memory"].append(mhx["memory"][0].detach().cpu().numpy())
        debug_obj["link_matrix"].append(mhx["link_matrix"][0][0].detach().cpu().numpy())
        debug_obj["precedence"].append(mhx["precedence"][0].detach().cpu().numpy())
        debug_obj["read_weights"].append(mhx["read_weights"][0].detach().cpu().numpy())
        debug_obj["write_weights"].append(mhx["write_weights"][0].detach().cpu().numpy())
        debug_obj["usage_vector"].append(mhx["usage_vector"][0].unsqueeze(0).detach().cpu().numpy())
        return debug_obj

    def _layer_forward(self, input, layer, hx, pass_through_memory=True):
        (chx, mhx, _) = hx

        input, chx = self.rnns[layer](input.unsqueeze(1), chx)
        input = input.squeeze(1)

        if self.clip != 0:
            output = torch.clamp(input, -self.clip, self.clip)
        else:
            output = input

        ξ = output

        if pass_through_memory:
            if self.share_memory_between_layers:
                read_vecs, mhx = self.memories[0](ξ, mhx)
            else:
                read_vecs, mhx = self.memories[layer](ξ, mhx)
            read_vectors = read_vecs.view(-1, self.w * self.r)
        else:
            read_vectors = cuda(torch.zeros(ξ.size(0), self.w * self.r), device=self.device)

        return output, (chx, mhx, read_vectors)

    def forward(self, input_data, hx=None, reset_experience=False, pass_through_memory=True):
        max_length: int
        if isinstance(input_data, PackedSequence):
            input, lengths = pad_packed_sequence(input_data, batch_first=self.batch_first)
            max_length = int(lengths.max().item())
        elif isinstance(input_data, torch.Tensor):
            input = input_data
            batch_size = input.size(0) if self.batch_first else input.size(1)
            max_length = input.size(1) if self.batch_first else input.size(0)
            lengths = torch.tensor([max_length] * batch_size, device=input.device)
        else:
            raise TypeError("input_data must be a PackedSequence or Tensor")

        if not self.batch_first:
            input = input.transpose(0, 1)

        controller_hidden, mem_hidden, last_read = self._init_hidden(
            hx, batch_size, reset_experience
        )

        inputs = [torch.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

        if self.debug:
            viz: dict[str, Any] | None = None

        outs = [None] * max_length
        read_vectors = None

        for time in range(max_length):
            for layer in range(self.num_layers):
                chx_layer = controller_hidden[layer]
                mem_layer = mem_hidden[0] if self.share_memory_between_layers else mem_hidden[layer]

                outs[time], (chx_layer_output, mem_layer_output, read_vectors) = (
                    self._layer_forward(
                        inputs[time],
                        layer,
                        (chx_layer, mem_layer, read_vectors),
                        pass_through_memory,
                    )
                )

                if self.debug:
                    viz = self._debug(mem_layer_output, viz)

                if self.share_memory_between_layers:
                    mem_hidden[0] = mem_layer_output
                else:
                    mem_hidden[layer] = mem_layer_output
                controller_hidden[layer] = chx_layer_output

                if read_vectors is not None:
                    outs[time] = torch.cat([outs[time], read_vectors], 1)
                else:
                    outs[time] = torch.cat([outs[time], last_read], 1)
                inputs[time] = outs[time]

        if self.debug and viz:
            viz = {k: [np.array(v) for v in vs] for k, vs in viz.items()}
            viz = {k: [v.reshape(v.shape[0], -1) for v in vs] for k, vs in viz.items()}

        inputs_tensor = torch.stack(inputs)
        outputs = self.output(inputs_tensor)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)

        if isinstance(input_data, PackedSequence):
            outputs = pack_padded_sequence(
                outputs, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False
            )

        if self.debug:
            return outputs, (controller_hidden, mem_hidden, read_vectors), viz
        else:
            return outputs, (controller_hidden, mem_hidden, read_vectors)


class _DNCWrapper(nn.Module):
    """Thin wrapper so tl.trace can call the module with a single positional tensor
    argument; the real DNC.forward(input_data, hx=None, ...) already defaults hx=None,
    so this simply forwards to it (kept as an explicit module for clarity/tracing)."""

    def __init__(self, dnc):
        super().__init__()
        self.dnc = dnc

    def forward(self, input_data):
        outputs, _hidden = self.dnc(input_data, None)
        return outputs


def build_dnc():
    # Tiny DNC: small controller + small memory (nr_cells x cell_size), single layer.
    return _DNCWrapper(
        DNC(
            input_size=8,
            hidden_size=16,
            rnn_type="lstm",
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=8,
            cell_size=6,
            read_heads=2,
        )
    )


def example_input_dnc():
    # (batch, seq_len, input_size) -- batch_first=True default.
    return torch.randn(2, 5, 8)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Differentiable Neural Computer (DNC) RL-agent controller",
        build_dnc,
        example_input_dnc,
        2016,
        MENAGERIE_ZOO,
    ),
]

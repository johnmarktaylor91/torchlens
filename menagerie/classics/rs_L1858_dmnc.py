# FAITHFUL PORT of thaihungle/DMNC @ master (original framework: TensorFlow 1.x)
#
# Source files transcribed:
#   memory.py       -> DNC external-memory read/write mechanics (content lookup,
#                       dynamic-allocation write weighting, temporal link matrix,
#                       forward/backward/content read-mode blending) -- Graves et al.
#                       "Differentiable Neural Computer" (Nature 2016), as implemented
#                       by thaihungle's DMNC repo.
#   controller.py, recurrent_controller.py
#                   -> LSTM controller wrapping the memory: parses the flat interface
#                       vector into read/write keys, strengths, gates, and read modes.
#   dual_dnc.py     -> Dual_DNC: two independent DNC memories/controllers encode two
#                       parallel input streams (diagnosis codes, procedure codes); a
#                       third "decoder" DNC controller (is_two_mem=2, double-width
#                       LSTM state formed by concatenating both encoder states) reads
#                       from BOTH memories and produces the medication-recommendation
#                       output. This is the DMNC architecture from Le, Tran & Venkatesh,
#                       "Dual Memory Neural Computer for Asynchronous Two-view
#                       Sequential Learning" (KDD 2018).
#
# The original repo is graph-mode TensorFlow 1.x (tf.placeholder / tf.get_variable /
# tf.variable_scope / tf.cond / tf.while_loop) and cannot run in a torch base env.
# This port transcribes the SAME memory equations and the SAME dual-encoder ->
# state-concatenation -> shared-decoder control flow into eager torch, unrolled with a
# plain Python time loop (replacing the TF tf.while_loop/tf.cond step machinery with
# ordinary Python control flow over a fixed-length step schedule, which is
# semantically identical for a fixed, statically-known sequence length). Encoder time
# gating (`encode1_point`/`encode2_point`) and `write_protect` (freeze memory writes
# once decoding starts) are preserved. Attention-over-encoder-history
# (`attend_dim > 0`), beam search, persist mode, and teacher forcing are training/
# inference infrastructure in the original repo, not part of the core dual-memory
# architecture, and are intentionally not ported (attend_dim=0 / greedy single pass,
# matching the repo's own default `args.attend=0` training configuration).

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DNCMemory(nn.Module):
    """External read/write memory bank, per memory.py in the original repo."""

    def __init__(self, words_num=16, word_size=8, read_heads=2, batch_size=1):
        super().__init__()
        self.words_num = words_num
        self.word_size = word_size
        self.read_heads = read_heads
        self.batch_size = batch_size
        self.register_buffer("I", torch.eye(words_num))

    def init_state(self, device, dtype):
        b, n, w, r = self.batch_size, self.words_num, self.word_size, self.read_heads
        return dict(
            memory=torch.full((b, n, w), 1e-6, device=device, dtype=dtype),
            usage=torch.zeros(b, n, device=device, dtype=dtype),
            precedence=torch.zeros(b, n, device=device, dtype=dtype),
            link=torch.zeros(b, n, n, device=device, dtype=dtype),
            write_weighting=torch.full((b, n), 1e-6, device=device, dtype=dtype),
            read_weightings=torch.full((b, n, r), 1e-6, device=device, dtype=dtype),
            read_vectors=torch.full((b, w, r), 1e-6, device=device, dtype=dtype),
        )

    @staticmethod
    def get_lookup_weighting(memory_matrix, keys, strengths):
        # cosine-similarity content addressing, softmax over memory slots
        normalized_memory = nn.functional.normalize(memory_matrix, dim=2)
        normalized_keys = nn.functional.normalize(keys, dim=1)
        similarity = torch.matmul(normalized_memory, normalized_keys)
        strengths = strengths.unsqueeze(1)
        return torch.softmax(similarity * strengths, dim=1)

    @staticmethod
    def update_usage_vector(usage_vector, read_weightings, write_weighting, free_gates):
        free_gates = free_gates.unsqueeze(1)
        retention_vector = torch.prod(1 - read_weightings * free_gates, dim=2)
        updated_usage = (
            usage_vector + write_weighting - usage_vector * write_weighting
        ) * retention_vector
        return updated_usage

    def get_allocation_weighting(self, sorted_usage, free_list):
        # exclusive cumulative product -> allocation weighting, then scatter back
        # to original (unsorted) memory-slot order via free_list indices.
        shifted_cumprod = torch.cumprod(
            torch.cat([torch.ones_like(sorted_usage[:, :1]), sorted_usage[:, :-1]], dim=1), dim=1
        )
        unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod
        allocation_weighting = torch.zeros_like(unordered_allocation_weighting)
        allocation_weighting.scatter_(1, free_list, unordered_allocation_weighting)
        return allocation_weighting

    @staticmethod
    def update_write_weighting(lookup_weighting, allocation_weighting, write_gate, allocation_gate):
        lookup_weighting = lookup_weighting.squeeze(-1)
        return write_gate * (
            allocation_gate * allocation_weighting + (1 - allocation_gate) * lookup_weighting
        )

    @staticmethod
    def update_memory(memory_matrix, write_weighting, write_vector, erase_vector):
        write_weighting = write_weighting.unsqueeze(2)
        write_vector = write_vector.unsqueeze(1)
        erase_vector = erase_vector.unsqueeze(1)
        erasing = memory_matrix * (1 - torch.matmul(write_weighting, erase_vector))
        writing = torch.matmul(write_weighting, write_vector)
        return erasing + writing

    @staticmethod
    def update_precedence_vector(precedence_vector, write_weighting):
        reset_factor = 1 - write_weighting.sum(dim=1, keepdim=True)
        return reset_factor * precedence_vector + write_weighting

    def update_link_matrix(self, precedence_vector, link_matrix, write_weighting):
        ww = write_weighting.unsqueeze(2)
        pv = precedence_vector.unsqueeze(1)
        reset_factor = 1 - (ww + ww.transpose(1, 2))
        updated_link_matrix = reset_factor * link_matrix + torch.matmul(ww, pv)
        updated_link_matrix = (1 - self.I) * updated_link_matrix
        return updated_link_matrix

    @staticmethod
    def get_directional_weightings(read_weightings, link_matrix):
        forward_weighting = torch.matmul(link_matrix, read_weightings)
        backward_weighting = torch.matmul(link_matrix.transpose(1, 2), read_weightings)
        return forward_weighting, backward_weighting

    @staticmethod
    def update_read_weightings(lookup_weightings, forward_weighting, backward_weighting, read_mode):
        backward_mode = read_mode[:, 0, :].unsqueeze(1) * backward_weighting
        lookup_mode = read_mode[:, 1, :].unsqueeze(1) * lookup_weightings
        forward_mode = read_mode[:, 2, :].unsqueeze(1) * forward_weighting
        return backward_mode + lookup_mode + forward_mode

    @staticmethod
    def update_read_vectors(memory_matrix, read_weightings):
        return torch.matmul(memory_matrix.transpose(1, 2), read_weightings)

    def write(
        self,
        memory_matrix,
        usage_vector,
        read_weightings,
        write_weighting,
        precedence_vector,
        link_matrix,
        key,
        strength,
        free_gates,
        allocation_gate,
        write_gate,
        write_vector,
        erase_vector,
    ):
        lookup_weighting = self.get_lookup_weighting(memory_matrix, key, strength)
        new_usage_vector = self.update_usage_vector(
            usage_vector, read_weightings, write_weighting, free_gates
        )

        sorted_usage, free_list = torch.topk(-new_usage_vector, self.words_num, dim=1)
        sorted_usage = -sorted_usage

        allocation_weighting = self.get_allocation_weighting(sorted_usage, free_list)
        new_write_weighting = self.update_write_weighting(
            lookup_weighting, allocation_weighting, write_gate, allocation_gate
        )
        new_memory_matrix = self.update_memory(
            memory_matrix, new_write_weighting, write_vector, erase_vector
        )
        new_link_matrix = self.update_link_matrix(
            precedence_vector, link_matrix, new_write_weighting
        )
        new_precedence_vector = self.update_precedence_vector(
            precedence_vector, new_write_weighting
        )

        return (
            new_usage_vector,
            new_write_weighting,
            new_memory_matrix,
            new_link_matrix,
            new_precedence_vector,
        )

    def read(self, memory_matrix, read_weightings, keys, strengths, link_matrix, read_modes):
        lookup_weighting = self.get_lookup_weighting(memory_matrix, keys, strengths)
        forward_weighting, backward_weighting = self.get_directional_weightings(
            read_weightings, link_matrix
        )
        new_read_weightings = self.update_read_weightings(
            lookup_weighting, forward_weighting, backward_weighting, read_modes
        )
        new_read_vectors = self.update_read_vectors(memory_matrix, new_read_weightings)
        return new_read_weightings, new_read_vectors


class DNCController(nn.Module):
    """LSTM controller + interface-vector parsing, per controller.py /
    recurrent_controller.py (StatelessRecurrentController) in the original repo."""

    def __init__(
        self,
        input_size,
        output_size,
        read_heads,
        word_size,
        batch_size=1,
        hidden_dim=32,
        is_two_mem=0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.read_heads = read_heads
        self.word_size = word_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.is_two_mem = is_two_mem

        nn_input_size = word_size * read_heads * (2 if is_two_mem > 0 else 1) + input_size
        self.lstm_cell = nn.LSTMCell(nn_input_size, hidden_dim)

        interface_vector_size = word_size * read_heads  # read keys
        interface_vector_size += 3 * word_size  # write key, erase, write vector
        interface_vector_size += 5 * read_heads  # read strengths, free gates, 3x read modes
        interface_vector_size += 3  # write strength, allocation gate, write gate
        self.interface_vector_size = interface_vector_size

        out_mult = 2 if is_two_mem == 2 else 1
        self.interface_weights = nn.Linear(hidden_dim, interface_vector_size * out_mult, bias=False)
        nn.init.normal_(self.interface_weights.weight, std=0.1)
        self.nn_output_weights = nn.Linear(hidden_dim, output_size, bias=False)
        nn.init.normal_(self.nn_output_weights.weight, std=0.1)
        mem_out_mult = 2 if is_two_mem > 0 else 1
        self.mem_output_weights = nn.Linear(
            word_size * read_heads * mem_out_mult, output_size, bias=False
        )
        nn.init.normal_(self.mem_output_weights.weight, std=0.1)

    def parse_interface_vector(self, interface_vector):
        w, r = self.word_size, self.read_heads
        b = interface_vector.shape[0]
        r_keys_end = w * r
        r_strengths_end = r_keys_end + r
        w_key_end = r_strengths_end + w
        erase_end = w_key_end + 1 + w
        write_end = erase_end + w
        free_end = write_end + r

        read_keys = interface_vector[:, :r_keys_end].reshape(b, w, r)
        read_strengths = interface_vector[:, r_keys_end:r_strengths_end].reshape(b, r)
        write_key = interface_vector[:, r_strengths_end:w_key_end].reshape(b, w, 1)
        write_strength = interface_vector[:, w_key_end : w_key_end + 1].reshape(b, 1)
        erase_vector = interface_vector[:, w_key_end + 1 : erase_end].reshape(b, w)
        write_vector = interface_vector[:, erase_end:write_end].reshape(b, w)
        free_gates = interface_vector[:, write_end:free_end].reshape(b, r)
        allocation_gate = interface_vector[:, free_end : free_end + 1]
        write_gate = interface_vector[:, free_end + 1 : free_end + 2]
        read_modes = interface_vector[:, free_end + 2 :].reshape(b, 3, r)

        return dict(
            read_keys=read_keys,
            read_strengths=1 + nn.functional.softplus(read_strengths),
            write_key=write_key,
            write_strength=1 + nn.functional.softplus(write_strength),
            erase_vector=torch.sigmoid(erase_vector),
            write_vector=write_vector,
            free_gates=torch.sigmoid(free_gates),
            allocation_gate=torch.sigmoid(allocation_gate),
            write_gate=torch.sigmoid(write_gate),
            read_modes=torch.softmax(read_modes, dim=1),
        )

    def process_input(self, x, last_read_vectors, state):
        flat_read = last_read_vectors.reshape(last_read_vectors.shape[0], -1)
        complete_input = torch.cat([x, flat_read], dim=1)
        h, c = self.lstm_cell(complete_input, state)
        pre_output = self.nn_output_weights(h)
        interface = self.interface_weights(h)
        if self.is_two_mem == 2:
            interface1, interface2 = torch.chunk(interface, 2, dim=-1)
            parsed = (
                self.parse_interface_vector(interface1),
                self.parse_interface_vector(interface2),
            )
        else:
            parsed = self.parse_interface_vector(interface)
        return pre_output, parsed, (h, c)

    def final_output(self, pre_output, new_read_vectors):
        flat_read = new_read_vectors.reshape(new_read_vectors.shape[0], -1)
        return pre_output + self.mem_output_weights(flat_read)


class DualDNC(nn.Module):
    """Dual-memory neural computer: two encoder DNCs (diagnosis / procedure code
    streams) feeding a shared decoder DNC that reads from both memories, per
    dual_dnc.py::Dual_DNC (write_protect=True, attend_dim=0 configuration)."""

    def __init__(
        self,
        input_size1=8,
        input_size2=8,
        output_size=12,
        words_num=16,
        word_size=8,
        read_heads=2,
        batch_size=1,
        hidden_dim=32,
        emb_size=16,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.words_num = words_num
        self.word_size = word_size
        self.read_heads = read_heads

        self.emb1 = nn.Linear(input_size1, emb_size, bias=False)
        self.emb2 = nn.Linear(input_size2, emb_size, bias=False)
        self.emb_out = nn.Linear(output_size, emb_size, bias=False)
        for lin in (self.emb1, self.emb2, self.emb_out):
            nn.init.uniform_(lin.weight, -1, 1)

        self.memory1 = DNCMemory(words_num, word_size, read_heads, batch_size)
        self.memory2 = DNCMemory(words_num, word_size, read_heads, batch_size)
        self.controller1 = DNCController(
            emb_size, output_size, read_heads, word_size, batch_size, hidden_dim
        )
        self.controller2 = DNCController(
            emb_size, output_size, read_heads, word_size, batch_size, hidden_dim
        )
        self.controller3 = DNCController(
            emb_size, output_size, read_heads, word_size, batch_size, hidden_dim * 2, is_two_mem=2
        )

    def forward(self, step1_seq, step2_seq):
        """step1_seq, step2_seq: (batch, T, input_size1|2) already-embedded-shape raw
        codes (one-hot/dense feature vectors), matching the original repo's
        `input_data1`/`input_data2` placeholders. Runs the fixed encode1/encode2 ->
        decode schedule used in the repo's own training loop: encoder 1 consumes
        the first half of the sequence, encoder 2 consumes the same span in
        parallel, and the decoder (write-protected) reads both memories for the
        remaining steps."""
        b, T, _ = step1_seq.shape
        device, dtype = step1_seq.device, step1_seq.dtype
        encode_point = max(1, T // 2)

        mem1 = self.memory1.init_state(device, dtype)
        mem2 = self.memory2.init_state(device, dtype)
        h1 = torch.zeros(b, self.hidden_dim, device=device, dtype=dtype)
        c1 = torch.zeros(b, self.hidden_dim, device=device, dtype=dtype)
        h2 = torch.zeros(b, self.hidden_dim, device=device, dtype=dtype)
        c2 = torch.zeros(b, self.hidden_dim, device=device, dtype=dtype)

        outputs = []
        for t in range(T):
            step1 = self.emb1(step1_seq[:, t, :])
            step2 = self.emb2(step2_seq[:, t, :])

            if t < encode_point:
                pre1, interface1, (h1, c1) = self.controller1.process_input(
                    step1, mem1["read_vectors"], (h1, c1)
                )
                pre2, interface2, (h2, c2) = self.controller2.process_input(
                    step2, mem2["read_vectors"], (h2, c2)
                )
                pre_output = pre1 + pre2

                uv1, ww1, mm1, lm1, pv1 = self.memory1.write(
                    mem1["memory"],
                    mem1["usage"],
                    mem1["read_weightings"],
                    mem1["write_weighting"],
                    mem1["precedence"],
                    mem1["link"],
                    interface1["write_key"],
                    interface1["write_strength"],
                    interface1["free_gates"],
                    interface1["allocation_gate"],
                    interface1["write_gate"],
                    interface1["write_vector"],
                    interface1["erase_vector"],
                )
                uv2, ww2, mm2, lm2, pv2 = self.memory2.write(
                    mem2["memory"],
                    mem2["usage"],
                    mem2["read_weightings"],
                    mem2["write_weighting"],
                    mem2["precedence"],
                    mem2["link"],
                    interface2["write_key"],
                    interface2["write_strength"],
                    interface2["free_gates"],
                    interface2["allocation_gate"],
                    interface2["write_gate"],
                    interface2["write_vector"],
                    interface2["erase_vector"],
                )

                rw1, rv1 = self.memory1.read(
                    mm1,
                    mem1["read_weightings"],
                    interface1["read_keys"],
                    interface1["read_strengths"],
                    lm1,
                    interface1["read_modes"],
                )
                rw2, rv2 = self.memory2.read(
                    mm2,
                    mem2["read_weightings"],
                    interface2["read_keys"],
                    interface2["read_strengths"],
                    lm2,
                    interface2["read_modes"],
                )

                mem1 = dict(
                    memory=mm1,
                    usage=uv1,
                    precedence=pv1,
                    link=lm1,
                    write_weighting=ww1,
                    read_weightings=rw1,
                    read_vectors=rv1,
                )
                mem2 = dict(
                    memory=mm2,
                    usage=uv2,
                    precedence=pv2,
                    link=lm2,
                    write_weighting=ww2,
                    read_weightings=rw2,
                    read_vectors=rv2,
                )

                out_t = (
                    self.controller3.mem_output_weights(torch.cat([rv1, rv2], dim=1).reshape(b, -1))
                    + pre_output
                )
            else:
                # decoder step: write-protected (memory frozen), reads both memories
                dec_step = (
                    self.emb_out(
                        torch.zeros(b, self.controller3.output_size, device=device, dtype=dtype)
                    )
                    if t == encode_point
                    else self.emb_out(outputs[-1])
                )
                ncontroller_state = (torch.cat([h1, h2], dim=-1), torch.cat([c1, c2], dim=-1))
                nread_vec = torch.cat([mem1["read_vectors"], mem2["read_vectors"]], dim=1)

                pre_output, (interface1, interface2), (nh, nc) = self.controller3.process_input(
                    dec_step, nread_vec, ncontroller_state
                )
                h1, h2 = torch.chunk(nh, 2, dim=-1)
                c1, c2 = torch.chunk(nc, 2, dim=-1)

                rw1, rv1 = self.memory1.read(
                    mem1["memory"],
                    mem1["read_weightings"],
                    interface1["read_keys"],
                    interface1["read_strengths"],
                    mem1["link"],
                    interface1["read_modes"],
                )
                rw2, rv2 = self.memory2.read(
                    mem2["memory"],
                    mem2["read_weightings"],
                    interface2["read_keys"],
                    interface2["read_strengths"],
                    mem2["link"],
                    interface2["read_modes"],
                )
                mem1 = dict(mem1, read_weightings=rw1, read_vectors=rv1)
                mem2 = dict(mem2, read_weightings=rw2, read_vectors=rv2)

                out_t = self.controller3.final_output(pre_output, torch.cat([rv1, rv2], dim=1))

            outputs.append(out_t)

        return torch.stack(outputs, dim=1)


def build_dmnc():
    torch.manual_seed(0)
    return DualDNC(
        input_size1=8,
        input_size2=8,
        output_size=12,
        words_num=16,
        word_size=8,
        read_heads=2,
        batch_size=1,
        hidden_dim=32,
        emb_size=16,
    )


def example_input_dmnc():
    torch.manual_seed(0)
    b, T = 1, 6
    step1 = torch.rand(b, T, 8)
    step2 = torch.rand(b, T, 8)
    return (step1, step2)


MENAGERIE_ENTRIES = [
    ("DMNC", build_dmnc, example_input_dmnc, 2018, "ported-pytorch"),
]

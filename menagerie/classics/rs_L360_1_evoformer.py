# SOURCE: vendored from aqlaboratory/openfold @ main
#
# EvoFormer (AlphaFold2 trunk module) -- row/column-wise gated multi-head attention
# over the MSA representation plus triangular multiplicative updates / triangular
# self-attention over the pair representation. This file concatenates the REAL
# openfold model-definition modules (openfold/model/{primitives,dropout,msa,
# outer_product_mean,pair_transition,triangular_attention,
# triangular_multiplicative_update,evoformer}.py and openfold/utils/{tensor_utils,
# chunk_utils,checkpointing,precision_utils}.py) verbatim into a single staging
# module. The only changes made: (1) internal `from openfold.xxx import yyy`
# cross-file imports are stripped since everything now lives in one namespace,
# (2) the top-of-file `import openfold.utils.kernel.attention_core` (a custom CUDA
# extension not available outside a compiled openfold install) is dropped along
# with the single call site that used it, forcing the pure-PyTorch `_attention()`
# fallback that the real code already takes whenever `use_deepspeed_evo_attention`,
# `use_cuequivariance_attention`, `use_lma`, and `use_flash` are all left at their
# real default of False. No architectural code was rewritten.
#
# Upstream license: Apache-2.0 (Copyright 2021 AlQuraishi Laboratory,
# Copyright 2021 DeepMind Technologies Limited, Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES).

import importlib
import logging
import math
import sys
from abc import ABC, abstractmethod
from functools import partial, partialmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from scipy.stats import truncnorm
from torch.fx._symbolic_trace import is_fx_tracing

MENAGERIE_ZOO = "vendored-pytorch"

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed


# ============================================================================
# --- source: openfold/utils/tensor_utils.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def maybe_to(x, dtype):
    return (
        x.to(dtype=dtype)
        if x is not None and x.dtype in [torch.float32, torch.float16, torch.bfloat16]
        else x
    )


def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    num_first_dims = len(tensor.shape) - len(inds)
    first_inds = list(range(num_first_dims))
    return tensor.permute(first_inds + [num_first_dims + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pts_to_distogram(pts, min_bin=2.3125, max_bin=21.6875, no_bins=64):
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=pts.device)
    dists = torch.sqrt(torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
    return torch.bucketize(dists, boundaries)


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


# ============================================================================
# --- source: openfold/utils/chunk_utils.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")

    return shapes


@torch.jit.ignore
def _flat_idx_to_idx(
    flat_idx: int,
    dims: Tuple[int],
) -> Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d

    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: int,
    start_edges: Optional[Sequence[bool]] = None,
    end_edges: Optional[Sequence[bool]] = None,
) -> Sequence[Tuple[int]]:
    """
    Produces an ordered sequence of tensor slices that, when used in
    sequence on a tensor with shape dims, yields tensors that contain every
    leaf in the contiguous range [start, end]. Care is taken to yield a
    short sequence of slices, and perhaps even the shortest possible (I'm
    pretty sure it's the latter).

    end is INCLUSIVE.
    """

    # start_edges and end_edges both indicate whether, starting from any given
    # dimension, the start/end index is at the top/bottom edge of the
    # corresponding tensor, modeled as a tree
    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]

    if start_edges is None:
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)

    # Base cases. Either start/end are empty and we're done, or the final,
    # one-dimensional tensor can be simply sliced
    if len(start) == 0:
        return [tuple()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    slices = []
    path = []

    # Dimensions common to start and end can be selected directly
    for s, e in zip(start, end):
        if s == e:
            path.append(slice(s, s + 1))
        else:
            break

    path = tuple(path)
    divergence_idx = len(path)

    # start == end, and we're done
    if divergence_idx == len(dims):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [
            path + (slice(sdi, sdi + 1),) + s
            for s in _get_minimal_slice_set(
                start[divergence_idx + 1 :],
                [d - 1 for d in dims[divergence_idx + 1 :]],
                dims[divergence_idx + 1 :],
                start_edges=start_edges[divergence_idx + 1 :],
                end_edges=[1 for _ in end_edges[divergence_idx + 1 :]],
            )
        ]

    def lower():
        edi = end[divergence_idx]
        return [
            path + (slice(edi, edi + 1),) + s
            for s in _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1 :]],
                end[divergence_idx + 1 :],
                dims[divergence_idx + 1 :],
                start_edges=[1 for _ in start_edges[divergence_idx + 1 :]],
                end_edges=end_edges[divergence_idx + 1 :],
            )
        ]

    # If both start and end are at the edges of the subtree rooted at
    # divergence_idx, we can just select the whole subtree at once
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    # If just start is at the edge, we can grab almost all of the subtree,
    # treating only the ragged bottom edge as an edge case
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    # Analogous to the previous case, but the top is ragged this time
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    # If both sides of the range are ragged, we need to handle both sides
    # separately. If there's contiguous meat in between them, we can index it
    # in one big chunk
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())

    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(
    t: torch.Tensor,
    flat_start: int,
    flat_end: int,
    no_batch_dims: int,
) -> torch.Tensor:
    """
    Equivalent to

        t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

    but without the need for the initial reshape call, which can be
    memory-intensive in certain situations. The only reshape operations
    in this function are performed on sub-tensors that scale with
    (flat_end - flat_start), the chunk size.
    """

    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # _get_minimal_slice_set is inclusive
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # Get an ordered list of slices to perform
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    sliced_tensors = [t[s] for s in slices]

    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple "pytrees,"
    consisting only of (arbitrarily nested) lists, tuples, and dicts with
    torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must
            be tensors and must share the same batch dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch
            dimensions are specified, a "sub-batch" is defined as a single
            indexing of all batch dimensions simultaneously (s.t. the
            number of sub-batches is the product of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can
            be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary
            in most cases, and is ever so slightly slower than the default
            setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)
    if no_chunks == 1:
        return layer(**inputs)

    def _prep_inputs(t):
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    prepped_inputs = tensor_tree_map(_prep_inputs, inputs)
    prepped_outputs = None
    if _out is not None:
        reshape_fn = lambda t: t.view([-1] + list(t.shape[no_batch_dims:]))
        prepped_outputs = tensor_tree_map(reshape_fn, _out)

    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        # Chunk the input
        if not low_mem:
            select_chunk = lambda t: t[i : i + chunk_size] if t.shape[0] != 1 else t
        else:
            select_chunk = partial(
                _chunk_slice,
                flat_start=i,
                flat_end=min(flat_batch_dim, i + chunk_size),
                no_batch_dims=len(orig_batch_dims),
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:

            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        if _add_into_out:
                            v[i : i + chunk_size] += d2[k]
                        else:
                            v[i : i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i : i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            if _add_into_out:
                out[i : i + chunk_size] += output_chunk
            else:
                out[i : i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out


class ChunkSizeTuner:
    def __init__(
        self,
        # Heuristically, runtimes for most of the modules in the network
        # plateau earlier than this on all GPUs I've run the model on.
        max_chunk_size=512,
    ):
        self.max_chunk_size = max_chunk_size
        self.cached_chunk_size = None
        self.cached_arg_data = None

    def _determine_favorable_chunk_size(self, fn, args, min_chunk_size):
        logging.info("Tuning chunk size...")

        if min_chunk_size >= self.max_chunk_size:
            return min_chunk_size

        candidates = [2**l for l in range(int(math.log(self.max_chunk_size, 2)) + 1)]
        candidates = [c for c in candidates if c > min_chunk_size]
        candidates = [min_chunk_size] + candidates
        candidates[-1] += 4

        def test_chunk_size(chunk_size):
            try:
                with torch.no_grad():
                    fn(*args, chunk_size=chunk_size)
                return True
            except RuntimeError:
                return False

        min_viable_chunk_size_index = 0
        i = len(candidates) - 1
        while i > min_viable_chunk_size_index:
            viable = test_chunk_size(candidates[i])
            if not viable:
                i = (min_viable_chunk_size_index + i) // 2
            else:
                min_viable_chunk_size_index = i
                i = (i + len(candidates) - 1) // 2

        return candidates[min_viable_chunk_size_index]

    def _compare_arg_caches(self, ac1, ac2):
        consistent = True
        for a1, a2 in zip(ac1, ac2):
            assert type(ac1) == type(ac2)
            if type(ac1) is list or type(ac1) is tuple:
                consistent &= self._compare_arg_caches(a1, a2)
            elif type(ac1) is dict:
                a1_items = [v for _, v in sorted(a1.items(), key=lambda x: x[0])]
                a2_items = [v for _, v in sorted(a2.items(), key=lambda x: x[0])]
                consistent &= self._compare_arg_caches(a1_items, a2_items)
            else:
                consistent &= a1 == a2

        return consistent

    def tune_chunk_size(
        self,
        representative_fn: Callable,
        args: Tuple[Any],
        min_chunk_size: int,
    ) -> int:
        consistent = True
        remove_tensors = lambda a: a.shape if type(a) is torch.Tensor else a
        arg_data = tree_map(remove_tensors, args, object)
        if self.cached_arg_data is not None:
            # If args have changed shape/value, we need to re-tune
            assert len(self.cached_arg_data) == len(arg_data)
            consistent = self._compare_arg_caches(self.cached_arg_data, arg_data)
        else:
            # Otherwise, we can reuse the precomputed value
            consistent = False

        if not consistent:
            self.cached_chunk_size = self._determine_favorable_chunk_size(
                representative_fn,
                args,
                min_chunk_size,
            )
            self.cached_arg_data = arg_data

        return self.cached_chunk_size


# ============================================================================
# --- source: openfold/utils/checkpointing.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed


BLOCK_ARG = Any
BLOCK_ARGS = List[BLOCK_ARG]


def get_checkpoint_fn():
    deepspeed_is_configured = deepspeed_is_installed and deepspeed.checkpointing.is_configured()
    if deepspeed_is_configured:
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint

    return checkpoint


@torch.jit.ignore
def checkpoint_blocks(
    blocks: List[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: Optional[int],
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    checkpoint = get_checkpoint_fn()

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint(chunker(s, e), *args)
        args = wrap(args)

    return args


# ============================================================================
# --- source: openfold/utils/precision_utils.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def cast_tensor(x, from_dtype, to_dtype):
    return x.to(dtype=to_dtype) if torch.is_tensor(x) and x.dtype == from_dtype else x


def cast_all(x, from_dtype, to_dtype):
    if isinstance(x, torch.Tensor):
        return cast_tensor(x, from_dtype=from_dtype, to_dtype=to_dtype)
    else:
        if isinstance(x, dict):
            new_dict = {}
            for k in x.keys():
                new_dict[k] = cast_all(x[k], from_dtype=from_dtype, to_dtype=to_dtype)
            return new_dict
        elif isinstance(x, tuple):
            return tuple(cast_all(y, from_dtype=from_dtype, to_dtype=to_dtype) for y in x)
        elif isinstance(x, list):
            return list(cast_all(y, from_dtype=from_dtype, to_dtype=to_dtype) for y in x)
        else:
            return x


class PrecisionWrapper(torch.nn.Module):
    def __init__(self, model, precision):
        super().__init__()
        self.precision = precision
        if self.precision == "bf16":
            print(f"Converting {model.__class__} to BF16 ...")
            model = model.bfloat16()
        elif self.precision == "fp16":
            print(f"Converting {model.__class__} to FP16 ...")
            model = model.half()
        self.model = model

    # TODO: generalize!!
    def forward(self, *args, **kwargs):
        if self.precision == "bf16":
            args = cast_all(args, from_dtype=torch.float32, to_dtype=torch.bfloat16)
            kwargs = cast_all(kwargs, from_dtype=torch.float32, to_dtype=torch.bfloat16)
        elif self.precision == "fp16":
            args = cast_all(args, from_dtype=torch.float32, to_dtype=torch.float16)
            kwargs = cast_all(kwargs, from_dtype=torch.float32, to_dtype=torch.float16)
        out = self.model(*args, **kwargs)
        if self.precision == "bf16":
            out = cast_all(out, from_dtype=torch.bfloat16, to_dtype=torch.float32)
        elif self.precision == "fp16":
            out = cast_all(out, from_dtype=torch.float16, to_dtype=torch.float32)

        return out


def wrap_for_precision(model, precision):
    return PrecisionWrapper(model, precision)


def is_fp16_enabled():
    # Autocast world
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()

    return fp16_enabled


# ============================================================================
# --- source: openfold/model/dropout.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        if not self.training:
            return x
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x *= mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


# ============================================================================
# --- source: openfold/model/primitives.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
ds4s_is_installed = (
    deepspeed_is_installed
    and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
)
if deepspeed_is_installed:
    import deepspeed

if ds4s_is_installed:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

fa_is_installed = importlib.util.find_spec("flash_attn") is not None
if fa_is_installed:
    from flash_attn.bert_padding import unpad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

cueq_is_installed = importlib.util.find_spec("cuequivariance_torch") is not None
if cueq_is_installed:
    from cuequivariance_ops_torch.triangle_attention import (
        CUEQ_TRIATTN_FALLBACK_THRESHOLD,
    )
    from cuequivariance_torch.primitives.triangle import triangle_attention

    def cueq_would_fall_back(n_token: int, hidden_dim: int, dtype: torch.dtype):
        # for q_x, dimension -2 is the context length
        if n_token <= CUEQ_TRIATTN_FALLBACK_THRESHOLD:
            return True
        if dtype == torch.float32:
            if hidden_dim > 32 or hidden_dim % 4 != 0:
                return True
        else:
            # float16, bfloat16
            if hidden_dim > 128 or hidden_dim % 8 != 0:
                return True
        return False


DEFAULT_LMA_Q_CHUNK_SIZE = 1024
DEFAULT_LMA_KV_CHUNK_SIZE = 4096


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
        precision=None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

        self.precision = precision

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.dtype
        deepspeed_is_initialized = deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        if self.precision is not None:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                return nn.functional.linear(
                    input.to(dtype=self.precision), self.weight.to(dtype=self.precision), bias
                ).to(dtype=d)

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast("cuda", enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return nn.functional.linear(input, self.weight.to(dtype=d), bias)

        return nn.functional.linear(input, self.weight, self.bias)


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast("cuda", enabled=False):
                out = nn.functional.layer_norm(
                    x, self.c_in, self.weight.to(dtype=d), self.bias.to(dtype=d), self.eps
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
    if d is torch.bfloat16 and not deepspeed_is_initialized:
        with torch.amp.autocast("cuda", enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


# @torch.jit.script
def _attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]
) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


@torch.jit.ignore
def _attention_chunked_trainable(
    query,
    key,
    value,
    biases,
    chunk_size,
    chunk_dim,
    checkpoint,
):
    if checkpoint and len(biases) > 2:
        raise ValueError("Checkpointed version permits only permits two bias terms")

    def _checkpointable_attention(q, k, v, b1, b2):
        bs = [b for b in [b1, b2] if b is not None]
        a = _attention(q, k, v, bs)
        return a

    o_chunks = []
    checkpoint_fn = get_checkpoint_fn()
    count = query.shape[chunk_dim]
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        idx = [slice(None)] * len(query.shape)
        idx[chunk_dim] = slice(start, end)
        idx_tup = tuple(idx)
        q_chunk = query[idx_tup]
        k_chunk = key[idx_tup]
        v_chunk = value[idx_tup]

        def _slice_bias(b):
            idx[chunk_dim] = slice(start, end) if b.shape[chunk_dim] != 1 else slice(None)
            return b[tuple(idx)]

        if checkpoint:
            bias_1_chunk, bias_2_chunk = [
                _slice_bias(b) if b is not None else None for b in (biases + [None, None])[:2]
            ]

            o_chunk = checkpoint_fn(
                _checkpointable_attention, q_chunk, k_chunk, v_chunk, bias_1_chunk, bias_2_chunk
            )
        else:
            bias_chunks = [_slice_bias(b) for b in biases]

            o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)

        o_chunk = o_chunk.transpose(-2, -3)
        o_chunks.append(o_chunk)

    o = torch.cat(o_chunks, dim=chunk_dim)
    return o


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        inf: float = 1e9,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.inf = inf

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, init="final")

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.no_heads, init="gating")

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        lma_kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE,
        use_flash: bool = False,
        flash_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel.
                This should be the default choice for most. If none of the
                "use_<...>" flags are True, a stock PyTorch implementation
                is used instead
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory-efficient attention kernel.
                If none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
            use_cuequivariance_attention:
                Whether to use cuEquivariance attention kernel.
                When on, biases[0] contains 0/1 mask tensor for cuEquivariance attention (0 for invalid positions)


        Returns
            [*, Q, C_q] attention update
        """
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError(
                "If use_lma is specified, lma_q_chunk_size and lma_kv_chunk_size must be provided"
            )

        if use_flash and biases is not None:
            raise ValueError(
                "use_flash is incompatible with the bias option. For masking, "
                "use flash_mask instead"
            )

        if use_cuequivariance_attention:
            if biases is None or len(biases) != 2:
                raise ValueError("cuEquivariance attention requires exactly two bias terms")

        attn_options = [
            use_memory_efficient_kernel,
            use_deepspeed_evo_attention or use_cuequivariance_attention,
            use_lma,
            use_flash,
        ]
        if sum(attn_options) > 1:
            raise ValueError("Choose at most one alternative attention algorithm")

        if biases is None:
            biases = []

        if is_fp16_enabled():
            use_memory_efficient_kernel = False

        if use_cuequivariance_attention:
            # cuEquivariance -> Torch fallback for small sequence length and some shapes
            if cueq_would_fall_back(q_x.shape[-2], q_x.shape[-1] // self.no_heads, q_x.dtype):
                # convert the mask from boolean to float pre-mul
                biases[0] = self.inf * (biases[0] - 1)
                use_cuequivariance_attention = False

        # The EvoformerAttention kernel can only be used for sequence lengths > 16
        if use_deepspeed_evo_attention and q_x.shape[-2] <= 16:
            use_deepspeed_evo_attention = False

        # DeepSpeed attention kernel applies scaling internally
        q, k, v = self._prep_qkv(
            q_x, kv_x, apply_scale=not (use_deepspeed_evo_attention or use_cuequivariance_attention)
        )

        # cuequivariance kernel takes precedence over use_deepspeed_evo_attention
        if use_cuequivariance_attention:
            if not cueq_is_installed:
                raise ValueError(
                    "Running with `use_cuequivariance_attention` but package is not "
                    "installed. See documentation for installation instructions."
                )
            o = _cuequivariance_attn(q, k, v, biases[1], biases[0])
        elif use_memory_efficient_kernel:
            if len(biases) > 2:
                raise ValueError(
                    "If use_memory_efficient_kernel is True, you may only "
                    "provide up to two bias terms"
                )
            o = attention_core(q, k, v, *((biases + [None] * 2)[:2]))  # noqa: F821 -- dead branch; the openfold custom CUDA kernel was intentionally not vendored, this path is never reached because MENAGERIE_ENTRIES calls with use_lma=True
            o = o.transpose(-2, -3)
        elif use_deepspeed_evo_attention:
            if len(biases) > 2:
                raise ValueError(
                    "If use_deepspeed_evo_attention is True, you may only "
                    "provide up to two bias terms"
                )
            o = _deepspeed_evo_attn(q, k, v, biases)
        elif use_lma:
            biases = [b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],)) for b in biases]
            o = _lma(q, k, v, biases, lma_q_chunk_size, lma_kv_chunk_size)
            o = o.transpose(-2, -3)
        elif use_flash:
            o = _flash_attn(q, k, v, flash_mask)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class GlobalAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, inf, eps):
        super(GlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(c_in, c_hidden * no_heads, bias=False, init="glorot")

        self.linear_k = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
        self.linear_v = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
        self.linear_g = Linear(c_in, c_hidden * no_heads, init="gating")
        self.linear_o = Linear(c_hidden * no_heads, c_in, init="final")

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        use_lma: bool = False,
    ) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + self.eps
        )

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= self.c_hidden ** (-0.5)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        bias = (self.inf * (mask - 1))[..., :, None, :]
        if not use_lma:
            # [*, N_res, H, N_seq]
            a = torch.matmul(
                q,
                k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
            )
            a += bias
            a = softmax_no_cast(a)

            # [*, N_res, H, C_hidden]
            o = torch.matmul(
                a,
                v,
            )
        else:
            o = _lma(q, k, v, [bias], DEFAULT_LMA_Q_CHUNK_SIZE, DEFAULT_LMA_KV_CHUNK_SIZE)

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m


@torch.jit.ignore
def _deepspeed_evo_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
):
    """ ""
    Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

    Args:
        q:
            [*, H, Q, C_hidden] query data
        k:
            [*, H, K, C_hidden] key data
        v:
            [*, H, V, C_hidden] value data
        biases:
            List of biases that broadcast to [*, H, Q, K]
    """

    if not ds4s_is_installed:
        raise ValueError(
            "_deepspeed_evo_attn requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )

    def reshape_dims(x):
        no_batch_dims = len(x.shape[:-3])
        if no_batch_dims < 2:
            return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
        if no_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    # [*, Q/K, H, C_hidden]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # Reshape tensors to match expected input shape [B, N, Q/K, H, C_hidden]
    # for DS4Sci_EvoformerAttention() by adding or flattening batch dims as needed.
    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dims(q)
        k = reshape_dims(k)
        v = reshape_dims(v)
        biases = [reshape_dims(b) for b in biases]

    # DeepSpeed attn. kernel requires inputs to be type bf16 or fp16
    # Cast to bf16 so kernel can be used during inference
    orig_dtype = q.dtype
    if orig_dtype not in [torch.bfloat16, torch.float16]:
        o = DS4Sci_EvoformerAttention(
            q.to(dtype=torch.bfloat16),
            k.to(dtype=torch.bfloat16),
            v.to(dtype=torch.bfloat16),
            [b.to(dtype=torch.bfloat16) for b in biases],
        )

        o = o.to(dtype=orig_dtype)
    else:
        o = DS4Sci_EvoformerAttention(q, k, v, biases)

    o = o.reshape(orig_shape)
    return o


def _lma(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
    q_chunk_size: int,
    kv_chunk_size: int,
):
    no_q, no_kv = q.shape[-2], k.shape[-2]

    # [*, H, Q, C_hidden]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s : q_s + q_chunk_size, :]
        large_bias_chunks = [b[..., q_s : q_s + q_chunk_size, :] for b in biases]

        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s : kv_s + kv_chunk_size, :]
            v_chunk = v[..., kv_s : kv_s + kv_chunk_size, :]
            small_bias_chunks = [b[..., kv_s : kv_s + kv_chunk_size] for b in large_bias_chunks]

            a = torch.einsum(
                "...hqd,...hkd->...hqk",
                q_chunk,
                k_chunk,
            )

            for b in small_bias_chunks:
                a += b

            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum("...hvf,...hqv->...hqf", v_chunk, exp_a)

            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)

        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)

        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values = chunk_values * max_diffs.unsqueeze(-1)
        chunk_weights = chunk_weights * max_diffs

        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)

        q_chunk_out = all_values / all_weights

        o[..., q_s : q_s + q_chunk_size, :] = q_chunk_out

    return o


@torch.jit.ignore
def _flash_attn(q, k, v, kv_mask):
    if not fa_is_installed:
        raise ValueError("_flash_attn requires that FlashAttention be installed")

    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype

    q = q.half()
    k = k.half()
    v = v.half()
    kv_mask = kv_mask.half()

    # [*, B, N, H, C]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]

    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])

    q_max_s = n
    q_cu_seqlens = torch.arange(0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device)

    # [B_flat, N, 2, H, C]
    kv = torch.stack([k, v], dim=-3)
    kv_shape = kv.shape

    # [B_flat, N, 2 * H * C]
    kv = kv.reshape(*kv.shape[:-3], -1)

    kv_unpad, _, kv_cu_seqlens, kv_max_s, _ = unpad_input(kv, kv_mask)
    kv_unpad = kv_unpad.reshape(-1, *kv_shape[-3:])

    out = flash_attn_varlen_kvpacked_func(
        q,
        kv_unpad,
        q_cu_seqlens,
        kv_cu_seqlens,
        q_max_s,
        kv_max_s,
        dropout_p=0.0,
        softmax_scale=1.0,  # q has been scaled already
    )

    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c)

    out = out.to(dtype=dtype)

    return out


@torch.jit.ignore
def _cuequivariance_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """
    Compute attention using the cuEquivariance triangle attention kernel.

    Args:
        q: [*, H, Q, C_hidden] query data
        k: [*, H, K, C_hidden] key data
        v: [*, H, V, C_hidden] value data
        bias: [*, H, Q, K] triangular bias
        mask: [*, Q, K] mask for masking invalid positions

    Returns:
        [*, H, Q, C_hidden] attention output
    """

    # Check input dimensionality
    qdim = len(q.shape)
    # If we have 4D tensors ([*, H, Q, D]), add batch dimension
    if qdim == 4:
        q = q.unsqueeze(0)  # [1, H, Q, D]
        k = k.unsqueeze(0)  # [1, H, K, D]
        v = v.unsqueeze(0)  # [1, H, V, D]
        bias = bias.unsqueeze(0)  # [1, H, Q, K]
        if mask is not None:
            mask = mask.unsqueeze(0)  # [1, Q, K]
    elif len(q.shape[:-3]) > 2:
        # If there are more than 2 leading dimensions, flatten them into B*N
        batch_shape = q.shape[:-3]
        flat_batch_size = 1
        for dim in batch_shape:
            flat_batch_size *= dim

        q = q.reshape(flat_batch_size, *q.shape[-3:])
        k = k.reshape(flat_batch_size, *k.shape[-3:])
        v = v.reshape(flat_batch_size, *v.shape[-3:])
        bias = bias.reshape(flat_batch_size, *bias.shape[-3:])
        if mask is not None:
            mask = mask.reshape(flat_batch_size, *mask.shape[-2:])

    # Apply cuEquivariance triangle attention
    o = triangle_attention(q=q, k=k, v=v, bias=bias, mask=mask)

    # If we added a batch dimension for 4D inputs, remove it
    if qdim == 4:
        o = o.squeeze(0)

    # Final transpose to match expected output format
    o = o.transpose(-2, -3)

    return o


# ============================================================================
# --- source: openfold/model/msa.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(self.c_z, self.no_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_in,
            self.c_in,
            self.c_in,
            self.c_hidden,
            self.no_heads,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
        use_memory_efficient_kernel: bool,
        use_deepspeed_evo_attention: bool,
        use_cuequivariance_attention: bool,
        use_lma: bool,
        use_flash: bool,
        flash_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        def fn(m, biases, flash_mask):
            m = self.layer_norm_m(m)
            return self.mha(
                q_x=m,
                kv_x=m,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=flash_mask,
            )

        inputs = {"m": m}
        if biases is not None:
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)
        if use_flash and flash_mask is not None:
            inputs["flash_mask"] = flash_mask
        else:
            fn = partial(fn, flash_mask=None)

        return chunk_layer(fn, inputs, chunk_size=chunk_size, no_batch_dims=len(m.shape[:-2]))

    def _prep_inputs(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
        use_cuequivariance_attention: bool = False,
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-1],
            )

        # [*, N_seq, 1, 1, N_res]
        if use_cuequivariance_attention:
            mask_bias = mask[..., :, None, None, :]
        else:
            # [*, I, 1, 1, J]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        if (
            self.pair_bias
            and z is not None  # For the
            and self.layer_norm_z is not None  # benefit of
            and self.linear_z is not None  # TorchScript
        ):
            if torch.onnx.is_in_onnx_export() or is_fx_tracing():
                inplace_safe = False
                chunk_size = None

            if chunk_size is None:
                z = self.layer_norm_z(z)
                z = self.linear_z(z)
            else:
                chunks = []

                for i in range(0, z.shape[-3], chunk_size):
                    z_chunk = z[..., i : i + chunk_size, :, :]

                    # [*, N_res, N_res, C_z]
                    z_chunk = self.layer_norm_z(z_chunk)

                    # [*, N_res, N_res, no_heads]
                    z_chunk = self.linear_z(z_chunk)

                    chunks.append(z_chunk)
                z = torch.cat(chunks, dim=-3)

            # [*, 1, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    @torch.jit.ignore
    def _chunked_msa_attn(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_logits: int,
        checkpoint: bool,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        MSA attention with training-time chunking of the softmax computation.
        Saves memory in the extra MSA stack. Probably obviated by our fused
        attention kernel, which is now used by default.
        """
        MSA_DIM = -4

        def _get_qkv(m, z):
            m, mask_bias, z = self._prep_inputs(m, z, mask, inplace_safe=inplace_safe)
            m = self.layer_norm_m(m)
            q, k, v = self.mha._prep_qkv(m, m)
            return m, q, k, v, mask_bias, z

        checkpoint_fn = get_checkpoint_fn()

        if torch.is_grad_enabled() and checkpoint:
            m, q, k, v, mask_bias, z = checkpoint_fn(_get_qkv, m, z)
        else:
            m, q, k, v, mask_bias, z = _get_qkv(m, z)

        o = _attention_chunked_trainable(
            query=q,
            key=k,
            value=v,
            biases=[mask_bias, z],
            chunk_size=chunk_logits,
            chunk_dim=MSA_DIM,
            checkpoint=checkpoint,
        )

        if torch.is_grad_enabled() and checkpoint:
            # Storing an additional m here is far from ideal
            m = checkpoint_fn(self.mha._wrap_up, o, m)
        else:
            m = self.mha._wrap_up(o, m)

        return m

    def forward(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _chunk_logits: Optional[int] = None,
        _checkpoint_chunks: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the
                cost of slower execution. Chunking is not performed by default.

        """
        if _chunk_logits is not None:
            return self._chunked_msa_attn(
                m=m,
                z=z,
                mask=mask,
                chunk_logits=_chunk_logits,
                checkpoint=_checkpoint_chunks,
                inplace_safe=inplace_safe,
            )

        if use_flash:
            assert z is None
            biases = None
        else:
            m, mask_bias, z = self._prep_inputs(
                m,
                z,
                mask,
                inplace_safe=inplace_safe,
                use_cuequivariance_attention=use_cuequivariance_attention,
            )

            biases = [mask_bias]
            if z is None and use_cuequivariance_attention:
                z = m.new_zeros(1, self.no_heads, m.shape[-2], m.shape[-2])

            if z is not None:
                biases.append(z)

        if chunk_size is not None:
            m = self._chunk(
                m,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )
        else:
            m = self.layer_norm_m(m)
            m = self.mha(
                q_x=m,
                kv_x=m,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.

    By rights, this should also be a subclass of MSAAttention. Alas,
    most inheritance isn't supported by TorchScript.
    """

    def __init__(self, c_m, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSAColumnAttention, self).__init__()

        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_in=c_m,
            c_hidden=c_hidden,
            no_heads=no_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the
                cost of slower execution. Chunking is not performed by default.
        """
        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m = self._msa_att(
            m,
            mask=mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_lma=use_lma,
            use_flash=use_flash,
        )

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        return m


class MSAColumnGlobalAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        inf=1e9,
        eps=1e-10,
    ):
        super(MSAColumnGlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_in)

        self.global_attention = GlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf,
            eps=eps,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        use_lma: bool = False,
    ) -> torch.Tensor:
        mha_input = {
            "m": m,
            "mask": mask,
        }

        def fn(m, mask):
            m = self.layer_norm_m(m)
            return self.global_attention(m, mask, use_lma=use_lma)

        return chunk_layer(
            fn,
            mha_input,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
    ) -> torch.Tensor:
        n_seq, n_res, c_in = m.shape[-3:]

        if mask is None:
            # [*, N_seq, N_res]
            mask = torch.ones(
                m.shape[:-1],
                dtype=m.dtype,
                device=m.device,
            ).detach()

        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size, use_lma=use_lma)
        else:
            m = self.layer_norm_m(m)
            m = self.global_attention(m=m, mask=mask, use_lma=use_lma)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m


# ============================================================================
# --- source: openfold/model/outer_product_mean.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden**2, c_z, init="final")

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def _forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask

        b = self.linear_2(ln)
        b = b * mask

        del ln

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        if inplace_safe:
            outer /= norm
        else:
            outer = outer / norm

        return outer

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(m.float(), mask, chunk_size, inplace_safe)
        else:
            return self._forward(m, mask, chunk_size, inplace_safe)


# ============================================================================
# --- source: openfold/model/pair_transition.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def _transition(self, z, mask):
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)
        z = z * mask

        return z

    @torch.jit.ignore
    def _chunk(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask in this module.
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z=z, mask=mask)

        return z


# ============================================================================
# --- source: openfold/model/triangular_attention.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")

        self.mha = Attention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        if use_cuequivariance_attention:
            mask_bias = mask[..., :, None, None, :]
        else:
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)


# ============================================================================
# --- source: openfold/model/triangular_multiplicative_update.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# cuEquivariance import handling
cuequivariance_is_installed = importlib.util.find_spec("cuequivariance_torch") is not None
if cuequivariance_is_installed:
    try:
        from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
    except ImportError:
        cuequivariance_is_installed = False


def _cuequivariance_triangular_mult(
    x: torch.Tensor,
    direction: str,
    mask: Optional[torch.Tensor],
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    p_in_bias: torch.Tensor,
    g_in_weight: torch.Tensor,
    g_in_bias: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    p_out_bias: torch.Tensor,
    g_out_weight: torch.Tensor,
    g_out_bias: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Wrapper function for cuEquivariance triangle multiplicative update.

    Args:
        x: [*, N, N, C] input tensor
        direction: "outgoing" or "incoming"
        mask: [*, N, N] mask tensor
        norm_in_weight: [C] input normalization weight
        norm_in_bias: [C] input normalization bias
        p_in_weight: [2*C, C] input projection weight
        g_in_weight: [2*C, C] input gating weight
        norm_out_weight: [C] output normalization weight
        norm_out_bias: [C] output normalization bias
        p_out_weight: [C, C] output projection weight
        g_out_weight: [C, C] output gating weight
        eps: epsilon for numerical stability

    Returns:
        [*, N, N, C] output tensor
    """
    if not cuequivariance_is_installed:
        raise ValueError(
            "_cuequivariance_triangular_mult requires that cuequivariance_torch be installed"
        )
    return triangle_multiplicative_update(
        x=x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        p_in_bias=p_in_bias,
        g_in_weight=g_in_weight,
        g_in_bias=g_in_bias,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        p_out_bias=p_out_bias,
        g_out_weight=g_out_weight,
        g_out_bias=g_out_bias,
        eps=eps,
    ).view(x.shape)


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Implements Algorithms 11 and 12.
    """

    @abstractmethod
    def __init__(self, c_z, c_hidden, _outgoing):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(BaseTriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self, a: torch.Tensor, b: torch.Tensor, _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.matmul(
                    a_chunk,
                    b_chunk,
                )

            p = a
        else:
            p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        _add_with_inplace: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        pass


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__(
            c_z=c_z, c_hidden=c_hidden, _outgoing=_outgoing
        )

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function.
        Uses in-place operations, fusion of the addition that happens after
        this module in the Evoformer, a smidge of recomputation, and
        a cache of overwritten values to lower peak memory consumption of this
        module from 5x the size of the input tensor z to 2.5x its size. Useful
        for inference on extremely long sequences.

        It works as follows. We will make reference to variables used in the
        default forward implementation below. Naively, triangle multiplication
        attention requires the manifestation of 5 tensors the size of z:
        1) z, the "square" input tensor, 2) a, the first projection of z,
        3) b, the second projection of b, 4) g, a z-sized mask, and 5) a
        z-sized tensor for intermediate computations. For large N, this is
        prohibitively expensive; for N=4000, for example, z is more than 8GB
        alone. To avoid this problem, we compute b, g, and all intermediate
        tensors in small chunks, noting that the chunks required to compute a
        chunk of the output depend only on the tensor a and corresponding
        vertical and horizontal chunks of z. This suggests an algorithm that
        loops over pairs of chunks of z: hereafter "columns" and "rows" of
        z, even though each "column" and "row" in fact contains
        inplace_chunk_size contiguous true columns and rows of z. Writing
        output chunks to a new tensor would bring total memory consumption
        down to 3x the size of z. However, more memory can be saved by writing
        output chunks directly to z in-place. WLOG, we choose to write output
        chunks vertically, overwriting the ith "column" of z at the end of
        the ith iteration of the main loop. Despite this overwriting, the
        ith column is always one column ahead of previously overwritten columns
        and can be recovered directly from z. After the first iteration,
        however, the ith row of z is always at least partially overwritten. For
        this reason, we introduce the z-cache, a tensor one-half the size of
        z. The z-cache initially contains the left half (2nd and 3rd quadrants)
        of z. For 0 < i < N/2, the missing left part of the ith row of z is
        recovered from this cache at the beginning of the ith iteration. Once i
        exceeds n/2, the cache is "reoriented" to encompass the 3rd and 4th
        quadrants of z instead. Though the 3rd quadrant of the original z is
        entirely overwritten at this point, it can be recovered from the z-cache
        itself. Thereafter, the ith row of z can be recovered in its entirety
        from the reoriented z-cache. After the final iteration, z has been
        completely overwritten and contains the triangular multiplicative
        update. If with_add is True, it instead contains the sum of z and the
        triangular multiplicative update. In either case, peak memory
        consumption is just 2.5x the size of z, disregarding memory used for
        chunks and other small variables.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask, a=True):
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p

            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p.sigmoid_()
            p *= linear_p(pair)
            p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True):
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a)
                if need_transpose:
                    p = p.transpose(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x
                # budget with a large intermediate tensor
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.bias.shape[-1]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = pair[..., i : i + inplace_chunk_size, :, :]
                    mask_chunk = mask[..., i : i + inplace_chunk_size, :, :]
                    pair_chunk = compute_projection_helper(
                        pair[..., i : i + inplace_chunk_size, :, :],
                        mask[..., i : i + inplace_chunk_size, :, :],
                        a,
                    )
                    if need_transpose:
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i : i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i : i + inplace_chunk_size, :] = pair_chunk

                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(z, mask, True, chunked=True)

        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t):
                return [slice(None) for _ in t.shape]

            def slice_tensor(t, start, end, dim):
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[s]

            def flip_z_cache_(z_cache, z):
                # "Reorient" the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z.
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)

                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., : (n // 2), :, :]

                # Move the 3rd quadrant of z into the
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[first_half_slicer] = quadrant_3

                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[quadrant_3_slicer] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[z_cache_slicer])
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(
                i_range + after_half, initial_offsets + after_half_offsets
            )
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(
                    z,
                    i,
                    i + offset,
                    b_chunk_dim,
                )
                mask_chunk = slice_tensor(
                    mask,
                    i,
                    i + offset,
                    b_chunk_dim,
                )

                z_chunk_b = z_chunk_b.clone()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else:  # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially
                    # overwritten at the end of each iteration. We need to
                    # restore the missing component from the z-cache.
                    if not z_cache_rotated:
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[z_chunk_slicer] = slice_tensor(
                            z_cache,
                            i,
                            i + offset,
                            row_dim,
                        )
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(
                            z_cache, z_cache_offset, z_cache_offset + offset, row_dim
                        )

                b_chunk = compute_projection(z_chunk_b, mask_chunk, a=False, chunked=False)
                del z_chunk_b

                x_chunk = torch.matmul(
                    a,
                    b_chunk,
                )
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)

                # The g dimension (col_dim) is parallel to and ahead of the
                # overwrites in z. We can extract the g chunk normally.
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                g_chunk.sigmoid_()
                del z_chunk_g

                x_chunk *= g_chunk

                # Write the columns into z in-place
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if with_add:
                    z[z_slicer] += x_chunk
                else:
                    z[z_slicer] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g.sigmoid_()
            x *= g
            if with_add:
                z += x
            else:
                z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """

        if use_cuequivariance_multiplicative_update:
            p_in_weight = torch.cat([self.linear_a_p.weight, self.linear_b_p.weight], dim=0)
            g_in_weight = torch.cat([self.linear_a_g.weight, self.linear_b_g.weight], dim=0)

            p_in_bias = torch.cat([self.linear_a_p.bias, self.linear_b_p.bias], dim=0)
            g_in_bias = torch.cat([self.linear_a_g.bias, self.linear_b_g.bias], dim=0)

            result = _cuequivariance_triangular_mult(
                z,
                direction="outgoing" if self._outgoing else "incoming",
                mask=mask,
                norm_in_weight=self.layer_norm_in.weight,
                norm_in_bias=self.layer_norm_in.bias,
                p_in_weight=p_in_weight,
                p_in_bias=p_in_bias,
                g_in_weight=g_in_weight,
                g_in_bias=g_in_bias,
                norm_out_weight=self.layer_norm_out.weight,
                norm_out_bias=self.layer_norm_out.bias,
                p_out_weight=self.linear_z.weight,
                p_out_bias=self.linear_z.bias,
                g_out_weight=self.linear_g.weight,
                g_out_bias=self.linear_g.bias,
                eps=1e-5,
            )
            # When not inplace_safe (training), caller should have set _add_with_inplace to False
            if inplace_safe and _add_with_inplace:
                result += z
            return result

        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        # Prevents overflow of torch.matmul in combine projections in
        # reduced-precision modes
        if is_fp16_enabled():
            a_std = a.std()
            b_std = b.std()
            if a_std != 0.0 and b_std != 0.0:
                a = a / a.std()
                b = b / b.std()
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class FusedTriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(FusedTriangleMultiplicativeUpdate, self).__init__(
            c_z=c_z, c_hidden=c_hidden, _outgoing=_outgoing
        )

        self.linear_ab_p = Linear(self.c_z, self.c_hidden * 2)
        self.linear_ab_g = Linear(self.c_z, self.c_hidden * 2, init="gating")

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        _inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask):
            p = self.linear_ab_g(pair)
            p.sigmoid_()
            p *= self.linear_ab_p(pair)
            p *= mask

            return p

        def compute_projection(pair, mask):
            p = compute_projection_helper(pair, mask)
            left = p[..., : self.c_hidden]
            right = p[..., self.c_hidden :]

            return left, right

        z_norm_in = self.layer_norm_in(z)
        a, b = compute_projection(z_norm_in, mask)
        x = self._combine_projections(a, b, _inplace_chunk_size=_inplace_chunk_size)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.linear_g(z_norm_in)
        g.sigmoid_()
        x *= g
        if with_add:
            z += x
        else:
            z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """

        if use_cuequivariance_multiplicative_update:
            direction = "outgoing" if self._outgoing else "incoming"
            result = _cuequivariance_triangular_mult(
                x=z,
                direction=direction,
                mask=mask,
                norm_in_weight=self.layer_norm_in.weight,
                norm_in_bias=self.layer_norm_in.bias,
                p_in_weight=self.linear_ab_p.weight,
                p_in_bias=self.linear_ab_p.bias,
                g_in_weight=self.linear_ab_g.weight,
                g_in_bias=self.linear_ab_g.bias,
                norm_out_weight=self.layer_norm_out.weight,
                norm_out_bias=self.layer_norm_out.bias,
                p_out_weight=self.linear_z.weight,
                p_out_bias=self.linear_z.bias,
                g_out_weight=self.linear_g.weight,
                g_out_bias=self.linear_g.bias,
                eps=1e-5,
            )
            # When not inplace_safe (training), caller should have set _add_with_inplace to False
            if inplace_safe and _add_with_inplace:
                result += z
            return result

        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                _inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        ab = mask
        ab = ab * self.sigmoid(self.linear_ab_g(z))
        ab = ab * self.linear_ab_p(z)

        a = ab[..., : self.c_hidden]
        b = ab[..., self.c_hidden :]

        # Prevents overflow of torch.matmul in combine projections in
        # reduced-precision modes
        a_std = a.std()
        b_std = b.std()
        if is_fp16_enabled() and a_std != 0.0 and b_std != 0.0:
            a = a / a.std()
            b = b / b.std()

        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class FusedTriangleMultiplicationOutgoing(FusedTriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """

    __init__ = partialmethod(FusedTriangleMultiplicativeUpdate.__init__, _outgoing=True)


class FusedTriangleMultiplicationIncoming(FusedTriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """

    __init__ = partialmethod(FusedTriangleMultiplicativeUpdate.__init__, _outgoing=False)


# ============================================================================
# --- source: openfold/model/evoformer.py ---
# ============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """

    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"m": m, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
    ):
        super(PairStack, self).__init__()

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        z = add(
            z,
            self.ps_dropout_row_layer(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if inplace_safe:
            z = z.contiguous()

        z = add(
            z,
            self.ps_dropout_row_layer(
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2),
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if inplace_safe:
            z = z.contiguous()

        z = add(
            z,
            self.pair_transition(
                z,
                mask=pair_trans_mask,
                chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        return z


class MSABlock(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
    ):
        super(MSABlock, self).__init__()

        self.opm_first = opm_first

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            eps=eps,
        )

    def _compute_opm(
        self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, z = input_tensors

        if _offload_inference and inplace_safe:
            # m: GPU, z: CPU
            del m, z
            assert sys.getrefcount(input_tensors[1]) == 2
            input_tensors[1] = input_tensors[1].cpu()
            m, z = input_tensors

        opm = self.outer_product_mean(
            m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
        )

        if _offload_inference and inplace_safe:
            # m: GPU, z: GPU
            del m, z
            assert sys.getrefcount(input_tensors[0]) == 2
            input_tensors[1] = input_tensors[1].to(opm.device)
            m, z = input_tensors

        z = add(z, opm, inplace=inplace_safe)
        del opm

        return m, z

    @abstractmethod
    def forward(
        self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class EvoformerBlock(MSABlock):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        no_column_attention: bool,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
    ):
        super(EvoformerBlock, self).__init__(
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            opm_first=opm_first,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            eps=eps,
        )

        # Specifically, seqemb mode does not use column attention
        self.no_column_attention = no_column_attention

        if not self.no_column_attention:
            self.msa_att_col = MSAColumnAttention(
                c_m,
                c_hidden_msa_att,
                no_heads_msa,
                inf=inf,
            )

    def forward(
        self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        msa_trans_mask = msa_mask if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        if _offload_inference and inplace_safe:
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        if self.opm_first:
            del m, z

            m, z = self._compute_opm(
                input_tensors=input_tensors,
                msa_mask=msa_mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
            )

        m = add(
            m,
            self.msa_dropout_layer(
                self.msa_att_row(
                    m,
                    z=z,
                    mask=msa_mask,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_lma=use_lma,
                )
            ),
            inplace=inplace_safe,
        )

        if _offload_inference and inplace_safe:
            # m: GPU, z: CPU
            del m, z
            assert sys.getrefcount(input_tensors[1]) == 2
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            m, z = input_tensors

        # Specifically, column attention is not used in seqemb mode.
        if not self.no_column_attention:
            m = add(
                m,
                self.msa_att_col(
                    m,
                    mask=msa_mask,
                    chunk_size=chunk_size,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_lma=use_lma,
                    use_flash=use_flash,
                ),
                inplace=inplace_safe,
            )

        m = add(
            m,
            self.msa_transition(
                m,
                mask=msa_trans_mask,
                chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        if not self.opm_first:
            if not inplace_safe:
                input_tensors = [m, z]

            del m, z

            m, z = self._compute_opm(
                input_tensors=input_tensors,
                msa_mask=msa_mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
            )

        if _offload_inference and inplace_safe:
            # m: CPU, z: GPU
            del m, z
            assert sys.getrefcount(input_tensors[0]) == 2
            device = input_tensors[0].device
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(device)
            m, z = input_tensors

        if not inplace_safe:
            input_tensors = [m, z]

        del m, z

        z = self.pair_stack(
            z=input_tensors[1],
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
        )

        if _offload_inference and inplace_safe:
            # m: GPU, z: GPU
            device = z.device
            assert sys.getrefcount(input_tensors[0]) == 2
            input_tensors[0] = input_tensors[0].to(device)
            m, _ = input_tensors
        else:
            m = input_tensors[0]

        return m, z


class ExtraMSABlock(MSABlock):
    """
    Almost identical to the standard EvoformerBlock, except in that the
    ExtraMSABlock uses GlobalAttention for MSA column attention and
    requires more fine-grained control over checkpointing. Separated from
    its twin to preserve the TorchScript-ability of the latter.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        ckpt: bool,
    ):
        super(ExtraMSABlock, self).__init__(
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            opm_first=opm_first,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            eps=eps,
        )

        self.ckpt = ckpt

        self.msa_att_col = MSAColumnGlobalAttention(
            c_in=c_m,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
            eps=eps,
        )

    def forward(
        self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        if _offload_inference and inplace_safe:
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        if self.opm_first:
            del m, z

            m, z = self._compute_opm(
                input_tensors=input_tensors,
                msa_mask=msa_mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
            )

        m = add(
            m,
            self.msa_dropout_layer(
                self.msa_att_row(
                    m.clone() if torch.is_grad_enabled() else m,
                    z=z.clone() if torch.is_grad_enabled() else z,
                    mask=msa_mask,
                    chunk_size=_attn_chunk_size,
                    use_lma=use_lma,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_memory_efficient_kernel=not (
                        use_lma or use_deepspeed_evo_attention or use_cuequivariance_attention
                    ),
                    _checkpoint_chunks=self.ckpt if torch.is_grad_enabled() else False,
                )
            ),
            inplace=inplace_safe,
        )

        if not inplace_safe:
            input_tensors = [m, z]

        del m, z

        def fn(input_tensors):
            m, z = input_tensors

            if _offload_inference and inplace_safe:
                # m: GPU, z: CPU
                del m, z
                assert sys.getrefcount(input_tensors[1]) == 2
                input_tensors[1] = input_tensors[1].cpu()
                torch.cuda.empty_cache()
                m, z = input_tensors

            m = add(
                m,
                self.msa_att_col(
                    m,
                    mask=msa_mask,
                    chunk_size=chunk_size,
                    use_lma=use_lma,
                ),
                inplace=inplace_safe,
            )

            m = add(
                m,
                self.msa_transition(
                    m,
                    mask=msa_mask,
                    chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
            )

            if not self.opm_first:
                if not inplace_safe:
                    input_tensors = [m, z]

                del m, z

                m, z = self._compute_opm(
                    input_tensors=input_tensors,
                    msa_mask=msa_mask,
                    chunk_size=chunk_size,
                    inplace_safe=inplace_safe,
                    _offload_inference=_offload_inference,
                )

            if _offload_inference and inplace_safe:
                # m: CPU, z: GPU
                del m, z
                assert sys.getrefcount(input_tensors[0]) == 2
                device = input_tensors[0].device
                input_tensors[0] = input_tensors[0].cpu()
                input_tensors[1] = input_tensors[1].to(device)
                m, z = input_tensors

            if not inplace_safe:
                input_tensors = [m, z]

            del m, z

            z = self.pair_stack(
                input_tensors[1],
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
                _attn_chunk_size=_attn_chunk_size,
            )

            m = input_tensors[0]
            if _offload_inference and inplace_safe:
                # m: GPU, z: GPU
                device = z.device
                del m
                assert sys.getrefcount(input_tensors[0]) == 2
                input_tensors[0] = input_tensors[0].to(device)
                m, _ = input_tensors

            return m, z

        if torch.is_grad_enabled() and self.ckpt:
            checkpoint_fn = get_checkpoint_fn()
            m, z = checkpoint_fn(fn, input_tensors)
        else:
            m, z = fn(input_tensors)

        return m, z


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        no_column_attention: bool,
        opm_first: bool,
        fuse_projection_weights: bool,
        blocks_per_ckpt: int,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            no_column_attention:
                When True, doesn't use column attention. Required for running
                sequence embedding mode
            opm_first:
                When True, Outer Product Mean is performed at the beginning of
                the Evoformer block instead of after the MSA Stack.
                Used in Multimer pipeline.
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                no_column_attention=no_column_attention,
                opm_first=opm_first,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner(2048)

    def _prep_blocks(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool,
        use_cuequivariance_attention: bool,
        use_cuequivariance_multiplicative_update: bool,
        use_lma: bool,
        use_flash: bool,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
                use_lma=use_lma,
                use_flash=use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:

            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # We don't want to write in-place during chunk tuning runs
                args=(
                    m.clone(),
                    z.clone(),
                ),
                min_chunk_size=chunk_size,
            )
            # A temporary measure to address torch's occasional
            # inability to allocate large tensors
            attn_chunk = (
                tuned_chunk_size if use_cuequivariance_attention else (tuned_chunk_size // 4)
            )
            blocks = [
                partial(
                    b,
                    chunk_size=tuned_chunk_size,
                    _attn_chunk_size=max(chunk_size, attn_chunk),
                )
                for b in blocks
            ]

        return blocks

    def _forward_offload(
        self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert not (self.training or torch.is_grad_enabled())
        blocks = self._prep_blocks(
            # We are very careful not to create references to these tensors in
            # this function
            m=input_tensors[0],
            z=input_tensors[1],
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=True,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            m, z = b(
                None,
                None,
                _offload_inference=True,
                _offloadable_inputs=input_tensors,
            )
            input_tensors[0] = m
            input_tensors[1] = z
            del m, z

        m, z = input_tensors

        s = self.linear(m[..., 0, :, :])

        return m, z, s

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            use_flash:
                Whether to use FlashAttention where possible. Mutually
                exclusive with use_lma and use_deepspeed_evo_attention.
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """

        if torch.onnx.is_in_onnx_export() or is_fx_tracing():
            inplace_safe = False
            chunk_size = None

        blocks = self._prep_blocks(
            m=m,
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        s = self.linear(m[..., 0, :, :])

        return m, z, s


class ExtraMSAStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
        eps: float,
        ckpt: bool,
        clear_cache_between_blocks: bool = False,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        super(ExtraMSAStack, self).__init__()

        self.ckpt = ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = ExtraMSABlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                eps=eps,
                ckpt=False,
            )
            self.blocks.append(block)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner(2048)

    def _prep_blocks(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool,
        use_cuequivariance_attention: bool,
        use_cuequivariance_multiplicative_update: bool,
        use_lma: bool,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if self.clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # Tensors cloned to avoid getting written to in-place
                # A corollary is that chunk size tuning should be disabled for
                # large N, when z gets really big
                args=(
                    m.clone(),
                    z.clone(),
                ),
                min_chunk_size=chunk_size,
            )

            # A temporary measure to address torch's occasional
            # inability to allocate large tensors
            attn_chunk = (
                tuned_chunk_size if use_cuequivariance_attention else (tuned_chunk_size // 4)
            )

            blocks = [
                partial(
                    b,
                    chunk_size=tuned_chunk_size,
                    _attn_chunk_size=max(chunk_size, attn_chunk),
                )
                for b in blocks
            ]

        return blocks

    def _forward_offload(
        self,
        input_tensors: Sequence[torch.Tensor],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        assert not (self.training or torch.is_grad_enabled())
        blocks = self._prep_blocks(
            # We are very careful not to create references to these tensors in
            # this function
            m=input_tensors[0],
            z=input_tensors[1],
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            use_lma=use_lma,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=True,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            m, z = b(
                None,
                None,
                _offload_inference=True,
                _offloadable_inputs=input_tensors,
            )
            input_tensors[0] = m
            input_tensors[1] = z
            del m, z

        return input_tensors[1]

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        chunk_size: int = None,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_res, C_m] extra MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            chunk_size: Inference-time subbatch size for Evoformer modules
            use_deepspeed_evo_attention: Whether to use DeepSpeed memory-efficient kernel
            use_lma: Whether to use low-memory attention during inference
            msa_mask:
                Optional [*, N_extra, N_res] MSA mask
            pair_mask:
                Optional [*, N_res, N_res] pair mask
        Returns:
            [*, N_res, N_res, C_z] pair update
        """

        if torch.onnx.is_in_onnx_export() or is_fx_tracing():
            inplace_safe = False
            chunk_size = None

        checkpoint_fn = get_checkpoint_fn()
        blocks = self._prep_blocks(
            m=m,
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            use_lma=use_lma,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            if self.ckpt and torch.is_grad_enabled():
                m, z = checkpoint_fn(b, m, z)
            else:
                m, z = b(m, z)

        return z


# --- MENAGERIE staging wrapper -----------------------------------------------
# EvoformerStack.forward() needs 4 tensor args (m, z, msa_mask, pair_mask); a
# recipe row supports only ONE concrete-tensor input, so this ships as a
# staging MODULE. EvoformerModuleWrapper.forward(m, z) builds the (all-ones)
# masks internally and forces `use_lma=True` -- the real chunked-softmax
# PyTorch attention path already in EvoformerBlock/MSA*Attention -- since the
# real code's *default* kwargs route through `use_memory_efficient_kernel=True`,
# which needs the openfold custom CUDA extension we deliberately did not vendor.


class EvoformerModuleWrapper(nn.Module):
    def __init__(self, evoformer_stack: "EvoformerStack"):
        super().__init__()
        self.evoformer_stack = evoformer_stack

    def forward(self, m: torch.Tensor, z: torch.Tensor):
        n_seq, n_res = m.shape[-3], m.shape[-2]
        msa_mask = m.new_ones(m.shape[:-3] + (n_seq, n_res))
        pair_mask = z.new_ones(z.shape[:-3] + (n_res, n_res))
        m_out, z_out, s_out = self.evoformer_stack(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=None,
            use_lma=True,
            inplace_safe=False,
            _mask_trans=True,
        )
        return m_out, z_out, s_out


def build_evoformer():
    stack = EvoformerStack(
        c_m=8,
        c_z=8,
        c_hidden_msa_att=4,
        c_hidden_opm=4,
        c_hidden_mul=8,
        c_hidden_pair_att=4,
        c_s=8,
        no_heads_msa=2,
        no_heads_pair=2,
        no_blocks=1,
        transition_n=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        no_column_attention=False,
        opm_first=False,
        fuse_projection_weights=False,
        blocks_per_ckpt=None,
        inf=1e5,
        eps=1e-8,
        clear_cache_between_blocks=False,
        tune_chunk_size=False,
    )
    stack.eval()
    return EvoformerModuleWrapper(stack)


def example_input_evoformer():
    n_seq, n_res = 3, 5
    m = torch.randn(1, n_seq, n_res, 8)
    z = torch.randn(1, n_res, n_res, 8)
    return (m, z)


MENAGERIE_ENTRIES = [
    (
        "EvoFormer",
        "build_evoformer",
        "example_input_evoformer",
        2021,
        "vendored-pytorch",
    ),
]

"""Pinned pretrained Flax checkpoints for the JAX ecosystem roster rows.

Transformers 5 removed its Flax model classes, but the checkpoint files are
still published: each repository below ships a ``flax_model.msgpack`` that
``flax.serialization`` reads directly. This module pins each file (repository,
commit, byte size, SHA-256, license), defines a small Flax NNX implementation
of the architecture whose attribute names are the checkpoint's own parameter
paths, and loads the weights through an exact two-way join: every checkpoint
tensor fills exactly one model parameter and every model parameter is filled,
with equal shapes, or the load raises. Nothing random survives a load.

The preprocessing is pinned too (token ids, image normalisation), so a row's
input does not depend on a tokenizer or image library at run time. The
independent reference for both families is the same repository revision run
through the PyTorch Transformers model
(``tests/test_parity_b_jcheck_reference.py``).

Framework imports occur only inside functions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

__tl_layer__ = "L9"


class CheckpointMismatchError(RuntimeError):
    """A downloaded file or its tensors differ from the pinned checkpoint."""


@dataclass(frozen=True)
class FlaxCheckpoint:
    """One pinned Flax checkpoint file and the reference weights beside it.

    Parameters
    ----------
    repo:
        Hugging Face repository id.
    revision:
        The full commit hash every download is pinned to.
    filename:
        The Flax weights file inside the repository.
    size:
        Its byte size.
    sha256:
        Its SHA-256 digest (the Git LFS object id).
    license:
        The repository's declared license.
    reference_filename:
        The PyTorch weights file at the same revision that the independent
        Transformers reference loads.
    reference_sha256:
        That file's SHA-256 digest.
    """

    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    license: str
    reference_filename: str
    reference_sha256: str

    @property
    def evidence(self) -> str:
        """The roster's ``checkpoint_evidence`` line for this file."""

        return (
            f"hf:{self.repo}@{self.revision} {self.filename}; "
            f"sha256:{self.sha256}; license:{self.license}"
        )


BERT_L2_H256 = FlaxCheckpoint(
    repo="google/bert_uncased_L-2_H-256_A-4",
    revision="4e937a8675e5afd9a4836735c186ec01695bc3ea",
    filename="flax_model.msgpack",
    size=38_365_627,
    sha256="eecb7af6123d287470edee59767cba257e77956c92abf31d4645210eeec61ace",
    license="apache-2.0",
    reference_filename="pytorch_model.bin",
    reference_sha256="67b710645b099510dba08b7605a60859fff59714628e1d9d1ffa9e0956b51783",
)

VIT_B16_IN21K = FlaxCheckpoint(
    repo="google/vit-base-patch16-224-in21k",
    revision="b4569560a39a0f1af58e3ddaf17facf20ab919b0",
    filename="flax_model.msgpack",
    size=345_564_359,
    sha256="aa00da5f7abb687576e6f9138d48cbb0fe7489f2bc9768793d262c3987b3db32",
    license="apache-2.0",
    reference_filename="model.safetensors",
    reference_sha256="fd4e1169c7aa6c2dbfa8a6448be13b35abc0ee256190857c90009d12c094619b",
)

#: The BERT row's sentence and its ids from the repository's ``vocab.txt``
#: (``BertTokenizer`` at the pinned revision; the reference test re-derives them).
BERT_PROMPT = "TorchLens traces JAX models."
BERT_INPUT_IDS: tuple[int, ...] = (101, 12723, 7770, 2015, 10279, 13118, 4275, 1012, 102)

#: The ViT row's image: seeded 224x224 RGB bytes, already at the model's size.
VIT_IMAGE_SEED = 20261112
VIT_IMAGE_SIZE = 224
#: ``preprocessor_config.json`` at the pinned revision: rescale by 1/255, then
#: normalise with mean 0.5 and std 0.5 per channel.
VIT_RESCALE = 1.0 / 255.0
VIT_MEAN = 0.5
VIT_STD = 0.5

#: Both configs set ``layer_norm_eps`` to 1e-12 and ``hidden_act`` to exact GELU.
_LAYER_NORM_EPS = 1e-12


def sha256_of(path: str) -> str:
    """The SHA-256 hex digest of the file at ``path``, read in chunks."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str, checkpoint: FlaxCheckpoint) -> None:
    """Raise :class:`CheckpointMismatchError` unless ``path`` is the pinned file."""

    import os

    size = os.path.getsize(path)
    if size != checkpoint.size:
        raise CheckpointMismatchError(
            f"{checkpoint.repo} {checkpoint.filename}: {size} bytes, pinned {checkpoint.size}"
        )
    digest = sha256_of(path)
    if digest != checkpoint.sha256:
        raise CheckpointMismatchError(
            f"{checkpoint.repo} {checkpoint.filename}: sha256 {digest}, pinned {checkpoint.sha256}"
        )


def fetch(checkpoint: FlaxCheckpoint, filename: str | None = None) -> str:
    """Download one file of ``checkpoint`` at its pinned revision; return the local path.

    The Flax weights file is verified against its pinned size and digest.
    """

    from huggingface_hub import hf_hub_download

    name = filename or checkpoint.filename
    path = str(hf_hub_download(checkpoint.repo, name, revision=checkpoint.revision))
    if name == checkpoint.filename:
        verify_file(path, checkpoint)
    return path


@cache
def load_flat_params(checkpoint: FlaxCheckpoint) -> dict[str, np.ndarray]:
    """The verified checkpoint's tensors keyed by ``/``-joined parameter path.

    Cached per process; the arrays are read-only so no row can alter another's.
    """

    from flax import serialization
    from flax.traverse_util import flatten_dict

    with open(fetch(checkpoint), "rb") as handle:
        tree = serialization.msgpack_restore(handle.read())
    flat: dict[str, np.ndarray] = {}
    for key, value in flatten_dict(tree).items():
        array = np.array(value, dtype=np.float32)
        array.setflags(write=False)
        flat["/".join(str(part) for part in key)] = array
    return flat


def assign_exact(model: Any, flat: dict[str, np.ndarray]) -> int:
    """Fill every ``nnx.Param`` of ``model`` from ``flat`` by identical path.

    Returns the number of tensors assigned. Raises
    :class:`CheckpointMismatchError` when a checkpoint tensor has no parameter,
    a parameter has no tensor, or a shape differs.
    """

    import jax.numpy as jnp
    from flax import nnx

    params = {
        "/".join(str(part) for part in path): variable
        for path, variable in nnx.iter_graph(model)
        if isinstance(variable, nnx.Param)
    }
    unused = sorted(set(flat) - set(params))
    unfilled = sorted(set(params) - set(flat))
    if unused or unfilled:
        raise CheckpointMismatchError(
            f"weight mapping incomplete: checkpoint-only {unused[:5]}, model-only {unfilled[:5]}"
        )
    for path, variable in params.items():
        if tuple(variable.value.shape) != flat[path].shape:
            raise CheckpointMismatchError(
                f"{path}: model shape {tuple(variable.value.shape)} != {flat[path].shape}"
            )
        variable.value = jnp.asarray(flat[path])
    return len(params)


# ----------------------------------------------------------------- NNX layers


@cache
def _layer_classes() -> dict[str, Any]:
    """The shared NNX building blocks, defined once on first use (Flax imported here)."""

    import jax
    import jax.numpy as jnp
    from flax import nnx

    def layer_norm(width: int, rngs: Any) -> Any:
        """Two-pass LayerNorm with the configs' epsilon (torch's variance form)."""

        return nnx.LayerNorm(width, epsilon=_LAYER_NORM_EPS, use_fast_variance=False, rngs=rngs)

    class Indexed(nnx.Module):
        """Children under the attributes ``0``, ``1``, ... (the checkpoint's layer keys)."""

        def __init__(self, children: list[Any]) -> None:
            for index, child in enumerate(children):
                setattr(self, str(index), child)
            self.count = len(children)

        def __call__(self, hidden: Any) -> Any:
            """Apply the children in index order."""

            for index in range(self.count):
                hidden = getattr(self, str(index))(hidden)
            return hidden

    class QKV(nnx.Module):
        """Multi-head scaled dot-product self-attention, no mask (one unpadded input)."""

        def __init__(self, width: int, heads: int, rngs: Any) -> None:
            self.query = nnx.Linear(width, width, rngs=rngs)
            self.key = nnx.Linear(width, width, rngs=rngs)
            self.value = nnx.Linear(width, width, rngs=rngs)
            self.heads = heads

        def __call__(self, hidden: Any) -> Any:
            """Attend every position to every position; return the merged heads."""

            batch, length, width = hidden.shape
            depth = width // self.heads

            def split(values: Any) -> Any:
                return values.reshape(batch, length, self.heads, depth)

            scores = jnp.einsum(
                "bqhd,bkhd->bhqk", split(self.query(hidden)), split(self.key(hidden))
            )
            probs = jax.nn.softmax(scores / math.sqrt(depth), axis=-1)
            context = jnp.einsum("bhqk,bkhd->bqhd", probs, split(self.value(hidden)))
            return context.reshape(batch, length, width)

    class Dense(nnx.Module):
        """One affine map under the checkpoint key ``dense``."""

        def __init__(self, width_in: int, width_out: int, rngs: Any) -> None:
            self.dense = nnx.Linear(width_in, width_out, rngs=rngs)

        def __call__(self, hidden: Any) -> Any:
            """Apply the affine map."""

            return self.dense(hidden)

    class Pooler(nnx.Module):
        """``tanh`` of an affine map of the first position."""

        def __init__(self, width: int, rngs: Any) -> None:
            self.dense = nnx.Linear(width, width, rngs=rngs)

        def __call__(self, hidden: Any) -> Any:
            """Pool the first token."""

            return jnp.tanh(self.dense(hidden[:, 0]))

    return {
        "layer_norm": layer_norm,
        "Indexed": Indexed,
        "QKV": QKV,
        "Dense": Dense,
        "Pooler": Pooler,
    }


@cache
def _bert_class() -> type:
    """The BERT encoder of ``google/bert_uncased_L-2_H-256_A-4`` with its pooler."""

    import jax
    import jax.numpy as jnp
    from flax import nnx

    blocks = _layer_classes()
    layer_norm, dense_cls = blocks["layer_norm"], blocks["Dense"]

    class BertEmbeddings(nnx.Module):
        """Word, position and token-type embeddings, summed and normalised."""

        def __init__(self, rngs: Any) -> None:
            self.word_embeddings = nnx.Embed(30522, 256, rngs=rngs)
            self.position_embeddings = nnx.Embed(512, 256, rngs=rngs)
            self.token_type_embeddings = nnx.Embed(2, 256, rngs=rngs)
            self.LayerNorm = layer_norm(256, rngs)

        def __call__(self, ids: Any) -> Any:
            """Embed one segment (token type 0) at positions 0..n-1."""

            positions = jnp.arange(ids.shape[1])[None, :]
            summed = (
                self.word_embeddings(ids)
                + self.position_embeddings(positions)
                + self.token_type_embeddings(jnp.zeros_like(ids))
            )
            return self.LayerNorm(summed)

    class ResidualNorm(nnx.Module):
        """``LayerNorm(dense(x) + residual)``, BERT's post-norm sublayer output."""

        def __init__(self, width_in: int, width_out: int, rngs: Any) -> None:
            self.dense = nnx.Linear(width_in, width_out, rngs=rngs)
            self.LayerNorm = layer_norm(width_out, rngs)

        def __call__(self, hidden: Any, residual: Any) -> Any:
            """Project, add the residual, normalise."""

            return self.LayerNorm(self.dense(hidden) + residual)

    class BertAttention(nnx.Module):
        """Self-attention (checkpoint key ``self``) and its residual output."""

        def __init__(self, rngs: Any) -> None:
            self.self = blocks["QKV"](256, 4, rngs)
            self.output = ResidualNorm(256, 256, rngs)

        def __call__(self, hidden: Any) -> Any:
            """Attend, then project with the residual."""

            return self.output(self.self(hidden), hidden)

    class BertLayer(nnx.Module):
        """Attention, exact-GELU feed-forward, two post-norm residuals."""

        def __init__(self, rngs: Any) -> None:
            self.attention = BertAttention(rngs)
            self.intermediate = dense_cls(256, 1024, rngs)
            self.output = ResidualNorm(1024, 256, rngs)

        def __call__(self, hidden: Any) -> Any:
            """One encoder block."""

            attended = self.attention(hidden)
            expanded = jax.nn.gelu(self.intermediate(attended), approximate=False)
            return self.output(expanded, attended)

    class BertEncoder(nnx.Module):
        """The two encoder blocks under ``layer/0`` and ``layer/1``."""

        def __init__(self, rngs: Any) -> None:
            self.layer = blocks["Indexed"]([BertLayer(rngs) for _ in range(2)])

        def __call__(self, hidden: Any) -> Any:
            """Run both blocks."""

            return self.layer(hidden)

    class Bert(nnx.Module):
        """``(last_hidden_state, pooler_output)`` for a batch of token ids."""

        def __init__(self, rngs: Any) -> None:
            self.embeddings = BertEmbeddings(rngs)
            self.encoder = BertEncoder(rngs)
            self.pooler = blocks["Pooler"](256, rngs)

        def __call__(self, ids: Any) -> Any:
            """Encode the ids and pool the first token."""

            sequence = self.encoder(self.embeddings(ids))
            return sequence, self.pooler(sequence)

    return Bert


@cache
def _vit_class() -> type:
    """The ViT-B/16 encoder of ``google/vit-base-patch16-224-in21k`` with its pooler."""

    import jax
    import jax.numpy as jnp
    from flax import nnx

    blocks = _layer_classes()
    layer_norm, dense_cls = blocks["layer_norm"], blocks["Dense"]
    width = 768

    class PatchEmbeddings(nnx.Module):
        """The 16x16 stride-16 patch projection (checkpoint key ``projection``)."""

        def __init__(self, rngs: Any) -> None:
            self.projection = nnx.Conv(
                3, width, kernel_size=(16, 16), strides=(16, 16), padding="VALID", rngs=rngs
            )

        def __call__(self, pixels: Any) -> Any:
            """NHWC pixels to a (batch, 196, 768) patch sequence."""

            patches = self.projection(pixels)
            return patches.reshape(patches.shape[0], -1, width)

    class ViTEmbeddings(nnx.Module):
        """Class token, patch sequence and learned position embeddings."""

        def __init__(self, rngs: Any) -> None:
            self.cls_token = nnx.Param(jnp.zeros((1, 1, width), jnp.float32))
            self.position_embeddings = nnx.Param(jnp.zeros((1, 197, width), jnp.float32))
            self.patch_embeddings = PatchEmbeddings(rngs)

        def __call__(self, pixels: Any) -> Any:
            """Prepend the class token and add positions."""

            patches = self.patch_embeddings(pixels)
            cls = jnp.broadcast_to(self.cls_token.value, (patches.shape[0], 1, width))
            return jnp.concatenate([cls, patches], axis=1) + self.position_embeddings.value

    class ViTAttention(nnx.Module):
        """Self-attention (checkpoint key ``attention``) and its output projection."""

        def __init__(self, rngs: Any) -> None:
            self.attention = blocks["QKV"](width, 12, rngs)
            self.output = dense_cls(width, width, rngs)

        def __call__(self, hidden: Any) -> Any:
            """Attend, then project."""

            return self.output(self.attention(hidden))

    class ViTLayer(nnx.Module):
        """Pre-norm block: attention and exact-GELU feed-forward, each residual."""

        def __init__(self, rngs: Any) -> None:
            self.attention = ViTAttention(rngs)
            self.intermediate = dense_cls(width, 3072, rngs)
            self.output = dense_cls(3072, width, rngs)
            self.layernorm_before = layer_norm(width, rngs)
            self.layernorm_after = layer_norm(width, rngs)

        def __call__(self, hidden: Any) -> Any:
            """One encoder block."""

            hidden = self.attention(self.layernorm_before(hidden)) + hidden
            expanded = jax.nn.gelu(
                self.intermediate(self.layernorm_after(hidden)), approximate=False
            )
            return self.output(expanded) + hidden

    class ViTEncoder(nnx.Module):
        """The twelve encoder blocks under ``layer/0`` .. ``layer/11``."""

        def __init__(self, rngs: Any) -> None:
            self.layer = blocks["Indexed"]([ViTLayer(rngs) for _ in range(12)])

        def __call__(self, hidden: Any) -> Any:
            """Run every block."""

            return self.layer(hidden)

    class ViT(nnx.Module):
        """``(last_hidden_state, pooler_output)`` for NHWC pixels."""

        def __init__(self, rngs: Any) -> None:
            self.embeddings = ViTEmbeddings(rngs)
            self.encoder = ViTEncoder(rngs)
            self.layernorm = layer_norm(width, rngs)
            self.pooler = blocks["Pooler"](width, rngs)

        def __call__(self, pixels: Any) -> Any:
            """Encode the image and pool the class token."""

            sequence = self.layernorm(self.encoder(self.embeddings(pixels)))
            return sequence, self.pooler(sequence)

    return ViT


# ----------------------------------------------------------------- inputs and builders


def vit_image_bytes() -> np.ndarray:
    """The ViT row's seeded (224, 224, 3) uint8 RGB image."""

    rng = np.random.default_rng(VIT_IMAGE_SEED)
    return rng.integers(0, 256, (VIT_IMAGE_SIZE, VIT_IMAGE_SIZE, 3), dtype=np.uint8)


def vit_pixels() -> np.ndarray:
    """The pinned preprocessing of :func:`vit_image_bytes`: a (1, 224, 224, 3) NHWC batch."""

    scaled = vit_image_bytes().astype(np.float32) * np.float32(VIT_RESCALE)
    return ((scaled - np.float32(VIT_MEAN)) / np.float32(VIT_STD))[None]


def build_bert() -> tuple[Any, Any]:
    """The pretrained BERT (every weight from the pinned file) and its token ids."""

    import jax.numpy as jnp
    from flax import nnx

    model = _bert_class()(nnx.Rngs(0))
    assign_exact(model, load_flat_params(BERT_L2_H256))
    return model, jnp.asarray([BERT_INPUT_IDS], dtype=jnp.int32)


def build_vit() -> tuple[Any, Any]:
    """The pretrained ViT-B/16 (every weight from the pinned file) and its pixels."""

    import jax.numpy as jnp
    from flax import nnx

    model = _vit_class()(nnx.Rngs(0))
    assign_exact(model, load_flat_params(VIT_B16_IN21K))
    return model, jnp.asarray(vit_pixels())


__all__ = (
    "BERT_INPUT_IDS",
    "BERT_L2_H256",
    "BERT_PROMPT",
    "VIT_B16_IN21K",
    "CheckpointMismatchError",
    "FlaxCheckpoint",
    "assign_exact",
    "build_bert",
    "build_vit",
    "fetch",
    "load_flat_params",
    "verify_file",
    "vit_image_bytes",
    "vit_pixels",
)

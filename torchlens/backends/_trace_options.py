"""Per-backend trace-option vocabularies (leaf module: no registry import).

``torchlens.backends.registry`` re-exports every name here; the protocol shells
read their option list from this leaf so constructing a shell never imports
the registry that is constructing it.
"""

from __future__ import annotations

__tl_layer__ = "L2"


TRACE_OPTION_CAPABILITY_EPOCHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "epoch1_module_modes_control_flow",
        (
            "jax_control_flow",
            "jax_max_control_flow_unroll",
            "module_identity_mode",
        ),
    ),
    ("epoch2_tinygrad_t1", ()),
    ("epoch3_codec_materialization", ("payload_policy", "save_preview")),
    ("epoch4_intermediate_derived_grads", ()),
    ("epoch5_container_structure", ("capture_container_structure",)),
    ("epoch6_inference_only", ("inference_only",)),
    ("epoch7_semantic_output_decode", ()),
    ("epoch8_forward_chunking", ("chunk_size", "chunk_paths")),
)
"""Ordered public trace-option capability epochs.

Each epoch must update registry capabilities, ``CaptureOptions``, cache-key
coverage, docs, and tests in one patch.
"""


PUBLIC_OPTION_SPINE_TRACE_OPTIONS: tuple[str, ...] = tuple(
    option for _epoch_name, options in TRACE_OPTION_CAPABILITY_EPOCHS for option in options
)
"""Trace options declared by the public-option API spine."""


TORCH_TRACE_OPTIONS: tuple[str, ...] = PUBLIC_OPTION_SPINE_TRACE_OPTIONS
"""Public trace-option names accepted by torch.

Torch implements container-structure capture, inference-only capture,
semantic output decode, and forward chunking. Backend-preview-only controls
such as JAX control-flow selection, non-default module identity modes,
payload-policy overrides, and save-preview mode are value-validated by the
torch entrypoint rather than silently treated as operational options.
"""


JAX_TRACE_OPTIONS: tuple[str, ...] = (
    "jax_static_argnums",
    "grad_options",
    "jax_control_flow",
    "jax_max_control_flow_unroll",
    "module_identity_mode",
)
"""Trace options implemented by the JAX preview backend."""


MLX_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the MLX preview backend."""


TINYGRAD_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the tinygrad preview backend."""


PADDLE_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the Paddle preview backend."""


TF_TRACE_OPTIONS: tuple[str, ...] = ("module_identity_mode", "grad_options")
"""Trace options implemented by the TensorFlow preview backend."""

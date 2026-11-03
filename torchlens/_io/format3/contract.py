"""Constants of the 3.0 saved format, pinned against the persisted-field contract.

Everything here is data: the envelope, the record index, the type map from
``StorageKind`` to a JSON encoding class, the typed-value tags, the tensor
reference fields and the refusal code table. The writer, reader and runnable
modules import these names; ``tests/test_format3_contract.py`` pins them to
``docs/maintainers/persisted_fields.md`` so the contract and the generated
field page cannot drift apart. The prose contract is
``docs/maintainers/saved_format_3.md``.
"""

from __future__ import annotations

from typing import Final, NoReturn

from ..._data_classes.field_policy import StorageKind
from ..format_errors import ArtifactError

#: Manifest ``format`` value; anything else is not this format.
FORMAT_NAME: Final = "torchlens.saved_format"
#: Manifest ``format_version``: the integer major of the saved format. The
#: reader accepts exactly this value.
FORMAT_MAJOR: Final = 3
#: The record contract (tlspec) version the constants below are pinned to.
#: The writer lane bumps it together with the added keys and the generated
#: field page; the pin test compares it with the page header.
RECORD_CONTRACT_VERSION: Final = 12
#: Lowest record contract version the 3.0 reader accepts. Nothing older than
#: the format itself is readable; a 2.x artifact refuses at the manifest.
RECORD_CONTRACT_FLOOR: Final = 12

#: Manifest file name at the artifact root; everything else is indexed from it.
MANIFEST_FILE: Final = "manifest.json"
#: Directory of one JSON file per record kind (``records/op.json``).
RECORDS_DIR: Final = "records"
#: Directory of safetensors files holding every tensor payload.
BLOBS_DIR: Final = "blobs"
#: The intervention recipe file (rules and built-in edits as typed JSON).
INTERVENTION_FILE: Final = "intervention.json"
#: Directory of the runnable component (descriptor, state dict, buffers).
RUNNABLE_DIR: Final = "runnable"

#: Manifest keys every writer must stamp (identity values are ledger data).
WRITER_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "torchlens_version",
    "python_version",
    "torch_version",
    "platform",
    "created_at",
)
#: Manifest keys the reader requires before it opens any other file.
MANIFEST_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "format",
    "format_version",
    "tlspec_version",
    "kind",
    "save_level",
    "payload_policy",
    "contract_digest",
    "codecs",
    "records",
    "files",
    *WRITER_IDENTITY_FIELDS,
)
#: Artifact kinds the manifest ``kind`` may carry.
ARTIFACT_KINDS: Final[tuple[str, ...]] = ("trace", "bundle", "intervention")
#: Payload codec ids the format writes; both are safetensors files.
PAYLOAD_CODEC_IDS: Final[tuple[str, ...]] = ("torch_safetensors_v1", "numpy_safetensors_v1")
#: Save levels, as ``torchlens._io.save_policy`` spells them.
SAVE_LEVELS: Final[tuple[str, ...]] = (
    "portable",
    "audit",
    "runnable",
    "executable_with_callables",
)

#: The twelve record kinds in page order: (record, class path, persisted keys).
#: The key counts are the tlspec 12 numbers of the generated field page.
RECORD_INDEX: Final[tuple[tuple[str, str, int], ...]] = (
    ("trace", "torchlens._data_classes.trace.Trace", 183),
    ("op", "torchlens._data_classes.op.Op", 176),
    ("layer", "torchlens._data_classes.layer.Layer", 92),
    ("aten_op", "torchlens._data_classes.aten_op.AtenOp", 32),
    ("backward_pass", "torchlens._data_classes.backward_pass.BackwardPass", 20),
    ("buffer", "torchlens._data_classes.buffer.Buffer", 21),
    ("func_call_location", "torchlens._data_classes.func_call_location.FuncCallLocation", 23),
    ("grad_fn", "torchlens._data_classes.grad_fn.GradFn", 53),
    ("grad_fn_call", "torchlens._data_classes.grad_fn_call.GradFnCall", 14),
    ("module", "torchlens._data_classes.module.Module", 82),
    ("module_call", "torchlens._data_classes.module.ModuleCall", 51),
    ("param", "torchlens._data_classes.param.Param", 34),
)
#: Persisted keys across the twelve records at ``RECORD_CONTRACT_VERSION``.
PERSISTED_KEY_COUNT: Final = sum(count for _, _, count in RECORD_INDEX)

#: Keys a later contract adds, per record, before they reach the bindings and
#: the generated page; a key leaves the plan in the change that persists it.
#: Empty at tlspec 12: ``op.arg_expressions`` and ``op.is_constant`` landed.
PLANNED_KEYS: Final[dict[str, tuple[str, ...]]] = {}
#: Persisted keys whose VALUE the writer settles on the live side so a loaded
#: trace equals the live one: today the portable scrub drops these values and
#: the loader settles the grouping policy after the fact.
LIVE_SIDE_VALUE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "op": ("equivalence_class", "func_duration", "edge_substitutions", "edge_replacement_stamps"),
    "trace": ("_options_grouping_policy",),
}

#: JSON encoding class per storage kind. ``value`` is the ordinary JSON
#: rendering of the Python value (null, bool, number, string, array, object,
#: with typed values tagged); ``id_array`` is an array of row references;
#: ``tensor_ref`` is a tensor reference object or a container of them;
#: ``never`` is never written and refuses as an unknown key when read.
JSON_TYPE_BY_STORAGE_KIND: Final[dict[StorageKind, str]] = {
    StorageKind.SCALAR: "value",
    StorageKind.INTERNED: "value",
    StorageKind.COMPUTED: "value",
    StorageKind.GROUP: "value",
    StorageKind.EDGE: "id_array",
    StorageKind.BITSET: "id_array",
    StorageKind.PAYLOAD: "tensor_ref",
    StorageKind.RUNTIME: "never",
}

#: Version of the built-in edit registry a recipe records (``helpers[].version``).
HELPER_REGISTRY_VERSION: Final = "1"

#: The key that marks a typed JSON object; its value is one of the tags below.
TYPED_VALUE_KEY: Final = "$tl"
#: Typed-value tags. Every Python value JSON cannot carry directly is one of
#: these objects; a user mapping that itself uses the marker key is written
#: through the ``dict`` form so the two never collide.
TYPED_VALUE_TAGS: Final[tuple[str, ...]] = (
    "tensor",
    "tuple",
    "set",
    "frozenset",
    "dict",
    "bytes",
    "float",
    "int",
    "complex",
    "dtype",
    "device",
    "type",
    "enum",
    "record",
    "ref",
    "unencodable",
    "callable_placeholder",
)
#: Where the ``record`` tag's type allowlist comes from: the writer contract's
#: ``component_specs`` classes (``field_registry.component_spec_tables``) plus
#: the value classes listed beside the codec in ``format3/values.py``
#: (quantities, dtype and device refs, ...); a ``record`` object of any other
#: type refuses. ``ref`` cells point at a row of another record table
#: (``{"record": kind, "row": index}``) where a live field holds a record object.
RECORD_TYPE_ALLOWLIST_SOURCE: Final = "torchlens._io.field_registry.component_spec_tables"
#: Fields of a tensor reference object (tag ``tensor``). ``blob`` and ``key``
#: are absent in a structure-only artifact; ``complex`` marks a tensor stored
#: as its real view.
TENSOR_REFERENCE_FIELDS: Final[tuple[str, ...]] = (
    "blob",
    "key",
    "dtype",
    "shape",
    "device",
    "requires_grad",
    "codec",
    "complex",
)
#: Fields of an ``unencodable`` disclosure object: the stringified user value.
UNENCODABLE_FIELDS: Final[tuple[str, ...]] = ("type", "repr", "repr_length", "at")
#: Longest ``repr`` text an ``unencodable`` disclosure keeps.
UNENCODABLE_REPR_LIMIT: Final = 256
#: Fields of a ``callable_placeholder`` in a saved intervention recipe.
CALLABLE_PLACEHOLDER_FIELDS: Final[tuple[str, ...]] = ("position", "qualname", "repr_length")
#: Public type an ``unencodable`` disclosure loads as (ruled 2 Nov 2026).
UNENCODABLE_VALUE_TYPE: Final = "UnencodableValue"
#: Load advisory attached when the manifest carries ``disclosures`` (ruled 2 Nov 2026).
UNENCODABLE_LOAD_ADVISORY: Final = "unencodable_values_stringified"
#: Labels ``tl.io.detect_tlspec_format`` returns: a 3.0 save, a 2.x save, not a
#: TorchLens save (ruled 2 Nov 2026).
FORMAT_DETECTION_LABELS: Final[tuple[str, ...]] = ("v3", "pre_3", "unknown")

#: Refusal codes the writer and reader raise, by situation. Every value is a
#: row of ``docs/reference/error_refusal_contract.md``; situations whose row
#: text the writer lane rewrites are noted in the prose contract.
REFUSAL_CODES: Final[dict[str, str]] = {
    "manifest_missing": "manifest_missing",
    "manifest_unreadable": "manifest_unreadable",
    "manifest_not_object": "manifest_not_json_object",
    "manifest_invalid": "manifest_schema_invalid",
    "format_older_than_floor": "artifact_version_below_floor",
    "format_newer_than_runtime": "artifact_version_above_runtime",
    "kind_mismatch": "artifact_kind_mismatch",
    "save_level_unknown": "artifact_save_level_invalid",
    "record_file_corrupt": "bundle_metadata_integrity_refused",
    "record_key_unknown": "unknown_persisted_field",
    "record_schema_invalid": "record_schema_invalid",
    "blob_missing": "bundle_load_failed",
    "structure_only_incoherent": "artifact_structure_only_incoherent",
    "symlink_path": "load_path_symlink_rejected",
    "payload_tensor_unencodable": "metadata_value_unportable",
    "payload_flag_conflict": "save_payload_level_conflict",
    "target_exists": "bundle_save_failed",
    "rerun_with_callable_placeholder": "run_intervention_spec_not_persisted",
    "writer_not_built": "bundle_save_failed",
    "reader_not_built": "bundle_load_failed",
}
#: Situations whose code is provisional: a private spelling builders can use
#: until the public code is ruled, each mapped to the existing code raised
#: meanwhile. Empty: every spelling the format needed has been ruled.
PROVISIONAL_REFUSALS: Final[dict[str, str]] = {}

#: Where the per-step timing keys come from: ``_phase_timings`` is keyed by the
#: ``name`` of each step of this tuple, nothing else.
PHASE_TIMING_KEY_SOURCE: Final = "torchlens._postprocess._pipeline.PIPELINE"


def refuse_writer_not_built(door: str, lane_module: str) -> NoReturn:
    """Refuse a 3.0 write door that its lane has not filled yet.

    Parameters
    ----------
    door:
        Public spelling of the door (``write_trace``).
    lane_module:
        Repo-relative module the lane fills (``torchlens/_io/format3/writer.py``).

    Raises
    ------
    ArtifactError
        ``bundle_save_failed``: the door exists as a contract stub only.
    """

    raise ArtifactError(
        f"The 3.0 saved-format door {door} is not built in this checkout; it is filled in "
        f"{lane_module}.",
        code="bundle_save_failed",
        remedy="save with Trace.save (the tlspec 11 writer) until the 3.0 writer lands",
    )


def refuse_reader_not_built(door: str, lane_module: str) -> NoReturn:
    """Refuse a 3.0 read door that its lane has not filled yet.

    Parameters
    ----------
    door:
        Public spelling of the door (``load_trace``).
    lane_module:
        Repo-relative module the lane fills (``torchlens/_io/format3/reader.py``).

    Raises
    ------
    ArtifactError
        ``bundle_load_failed``: the door exists as a contract stub only.
    """

    raise ArtifactError(
        f"The 3.0 saved-format door {door} is not built in this checkout; it is filled in "
        f"{lane_module}.",
        code="bundle_load_failed",
        remedy="load with tl.load (the tlspec 11 reader) until the 3.0 reader lands",
    )

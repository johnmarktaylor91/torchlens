"""r7 R80 (fable b10 MED x2): freeze the sprint-token test-naming namespace.

Two naming diseases grew unchecked through six rounds:

1. **rN-token filenames**: 141 test files carry a bare sprint-round
   token (``test_r55_*``, ``*_r10r60``), and 32 distinct tokens appear in MORE
   THAN ONE file under DIFFERENT numbering schemes (grind rounds, W-lanes,
   hunt batches) -- ``test_r55_entry_and_context.py`` and
   ``test_tlspec_uninit_consumer_r55.py`` are unrelated. The token carries no
   meaning once its sprint ends and actively misleads (a reader cannot tell
   which "r55" a file belongs to).
2. **sprint-lane grab-bags**: ``test_grind_b4_lane_p.py`` grew 516 -> 591
   lines across fixwave-5 as unrelated deposits accreted onto a lane-named
   dump file.

This lint GRANDFATHERS the existing census (shrink-only: a rename that drops
the token deletes its ledger row) and refuses NEW deposits: name new test
files after the SUBSYSTEM AND CONTRACT under test, not the sprint that found
the bug.
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parent

#: Filename stems carrying a sprint-round token (r7 R80 census, 2026-08-16).
#: SHRINK-ONLY: never add a row -- name new files after subsystem + contract.
_LEDGERED_SPRINT_TOKEN_FILES: frozenset[str] = frozenset(
    {
        "test_generative_properties_r73.py",
        "test_io_error_typing_r65.py",
        "test_io_reconstructible_predicate_r10.py",
        "test_io_rollup_hardening_r10r60.py",
        "test_io_runnable_slot_id_uniqueness_r10.py",
        "test_io_security_r27.py",
        "test_postprocess_r14a_regressions.py",
        "test_r10_container_rce.py",
        "test_r11_dup_resolver_rce.py",
        "test_r12_import_ref_rce.py",
        "test_r18a_compat_report.py",
        "test_r18b_bridge_hf.py",
        "test_r18bauto_autoroute.py",
        "test_r18c_bridge_adapters.py",
        "test_r18cg_capture_gap.py",
        "test_r18d_report.py",
        "test_r18e_debug.py",
        "test_r18f_semantic_observers.py",
        "test_r18g_tensor_rng.py",
        "test_r18h_introspect_hash.py",
        "test_r18i2_options_residue.py",
        "test_r18i3_utils_residue.py",
        "test_r18i_utils_options.py",
        "test_r18j_recurrent_render.py",
        "test_r18l_viz_rank.py",
        "test_r18m_safe_unpickle_alloc.py",
        "test_r18n_bundle_save_atomic.py",
        "test_r18o_scrub.py",
        "test_r18p_streaming_lazy.py",
        "test_r18rf_render_flow.py",
        "test_r19a_attribution.py",
        "test_r19b_rf_gradient_verify.py",
        "test_r19c_cache_capability.py",
        "test_r19d_hash_scalar.py",
        "test_r19f_rf_view_table.py",
        "test_r20a_extras.py",
        "test_r20capprov_witness.py",
        "test_r20i_ir_livelane.py",
        "test_r20j_container.py",
        "test_r20l_ir_refs.py",
        "test_r23_type_reduce_rce.py",
        "test_r24_container_construction_gadget.py",
        "test_r25_container_final.py",
        "test_r26_denylist_completion.py",
        "test_r27_dotted_walk_rce.py",
        "test_r28_capture_attribution.py",
        "test_r28_rebuild_dotted_walk_rce.py",
        "test_r28_validation_identity_witness.py",
        "test_r29_capval_hardening.py",
        "test_r31_stdlib_deny.py",
        "test_r33_operator_gadget_carveout.py",
        "test_r34_security_io.py",
        "test_r36_cuda_hygiene.py",
        "test_r36_tensor_method_smuggling.py",
        "test_r39_callable_invoke_denylist.py",
        "test_r3_torch_namespace_rce.py",
        "test_r41_gate_completeness.py",
        "test_r43_gate_inversion.py",
        "test_r45_property_classification.py",
        "test_r45_torch_symbol_decode.py",
        "test_r47_crossthread_op_surface.py",
        "test_r47_dataclass_foreign_init.py",
        "test_r47_forward_dunder_gate.py",
        "test_r47_generator_container_subclass.py",
        "test_r49_empty_mp_queue.py",
        "test_r49_private_c_op_surface.py",
        "test_r49_reconstruction_substitution.py",
        "test_r49_unpickler_weights_only_baseline.py",
        "test_r4_metadata_unpickle_rce.py",
        "test_r55_entry_and_context.py",
        "test_r58_capture_cache_rce.py",
        "test_r58_merged_distributed_rce.py",
        "test_r59_writer_toctou.py",
        "test_r5_from_file_io_rce.py",
        "test_r63_construct_deny.py",
        "test_r6_state_mutator_rce.py",
        "test_r7_torch_type_io_rce.py",
        "test_r8_operator_gate_rce.py",
        "test_r9_appliance_import_rce.py",
        "test_rng_monitor_escapes_r3.py",
        "test_rng_seal_r57.py",
        "test_runnable_r36_regressions.py",
        "test_tlspec_runnable_r15_writeclass.py",
        "test_tlspec_runnable_r16_writeclass_temporal.py",
        "test_tlspec_runnable_r18_param_escape_parity.py",
        "test_tlspec_runnable_r19_param_buffer_writemask.py",
        "test_tlspec_runnable_r20_state_toctou_attest.py",
        "test_tlspec_runnable_r21_attest_mha_dropout.py",
        "test_tlspec_runnable_r22_attest_tripwire_inf.py",
        "test_tlspec_runnable_r23_structseq_invert.py",
        "test_tlspec_runnable_r24_idiv_bufwrite.py",
        "test_tlspec_runnable_r25_producer.py",
        "test_tlspec_runnable_r29_aliasing.py",
        "test_tlspec_runnable_r29_input_metadata_class.py",
        "test_tlspec_runnable_r29_norm_attestation.py",
        "test_tlspec_runnable_r29_structural_tree.py",
        "test_tlspec_runnable_r31_input_metadata_witness.py",
        "test_tlspec_runnable_r35_admission_context.py",
        "test_tlspec_runnable_r35_attestation_lattice.py",
        "test_tlspec_runnable_r35_exact_semantics.py",
        "test_tlspec_runnable_r35_nonpersistent_buffers.py",
        "test_tlspec_runnable_r39_bounded_holders.py",
        "test_tlspec_runnable_r39_witness_exec.py",
        "test_tlspec_runnable_r41_crossthread_witness.py",
        "test_tlspec_runnable_r45_crossthread_provenance.py",
        "test_tlspec_runnable_r45_rng_inventory.py",
        "test_tlspec_runnable_r53_alloc_bomb.py",
        "test_tlspec_runnable_r53_ambient_grad.py",
        "test_tlspec_runnable_r53_uninit_alloc.py",
        "test_tlspec_runnable_r59_alloc_ceiling.py",
        "test_tlspec_runnable_r59_slots_rng.py",
        "test_tlspec_runnable_r61_alloc_projection.py",
        "test_tlspec_runnable_r61_gc_inert_edges.py",
        "test_tlspec_runnable_r63_state_metadata.py",
        "test_tlspec_runnable_r65_state_metadata_parity.py",
        "test_tlspec_runnable_r65_torch_rng.py",
        "test_tlspec_runnable_r65_walker_parity.py",
        "test_tlspec_runnable_r67_defensive_clone_context.py",
        "test_tlspec_runnable_r67_input_structure.py",
        "test_tlspec_runnable_r67_storage_metadata.py",
        "test_tlspec_runnable_r69_input_contract.py",
        "test_tlspec_runnable_r71_witness_obligations.py",
        "test_tlspec_runnable_r73_derived_layout.py",
        "test_tlspec_runnable_r75_data_alias_layout.py",
        "test_tlspec_runnable_r75_envelope_path.py",
        "test_tlspec_runnable_r75_param_identity.py",
        "test_tlspec_runnable_r77_buffer_universe.py",
        "test_tlspec_runnable_r77_fact_components.py",
        "test_tlspec_runnable_r77_param_provenance.py",
        "test_tlspec_runnable_r77_seed_typing.py",
        "test_tlspec_runnable_r79_empty_name_param.py",
        "test_tlspec_runnable_r79_seed_typing.py",
        "test_tlspec_runnable_r79_session_leak.py",
        "test_tlspec_runnable_r81_buffer_rung.py",
        "test_tlspec_runnable_r83_artifact_coherence.py",
        "test_tlspec_runnable_r83_buffer_address_authority.py",
        "test_tlspec_runnable_r83_label_anchoring.py",
        "test_tlspec_runnable_r85_storage_integrity.py",
        "test_tlspec_runnable_r87_backend_address_writeside.py",
        "test_tlspec_uninit_consumer_r55.py",
    }
)

#: Sprint-lane grab-bag files (shrink-only; do not add deposits or siblings).
_LEDGERED_GRAB_BAG_FILES: frozenset[str] = frozenset({"test_grind_b4_lane_p.py"})

_SPRINT_TOKEN = re.compile(r"(^|_)r\d+")
_GRAB_BAG = re.compile(r"^test_(grind|hunt|lane|fixwave|sprint)_")


def _census() -> tuple[set[str], set[str]]:
    """Return (sprint-token files, grab-bag files) relative to tests/."""

    token_files: set[str] = set()
    grab_bag_files: set[str] = set()
    for path in _TESTS_ROOT.rglob("test_*.py"):
        relative = path.relative_to(_TESTS_ROOT).as_posix()
        if _SPRINT_TOKEN.search(path.stem):
            token_files.add(relative)
        if _GRAB_BAG.match(path.name):
            grab_bag_files.add(relative)
    return token_files, grab_bag_files


def test_no_new_sprint_token_filenames() -> None:
    """New test files must not carry sprint-round tokens (ledger is shrink-only)."""

    token_files, grab_bag_files = _census()
    new_tokens = token_files - _LEDGERED_SPRINT_TOKEN_FILES
    assert not new_tokens, (
        "new test filename(s) carry a sprint-round token -- name the file "
        f"after the subsystem and contract under test instead: {sorted(new_tokens)}"
    )
    new_grab_bags = grab_bag_files - _LEDGERED_GRAB_BAG_FILES
    assert not new_grab_bags, (
        f"new sprint-lane grab-bag file(s): {sorted(new_grab_bags)} -- "
        "tests belong in subsystem-named files"
    )


def test_sprint_token_ledger_has_no_stale_rows() -> None:
    """A renamed/deleted file leaves the ledger, so the census only shrinks."""

    token_files, grab_bag_files = _census()
    stale = (_LEDGERED_SPRINT_TOKEN_FILES - token_files) | (
        _LEDGERED_GRAB_BAG_FILES - grab_bag_files
    )
    assert not stale, (
        f"ledger rows for files that no longer exist (delete the rows): {sorted(stale)}"
    )


def test_naming_lint_is_red_capable() -> None:
    """The matchers fire on each disease and spare descriptive names."""

    assert _SPRINT_TOKEN.search("test_r99_new_thing")
    assert _SPRINT_TOKEN.search("test_io_hardening_r10r60")
    assert not _SPRINT_TOKEN.search("test_receptive_field")
    assert not _SPRINT_TOKEN.search("test_rng_seal")  # rng is not an rN token
    assert _GRAB_BAG.match("test_grind_b9_lane_q.py")
    assert _GRAB_BAG.match("test_fixwave_leftovers.py")
    assert not _GRAB_BAG.match("test_gradient_flow.py")

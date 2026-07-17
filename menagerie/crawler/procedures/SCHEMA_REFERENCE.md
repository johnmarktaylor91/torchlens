# Crawler schema reference

This data dictionary is the human companion to the executable JSON Schemas in `menagerie/crawler/schemas/`. “Mandatory” means the field is required by its enclosing schema object; “Best-effort” means it may be null, omitted by a branch, or records an observation that is not guaranteed to exist. Closed vocabularies and cross-field conditions remain authoritative in the schemas.

Current v3 amendment: `input_contract.code_path` is absent from `model.v3`,
`author-proposal.v3`, and embedded `author-result.v3`; either null or string presence rejects. The field
listed below belongs only to readable untrusted `model.v2` history. The distinct current-v3 fields
`implementation.code_path` and `implementation.source_to_code_map[].code_path` remain mandatory where
their enclosing recipe requires them.

## `model.v2`

### Bookkeeping

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `schema_version` | const `menagerie.crawler.model.v2` | Mandatory | Mandatory versioned schema identifier. |
| `stable_id` | string | Mandatory | Mandatory stable model identifier. |
| `record_seq` | integer | Mandatory | Mandatory monotonic model-ledger sequence. |
| `record_revision` | string | Mandatory | Mandatory hash of this model revision. |
| `parent_revision` | string \| null | Mandatory | Best-effort hash of the superseded model revision. |
| `created_at` | string | Mandatory | Mandatory UTC creation timestamp. |
| `revised_by` | object | Mandatory | Mandatory actor that produced this revision. |
| `revised_by.actor` | const `driver` or string | Mandatory | Mandatory revision-producing actor. |
| `revised_by.model` | string | Mandatory for non-driver revisions | Model identifier for a non-driver revision producer. |
| `revised_by.version` | string | Mandatory for non-driver revisions | Version of a non-driver revision producer. |
| `authored_metadata_state` | enum | Mandatory | Mandatory acceptance state for source-read metadata. |
| `intake` | object | Mandatory | Mandatory preserved intake provenance. |
| `intake.snapshot_id` | string | Mandatory | Mandatory snapshot id. |
| `intake.snapshot_sha256` | string | Mandatory | Mandatory snapshot sha256. |
| `intake.legacy_row_sha256` | string \| null | Mandatory | Mandatory legacy row sha256. |
| `intake.legacy_recipe_sha256` | string \| null | Mandatory | Mandatory legacy recipe sha256. |
| `intake.legacy_module_sha256` | string \| null | Mandatory | Mandatory legacy module sha256. |
| `intake.legacy_claims_untrusted` | const `True` | Mandatory | Mandatory legacy claims untrusted. |
| `intake.preserved_legacy_flags` | array<string> | Mandatory | Mandatory preserved legacy flags. |
| `intake.discovery_sources` | array<string> | Mandatory | Mandatory discovery sources. |
| `provenance` | object | Mandatory | Mandatory production provenance. |
| `provenance.author_model` | string | Mandatory | Mandatory author model. |
| `provenance.author_version` | string | Mandatory | Mandatory author version. |
| `provenance.author_prompt_sha256` | string | Mandatory | Mandatory author prompt sha256. |
| `provenance.checker_model` | string | Mandatory | Mandatory checker model. |
| `provenance.checker_version` | string | Mandatory | Mandatory checker version. |
| `provenance.producer_run_id` | string | Mandatory | Mandatory producer run id. |
| `provenance.machine_id` | string | Mandatory | Mandatory machine id. |
| `budget` | object | Mandatory | Mandatory bounded-work budget accounting. |
| `budget.author_sessions_used` | integer | Mandatory | Mandatory author sessions used. |
| `budget.author_sessions_max` | integer | Mandatory | Mandatory author sessions max. |
| `budget.gate_rounds_used` | integer | Mandatory | Mandatory gate rounds used. |
| `budget.run_revisions_used` | integer | Mandatory | Mandatory run revisions used. |
| `budget.explicit_grants` | array<string> | Mandatory | Mandatory explicit grants. |
| `flags` | array<string> | Mandatory | Mandatory machine-readable record flags. |
| `notes` | string | Mandatory | Mandatory free-form record notes. |
| `scar_history` | array<string> | Mandatory | Mandatory immutable history of accepted deviations. |
| `completeness` | object | Mandatory | Mandatory release-completeness checks. |
| `completeness.schema_valid` | boolean | Mandatory | Mandatory schema valid. |
| `completeness.mandatory_source_present` | boolean | Mandatory | Mandatory mandatory source present. |
| `completeness.source_read_fields_complete` | boolean | Mandatory | Mandatory source read fields complete. |
| `completeness.evidence_coverage_complete` | boolean | Mandatory | Mandatory evidence coverage complete. |
| `completeness.accuracy_gate_current` | boolean | Mandatory | Mandatory accuracy gate current. |
| `completeness.required_fidelity_current` | boolean | Mandatory | Mandatory required fidelity current. |
| `completeness.execution_current` | boolean | Mandatory | Mandatory execution current. |
| `completeness.family_template_valid` | boolean | Mandatory | Mandatory family template valid. |
| `completeness.release_eligible` | boolean | Mandatory | Mandatory release eligible. |
| `completeness.issues` | array<string> | Mandatory | Mandatory issues. |

### Identity and taxonomy

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `identity` | object | Mandatory | Mandatory canonical model identity. |
| `identity.canonical_name` | string | Mandatory | Mandatory canonical name. |
| `identity.aliases` | array<string> | Mandatory | Mandatory aliases. |
| `identity.acronym` | string \| null | Mandatory | Mandatory acronym. |
| `identity.variant` | string | Mandatory | Mandatory variant. |
| `identity.variant_scope` | string | Mandatory | Mandatory variant scope. |
| `identity.family_representative_id` | string | Mandatory | Mandatory family representative id. |
| `identity.duplicate_of` | string \| null | Mandatory | Mandatory duplicate of. |
| `identity.alias_of` | string \| null | Mandatory | Mandatory alias of. |
| `taxonomy` | object | Mandatory | Best-effort gated architecture taxonomy. |
| `taxonomy.family` | string | Mandatory | Mandatory family. |
| `taxonomy.domains` | array<string> | Mandatory | Mandatory domains. |
| `taxonomy.tasks` | array<string> | Mandatory | Mandatory tasks. |
| `taxonomy.modalities` | array<string> | Mandatory | Mandatory modalities. |
| `taxonomy.era` | string | Mandatory | Mandatory era. |
| `taxonomy.architecture_tags` | array<string> | Mandatory | Mandatory architecture tags. |
| `taxonomy.novel_ops` | array<string> | Mandatory | Mandatory novel ops. |

### External metadata

External metadata is captured and gated now because it requires source reading, web research, or human judgment. Parameter counts, input/output shapes, operation types, FLOPs, and graph structure are TorchLens-derivable; they are optional observations, never a reason to re-crawl external sources.

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `external_metadata` | object | Mandatory | Best-effort gated externally sourced catalog metadata. |
| `external_metadata.modality` | array<string> | Mandatory | Mandatory modality. |
| `external_metadata.architecture_class` | array<string> | Mandatory | Mandatory architecture class. |
| `external_metadata.domain` | array<string> | Mandatory | Mandatory domain. |
| `external_metadata.task` | array<string> | Mandatory | Mandatory task. |
| `external_metadata.field` | string \| null | Mandatory | Mandatory field. |
| `external_metadata.subfield` | string \| null | Mandatory | Mandatory subfield. |
| `external_metadata.paradigm` | array<string> | Mandatory | Mandatory paradigm. |
| `external_metadata.lineage` | array<string> | Mandatory | Mandatory lineage. |
| `external_metadata.predecessors` | array<string> | Mandatory | Mandatory predecessors. |
| `external_metadata.tags` | array<string> | Mandatory | Mandatory tags. |
| `external_metadata.keywords` | array<string> | Mandatory | Mandatory keywords. |
| `external_metadata.venue` | string \| null | Mandatory | Mandatory venue. |
| `external_metadata.family` | string | Mandatory | Mandatory family. |
| `external_metadata.era` | string | Mandatory | Mandatory era. |
| `external_metadata.year` | integer \| null | Mandatory | Mandatory year. |
| `external_metadata.country` | string \| null | Mandatory | Mandatory country. |
| `external_metadata.authors` | array<string> | Mandatory | Mandatory authors. |
| `external_metadata.institution` | array<string> | Mandatory | Mandatory institution. |
| `external_metadata.citation` | object | Mandatory | Best-effort gated citation metadata. |
| `external_metadata.citation.status` | enum | Mandatory | Mandatory closed current disposition. |
| `external_metadata.citation.title` | string \| null | Mandatory | Mandatory title. |
| `external_metadata.citation.authors` | array<string> | Mandatory | Mandatory authors. |
| `external_metadata.citation.year` | integer \| null | Mandatory | Mandatory year. |
| `external_metadata.citation.venue` | string \| null | Mandatory | Mandatory venue. |
| `external_metadata.citation.arxiv_id` | string \| null | Mandatory | Mandatory arxiv id. |
| `external_metadata.citation.doi` | string \| null | Mandatory | Mandatory doi. |
| `external_metadata.citation.openreview_id` | string \| null | Mandatory | Mandatory openreview id. |
| `external_metadata.citation.url` | string \| null | Mandatory | Mandatory public source URL when available. |
| `external_metadata.citation.bibtex` | string \| null | Mandatory | Mandatory bibtex. |
| `external_metadata.citation.source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |
| `external_metadata.license` | string \| null | Mandatory | Mandatory license. |
| `external_metadata.key_contribution` | string | Mandatory | Mandatory key contribution. |
| `external_metadata.description` | string | Mandatory | Mandatory description. |
| `external_metadata.original_framework` | string | Mandatory | Mandatory original framework. |
| `external_metadata.run_framework` | string | Mandatory | Mandatory run framework. |
| `external_metadata.modes` | object | Mandatory | Mandatory meaningful runtime-mode record. |
| `external_metadata.modes.meaningful_modes` | array<enum> | Mandatory | Mandatory meaningful modes. |
| `external_metadata.modes.train_eval_divergence` | enum | Mandatory | Mandatory train eval divergence. |

### Website

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `website` | object | Mandatory | Best-effort presentation metadata for the catalog page. |
| `website.kind` | enum | Mandatory | Mandatory record or status kind. |
| `website.tagline` | string | Mandatory | Mandatory tagline. |
| `website.description` | string | Mandatory | Mandatory description. |
| `website.key_contribution` | string | Mandatory | Mandatory key contribution. |
| `website.voice_version` | string | Mandatory | Mandatory voice version. |
| `website.family_grounding_id` | string | Mandatory | Mandatory family grounding id. |
| `website.template_source_model_id` | string \| null | Mandatory | Mandatory template source model id. |
| `website.variant_parameter_input_line` | string \| null | Mandatory | Mandatory variant parameter input line. |
| `website.template_hash` | string \| null | Mandatory | Mandatory template hash. |

### People, origin, dates, and citation

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `people_and_origin` | object | Mandatory | Best-effort gated people and origin metadata. |
| `people_and_origin.authors` | array<string> | Mandatory | Mandatory authors. |
| `people_and_origin.labs` | array<string> | Mandatory | Mandatory labs. |
| `people_and_origin.institutions` | array<string> | Mandatory | Mandatory institutions. |
| `people_and_origin.origin_countries` | array<string> | Mandatory | Mandatory origin countries. |
| `people_and_origin.country_basis` | string | Mandatory | Mandatory country basis. |
| `people_and_origin.country_confidence` | enum | Mandatory | Mandatory country confidence. |
| `people_and_origin.country_note` | string | Mandatory | Mandatory country note. |
| `dates` | object | Mandatory | Best-effort gated publication-date metadata. |
| `dates.year` | integer \| null | Mandatory | Mandatory year. |
| `dates.year_basis` | string | Mandatory | Mandatory year basis. |
| `dates.first_public_date` | string \| null | Mandatory | Mandatory first public date. |
| `dates.first_public_date_basis` | string | Mandatory | Mandatory first public date basis. |
| `citation` | object | Mandatory | Best-effort gated citation metadata. |
| `citation.status` | enum | Mandatory | Mandatory closed current disposition. |
| `citation.title` | string \| null | Mandatory | Mandatory title. |
| `citation.authors` | array<string> | Mandatory | Mandatory authors. |
| `citation.year` | integer \| null | Mandatory | Mandatory year. |
| `citation.venue` | string \| null | Mandatory | Mandatory venue. |
| `citation.arxiv_id` | string \| null | Mandatory | Mandatory arxiv id. |
| `citation.doi` | string \| null | Mandatory | Mandatory doi. |
| `citation.openreview_id` | string \| null | Mandatory | Mandatory openreview id. |
| `citation.url` | string \| null | Mandatory | Mandatory public source URL when available. |
| `citation.bibtex` | string \| null | Mandatory | Mandatory bibtex. |
| `citation.source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |

### Licenses

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `licenses` | object | Mandatory | Best-effort gated source-license metadata. |
| `licenses.code` | object | Mandatory | Mandatory closed status code. |
| `licenses.code.spdx` | string | Mandatory | Mandatory spdx. |
| `licenses.code.status` | enum | Mandatory | Mandatory closed current disposition. |
| `licenses.code.source_id` | string | Mandatory | Mandatory source identifier. |
| `licenses.code.locator` | string | Mandatory | Mandatory exact location within the source. |
| `licenses.code.evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `licenses.paper_text` | object | Mandatory | Mandatory paper text. |
| `licenses.paper_text.status` | enum | Mandatory | Mandatory closed current disposition. |
| `licenses.paper_text.source_id` | string \| null | Mandatory | Mandatory source identifier. |
| `licenses.weights` | object | Mandatory | Mandatory weights. |
| `licenses.weights.status` | const `not-used` | Mandatory | Mandatory closed current disposition. |
| `licenses.data` | object | Mandatory | Mandatory data. |
| `licenses.data.spdx` | string \| null | Mandatory | Mandatory spdx. |
| `licenses.data.status` | string | Mandatory | Mandatory closed current disposition. |
| `licenses.data.source_id` | string \| null | Mandatory | Mandatory source identifier. |
| `licenses.data.evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `licenses.redistribution_class` | enum | Mandatory | Mandatory redistribution class. |

### Source resolution

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `source_resolution` | object | Mandatory | Mandatory selected source rung and search record. |
| `source_resolution.rung` | enum | Mandatory | Mandatory source-resolution ladder rung. |
| `source_resolution.decision` | string | Mandatory | Mandatory source-resolution decision. |
| `source_resolution.rung_evidence` | string | Mandatory | Mandatory rung evidence. |
| `source_resolution.sufficiency_gap` | string \| null | Mandatory | Mandatory for insufficient-description skips; names missing implementation detail. |
| `source_resolution.searched_at` | string | Mandatory | Mandatory searched at. |
| `source_resolution.attempted_rungs` | array<object> | Mandatory | Mandatory attempted rungs. |
| `source_resolution.attempted_rungs[].rung` | enum | Mandatory | Mandatory source-resolution ladder rung. |
| `source_resolution.attempted_rungs[].result` | string | Mandatory | Mandatory immutable attempt outcome. |
| `source_resolution.attempted_rungs[].reason_code` | string | Mandatory | Mandatory closed reason code when applicable. |
| `source_resolution.attempted_rungs[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `source_resolution.search_report` | object | Mandatory | Mandatory search report. |
| `source_resolution.search_report.queries` | array<string> | Mandatory | Mandatory queries. |
| `source_resolution.search_report.places_checked` | array<string> | Mandatory | Mandatory places checked. |
| `source_resolution.search_report.links_checked` | array<string> | Mandatory | Mandatory links checked. |
| `source_resolution.search_report.languages_checked` | array<string> | Mandatory | Mandatory languages checked. |
| `source_resolution.search_report.archives_checked` | array<string> | Mandatory | Mandatory archives checked. |
| `source_resolution.search_report.started_at` | string | Mandatory | Mandatory UTC start timestamp. |
| `source_resolution.search_report.finished_at` | string | Mandatory | Mandatory UTC completion timestamp. |
| `source_resolution.search_report.conclusion` | string | Mandatory | Mandatory conclusion. |
| `source_resolution.mandatory_link_status` | enum | Mandatory | Mandatory mandatory link status. |
| `source_resolution.primary_source_id` | string | Mandatory | Mandatory primary source id. |
| `source_resolution.sources` | array<object> | Mandatory | Mandatory resolved public sources. |
| `source_resolution.sources[].source_id` | string | Mandatory | Mandatory source identifier. |
| `source_resolution.sources[].role` | enum | Mandatory | Mandatory role. |
| `source_resolution.sources[].kind` | enum | Mandatory | Mandatory record or status kind. |
| `source_resolution.sources[].url` | string | Mandatory | Mandatory public source URL when available. |
| `source_resolution.sources[].revision_kind` | string | Mandatory | Mandatory revision kind. |
| `source_resolution.sources[].revision` | string | Mandatory | Mandatory revision. |
| `source_resolution.sources[].locator` | string | Mandatory | Mandatory exact location within the source. |
| `source_resolution.sources[].content_sha256` | string | Mandatory | Mandatory content sha256. |
| `source_resolution.sources[].byte_count` | integer | Mandatory | Mandatory byte count. |
| `source_resolution.sources[].media_type` | string | Mandatory | Mandatory media type. |
| `source_resolution.sources[].retrieved_at` | string | Mandatory | Mandatory retrieved at. |
| `source_resolution.sources[].fetch_recipe` | string | Mandatory | Mandatory fetch recipe. |
| `source_resolution.sources[].mirror_class` | string | Mandatory | Mandatory mirror class. |
| `source_resolution.sources[].mirror_digest` | string \| null | Mandatory | Mandatory mirror digest. |

### Evidence

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `evidence` | object | Mandatory | Mandatory literal evidence and coverage record. |
| `evidence.excerpts` | array<object> | Mandatory | Mandatory literal source excerpts. |
| `evidence.excerpts[].evidence_id` | string | Mandatory | Mandatory evidence identifier. |
| `evidence.excerpts[].source_id` | string | Mandatory | Mandatory source identifier. |
| `evidence.excerpts[].locator` | string | Mandatory | Mandatory exact location within the source. |
| `evidence.excerpts[].text` | string | Mandatory | Mandatory verbatim retained source text. |
| `evidence.excerpts[].text_sha256` | string | Mandatory | Mandatory hash of the verbatim excerpt. |
| `evidence.excerpts[].supports` | array<string> | Mandatory | Mandatory supports. |
| `evidence.excerpts[].family_level` | boolean | Mandatory | Mandatory family level. |
| `evidence.excerpts[].disposition` | enum | Mandatory | Mandatory excerpt role, including insufficient reimplementation evidence. |
| `evidence.excerpts[].license_disposition` | string | Mandatory | Mandatory license handling for the retained excerpt. |
| `evidence.coverage` | object | Mandatory | Mandatory coverage. |
| `evidence.coverage.all_agent_fields_have_support` | boolean | Mandatory | Mandatory all agent fields have support. |
| `evidence.coverage.missing_support` | array<string> | Mandatory | Mandatory missing support. |
| `evidence.coverage.family_grounding_complete` | boolean | Mandatory | Mandatory family grounding complete. |
| `evidence.evidence_identity` | string | Mandatory | Mandatory evidence identity. |
| `evidence.family_grounding_path` | string \| null | Mandatory | Mandatory family grounding path. |

### Implementation

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `implementation` | object | Mandatory | Mandatory executable implementation recipe. |
| `implementation.original_framework` | string | Mandatory | Mandatory original framework. |
| `implementation.run_framework` | string | Mandatory | Mandatory run framework. |
| `implementation.native_object_type` | string | Mandatory | Mandatory native object type. |
| `implementation.native_call_method` | string | Mandatory | Mandatory native call method. |
| `implementation.transparent_forward_adapter` | boolean | Mandatory | Mandatory transparent forward adapter. |
| `implementation.recipe_type` | enum | Mandatory | Mandatory recipe type. |
| `implementation.code_path` | string \| null | Mandatory | Mandatory code path. |
| `implementation.code_sha256` | string \| null | Mandatory | Mandatory code sha256. |
| `implementation.builder_symbol` | const `build_model` \| null | Mandatory | Mandatory builder symbol. |
| `implementation.dummy_call_symbol` | const `make_dummy_call` \| null | Mandatory | Mandatory dummy call symbol. |
| `implementation.library_recipe` | object | Mandatory | Mandatory library recipe. |
| `implementation.library_recipe.distribution` | string | Mandatory | Mandatory distribution. |
| `implementation.library_recipe.version` | string | Mandatory | Mandatory version. |
| `implementation.library_recipe.artifact_sha256` | string | Mandatory | Mandatory artifact sha256. |
| `implementation.library_recipe.module` | string | Mandatory | Mandatory module. |
| `implementation.library_recipe.symbol` | string | Mandatory | Mandatory symbol. |
| `implementation.library_recipe.kwargs` | object | Mandatory | Mandatory kwargs. |
| `implementation.library_recipe.pretrained_disable_fields` | array<string> | Mandatory | Mandatory pretrained disable fields. |
| `implementation.upstream_files` | array<object> | Mandatory | Mandatory upstream files. |
| `implementation.upstream_files[].source_id` | string | Mandatory | Mandatory source identifier. |
| `implementation.upstream_files[].path` | string | Mandatory | Mandatory path. |
| `implementation.upstream_files[].sha256` | string | Mandatory | Mandatory sha256. |
| `implementation.upstream_files[].use` | string | Mandatory | Mandatory use. |
| `implementation.patches` | array<object> | Mandatory | Mandatory patches. |
| `implementation.patches[].path` | string | Mandatory | Mandatory path. |
| `implementation.patches[].sha256` | string | Mandatory | Mandatory sha256. |
| `implementation.patches[].classification` | string | Mandatory | Mandatory classification. |
| `implementation.patches[].semantic` | boolean | Mandatory | Mandatory semantic. |
| `implementation.patches[].rationale` | string | Mandatory | Mandatory rationale. |
| `implementation.patches[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `implementation.source_to_code_map` | array<object> | Mandatory | Mandatory source to code map. |
| `implementation.source_to_code_map[].material_item` | string | Mandatory | Mandatory material item. |
| `implementation.source_to_code_map[].source_id` | string | Mandatory | Mandatory source identifier. |
| `implementation.source_to_code_map[].source_locator` | string | Mandatory | Mandatory source locator. |
| `implementation.source_to_code_map[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `implementation.source_to_code_map[].code_path` | string | Mandatory | Mandatory code path. |
| `implementation.source_to_code_map[].code_locator` | string | Mandatory | Mandatory code locator. |
| `implementation.source_to_code_map[].disposition` | string | Mandatory | Mandatory excerpt role, including insufficient reimplementation evidence. |
| `implementation.declared_choices` | array<object> | Mandatory | Mandatory declared choices. |
| `implementation.declared_choices[].field` | string | Mandatory | Mandatory field. |
| `implementation.declared_choices[].value` | value | Mandatory | Mandatory value. |
| `implementation.declared_choices[].source_status` | string | Mandatory | Mandatory source status. |
| `implementation.declared_choices[].material` | boolean | Mandatory | Mandatory material. |
| `implementation.declared_choices[].rationale` | string | Mandatory | Mandatory rationale. |
| `implementation.declared_choices[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `implementation.initialization` | object | Mandatory | Mandatory initialization. |
| `implementation.initialization.policy` | const `random` | Mandatory | Mandatory policy. |
| `implementation.initialization.pretrained_disabled` | const `True` | Mandatory | Mandatory pretrained disabled. |
| `implementation.initialization.source_specified_choices` | array<string> | Mandatory | Mandatory source specified choices. |
| `implementation.mode` | string | Mandatory | Mandatory mode. |
| `implementation.device_policy` | string | Mandatory | Mandatory device policy. |
| `implementation.required_construct_asset` | null | Mandatory | Mandatory required construct asset. |
| `implementation.recipe_revision` | string | Mandatory | Mandatory recipe revision. |
| `implementation.torchlens_import_static_check` | const `passed` | Mandatory | Mandatory torchlens import static check. |

### Input contract

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `input_contract` | object | Mandatory | Mandatory source-valid dummy-input contract. |
| `input_contract.code_path` | string \| null | Mandatory | Historical v2 input-builder path retained only as untrusted history; absent and rejected in v3. |
| `input_contract.builder_symbol` | string | Mandatory | Mandatory builder symbol. |
| `input_contract.seed` | integer | Mandatory | Mandatory seed. |
| `input_contract.semantic_description` | string | Mandatory | Mandatory semantic description. |
| `input_contract.source_basis` | array<string> | Mandatory | Mandatory source basis. |
| `input_contract.smallest_valid_probe_rationale` | string | Mandatory | Mandatory smallest valid probe rationale. |
| `input_contract.args` | array<object> | Mandatory | Mandatory args. |
| `input_contract.args[].path` | string | Mandatory | Mandatory path. |
| `input_contract.args[].kind` | string | Mandatory | Mandatory record or status kind. |
| `input_contract.args[].semantic_role` | string | Mandatory | Mandatory semantic role. |
| `input_contract.args[].shape` | array<integer \| string> | Mandatory | Mandatory shape. |
| `input_contract.args[].dtype` | string | Mandatory | Mandatory dtype. |
| `input_contract.args[].device_policy` | string | Mandatory | Mandatory device policy. |
| `input_contract.args[].distribution` | enum | Mandatory | Mandatory distribution. |
| `input_contract.args[].constraints` | array<string> | Mandatory | Mandatory constraints. |
| `input_contract.args[].source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |
| `input_contract.kwargs` | array<object> | Mandatory | Mandatory kwargs. |
| `input_contract.kwargs[].path` | string | Mandatory | Mandatory path. |
| `input_contract.kwargs[].kind` | string | Mandatory | Mandatory record or status kind. |
| `input_contract.kwargs[].semantic_role` | string | Mandatory | Mandatory semantic role. |
| `input_contract.kwargs[].shape` | array<integer \| string> | Mandatory | Mandatory shape. |
| `input_contract.kwargs[].dtype` | string | Mandatory | Mandatory dtype. |
| `input_contract.kwargs[].device_policy` | string | Mandatory | Mandatory device policy. |
| `input_contract.kwargs[].distribution` | enum | Mandatory | Mandatory distribution. |
| `input_contract.kwargs[].constraints` | array<string> | Mandatory | Mandatory constraints. |
| `input_contract.kwargs[].source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |
| `input_contract.non_tensor_values` | array<object> | Mandatory | Mandatory non tensor values. |
| `input_contract.non_tensor_values[].path` | string | Mandatory | Mandatory path. |
| `input_contract.non_tensor_values[].type` | string | Mandatory | Mandatory type. |
| `input_contract.non_tensor_values[].value` | value | Mandatory | Mandatory value. |
| `input_contract.non_tensor_values[].semantic_role` | string | Mandatory | Mandatory semantic role. |
| `input_contract.non_tensor_values[].constraints` | array<string> | Mandatory | Mandatory constraints. |
| `input_contract.non_tensor_values[].source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |
| `input_contract.masks_state_and_control` | array<string> | Mandatory | Mandatory masks state and control. |
| `input_contract.expected_output_semantics` | string | Mandatory | Mandatory expected output semantics. |

### Observed

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `observed` | object | Mandatory | Mandatory best-effort observed runtime facts. |
| `observed.parameter_count_total` | integer | Mandatory | Mandatory parameter count total. |
| `observed.parameter_count_trainable` | integer | Mandatory | Mandatory parameter count trainable. |
| `observed.output_signature` | object | Mandatory | Mandatory output signature. |
| `observed.output_signature.tree` | value | Mandatory | Mandatory tree. |
| `observed.output_signature.leaves` | array<object> | Mandatory | Mandatory leaves. |
| `observed.output_signature.leaves[].path` | string | Mandatory | Mandatory path. |
| `observed.output_signature.leaves[].kind` | string | Mandatory | Mandatory record or status kind. |
| `observed.output_signature.leaves[].shape` | array<integer \| string> \| null | Mandatory | Mandatory shape. |
| `observed.output_signature.leaves[].dtype` | string \| null | Mandatory | Mandatory dtype. |
| `observed.output_signature.leaves[].device` | string \| null | Mandatory | Mandatory device. |
| `observed.output_signature.leaves[].python_type` | string | Mandatory | Mandatory python type. |
| `observed.input_kind` | string | Mandatory | Mandatory input kind. |
| `observed.input_asset` | string \| null | Mandatory | Mandatory input asset. |
| `observed.input_note` | string | Mandatory | Mandatory input note. |
| `observed.constructor_seconds` | number | Mandatory | Mandatory constructor seconds. |
| `observed.forward_seconds` | number | Mandatory | Mandatory forward seconds. |
| `observed.peak_rss_bytes` | integer | Mandatory | Mandatory peak rss bytes. |
| `observed.measurement_attempt_ids` | array<string> | Mandatory | Mandatory measurement attempt ids. |
| `observed.snippet` | string | Mandatory | Mandatory snippet. |
| `observed.snippet_sha256` | string | Mandatory | Mandatory snippet sha256. |

### Modes and verification state

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `modes` | object | Mandatory | Mandatory meaningful runtime-mode record. |
| `modes.meaningful_modes` | array<enum> | Mandatory | Mandatory meaningful modes. |
| `modes.per_mode_run` | object | Mandatory | Mandatory per mode run. |
| `modes.per_mode_run.train` | object | Best-effort | Best-effort train. |
| `modes.per_mode_run.train.attempt_id` | string | Mandatory | Mandatory immutable attempt identifier. |
| `modes.per_mode_run.train.status` | enum | Mandatory | Mandatory closed current disposition. |
| `modes.per_mode_run.eval` | object | Best-effort | Best-effort eval. |
| `modes.per_mode_run.eval.attempt_id` | string | Mandatory | Mandatory immutable attempt identifier. |
| `modes.per_mode_run.eval.status` | enum | Mandatory | Mandatory closed current disposition. |
| `modes.train_eval_divergence` | enum | Mandatory | Mandatory train eval divergence. |
| `modes.divergence_evidence` | string | Mandatory | Mandatory divergence evidence. |
| `fidelity` | object | Mandatory | Mandatory implementation-fidelity gate state. |
| `fidelity.required` | boolean | Mandatory | Mandatory required. |
| `fidelity.reason` | string | Mandatory | Mandatory reason. |
| `fidelity.verdict` | enum \| null | Mandatory | Mandatory checker verdict. |
| `fidelity.fidelity_identity` | string \| null | Mandatory | Mandatory fidelity identity. |
| `fidelity.gate_id` | string \| null | Mandatory | Mandatory immutable gate identifier. |
| `fidelity.current` | boolean | Mandatory | Mandatory current. |
| `fidelity.permanent_scar` | boolean | Mandatory | Mandatory permanent scar. |
| `fidelity.deviations` | array<string> | Mandatory | Mandatory deviations. |
| `accuracy_gate` | object | Mandatory | Mandatory metadata-accuracy gate state. |
| `accuracy_gate.required` | const `True` | Mandatory | Mandatory required. |
| `accuracy_gate.vet_identity` | string \| null | Mandatory | Mandatory vet identity. |
| `accuracy_gate.gate_id` | string \| null | Mandatory | Mandatory immutable gate identifier. |
| `accuracy_gate.verdict` | enum \| null | Mandatory | Mandatory checker verdict. |
| `accuracy_gate.current` | boolean | Mandatory | Mandatory current. |
| `accuracy_gate.checker_model` | string | Mandatory | Mandatory checker model. |
| `accuracy_gate.checker_version` | string | Mandatory | Mandatory checker version. |
| `accuracy_gate.prompt_sha256` | string | Mandatory | Mandatory prompt sha256. |
| `execution` | object | Mandatory | Mandatory execution identity and currentness state. |
| `execution.execution_identity` | string | Mandatory | Mandatory execution identity. |
| `execution.environment_id` | string | Mandatory | Mandatory environment id. |
| `execution.env_generation` | string | Mandatory | Mandatory env generation. |
| `execution.accepted_attempt_ids` | array<string> | Mandatory | Mandatory accepted attempt ids. |
| `execution.confirmation_policy` | enum | Mandatory | Mandatory confirmation policy. |
| `execution.network_attempted` | const `False` | Mandatory | Mandatory network attempted. |
| `execution.checkpoint_accessed` | const `False` | Mandatory | Mandatory checkpoint accessed. |
| `execution.last_verified_at` | string | Mandatory | Mandatory last verified at. |
| `execution.current` | boolean | Mandatory | Mandatory current. |
| `status` | object | Mandatory | Mandatory closed current disposition. |
| `status.kind` | enum | Mandatory | Mandatory record or status kind. |
| `status.code` | enum | Mandatory | Mandatory closed status code. |
| `status.stage` | enum \| null | Mandatory | Mandatory processing or failure stage. |
| `status.reason_code` | referenced value \| null | Mandatory | Mandatory closed reason code when applicable. |
| `status.detail` | string \| null | Mandatory | Best-effort human-readable detail. |
| `status.traceback` | string \| redacted-text object \| null | Mandatory | Best-effort local diagnostic reference for a captured traceback. |
| `status.no_traceback_reason` | string \| null | Mandatory | Mandatory no traceback reason. |
| `status.attempted_rungs` | array<enum> | Mandatory | Mandatory attempted rungs. |
| `status.retries` | object | Mandatory | Mandatory retries. |
| `status.retries.source` | integer | Mandatory | Mandatory source. |
| `status.retries.fetch` | integer | Mandatory | Mandatory fetch. |
| `status.retries.evidence` | integer | Mandatory | Mandatory literal evidence and coverage record. |
| `status.retries.author` | integer | Mandatory | Mandatory author model identity. |
| `status.retries.gate` | integer | Mandatory | Mandatory gate. |
| `status.retries.environment` | integer | Mandatory | Mandatory environment. |
| `status.retries.import` | integer | Mandatory | Mandatory import. |
| `status.retries.constructor` | integer | Mandatory | Mandatory constructor. |
| `status.retries.input` | integer | Mandatory | Mandatory input. |
| `status.retries.forward` | integer | Mandatory | Mandatory forward. |
| `status.retries.fidelity` | integer | Mandatory | Mandatory implementation-fidelity gate state. |
| `status.environment` | string \| null | Mandatory | Mandatory environment. |
| `status.timestamp` | string | Mandatory | Mandatory timestamp. |
| `status.attempt_ids` | array<string> | Mandatory | Mandatory attempt ids. |
| `status.root_cause_fingerprint` | string \| null | Mandatory | Mandatory root cause fingerprint. |
| `status.supersedes_revision` | string \| null | Mandatory | Mandatory supersedes revision. |
| `status.human_review` | object | Mandatory | Mandatory human review. |
| `status.human_review.required` | boolean | Mandatory | Mandatory required. |
| `status.human_review.reason` | string \| null | Mandatory | Mandatory reason. |
| `status.human_review.queue` | string \| null | Mandatory | Mandatory queue. |
| `status.human_review.requested_at` | string \| null | Mandatory | Mandatory requested at. |

## `attempt.v2` receipt fields

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `schema_version` | const `menagerie.crawler.attempt.v2` | Mandatory | Mandatory versioned schema identifier. |
| `attempt_id` | string | Mandatory | Mandatory immutable attempt identifier. |
| `ledger_seq` | integer | Mandatory | Mandatory monotonic ledger sequence. |
| `payload_sha256` | string | Mandatory | Mandatory hash of the complete ledger payload. |
| `work_id` | string | Mandatory | Mandatory deterministic work identifier. |
| `stable_id` | string \| null | Mandatory | Mandatory stable model identifier. |
| `attempt_no` | integer | Mandatory | Mandatory ordinal attempt number. |
| `parent_attempt_id` | string \| null | Mandatory | Best-effort parent attempt identifier. |
| `actor` | string | Mandatory | Mandatory producing actor. |
| `stage` | enum | Mandatory | Mandatory processing or failure stage. |
| `mode` | enum \| null | Mandatory | Mandatory mode. |
| `started_at` | string | Mandatory | Mandatory UTC start timestamp. |
| `finished_at` | string | Mandatory | Mandatory UTC completion timestamp. |
| `result` | enum | Mandatory | Mandatory immutable attempt outcome. |
| `attempted_rungs` | array<enum> | Mandatory | Mandatory attempted rungs. |
| `retries` | object | Mandatory | Mandatory retries. |
| `retries.stage_attempt` | integer | Mandatory | Mandatory stage attempt. |
| `retries.root_cause_repeat` | integer | Mandatory | Mandatory root cause repeat. |
| `retries.author_round` | integer | Mandatory | Mandatory author round. |
| `retries.gate_round` | integer | Mandatory | Mandatory checker repair round. |
| `identities` | object | Mandatory | Mandatory identities. |
| `identities.source` | string \| null | Mandatory | Mandatory source. |
| `identities.evidence` | string \| null | Mandatory | Mandatory literal evidence and coverage record. |
| `identities.recipe` | string \| null | Mandatory | Mandatory recipe. |
| `identities.environment` | string \| null | Mandatory | Mandatory environment. |
| `identities.execution` | string \| null | Mandatory | Mandatory execution identity and currentness state. |
| `identities.runner` | string \| null | Mandatory | Mandatory runner. |
| `identities.author_prompt` | string \| null | Mandatory | Mandatory author prompt. |
| `identities.checker_prompt` | string \| null | Mandatory | Mandatory checker prompt. |
| `environment` | object | Mandatory | Mandatory environment. |
| `environment.family` | string | Mandatory | Mandatory family. |
| `environment.target` | string | Mandatory | Mandatory target. |
| `environment.env_id` | string | Mandatory | Mandatory env id. |
| `environment.lock_sha256` | string | Mandatory | Mandatory lock sha256. |
| `environment.resolved_export_sha256` | string | Mandatory | Mandatory resolved export sha256. |
| `environment.python` | string | Mandatory | Mandatory python. |
| `environment.packages_manifest_sha256` | string | Mandatory | Mandatory packages manifest sha256. |
| `environment.compiler_identity` | string | Mandatory | Mandatory compiler identity. |
| `environment.sdk_identity` | string | Mandatory | Mandatory sdk identity. |
| `host` | object | Mandatory | Mandatory host. |
| `host.machine_id` | string | Mandatory | Mandatory machine id. |
| `host.os` | string | Mandatory | Mandatory os. |
| `host.os_build` | string | Mandatory | Mandatory os build. |
| `host.architecture` | string | Mandatory | Mandatory architecture. |
| `host.cpu` | string | Mandatory | Mandatory cpu. |
| `host.ram_bytes` | integer | Mandatory | Mandatory ram bytes. |
| `host.accelerator` | string \| null | Mandatory | Mandatory accelerator. |
| `host.accelerator_runtime` | string \| null | Mandatory | Mandatory accelerator runtime. |
| `invocation` | object | Mandatory | Mandatory invocation. |
| `invocation.argv` | array<string> | Mandatory | Mandatory argv. |
| `invocation.cwd` | string | Mandatory | Mandatory cwd. |
| `invocation.safe_env` | object map | Mandatory | Mandatory safe env. |
| `invocation.seed` | integer | Mandatory | Mandatory seed. |
| `invocation.device` | string | Mandatory | Mandatory device. |
| `invocation.mode` | enum \| null | Mandatory | Mandatory mode. |
| `invocation.network_policy` | string | Mandatory | Mandatory network policy. |
| `invocation.timeout_seconds` | integer | Mandatory | Mandatory timeout seconds. |
| `invocation.rss_limit_bytes` | integer | Mandatory | Mandatory rss limit bytes. |
| `invocation.scratch_limit_bytes` | integer | Mandatory | Mandatory scratch limit bytes. |
| `worker_receipt` | object | Mandatory | Mandatory worker receipt. |
| `worker_receipt.present` | boolean | Mandatory | Mandatory present. |
| `worker_receipt.receipt_sha256` | string \| null | Mandatory | Mandatory receipt sha256. |
| `worker_receipt.constructor_started` | boolean | Mandatory | Mandatory constructor started. |
| `worker_receipt.constructor_completed` | boolean | Mandatory | Mandatory constructor completed. |
| `worker_receipt.input_completed` | boolean | Mandatory | Mandatory input completed. |
| `worker_receipt.forward_started` | boolean | Mandatory | Mandatory forward started. |
| `worker_receipt.forward_completed` | boolean | Mandatory | Mandatory forward completed. |
| `worker_receipt.mode` | enum \| null | Mandatory | Mandatory mode. |
| `worker_receipt.input_signature` | value | Mandatory | Mandatory input signature. |
| `worker_receipt.output_signature` | value | Mandatory | Mandatory output signature. |
| `worker_receipt.input_kind` | string \| null | Mandatory | Mandatory input kind. |
| `worker_receipt.input_asset` | string \| null | Mandatory | Mandatory input asset. |
| `worker_receipt.input_note` | string | Mandatory | Mandatory input note. |
| `worker_receipt.parameter_count_total` | integer \| null | Mandatory | Mandatory parameter count total. |
| `worker_receipt.parameter_count_trainable` | integer \| null | Mandatory | Mandatory parameter count trainable. |
| `worker_receipt.native_framework` | string \| null | Mandatory | Mandatory native framework. |
| `worker_receipt.delegated_method` | string \| null | Mandatory | Mandatory delegated method. |
| `supervisor_observation` | object | Mandatory | Mandatory supervisor observation. |
| `supervisor_observation.exit_code` | integer \| null | Mandatory | Mandatory exit code. |
| `supervisor_observation.signal` | integer \| null | Mandatory | Mandatory signal. |
| `supervisor_observation.wall_seconds` | number | Mandatory | Mandatory wall seconds. |
| `supervisor_observation.cpu_seconds` | number | Mandatory | Mandatory cpu seconds. |
| `supervisor_observation.peak_rss_bytes` | integer | Mandatory | Mandatory peak rss bytes. |
| `supervisor_observation.stdout_sha256` | string | Mandatory | Mandatory stdout sha256. |
| `supervisor_observation.stdout_bytes` | integer | Mandatory | Mandatory stdout bytes. |
| `supervisor_observation.stdout_tail` | string \| redacted-text object | Mandatory | Empty string or hash-bound local diagnostic reference; raw worker stdout is forbidden. |
| `supervisor_observation.stdout_completion_line` | string \| null | Optional | Parent-attested TorchLens-owned completion marker used for run-award verification. |
| `supervisor_observation.stderr_sha256` | string | Mandatory | Mandatory stderr sha256. |
| `supervisor_observation.stderr_bytes` | integer | Mandatory | Mandatory stderr bytes. |
| `supervisor_observation.stderr_tail` | string \| redacted-text object | Mandatory | Empty string or hash-bound local diagnostic reference; raw worker stderr is forbidden. |
| `supervisor_observation.full_log_local_path` | string | Mandatory | Mandatory full log local path. |
| `supervisor_observation.full_log_retention` | string | Mandatory | Mandatory full log retention. |
| `policy_observation` | object | Mandatory | Mandatory policy observation. |
| `policy_observation.network_attempted` | boolean | Mandatory | Mandatory network attempted. |
| `policy_observation.socket_targets` | array<string> | Mandatory | Mandatory socket targets. |
| `policy_observation.checkpoint_or_weight_read_attempted` | boolean | Mandatory | Mandatory checkpoint or weight read attempted. |
| `policy_observation.checkpoint_paths` | array<string> | Mandatory | Mandatory checkpoint paths. |
| `policy_observation.write_outside_scratch_attempted` | boolean | Mandatory | Mandatory write outside scratch attempted. |
| `policy_observation.write_paths` | array<string> | Mandatory | Mandatory write paths. |
| `policy_observation.credentials_present` | boolean | Mandatory | Mandatory credentials present. |
| `policy_observation.torchlens_import_attempted` | boolean | Mandatory | Mandatory torchlens import attempted. |
| `policy_observation.cache_read_attempted` | boolean | Mandatory | Mandatory cache read attempted. |
| `error` | object | Mandatory | Mandatory error. |
| `error.stage` | enum | Mandatory | Mandatory processing or failure stage. |
| `error.reason_code` | enum | Mandatory | Mandatory closed reason code when applicable. |
| `error.exception_type` | string \| null | Mandatory | Mandatory exception type. |
| `error.message` | string \| redacted-text object | Mandatory | Empty string or a hash-bound local diagnostic reference; raw worker text is forbidden. |
| `error.traceback` | string \| redacted-text object \| null | Mandatory | Best-effort local diagnostic reference for a captured traceback. |
| `error.no_traceback_reason` | string \| null | Mandatory | Mandatory no traceback reason. |
| `error.native_crash` | boolean | Mandatory | Mandatory native crash. |
| `error.root_cause_fingerprint` | string | Mandatory | Mandatory root cause fingerprint. |
| `error.details` | object | Mandatory | Mandatory structured event details. |
| `defer_evidence` | object | Mandatory | Mandatory defer evidence. |
| `defer_evidence.target_status` | enum | Mandatory | Mandatory target status. |
| `defer_evidence.source_ids` | array<string> | Mandatory | Mandatory source ids. |
| `defer_evidence.probe_attempt_ids` | array<string> | Mandatory | Mandatory probe attempt ids. |
| `defer_evidence.explanation` | string | Mandatory | Mandatory explanation. |

## `gate.v2` verdict fields

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `schema_version` | const `menagerie.crawler.gate.v2` | Mandatory | Mandatory versioned schema identifier. |
| `gate_id` | string | Mandatory | Mandatory immutable gate identifier. |
| `ledger_seq` | integer | Mandatory | Mandatory monotonic ledger sequence. |
| `payload_sha256` | string | Mandatory | Mandatory hash of the complete ledger payload. |
| `gate_kind` | enum | Mandatory | Mandatory checker gate kind. |
| `batch_size` | integer | Mandatory | Mandatory number of items in this gate. |
| `gate_round` | integer | Mandatory | Mandatory checker repair round. |
| `gate_identity` | string | Mandatory | Mandatory hash binding the gate inputs and checker. |
| `checker` | object | Mandatory | Mandatory checker identity and timing. |
| `checker.provider` | const `openai` | Mandatory | Best-effort provider associated with the event. |
| `checker.model` | string | Mandatory | Mandatory model. |
| `checker.version` | string | Mandatory | Mandatory version. |
| `checker.prompt_sha256` | string | Mandatory | Mandatory prompt sha256. |
| `checker.started_at` | string | Mandatory | Mandatory UTC start timestamp. |
| `checker.finished_at` | string | Mandatory | Mandatory UTC completion timestamp. |
| `items` | array<object> | Mandatory | Mandatory gate verdict items. |
| `items[].work_id` | string | Mandatory | Mandatory deterministic work identifier. |
| `items[].stable_id` | string | Mandatory | Mandatory stable model identifier. |
| `items[].family_representative_id` | string | Mandatory | Mandatory family representative id. |
| `items[].fidelity_identity` | string \| null | Mandatory | Mandatory fidelity identity. |
| `items[].vet_identity` | string | Mandatory | Mandatory vet identity. |
| `items[].verified_hashes` | object | Mandatory | Mandatory verified hashes. |
| `items[].verified_hashes.proposal` | string | Mandatory | Mandatory proposal. |
| `items[].verified_hashes.source_manifest` | string | Mandatory | Mandatory source manifest. |
| `items[].verified_hashes.evidence` | string | Mandatory | Mandatory literal evidence and coverage record. |
| `items[].verified_hashes.code` | string \| null | Mandatory | Mandatory closed status code. |
| `items[].verified_hashes.source_to_code_map` | string | Mandatory | Mandatory source to code map. |
| `items[].verified_hashes.family_template` | string \| null | Mandatory | Mandatory family template. |
| `items[].integrity` | object | Mandatory | Mandatory integrity. |
| `items[].integrity.verdict` | enum | Mandatory | Mandatory checker verdict. |
| `items[].integrity.hash_mismatches` | array<string> | Mandatory | Mandatory hash mismatches. |
| `items[].integrity.excerpt_discrepancies` | array<string> | Mandatory | Mandatory excerpt discrepancies. |
| `items[].integrity.locator_failures` | array<string> | Mandatory | Mandatory locator failures. |
| `items[].verdict` | enum | Mandatory | Mandatory checker verdict. |
| `items[].field_checks` | array<object> | Mandatory | Mandatory per-field accuracy findings. |
| `items[].field_checks[].field` | string | Mandatory | Mandatory field. |
| `items[].field_checks[].verdict` | enum | Mandatory | Mandatory checker verdict. |
| `items[].field_checks[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `items[].field_checks[].checked_source_ids` | array<string> | Mandatory | Mandatory checked source ids. |
| `items[].field_checks[].reason` | string | Mandatory | Mandatory reason. |
| `items[].field_checks[].required_repair` | string \| null | Mandatory | Mandatory required repair. |
| `items[].fidelity` | object | Mandatory | Mandatory implementation-fidelity gate state. |
| `items[].fidelity.required` | boolean | Mandatory | Mandatory required. |
| `items[].fidelity.verdict` | enum | Mandatory | Mandatory checker verdict. |
| `items[].fidelity.material_checks` | array<object> | Mandatory | Mandatory material fidelity findings. |
| `items[].fidelity.material_checks[].category` | string | Mandatory | Mandatory category. |
| `items[].fidelity.material_checks[].verdict` | enum | Mandatory | Mandatory checker verdict. |
| `items[].fidelity.material_checks[].source_id` | string | Mandatory | Mandatory source identifier. |
| `items[].fidelity.material_checks[].source_locator` | string | Mandatory | Mandatory source locator. |
| `items[].fidelity.material_checks[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `items[].fidelity.material_checks[].code_path` | string \| null | Mandatory | Mandatory code path. |
| `items[].fidelity.material_checks[].code_locator` | string \| null | Mandatory | Mandatory code locator. |
| `items[].fidelity.material_checks[].reason` | string | Mandatory | Mandatory reason. |
| `items[].fidelity.unsupported_choices` | array<string> | Mandatory | Mandatory unsupported choices. |
| `items[].fidelity.contradictions` | array<string> | Mandatory | Mandatory contradictions. |
| `items[].fidelity.omissions` | array<string> | Mandatory | Mandatory omissions. |
| `items[].fidelity.permanent_scar` | boolean | Mandatory | Mandatory permanent scar. |
| `items[].rung_check` | object | Mandatory | Mandatory rung check. |
| `items[].rung_check.selected_rung` | enum | Mandatory | Mandatory selected rung. |
| `items[].rung_check.highest_applicable` | enum | Mandatory | Mandatory highest applicable. |
| `items[].rung_check.verdict` | enum | Mandatory | Mandatory checker verdict. |
| `items[].rung_check.findings` | array<string> | Mandatory | Mandatory findings. |
| `items[].unsupported_claims` | array<string> | Mandatory | Mandatory unsupported claims. |
| `items[].required_repairs` | array<string> | Mandatory | Mandatory repairs required before acceptance. |
| `items[].confidence` | enum | Mandatory | Mandatory checker confidence level. |
| `result_envelope_sha256` | string | Mandatory | Mandatory result envelope sha256. |

The `author-proposal.v2` and `operational-event.v1` schemas are also self-documenting: every property carries its own executable JSON Schema description.

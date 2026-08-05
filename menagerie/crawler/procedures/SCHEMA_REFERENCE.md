# Crawler schema reference

This data dictionary is the human companion to the executable JSON Schemas in
`menagerie/crawler/schemas/`. Closed vocabularies and cross-field conditions remain
authoritative in the schemas; this document restates them, it does not extend them.

**Every field table below is generated.** The rows are a mechanical projection of the
executable schemas -- the field path, its rendered type, whether the enclosing object
requires it, and the leaf's own schema `description` verbatim. Do not hand-edit anything
between a `BEGIN GENERATED SCHEMA TABLE` / `END GENERATED SCHEMA TABLE` marker pair: the
prose outside those markers is hand-written, the tables inside them are owned by
`menagerie/crawler/tools/render_schema_reference.py` and re-derived by the test suite on
every run. After a schema change, regenerate with:

```bash
python -m menagerie.crawler.tools.render_schema_reference --write
```

The `Presence` column is mechanically derived from the enclosing object's `required` list:

- **Mandatory** -- required by every alternative shape of its enclosing object.
- **Optional** -- declared by every alternative shape and required by none.
- **Branch-dependent** -- declared or required by only some of the alternative shapes
  (`oneOf`/`anyOf` branches). The branch condition itself is authoritative in the schema.

Presence is a structural fact only. A field can be `Mandatory` and still carry `null`, and
its `Meaning` text is where the schema says so -- descriptions that begin "Best-effort"
mark exactly those observations that are required to be present but not guaranteed to
exist.

Named closed vocabularies referenced by the `Type` column are expanded once under
"Closed vocabularies" rather than repeated in every row.

Current v3 amendment: `input_contract.code_path` is absent from `model.v3`,
`author-proposal.v3`, and embedded `author-result.v4`; either null or string presence
rejects. The field listed below belongs only to readable untrusted `model.v2` history. The
distinct current-v3 fields `implementation.code_path` and
`implementation.source_to_code_map[].code_path` remain mandatory where their enclosing
recipe requires them.

## `model.v2`

### Bookkeeping

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/bookkeeping -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `schema_version` | const `menagerie.crawler.model.v2` | Mandatory | Mandatory versioned schema identifier. |
| `stable_id` | string | Mandatory | Mandatory stable model identifier. |
| `record_seq` | integer | Mandatory | Mandatory monotonic model-ledger sequence. |
| `record_revision` | string | Mandatory | Mandatory hash of this model revision. |
| `parent_revision` | string \| null | Mandatory | Best-effort hash of the superseded model revision. |
| `created_at` | string | Mandatory | Mandatory UTC creation timestamp. |
| `revised_by` | object | Mandatory | Mandatory actor that produced this revision. |
| `revised_by.actor` | const `driver` \| string | Mandatory | Mandatory producing actor. |
| `revised_by.model` | string | Branch-dependent | Mandatory model. |
| `revised_by.version` | string | Branch-dependent | Mandatory version. |
| `authored_metadata_state` | enum: `pending` \| `accepted` \| `failed` | Mandatory | Mandatory acceptance state for source-read metadata. |
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
| `untrusted_attempt` | object \| null | Optional | Explicitly untrusted authored attempt facts excluded from authority fields. |
| `untrusted_attempt.proposal_sha256` | string | Mandatory | Hash of the retained unaccepted author proposal. |
| `untrusted_attempt.proposal` | object map | Mandatory | Raw author proposal retained only as untrusted attempt evidence. |
<!-- END GENERATED SCHEMA TABLE: model.v2/bookkeeping -->

### Identity and taxonomy

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/identity-and-taxonomy -->
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
| `taxonomy` | object \| null | Mandatory | Best-effort gated architecture taxonomy. |
| `taxonomy.family` | string | Mandatory | Mandatory family. |
| `taxonomy.domains` | array<string> | Mandatory | Mandatory domains. |
| `taxonomy.tasks` | array<string> | Mandatory | Mandatory tasks. |
| `taxonomy.modalities` | array<string> | Mandatory | Mandatory modalities. |
| `taxonomy.era` | string \| null | Mandatory | Mandatory era, or null only with a typed taxonomy.era availability record. |
| `taxonomy.architecture_tags` | array<string> | Mandatory | Mandatory architecture tags. |
| `taxonomy.novel_ops` | array<string> | Mandatory | Mandatory novel ops. |
| `family_variant_derivation` | object \| null | Optional | Reducer-verifiable mechanical family recipe specialization proof. |
| `family_variant_derivation.variant_token` | string | Mandatory | Exact trusted intake token selecting the variant. |
| `family_variant_derivation.template_source_model_id` | string | Mandatory | Exact representative stable ID. |
| `family_variant_derivation.template_source_revision` | string | Mandatory | Exact representative record revision. |
| `family_variant_derivation.representative_recipe_revision` | string | Mandatory | Exact representative recipe identity. |
| `family_variant_derivation.representative_library_recipe` | object | Mandatory | Byte-exact representative declarative library recipe. |
| `family_variant_derivation.representative_library_recipe.distribution` | string | Mandatory | Mandatory distribution. |
| `family_variant_derivation.representative_library_recipe.version` | string | Mandatory | Mandatory version. |
| `family_variant_derivation.representative_library_recipe.artifact_sha256` | string \| null | Optional | Machine-derived installed-distribution artifact digest. The author has no package inventory and cannot derive it, so the leaf is optional in an author proposal. The driver resolves it from the routed intent's exact resolved export before gating, matching the declared Python distribution against the inventory package that provides it, and refuses a conflicting or unverifiable supplied value. It is null ONLY when the routed target exposes no package inventory at all; an inventory that does not install the pinned distribution is a typed refusal, never a null. |
| `family_variant_derivation.representative_library_recipe.module` | string | Mandatory | Mandatory module. |
| `family_variant_derivation.representative_library_recipe.symbol` | string | Mandatory | Mandatory symbol. |
| `family_variant_derivation.representative_library_recipe.kwargs` | object map | Mandatory | Mandatory kwargs. A mapping value carrying the single '__construct__' key is a closed construct node ({module, symbol, kwargs}) resolved from the declared distribution at build time; every other value is literal JSON. Generic torch.nn containers are refused as construct-node symbols. |
| `family_variant_derivation.representative_library_recipe.pretrained_disable_fields` | array<string> | Mandatory | Mandatory pretrained disable fields. Names EVERY pinned-constructor keyword that would otherwise load pretrained weights or a checkpoint. This is checked, not merely recorded: every name here must also be a key of kwargs carrying a disabling value (null, false, empty string, 'none', or 'random'), names must not repeat, and each name must be a real keyword parameter of the pinned constructor -- a name absent from that signature is refused when the recipe is loaded. R1_LIBRARY must state its pretrained disposition positively, so an empty array is accepted ONLY together with pretrained_fields_absent true; an empty array with no assertion is refused as silence. Three further checks run whether or not you declare anything, and none can be satisfied by wording: a kwargs key naming a known pretrained keyword (pretrained, weights, encoder_weights, pretrained_backbone, weights_backbone, pretrained_cfg, checkpoint_path and similar) with an enabling value is refused outright; a kwargs key with a known pretrained-asset name that is not listed here is refused until it is declared, so an enabling value can never ride through unlisted; and at load time the real constructor signature is read so that a pretrained keyword left at an ENABLING DEFAULT you never mentioned -- encoder_weights='imagenet' -- is refused before construction. |
| `family_variant_derivation.representative_library_recipe.pretrained_fields_absent` | boolean | Optional | Optional positive assertion that the pinned constructor exposes NO pretrained, weights, or checkpoint keyword at all, so there is nothing to disable. This is the honest spelling for a constructor such as MiniMaxForCausalLM(config), a GNN layer, or an SNN, and it is the ONLY way an empty pretrained_disable_fields is accepted. It is checked, not merely recorded: it cannot be combined with a non-empty pretrained_disable_fields (asserting both is a refused contradiction), and it is not taken on trust: the driver re-derives it from the real constructor signature when the recipe is loaded and refuses the recipe if that signature exposes any known pretrained keyword. Absent means false. |
| `family_variant_derivation.representative_library_recipe.entrypoint` | string \| null | Optional | Optional public non-forward call method for the declarative model, delegated through the crawler-owned transparent adapter and receipted as delegated_method. Null or absent means the native forward. Underscore-prefixed names are refused. |
| `family_variant_derivation.representative_library_recipe.post_construct` | array<object> | Optional | Optional bounded declarative post-construction configuration calls applied to the constructed model in order, before the runtime provenance tripwire. Plain JSON arguments only; construct nodes are refused here. |
| `family_variant_derivation.representative_library_recipe.post_construct[].method` | string | Mandatory | Mandatory public method name invoked on the constructed model; underscore-prefixed names are refused. |
| `family_variant_derivation.representative_library_recipe.post_construct[].args` | array | Mandatory | Mandatory plain-JSON positional arguments for the configuration call. |
| `family_variant_derivation.representative_library_recipe.post_construct[].kwargs` | object map | Mandatory | Mandatory plain-JSON keyword arguments for the configuration call. |
| `family_variant_derivation.selector_rule` | object | Mandatory | Closed selector-key or direct-symbol specialization rule. |
| `family_variant_derivation.selector_rule.kind` | enum: `kwarg` \| `symbol` | Mandatory | Closed mechanical selector rule kind. |
| `family_variant_derivation.selector_rule.key` | string \| null | Mandatory | Exact selector kwarg, or null for a symbol rule. |
| `family_variant_derivation.allowed_recipe_delta` | object | Mandatory | The sole permitted representative recipe change. |
| `family_variant_derivation.allowed_recipe_delta.path` | array<string> | Mandatory | Exact recipe path changed by specialization. |
| `family_variant_derivation.allowed_recipe_delta.previous_value` | value | Mandatory | Exact representative value replaced by specialization. |
| `family_variant_derivation.allowed_recipe_delta.new_value` | string | Mandatory | Exact trusted intake token written at the recipe path. |
| `family_variant_derivation.allowed_input_delta` | const `unchanged` | Mandatory | Input contract must remain byte-identical to the representative. |
<!-- END GENERATED SCHEMA TABLE: model.v2/identity-and-taxonomy -->

### External metadata

External metadata is captured and gated now because it requires source reading, web
research, or human judgment. Parameter counts, input/output shapes, operation types,
FLOPs, and graph structure are TorchLens-derivable; they are optional observations, never
a reason to re-crawl external sources.

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/external-metadata -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `external_metadata` | object \| null | Mandatory | Best-effort gated externally sourced catalog metadata. |
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
| `external_metadata.keywords` | array<string> | Mandatory | Mandatory non-empty English user search terms. These are relevance-checked model/family/task/domain aliases and phrases, not verbatim-source quotations. |
| `external_metadata.venue` | string \| null | Mandatory | Mandatory venue. |
| `external_metadata.family` | string | Mandatory | Mandatory family. |
| `external_metadata.era` | string | Mandatory | Mandatory era. |
| `external_metadata.year` | integer \| null | Mandatory | Mandatory year. |
| `external_metadata.country` | string \| null | Mandatory | Mandatory country. |
| `external_metadata.authors` | array<string> | Mandatory | Mandatory authors. |
| `external_metadata.institution` | array<string> | Mandatory | Mandatory institution. |
| `external_metadata.citation` | object | Mandatory | Best-effort gated citation metadata. |
| `external_metadata.citation.status` | enum: `present` \| `not-found-after-search` \| `not-applicable` | Mandatory | Mandatory closed current disposition. Declaring 'present' commits to evidence, not just to values: the frozen manifest must carry a controlled-fetched paper-role source (declared role introducing-paper, supplement, or project-page) and at least one excerpt from it must name external_metadata.citation in its supports. Paper metadata -- authors, venue, institution, country, year -- does not occur in implementation code and cannot be grounded from it, and a BibTeX block quoted from a repository README is a documentation source, not a paper source. Request quotable paper bytes as a raw-url descriptor; a paper-kind descriptor derives citation metadata only and never becomes a manifest row you can excerpt. |
| `external_metadata.citation.title` | string \| null | Mandatory | Mandatory title. |
| `external_metadata.citation.authors` | array<string> | Mandatory | Mandatory authors. EVERY listed author is checked individually: each name's components (given name AND surname, diacritics folded, order-insensitive) must all occur inside the text of an excerpt named by source_evidence_ids and drawn from a paper-role source. A name appearing elsewhere in the fetched page grounds nothing, and an excerpt from a documentation source (a README or model-doc page that happens to name the authors) is not a paper-role excerpt and does not count. So excerpt the paper page's COMPLETE author list: the abs-page summary line 'by First Author and N other authors' grounds ONLY the named first author, and a single author's search-link entry grounds only that one author -- the 2026-08-05 rung killed three otherwise-correct models (m5273, m538, m5445) on exactly those two shapes while the full list sat in the already-fetched bytes. Quote lines as printed, including markup: when a rendering fuses a name with a name-derived email address (an ar5iv \\addauthor failure printing 'Hanchao Lilihanchao@bit.edu.com1'), the fused run itself grounds the name, so do not repair or re-space it. |
| `external_metadata.citation.year` | integer \| null | Mandatory | Mandatory year. When non-null it must be grounded like every other citation leaf: the four-digit year must occur inside the text of an excerpt named by source_evidence_ids and drawn from a paper-role source. A rendering of the paper BODY (ar5iv HTML, a PDF-to-text page) does NOT print the work its own year -- the only years such a page carries are other works' years in its reference list, and binding one of those grounds nothing. Fetch the landing page (arxiv.org/abs/..., the PMC article, the OpenReview forum) as a second paper-role source and bind the line that states the date. One entailment is granted instead: when a modern arXiv identifier YYMM.NNNNN is itself grounded, it encodes the announcement month, so the announcement year or the single following venue-publication year is established without an excerpt. An earlier year, or one two or more years later, is not entailed and stays refused. |
| `external_metadata.citation.venue` | string \| null | Mandatory | Mandatory venue. |
| `external_metadata.citation.arxiv_id` | string \| null | Mandatory | Mandatory arxiv id. When non-null the identifier must appear inside the text of an excerpt named by source_evidence_ids; occurring elsewhere in the fetched page grounds nothing, so bind the line that carries it as its own excerpt. A page mention that adds a revision selector (arXiv:2111.11418v3) grounds the unversioned id, but a bare mention does not ground a version you declare. |
| `external_metadata.citation.doi` | string \| null | Mandatory | Mandatory doi. When non-null the identifier must appear inside the text of an excerpt named by source_evidence_ids; occurring elsewhere in the fetched page grounds nothing, so bind the line that carries it as its own excerpt. A DOI suffix is opaque, so it is matched exactly and no near variant grounds it. |
| `external_metadata.citation.openreview_id` | string \| null | Mandatory | Mandatory openreview id. When non-null the identifier must appear inside the text of an excerpt named by source_evidence_ids; occurring elsewhere in the fetched page grounds nothing, so bind the line that carries it as its own excerpt. An OpenReview id is opaque, so it is matched exactly and no near variant grounds it. |
| `external_metadata.citation.url` | string \| null | Mandatory | Mandatory public source URL when available. |
| `external_metadata.citation.bibtex` | string \| null | Mandatory | Mandatory bibtex. |
| `external_metadata.citation.source_evidence_ids` | array<string> | Mandatory | Mandatory ids of the excerpts that ground this citation. Every non-null citation leaf is value-checked against the concatenated text of these excerpts and nothing else: title, venue, year, each author, and each declared identifier must occur inside that bound text. A value present in the fetched source bytes but not inside a bound excerpt is refused, so bind an excerpt that literally carries each value declared here, adding a second excerpt when the identifier and the title sit on different lines. |
| `external_metadata.license` | string \| null | Mandatory | Mandatory license. |
| `external_metadata.key_contribution` | string | Mandatory | Mandatory key contribution. |
| `external_metadata.description` | string | Mandatory | Mandatory description. |
| `external_metadata.original_framework` | string | Mandatory | Mandatory original framework. |
| `external_metadata.run_framework` | string | Mandatory | Mandatory run framework. |
| `external_metadata.modes` | object | Mandatory | Mandatory meaningful runtime-mode record. |
| `external_metadata.modes.meaningful_modes` | array<enum: `train` \| `eval`> | Mandatory | Mandatory meaningful modes. |
| `external_metadata.modes.train_eval_divergence` | enum: `none` \| `statistical` \| `structural` | Mandatory | Mandatory train eval divergence. |
<!-- END GENERATED SCHEMA TABLE: model.v2/external-metadata -->

### Website

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/website -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `website` | object \| null | Mandatory | Best-effort presentation metadata for the catalog page. |
| `website.kind` | enum: `family-representative` \| `size-variant-template` | Mandatory | Mandatory record or status kind. |
| `website.tagline` | string | Mandatory | Mandatory tagline. |
| `website.description` | string | Mandatory | Mandatory description. |
| `website.key_contribution` | string | Mandatory | Mandatory key contribution. |
| `website.voice_version` | string | Mandatory | Mandatory voice version. |
| `website.family_grounding_id` | string | Mandatory | Mandatory family grounding id: the evidence_id of a family_level excerpt in this record's own evidence block, validated referentially at proposal admission. |
| `website.template_source_model_id` | string \| null | Mandatory | Mandatory template source model id. |
| `website.variant_parameter_input_line` | string \| null | Mandatory | Mandatory variant parameter input line. |
| `website.template_hash` | string \| null | Mandatory | Mandatory template hash. |
<!-- END GENERATED SCHEMA TABLE: model.v2/website -->

### People, origin, dates, and citation

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/people-origin-dates-citation -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `people_and_origin` | object \| null | Mandatory | Best-effort gated people and origin metadata. |
| `people_and_origin.authors` | array<string> | Mandatory | Mandatory authors. |
| `people_and_origin.labs` | array<string> | Mandatory | Mandatory labs. |
| `people_and_origin.institutions` | array<string> | Mandatory | Mandatory institutions. |
| `people_and_origin.origin_countries` | array<string> | Mandatory | Mandatory origin countries. |
| `people_and_origin.country_basis` | string | Mandatory | Mandatory country basis. |
| `people_and_origin.country_confidence` | enum: `high` \| `medium` \| `low` \| `cannot-determine` | Mandatory | Mandatory country confidence. |
| `people_and_origin.country_note` | string | Mandatory | Mandatory country note. |
| `dates` | object \| null | Mandatory | Best-effort gated publication-date metadata. |
| `dates.year` | integer \| null | Mandatory | Mandatory year. |
| `dates.year_basis` | string | Mandatory | Mandatory year basis. |
| `dates.first_public_date` | string \| null | Mandatory | Mandatory first public date. |
| `dates.first_public_date_basis` | string | Mandatory | Mandatory first public date basis. |
| `citation` | object \| null | Mandatory | Best-effort gated citation metadata. |
| `citation.status` | enum: `present` \| `not-found-after-search` \| `not-applicable` | Mandatory | Mandatory closed current disposition. Declaring 'present' commits to evidence, not just to values: the frozen manifest must carry a controlled-fetched paper-role source (declared role introducing-paper, supplement, or project-page) and at least one excerpt from it must name external_metadata.citation in its supports. Paper metadata -- authors, venue, institution, country, year -- does not occur in implementation code and cannot be grounded from it, and a BibTeX block quoted from a repository README is a documentation source, not a paper source. Request quotable paper bytes as a raw-url descriptor; a paper-kind descriptor derives citation metadata only and never becomes a manifest row you can excerpt. |
| `citation.title` | string \| null | Mandatory | Mandatory title. |
| `citation.authors` | array<string> | Mandatory | Mandatory authors. EVERY listed author is checked individually: each name's components (given name AND surname, diacritics folded, order-insensitive) must all occur inside the text of an excerpt named by source_evidence_ids and drawn from a paper-role source. A name appearing elsewhere in the fetched page grounds nothing, and an excerpt from a documentation source (a README or model-doc page that happens to name the authors) is not a paper-role excerpt and does not count. So excerpt the paper page's COMPLETE author list: the abs-page summary line 'by First Author and N other authors' grounds ONLY the named first author, and a single author's search-link entry grounds only that one author -- the 2026-08-05 rung killed three otherwise-correct models (m5273, m538, m5445) on exactly those two shapes while the full list sat in the already-fetched bytes. Quote lines as printed, including markup: when a rendering fuses a name with a name-derived email address (an ar5iv \\addauthor failure printing 'Hanchao Lilihanchao@bit.edu.com1'), the fused run itself grounds the name, so do not repair or re-space it. |
| `citation.year` | integer \| null | Mandatory | Mandatory year. When non-null it must be grounded like every other citation leaf: the four-digit year must occur inside the text of an excerpt named by source_evidence_ids and drawn from a paper-role source. A rendering of the paper BODY (ar5iv HTML, a PDF-to-text page) does NOT print the work its own year -- the only years such a page carries are other works' years in its reference list, and binding one of those grounds nothing. Fetch the landing page (arxiv.org/abs/..., the PMC article, the OpenReview forum) as a second paper-role source and bind the line that states the date. One entailment is granted instead: when a modern arXiv identifier YYMM.NNNNN is itself grounded, it encodes the announcement month, so the announcement year or the single following venue-publication year is established without an excerpt. An earlier year, or one two or more years later, is not entailed and stays refused. |
| `citation.venue` | string \| null | Mandatory | Mandatory venue. |
| `citation.arxiv_id` | string \| null | Mandatory | Mandatory arxiv id. When non-null the identifier must appear inside the text of an excerpt named by source_evidence_ids; occurring elsewhere in the fetched page grounds nothing, so bind the line that carries it as its own excerpt. A page mention that adds a revision selector (arXiv:2111.11418v3) grounds the unversioned id, but a bare mention does not ground a version you declare. |
| `citation.doi` | string \| null | Mandatory | Mandatory doi. When non-null the identifier must appear inside the text of an excerpt named by source_evidence_ids; occurring elsewhere in the fetched page grounds nothing, so bind the line that carries it as its own excerpt. A DOI suffix is opaque, so it is matched exactly and no near variant grounds it. |
| `citation.openreview_id` | string \| null | Mandatory | Mandatory openreview id. When non-null the identifier must appear inside the text of an excerpt named by source_evidence_ids; occurring elsewhere in the fetched page grounds nothing, so bind the line that carries it as its own excerpt. An OpenReview id is opaque, so it is matched exactly and no near variant grounds it. |
| `citation.url` | string \| null | Mandatory | Mandatory public source URL when available. |
| `citation.bibtex` | string \| null | Mandatory | Mandatory bibtex. |
| `citation.source_evidence_ids` | array<string> | Mandatory | Mandatory ids of the excerpts that ground this citation. Every non-null citation leaf is value-checked against the concatenated text of these excerpts and nothing else: title, venue, year, each author, and each declared identifier must occur inside that bound text. A value present in the fetched source bytes but not inside a bound excerpt is refused, so bind an excerpt that literally carries each value declared here, adding a second excerpt when the identifier and the title sit on different lines. |
<!-- END GENERATED SCHEMA TABLE: model.v2/people-origin-dates-citation -->

### Licenses

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/licenses -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `licenses` | object \| null | Mandatory | Best-effort gated source-license metadata. |
| `licenses.code` | object | Mandatory | Mandatory closed status code. |
| `licenses.code.spdx` | string | Mandatory | Mandatory spdx. |
| `licenses.code.status` | enum: `declared` \| `not-found` \| `custom` \| `not-applicable` | Mandatory | Mandatory closed current disposition. |
| `licenses.code.source_id` | string | Mandatory | Mandatory source identifier. |
| `licenses.code.locator` | string | Mandatory | Mandatory exact location within the source. |
| `licenses.code.evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `licenses.paper_text` | object | Mandatory | Mandatory paper text. |
| `licenses.paper_text.status` | enum: `linked-not-redistributed` \| `short-excerpt-committed` \| `not-applicable` | Mandatory | Mandatory closed current disposition. |
| `licenses.paper_text.source_id` | string \| null | Mandatory | Mandatory source identifier. |
| `licenses.weights` | object | Mandatory | Mandatory weights. |
| `licenses.weights.status` | const `not-used` | Mandatory | Mandatory closed current disposition. |
| `licenses.data` | object | Mandatory | Mandatory data. |
| `licenses.data.spdx` | string \| null | Mandatory | Mandatory spdx. |
| `licenses.data.status` | string | Mandatory | Mandatory closed current disposition. |
| `licenses.data.source_id` | string \| null | Mandatory | Mandatory source identifier. |
| `licenses.data.evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `licenses.redistribution_class` | enum: `public-compatible` \| `restricted-private` \| `manifest-only` \| `not-applicable` | Mandatory | Mandatory redistribution class. |
| `licenses.source_dispositions` | array<object> | Optional | Optional per-source dispositions for heterogeneous fetched manifests. |
| `licenses.source_dispositions[].spdx` | string | Mandatory | Exact SPDX or NOASSERTION disposition. |
| `licenses.source_dispositions[].status` | enum: `declared` \| `not-found` \| `custom` \| `not-applicable` | Mandatory | Evidence-backed disposition status. |
| `licenses.source_dispositions[].source_id` | string | Mandatory | Exactly one source-manifest member identity. |
| `licenses.source_dispositions[].locator` | string | Mandatory | Exact license locator in the named source. |
| `licenses.source_dispositions[].evidence_ids` | array<string> | Mandatory | Literal license excerpts supporting this source. |
<!-- END GENERATED SCHEMA TABLE: model.v2/licenses -->

### Source resolution

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/source-resolution -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `source_resolution` | object | Mandatory | Mandatory selected source rung and search record. |
| `source_resolution.rung` | enum `model-common.rung_or_no_selection` | Mandatory | Mandatory source-resolution ladder rung, or NO_RUNG_SELECTED when work ended before any rung was selected. |
| `source_resolution.decision` | string | Mandatory | Mandatory source-resolution decision. |
| `source_resolution.rung_evidence` | string | Mandatory | Mandatory rung evidence. |
| `source_resolution.sufficiency_gap` | string \| null | Mandatory | Mandatory for insufficient-description skips; names missing implementation detail. |
| `source_resolution.searched_at` | string | Mandatory | Mandatory searched at. |
| `source_resolution.attempted_rungs` | array<object> | Mandatory | Mandatory attempted rungs. |
| `source_resolution.attempted_rungs[].rung` | enum `model-common.rung_or_no_selection` | Mandatory | Mandatory source-resolution ladder rung, or NO_RUNG_SELECTED when the ladder was never walked. |
| `source_resolution.attempted_rungs[].result` | string | Mandatory | Mandatory immutable attempt outcome. |
| `source_resolution.attempted_rungs[].reason_code` | string | Mandatory | Mandatory closed reason code when applicable. |
| `source_resolution.attempted_rungs[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. |
| `source_resolution.search_report` | object | Mandatory | Mandatory search report. |
| `source_resolution.search_report.queries` | array<string> | Mandatory | Mandatory queries. |
| `source_resolution.search_report.places_checked` | array<string> | Mandatory | Mandatory places checked. |
| `source_resolution.search_report.links_checked` | array<string> | Mandatory | Mandatory exact content URLs checked. For R2/R4 every entry must bind to fetched, hash-inventoried CAS bytes; unfetched candidate repositories are a coverage gap. |
| `source_resolution.search_report.languages_checked` | array<string> | Mandatory | Mandatory languages checked. |
| `source_resolution.search_report.archives_checked` | array<string> | Mandatory | Mandatory archives checked. |
| `source_resolution.search_report.started_at` | string | Mandatory | Mandatory UTC start timestamp. |
| `source_resolution.search_report.finished_at` | string | Mandatory | Mandatory UTC completion timestamp. |
| `source_resolution.search_report.conclusion` | string | Mandatory | Mandatory conclusion. |
| `source_resolution.mandatory_link_status` | enum: `ok` \| `failed` | Mandatory | Mandatory mandatory link status. |
| `source_resolution.primary_source_id` | string | Mandatory | Mandatory primary source id. |
| `source_resolution.sources` | array<object> | Mandatory | Mandatory resolved public sources. This is a CUSTODY ECHO, not a bibliography: it must name EXACTLY the set the machine fetched for this model -- every row of the frozen source_manifest, plus every row of the supplementary manifest if the one supplementary source round was granted -- with no row added and NO ROW DROPPED. Echo a source you ended up not using anyway; silently omitting one is refused as 'proposal and source manifest source sets differ', and so is naming a source_id that is in neither manifest. Per row, url, revision, content_sha256, byte_count, and media_type must be the manifest's exact bytes verbatim; role, locator, and the remaining leaves are your own judgement about that source. |
| `source_resolution.sources[].source_id` | string | Mandatory | Mandatory source identifier. |
| `source_resolution.sources[].role` | enum: `implementation` \| `introducing-paper` \| `supplement` \| `project-page` \| `documentation` \| `license` \| `affiliation` \| `archive` | Mandatory | Mandatory role. |
| `source_resolution.sources[].kind` | enum: `repository` \| `package` \| `paper` \| `web-page` \| `archive` \| `intake-snapshot` \| `discovery-evidence` | Mandatory | Mandatory record or status kind. |
| `source_resolution.sources[].url` | string | Mandatory | Mandatory public source URL, except for typed machine discovery evidence. |
| `source_resolution.sources[].revision_kind` | string | Mandatory | Mandatory revision kind. |
| `source_resolution.sources[].revision` | string | Mandatory | Mandatory revision. |
| `source_resolution.sources[].locator` | string | Mandatory | Mandatory exact location within the source. |
| `source_resolution.sources[].content_sha256` | string \| null | Mandatory | Mandatory content sha256. |
| `source_resolution.sources[].byte_count` | integer | Mandatory | Mandatory byte count. |
| `source_resolution.sources[].media_type` | string | Mandatory | Mandatory media type. |
| `source_resolution.sources[].retrieved_at` | string | Mandatory | Mandatory retrieved at. |
| `source_resolution.sources[].fetch_recipe` | string | Mandatory | Mandatory fetch recipe. |
| `source_resolution.sources[].mirror_class` | string | Mandatory | Mandatory mirror class. |
| `source_resolution.sources[].mirror_digest` | string \| null | Mandatory | Mandatory mirror digest. |
<!-- END GENERATED SCHEMA TABLE: model.v2/source-resolution -->

### Evidence

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/evidence -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `evidence` | object | Mandatory | Mandatory literal evidence and coverage record. |
| `evidence.excerpts` | array<object> | Mandatory | Mandatory literal source excerpts. |
| `evidence.excerpts[].evidence_id` | string | Mandatory | Mandatory evidence identifier. |
| `evidence.excerpts[].source_id` | string | Mandatory | Mandatory source identifier. |
| `evidence.excerpts[].locator` | string | Mandatory | Mandatory exact location within the source. |
| `evidence.excerpts[].text` | string | Mandatory | Mandatory verbatim retained source text. |
| `evidence.excerpts[].text_sha256` | string | Mandatory | Mandatory hash of the verbatim excerpt. |
| `evidence.excerpts[].supports` | array<string> | Mandatory | Mandatory supports; claim-category strings matched by exact equality, never by prefix roll-up. |
| `evidence.excerpts[].family_level` | boolean | Mandatory | Mandatory family level. |
| `evidence.excerpts[].disposition` | enum: `supporting` \| `insufficient-for-faithful-reimpl` | Mandatory | Mandatory excerpt role, including insufficient reimplementation evidence. |
| `evidence.excerpts[].license_disposition` | string | Mandatory | Mandatory license handling for the retained excerpt. |
| `evidence.coverage` | object | Mandatory | Mandatory coverage. |
| `evidence.coverage.all_agent_fields_have_support` | boolean | Mandatory | Mandatory all agent fields have support. |
| `evidence.coverage.missing_support` | array<string> | Mandatory | Mandatory missing support. |
| `evidence.coverage.family_grounding_complete` | boolean | Mandatory | Mandatory family grounding complete. |
| `evidence.evidence_identity` | string | Mandatory | Mandatory evidence identity. |
| `evidence.family_grounding_path` | string \| null | Mandatory | Mandatory family grounding path. |
<!-- END GENERATED SCHEMA TABLE: model.v2/evidence -->

### Implementation

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/implementation -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `implementation` | object | Mandatory | Mandatory executable implementation recipe. |
| `implementation.original_framework` | string | Mandatory | Mandatory original framework. |
| `implementation.run_framework` | string | Mandatory | Mandatory run framework. |
| `implementation.native_object_type` | string | Mandatory | Mandatory native object type. |
| `implementation.native_call_method` | string | Mandatory | Mandatory native call method. |
| `implementation.transparent_forward_adapter` | boolean | Mandatory | Mandatory transparent forward adapter. |
| `implementation.recipe_type` | enum: `declarative-library` \| `typed-adapter` \| `port` \| `reimplementation` \| `none` | Mandatory | Mandatory recipe type. |
| `implementation.code_path` | string \| null | Mandatory | Mandatory code path. Model-root-relative entry module for a staged rung; null for R1_LIBRARY, which refuses staged code and must use library_recipe instead. Staged code is checked by AST before acceptance and the code YOU WRITE must be FULLY TYPED: every function and method defined in this module, and in every AUTHOR-WRITTEN member of its recursive model-local import closure listed in code_manifest, carries an annotation on every parameter other than self/cls -- including *args and **kwargs -- and a return annotation. One unannotated authored definition anywhere in that closure refuses the whole proposal, and this entry module is never exempt whatever it contains. A closure member that is VERBATIM VENDORED upstream source is exempt from the annotation rule only, and only on proof: it must be declared in upstream_files and its staged bytes must hash to a content_sha256 the controlled-fetch manifest already froze. Editing one character breaks that digest and the file goes straight back under the annotation rule, so vendored bytes must be copied exactly and never adjusted to pass a check. The exemption is for LEGIBILITY only: the dynamic-execution and out-of-sandbox-write checks cover every closure member with no exemption at all, vendored or authored. |
| `implementation.code_sha256` | string \| null | Mandatory | Mandatory code sha256. |
| `implementation.code_manifest` | array<object> | Optional | Closed recursive model-local Python import manifest. Every AUTHOR-WRITTEN member listed here is subject to the same full-annotation AST check as code_path; a member proven to be verbatim vendored upstream bytes (declared in upstream_files, staged digest equal to a frozen controlled-fetch content_sha256) is exempt from that check alone. Every member without exception is checked for dynamic execution and out-of-sandbox writes. |
| `implementation.code_manifest[].path` | string | Mandatory | Model-root-relative imported code path. |
| `implementation.code_manifest[].sha256` | string | Mandatory | Exact imported code member digest. |
| `implementation.builder_symbol` | const `build_model` \| null | Mandatory | Mandatory builder symbol. Exactly build_model for a staged rung, null otherwise. It must be defined fully annotated -- 'def build_model() -> torch.nn.Module:', not 'def build_model():'; see code_path for the annotation rule that covers every staged function. |
| `implementation.dummy_call_symbol` | const `make_dummy_call` \| null | Mandatory | Mandatory dummy call symbol. Exactly make_dummy_call for a staged rung, null otherwise. Both parameters and the return need annotations -- 'def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:', not 'def make_dummy_call(seed, device):'; see code_path. |
| `implementation.library_recipe` | object \| null | Mandatory | Mandatory library recipe. |
| `implementation.library_recipe.distribution` | string | Mandatory | Mandatory distribution. |
| `implementation.library_recipe.version` | string | Mandatory | Mandatory version. |
| `implementation.library_recipe.artifact_sha256` | string \| null | Optional | Machine-derived installed-distribution artifact digest. The author has no package inventory and cannot derive it, so the leaf is optional in an author proposal. The driver resolves it from the routed intent's exact resolved export before gating, matching the declared Python distribution against the inventory package that provides it, and refuses a conflicting or unverifiable supplied value. It is null ONLY when the routed target exposes no package inventory at all; an inventory that does not install the pinned distribution is a typed refusal, never a null. |
| `implementation.library_recipe.module` | string | Mandatory | Mandatory module. |
| `implementation.library_recipe.symbol` | string | Mandatory | Mandatory symbol. |
| `implementation.library_recipe.kwargs` | object map | Mandatory | Mandatory kwargs. A mapping value carrying the single '__construct__' key is a closed construct node ({module, symbol, kwargs}) resolved from the declared distribution at build time; every other value is literal JSON. Generic torch.nn containers are refused as construct-node symbols. |
| `implementation.library_recipe.pretrained_disable_fields` | array<string> | Mandatory | Mandatory pretrained disable fields. Names EVERY pinned-constructor keyword that would otherwise load pretrained weights or a checkpoint. This is checked, not merely recorded: every name here must also be a key of kwargs carrying a disabling value (null, false, empty string, 'none', or 'random'), names must not repeat, and each name must be a real keyword parameter of the pinned constructor -- a name absent from that signature is refused when the recipe is loaded. R1_LIBRARY must state its pretrained disposition positively, so an empty array is accepted ONLY together with pretrained_fields_absent true; an empty array with no assertion is refused as silence. Three further checks run whether or not you declare anything, and none can be satisfied by wording: a kwargs key naming a known pretrained keyword (pretrained, weights, encoder_weights, pretrained_backbone, weights_backbone, pretrained_cfg, checkpoint_path and similar) with an enabling value is refused outright; a kwargs key with a known pretrained-asset name that is not listed here is refused until it is declared, so an enabling value can never ride through unlisted; and at load time the real constructor signature is read so that a pretrained keyword left at an ENABLING DEFAULT you never mentioned -- encoder_weights='imagenet' -- is refused before construction. |
| `implementation.library_recipe.pretrained_fields_absent` | boolean | Optional | Optional positive assertion that the pinned constructor exposes NO pretrained, weights, or checkpoint keyword at all, so there is nothing to disable. This is the honest spelling for a constructor such as MiniMaxForCausalLM(config), a GNN layer, or an SNN, and it is the ONLY way an empty pretrained_disable_fields is accepted. It is checked, not merely recorded: it cannot be combined with a non-empty pretrained_disable_fields (asserting both is a refused contradiction), and it is not taken on trust: the driver re-derives it from the real constructor signature when the recipe is loaded and refuses the recipe if that signature exposes any known pretrained keyword. Absent means false. |
| `implementation.library_recipe.entrypoint` | string \| null | Optional | Optional public non-forward call method for the declarative model, delegated through the crawler-owned transparent adapter and receipted as delegated_method. Null or absent means the native forward. Underscore-prefixed names are refused. |
| `implementation.library_recipe.post_construct` | array<object> | Optional | Optional bounded declarative post-construction configuration calls applied to the constructed model in order, before the runtime provenance tripwire. Plain JSON arguments only; construct nodes are refused here. |
| `implementation.library_recipe.post_construct[].method` | string | Mandatory | Mandatory public method name invoked on the constructed model; underscore-prefixed names are refused. |
| `implementation.library_recipe.post_construct[].args` | array | Mandatory | Mandatory plain-JSON positional arguments for the configuration call. |
| `implementation.library_recipe.post_construct[].kwargs` | object map | Mandatory | Mandatory plain-JSON keyword arguments for the configuration call. |
| `implementation.upstream_files` | array<object> | Mandatory | Mandatory upstream files. Exact verbatim upstream bytes staged beside the adapter, each bound to a frozen controlled-fetch source. This array is also what proves a closure member was not written by you: a member declared here whose staged bytes hash to the declared, frozen content_sha256 is exempt from the full-annotation AST check that covers authored code. Declare every vendored file; an undeclared one is treated as yours. |
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
| `implementation.source_to_code_map[].source_id` | string | Mandatory | Mandatory source identifier. Under R2_VENDOR every row must name one of implementation.upstream_files[].source_id -- the map exists to bind vendored upstream bytes to the staged code, so a row pointing at a README, a requirements file, or any other non-vendored source is refused even when its disposition is authored-adapter. Support that an adapter draws from documentation belongs in evidence.excerpts, not here. |
| `implementation.source_to_code_map[].source_locator` | string | Mandatory | Mandatory source locator. It must be byte-identical to the locator of one excerpt in evidence.excerpts that also carries this row's source_id and is named in this row's evidence_ids; the three are matched as one exact triple, so "pit.py lines 54-62" does not bind an excerpt recorded at "pit.py lines 54-69". |
| `implementation.source_to_code_map[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers. At least one must name an excerpt whose (evidence_id, source_id, locator) triple matches this row exactly. |
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
| `implementation.torchlens_import_static_check` | enum: `passed` \| `not-checked` \| `not-applicable-no-code` | Mandatory | Mandatory torchlens import static-check disposition. 'passed' is an attestation that only a party with an interpreter may make; the author stage has none, so an author declares the honest 'not-checked'. |
| `implementation.declared_timeout_seconds` | integer \| null | Optional | Optional author-declared per-forward timeout seconds in [1,1800]; null uses the default. |
<!-- END GENERATED SCHEMA TABLE: model.v2/implementation -->

### Input contract

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/input-contract -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `input_contract` | object | Mandatory | Mandatory source-valid dummy-input contract. |
| `input_contract.code_path` | string \| null | Mandatory | Mandatory code path. |
| `input_contract.builder_symbol` | string | Mandatory | Mandatory builder symbol: the exact thing this contract's args/kwargs are passed to, named as a DOTTED PYTHON PATH and nothing else. The grammar is enforced and is exactly identifier('.'identifier)* -- so 'timm.models.dla.dla60' and 'build_model' are legal, while 'timm.models.dla:dla60' (entry-point colon), 'declarative-library-recipe' or any other hyphenated placeholder, a slashed path, a call expression, and a bare description are all refused. Derive it, never invent it: for recipe_type 'declarative-library' it is implementation.library_recipe.module + '.' + implementation.library_recipe.symbol; for a staged rung it is exactly implementation.builder_symbol, i.e. 'build_model'. This field names the constructor for the record's readers -- it is never dereferenced as a filesystem path, and the identifier grammar is what keeps it from becoming one. |
| `input_contract.seed` | integer | Mandatory | Mandatory seed. |
| `input_contract.semantic_description` | string | Mandatory | Mandatory semantic description. |
| `input_contract.source_basis` | array<string> | Mandatory | Mandatory source basis. |
| `input_contract.smallest_valid_probe_rationale` | string | Mandatory | Mandatory smallest valid probe rationale. |
| `input_contract.args` | array<object> | Mandatory | Mandatory args. |
| `input_contract.args[].path` | string | Mandatory | Mandatory path. |
| `input_contract.args[].kind` | enum: `tensor` \| `constructed` | Mandatory | Mandatory materialization kind, and a CLOSED vocabulary because the executor's is closed: worker._materialize_declarative_call materializes exactly 'tensor' and 'constructed' and refuses every other value. Publishing it as an open string let a schema-valid, checker-passed proposal name a descriptive kind ('standard-image-tensor') and die at execution on an ordinary image tensor, after the one authoring visit this catalog allows. The vocabulary is deliberately about MATERIALIZATION, not modality: modality lives in the recipe, and shape/dtype/distribution already carry the rest. |
| `input_contract.args[].semantic_role` | string | Mandatory | Mandatory semantic role. |
| `input_contract.args[].shape` | array<integer \| string> | Mandatory | Mandatory shape. |
| `input_contract.args[].dtype` | string | Mandatory | Mandatory authored, source-read input dtype. Unlike observed.* dtype/device facts, this input_contract leaf is accuracy-gated and contributes to vet identity. |
| `input_contract.args[].device_policy` | string | Mandatory | Mandatory device policy. |
| `input_contract.args[].distribution` | enum: `normal` \| `uniform` \| `integer-range` \| `zeros` \| `ones` \| `categorical` \| `constructor` | Mandatory | Mandatory distribution. |
| `input_contract.args[].constraints` | array<string> | Mandatory | Mandatory constraints. |
| `input_contract.args[].source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |
| `input_contract.args[].constructor` | object | Optional | Closed declarative constructor for a non-tensor input object (for example a graph). Required exactly when distribution is 'constructor' and kind is 'constructed'; forbidden otherwise. |
| `input_contract.args[].constructor.module` | string | Mandatory | Mandatory dotted import module defining the input constructor. |
| `input_contract.args[].constructor.symbol` | string | Mandatory | Mandatory direct constructor attribute resolved on the module. |
| `input_contract.args[].constructor.kwargs` | object map | Mandatory | Mandatory JSON-only constructor keyword arguments; nested construct nodes follow the same closed grammar and bounds as declarative recipe kwargs. |
| `input_contract.kwargs` | array<object> | Mandatory | Mandatory kwargs. |
| `input_contract.kwargs[].path` | string | Mandatory | Mandatory path. |
| `input_contract.kwargs[].kind` | enum: `tensor` \| `constructed` | Mandatory | Mandatory materialization kind, and a CLOSED vocabulary because the executor's is closed: worker._materialize_declarative_call materializes exactly 'tensor' and 'constructed' and refuses every other value. Publishing it as an open string let a schema-valid, checker-passed proposal name a descriptive kind ('standard-image-tensor') and die at execution on an ordinary image tensor, after the one authoring visit this catalog allows. The vocabulary is deliberately about MATERIALIZATION, not modality: modality lives in the recipe, and shape/dtype/distribution already carry the rest. |
| `input_contract.kwargs[].semantic_role` | string | Mandatory | Mandatory semantic role. |
| `input_contract.kwargs[].shape` | array<integer \| string> | Mandatory | Mandatory shape. |
| `input_contract.kwargs[].dtype` | string | Mandatory | Mandatory authored, source-read input dtype. Unlike observed.* dtype/device facts, this input_contract leaf is accuracy-gated and contributes to vet identity. |
| `input_contract.kwargs[].device_policy` | string | Mandatory | Mandatory device policy. |
| `input_contract.kwargs[].distribution` | enum: `normal` \| `uniform` \| `integer-range` \| `zeros` \| `ones` \| `categorical` \| `constructor` | Mandatory | Mandatory distribution. |
| `input_contract.kwargs[].constraints` | array<string> | Mandatory | Mandatory constraints. |
| `input_contract.kwargs[].source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |
| `input_contract.kwargs[].constructor` | object | Optional | Closed declarative constructor for a non-tensor input object (for example a graph). Required exactly when distribution is 'constructor' and kind is 'constructed'; forbidden otherwise. |
| `input_contract.kwargs[].constructor.module` | string | Mandatory | Mandatory dotted import module defining the input constructor. |
| `input_contract.kwargs[].constructor.symbol` | string | Mandatory | Mandatory direct constructor attribute resolved on the module. |
| `input_contract.kwargs[].constructor.kwargs` | object map | Mandatory | Mandatory JSON-only constructor keyword arguments; nested construct nodes follow the same closed grammar and bounds as declarative recipe kwargs. |
| `input_contract.non_tensor_values` | array<object> | Mandatory | Mandatory non tensor values. |
| `input_contract.non_tensor_values[].path` | string | Mandatory | Mandatory path. |
| `input_contract.non_tensor_values[].type` | string | Mandatory | Mandatory type. |
| `input_contract.non_tensor_values[].value` | value | Mandatory | Mandatory value. |
| `input_contract.non_tensor_values[].semantic_role` | string | Mandatory | Mandatory semantic role. |
| `input_contract.non_tensor_values[].constraints` | array<string> | Mandatory | Mandatory constraints. |
| `input_contract.non_tensor_values[].source_evidence_ids` | array<string> | Mandatory | Mandatory source evidence ids. |
| `input_contract.masks_state_and_control` | array<string> | Mandatory | Mandatory masks state and control. |
| `input_contract.expected_output_semantics` | string | Mandatory | Mandatory expected output semantics. |
<!-- END GENERATED SCHEMA TABLE: model.v2/input-contract -->

### Observed

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/observed -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `observed` | object | Mandatory | Mandatory best-effort observed runtime facts. |
| `observed.parameter_count_total` | integer | Mandatory | Mandatory parameter count total. |
| `observed.parameter_count_trainable` | integer | Mandatory | Mandatory parameter count trainable. |
| `observed.native_framework` | string \| null | Mandatory | Worker-observed native framework for the designated measurement attempt. |
| `observed.delegated_method` | string \| null | Mandatory | Worker-observed native call method for the designated measurement attempt. |
| `observed.output_signature` | object | Mandatory | Mandatory output signature. |
| `observed.output_signature.tree` | value | Mandatory | Mandatory tree. |
| `observed.output_signature.leaves` | array<object> | Mandatory | Mandatory leaves. |
| `observed.output_signature.leaves[].path` | string | Mandatory | Mandatory path. |
| `observed.output_signature.leaves[].kind` | string | Mandatory | Mandatory record or status kind. |
| `observed.output_signature.leaves[].shape` | array<integer \| string> \| null | Mandatory | Mandatory shape. |
| `observed.output_signature.leaves[].dtype` | string \| null | Mandatory | Mandatory mechanically observed output dtype under observed.*; this root-path placement is TorchLens-derivable and excluded from authored metadata gating. |
| `observed.output_signature.leaves[].device` | string \| null | Mandatory | Mandatory mechanically observed output device under observed.*; this root-path placement is TorchLens-derivable and excluded from authored metadata gating. |
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
<!-- END GENERATED SCHEMA TABLE: model.v2/observed -->

### Modes and verification state

<!-- BEGIN GENERATED SCHEMA TABLE: model.v2/modes-and-verification-state -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `modes` | object | Mandatory | Mandatory meaningful runtime-mode record. |
| `modes.meaningful_modes` | array<enum: `train` \| `eval`> | Mandatory | Mandatory meaningful modes. |
| `modes.per_mode_run` | object | Mandatory | Mandatory per mode run. |
| `modes.per_mode_run.train` | object | Optional | Best-effort train. |
| `modes.per_mode_run.train.attempt_id` | string | Mandatory | Mandatory immutable attempt identifier. |
| `modes.per_mode_run.train.status` | enum: `succeeded` \| `failed` \| `observed` | Mandatory | Mandatory closed current disposition. |
| `modes.per_mode_run.eval` | object | Optional | Best-effort eval. |
| `modes.per_mode_run.eval.attempt_id` | string | Mandatory | Mandatory immutable attempt identifier. |
| `modes.per_mode_run.eval.status` | enum: `succeeded` \| `failed` \| `observed` | Mandatory | Mandatory closed current disposition. |
| `modes.train_eval_divergence` | enum: `none` \| `statistical` \| `structural` | Mandatory | Mandatory train eval divergence. |
| `modes.divergence_evidence` | string | Mandatory | Mandatory divergence evidence. |
| `fidelity` | object | Mandatory | Mandatory implementation-fidelity gate state. |
| `fidelity.required` | boolean | Mandatory | Mandatory required. AUTHOR-OWNED: whether an implementation-fidelity gate must adjudicate this record, subject to the machine floor that rungs R3 and R4 always require one. |
| `fidelity.reason` | string | Mandatory | Mandatory reason. AUTHOR-OWNED prose saying why fidelity is or is not required here. |
| `fidelity.verdict` | enum: `match` \| `minor-drift` \| `major-drift` \| `slop` \| `cannot-verify` \| null | Mandatory | Mandatory checker verdict. GATE-OWNED: no fidelity gate has run when a proposal is authored, so an authored proposal always writes null. |
| `fidelity.fidelity_identity` | string \| null | Mandatory | Mandatory fidelity identity, bound when a fidelity gate runs. GATE-OWNED: an authored proposal always writes null. This is NOT the proposal's own top-level fidelity_identity, which the author computes with the identity calculator. |
| `fidelity.gate_id` | string \| null | Mandatory | Mandatory immutable gate identifier. GATE-OWNED: an authored proposal always writes null. |
| `fidelity.current` | boolean | Mandatory | Mandatory current: whether a fidelity verdict is current for these exact facts. GATE-OWNED: an authored proposal always writes false. |
| `fidelity.permanent_scar` | boolean | Mandatory | Mandatory permanent scar: an indelible mark that this record was once judged slop. GATE-OWNED and set only by a fidelity gate -- true for a 'slop' verdict or a checker-preserved prior scar, never an author's self-assessment. It is REQUIRED in every fidelity block, including an authored proposal's, and an authored proposal always writes false, because no gate has run yet. Omitting it because it has no value yet is a refusal, and the authoring campaign runs once. |
| `fidelity.deviations` | array<string> | Mandatory | Mandatory deviations a fidelity gate recorded. GATE-OWNED: an authored proposal always writes []. |
| `accuracy_gate` | object | Mandatory | Mandatory metadata-accuracy gate state. |
| `accuracy_gate.required` | const `True` | Mandatory | Mandatory required. |
| `accuracy_gate.vet_identity` | string \| null | Mandatory | Mandatory vet identity. |
| `accuracy_gate.gate_id` | string \| null | Mandatory | Mandatory immutable gate identifier. |
| `accuracy_gate.verdict` | enum: `accurate` \| `inaccurate` \| `cannot-verify` \| null | Mandatory | Mandatory checker verdict. |
| `accuracy_gate.current` | boolean | Mandatory | Mandatory current. |
| `accuracy_gate.checker_model` | string | Mandatory | Mandatory checker model. |
| `accuracy_gate.checker_version` | string | Mandatory | Mandatory checker version. |
| `accuracy_gate.prompt_sha256` | string | Mandatory | Mandatory prompt sha256. |
| `execution` | object | Mandatory | Mandatory execution identity and currentness state. |
| `execution.execution_identity` | string | Mandatory | Mandatory execution identity. |
| `execution.environment_id` | string | Mandatory | Mandatory environment id. |
| `execution.env_generation` | string | Mandatory | Mandatory env generation. |
| `execution.accepted_attempt_ids` | array<string> | Mandatory | Mandatory accepted attempt ids. |
| `execution.confirmation_policy` | enum: `two-cold-r3-r4` \| `single-mechanical` \| `mechanical-canary` | Mandatory | Mandatory confirmation policy. |
| `execution.network_attempted` | const `False` | Mandatory | Mandatory network attempted. |
| `execution.checkpoint_accessed` | const `False` | Mandatory | Mandatory checkpoint accessed. |
| `execution.last_verified_at` | string | Mandatory | Mandatory last verified at. |
| `execution.current` | boolean | Mandatory | Mandatory current. |
| `status` | object | Mandatory | Mandatory closed current disposition. |
| `status.kind` | enum: `runs` \| `deferred` \| `skipped` \| `failed` | Mandatory | Mandatory record or status kind. |
| `status.code` | enum `model-common.status_code` | Mandatory | Mandatory closed status code. |
| `status.stage` | enum `model-common.failure_stage` \| null | Mandatory | Mandatory processing or failure stage. |
| `status.reason_code` | enum `attempt-common.reason_code` \| null | Mandatory | Mandatory closed reason code when applicable. |
| `status.detail` | string \| null | Mandatory | Best-effort human-readable detail. |
| `status.traceback` | string \| null \| object | Mandatory | Best-effort captured traceback. |
| `status.traceback.redaction` | const `externally-controlled-text-v1` | Mandatory | Closed marker for externally controlled text removed from public records. |
| `status.traceback.content_sha256` | string | Mandatory | Digest of the exact canonical JSON diagnostic value. |
| `status.traceback.local_path` | string | Mandatory | Gitignored local diagnostic sidecar locator. |
| `status.traceback.diagnostic_key` | string | Mandatory | JSON-style key locating the exact value inside the sidecar. |
| `status.traceback.stream_sha256` | string | Optional | Exact parent-observed stream digest when the value is a stdio tail. |
| `status.no_traceback_reason` | string \| null | Mandatory | Mandatory no traceback reason. |
| `status.attempted_rungs` | array<enum `model-common.rung_or_no_selection`> | Mandatory | Mandatory attempted rungs. |
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
<!-- END GENERATED SCHEMA TABLE: model.v2/modes-and-verification-state -->

## `attempt.v2` receipt fields

<!-- BEGIN GENERATED SCHEMA TABLE: attempt.v2/receipt -->
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
| `stage` | enum `attempt-common.stage` | Mandatory | Mandatory processing or failure stage. |
| `mode` | enum: `train` \| `eval` \| null | Mandatory | Mandatory mode. |
| `started_at` | string | Mandatory | Mandatory UTC start timestamp. |
| `finished_at` | string | Mandatory | Mandatory UTC completion timestamp. |
| `result` | enum: `succeeded` \| `failed` \| `observed` | Mandatory | Mandatory immutable attempt outcome. |
| `attempted_rungs` | array<enum `attempt-common.rung_or_no_selection`> | Mandatory | Mandatory attempted rungs. |
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
| `environment` | object \| null | Mandatory | Mandatory environment. |
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
| `invocation.mode` | enum: `train` \| `eval` \| null | Mandatory | Mandatory mode. |
| `invocation.network_policy` | string | Mandatory | Mandatory network policy. |
| `invocation.timeout_seconds` | integer | Mandatory | Mandatory timeout seconds. |
| `invocation.rss_limit_bytes` | integer | Mandatory | Mandatory rss limit bytes. |
| `invocation.scratch_limit_bytes` | integer | Mandatory | Mandatory scratch limit bytes. |
| `worker_receipt` | object | Mandatory | Mandatory worker receipt. |
| `worker_receipt.present` | boolean | Mandatory | Mandatory present. |
| `worker_receipt.receipt_sha256` | string \| null | Mandatory | Mandatory receipt sha256. |
| `worker_receipt.observed_recipe_revision` | string \| null | Mandatory | Worker-observed recipe revision, required for parent comparison. |
| `worker_receipt.observed_adapter_sha256` | string \| null | Mandatory | Worker-observed typed-adapter digest, or null for declarative recipes. |
| `worker_receipt.observed_code_manifest_sha256` | string \| null | Mandatory | Worker-observed digest over the recursive code_manifest bytes, or null when the recipe has no code manifest. |
| `worker_receipt.observed_input_asset_sha256` | string \| null | Mandatory | Worker-observed digest of the materialized standard input asset, or null when no standard asset was selected. |
| `worker_receipt.constructor_seconds` | number \| null | Optional | Worker-observed model-construction wall seconds, or null when the constructor did not complete. |
| `worker_receipt.forward_seconds` | number \| null | Optional | Worker-observed forward wall seconds for this mode, or null when the forward did not run. |
| `worker_receipt.constructor_started` | boolean | Mandatory | Mandatory constructor started. |
| `worker_receipt.constructor_completed` | boolean | Mandatory | Mandatory constructor completed. |
| `worker_receipt.input_completed` | boolean | Mandatory | Mandatory input completed. |
| `worker_receipt.forward_started` | boolean | Mandatory | Mandatory forward started. |
| `worker_receipt.forward_completed` | boolean | Mandatory | Mandatory forward completed. |
| `worker_receipt.mode` | enum: `train` \| `eval` \| null | Mandatory | Mandatory mode. |
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
| `supervisor_observation.stdout_sha256` | string \| null | Mandatory | Mandatory stdout sha256. |
| `supervisor_observation.stdout_bytes` | integer | Mandatory | Mandatory stdout bytes. |
| `supervisor_observation.stdout_tail` | string \| object | Mandatory | Mandatory stdout tail. |
| `supervisor_observation.stdout_tail.redaction` | const `externally-controlled-text-v1` | Mandatory | Closed marker for externally controlled text removed from public records. |
| `supervisor_observation.stdout_tail.content_sha256` | string | Mandatory | Digest of the exact canonical JSON diagnostic value. |
| `supervisor_observation.stdout_tail.local_path` | string | Mandatory | Gitignored local diagnostic sidecar locator. |
| `supervisor_observation.stdout_tail.diagnostic_key` | string | Mandatory | JSON-style key locating the exact value inside the sidecar. |
| `supervisor_observation.stdout_tail.stream_sha256` | string | Optional | Exact parent-observed stream digest when the value is a stdio tail. |
| `supervisor_observation.stdout_completion_line` | string \| null | Optional | Parent-attested TorchLens-owned completion marker. |
| `supervisor_observation.stderr_sha256` | string \| null | Mandatory | Mandatory stderr sha256. |
| `supervisor_observation.stderr_bytes` | integer | Mandatory | Mandatory stderr bytes. |
| `supervisor_observation.stderr_tail` | string \| object | Mandatory | Mandatory stderr tail. |
| `supervisor_observation.stderr_tail.redaction` | const `externally-controlled-text-v1` | Mandatory | Closed marker for externally controlled text removed from public records. |
| `supervisor_observation.stderr_tail.content_sha256` | string | Mandatory | Digest of the exact canonical JSON diagnostic value. |
| `supervisor_observation.stderr_tail.local_path` | string | Mandatory | Gitignored local diagnostic sidecar locator. |
| `supervisor_observation.stderr_tail.diagnostic_key` | string | Mandatory | JSON-style key locating the exact value inside the sidecar. |
| `supervisor_observation.stderr_tail.stream_sha256` | string | Optional | Exact parent-observed stream digest when the value is a stdio tail. |
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
| `error` | value \| null | Mandatory | Mandatory error. |
| `error.stage` | enum `attempt-common.stage` | Branch-dependent | Mandatory processing or failure stage. |
| `error.reason_code` | enum `attempt-common.reason_code` | Branch-dependent | Mandatory closed reason code when applicable. |
| `error.exception_type` | string \| null | Branch-dependent | Mandatory exception type. |
| `error.message` | string \| object | Branch-dependent | Mandatory message. |
| `error.message.redaction` | const `externally-controlled-text-v1` | Mandatory | Closed marker for externally controlled text removed from public records. |
| `error.message.content_sha256` | string | Mandatory | Digest of the exact canonical JSON diagnostic value. |
| `error.message.local_path` | string | Mandatory | Gitignored local diagnostic sidecar locator. |
| `error.message.diagnostic_key` | string | Mandatory | JSON-style key locating the exact value inside the sidecar. |
| `error.message.stream_sha256` | string | Optional | Exact parent-observed stream digest when the value is a stdio tail. |
| `error.traceback` | string \| null \| object \| string \| object \| null | Branch-dependent | Best-effort captured traceback. |
| `error.traceback.redaction` | const `externally-controlled-text-v1` | Mandatory | Closed marker for externally controlled text removed from public records. |
| `error.traceback.content_sha256` | string | Mandatory | Digest of the exact canonical JSON diagnostic value. |
| `error.traceback.local_path` | string | Mandatory | Gitignored local diagnostic sidecar locator. |
| `error.traceback.diagnostic_key` | string | Mandatory | JSON-style key locating the exact value inside the sidecar. |
| `error.traceback.stream_sha256` | string | Optional | Exact parent-observed stream digest when the value is a stdio tail. |
| `error.no_traceback_reason` | string \| null \| null \| string | Branch-dependent | Mandatory no traceback reason. |
| `error.native_crash` | boolean | Branch-dependent | Mandatory native crash. |
| `error.root_cause_fingerprint` | string | Branch-dependent | Mandatory root cause fingerprint. |
| `error.details` | object map | Branch-dependent | Mandatory structured event details. |
| `defer_evidence` | object \| null | Mandatory | Mandatory defer evidence. |
| `defer_evidence.target_status` | enum: `deferred:needs-cuda` \| `deferred:needs-x86` | Mandatory | Mandatory target status. |
| `defer_evidence.source_ids` | array<string> | Mandatory | Mandatory source ids. |
| `defer_evidence.probe_attempt_ids` | array<string> | Mandatory | Mandatory probe attempt ids. |
| `defer_evidence.explanation` | string | Mandatory | Mandatory explanation. |
<!-- END GENERATED SCHEMA TABLE: attempt.v2/receipt -->

## `gate.v2` verdict fields

<!-- BEGIN GENERATED SCHEMA TABLE: gate.v2/verdict -->
| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `schema_version` | const `menagerie.crawler.gate.v2` | Mandatory | Mandatory versioned schema identifier. |
| `gate_id` | string | Mandatory | Mandatory immutable gate identifier. |
| `ledger_seq` | integer | Mandatory | Mandatory monotonic ledger sequence. |
| `payload_sha256` | string | Mandatory | Mandatory hash of the complete ledger payload. |
| `gate_kind` | enum: `metadata_batch` \| `fidelity` | Mandatory | Mandatory checker gate kind. |
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
| `items[].campaign_root_work_id` | string | Mandatory | Stable campaign/root-work lineage across repaired proposals. |
| `items[].stable_id` | string | Mandatory | Mandatory stable model identifier. |
| `items[].family_representative_id` | string | Mandatory | Mandatory family representative id. |
| `items[].fidelity_identity` | string \| null | Mandatory | Mandatory fidelity identity. |
| `items[].vet_identity` | string | Mandatory | Mandatory vet identity. |
| `items[].verified_hashes` | object | Mandatory | Mandatory verified hashes. |
| `items[].verified_hashes.proposal` | string | Mandatory | Mandatory proposal. |
| `items[].verified_hashes.source_manifest` | string | Mandatory | Mandatory source manifest. |
| `items[].verified_hashes.evidence` | string | Mandatory | Mandatory literal evidence and coverage record. |
| `items[].verified_hashes.code` | string \| null | Mandatory | Mandatory closed status code. |
| `items[].verified_hashes.code_manifest` | string | Optional | Closed recursive model-code path-and-byte manifest digest. |
| `items[].verified_hashes.source_to_code_map` | string | Mandatory | Mandatory source to code map. |
| `items[].verified_hashes.family_template` | string \| null | Mandatory | Mandatory family template. |
| `items[].integrity` | object | Mandatory | Mandatory integrity. |
| `items[].integrity.verdict` | enum `gate-common.accuracy_verdict` | Mandatory | Mandatory checker verdict. |
| `items[].integrity.hash_mismatches` | array<string> | Mandatory | Mandatory hash mismatches. |
| `items[].integrity.excerpt_discrepancies` | array<string> | Mandatory | Mandatory excerpt discrepancies. |
| `items[].integrity.locator_failures` | array<string> | Mandatory | Mandatory locator failures. |
| `items[].verdict` | enum `gate-common.accuracy_verdict` | Mandatory | Mandatory checker verdict. |
| `items[].field_checks` | array<object> | Mandatory | Mandatory per-field accuracy findings. |
| `items[].field_checks[].field` | string | Mandatory | Mandatory field: ONE gated claim path copied VERBATIM from the envelope item's machine-derived 'required_field_checks' list (e.g. 'external_metadata.citation', 'taxonomy.novel_ops', 'input_contract'). An optional 'proposed_facts.' prefix is accepted and stripped; nothing else is. Anything not in that list is refused as an extraneous check: a section name ('identity', 'licenses'), a grouped spelling ('citation; dates' -- one verdict cannot carry two claims' provenance), and a per-leaf expansion of a claim ('external_metadata.citation.year') alike. Coverage is exhaustive and one-to-one: every listed claim gets exactly one check, a repeated claim is refused as a duplicate, and any claim left unchecked is refused as an ungated authored fact. |
| `items[].field_checks[].verdict` | enum `gate-common.accuracy_verdict` | Mandatory | Mandatory checker verdict. |
| `items[].field_checks[].evidence_ids` | array<string> | Mandatory | Mandatory supporting evidence identifiers; may be empty only for external_metadata.keywords relevance judgments, which still require checked_source_ids and a reason. |
| `items[].field_checks[].checked_source_ids` | array<string> | Mandatory | Mandatory checked source ids. |
| `items[].field_checks[].reason` | string | Mandatory | Mandatory reason. |
| `items[].field_checks[].required_repair` | string \| null | Mandatory | Mandatory required repair. |
| `items[].fidelity` | object | Mandatory | Mandatory implementation-fidelity gate state. |
| `items[].fidelity.required` | boolean | Mandatory | Mandatory required. |
| `items[].fidelity.verdict` | enum: `match` \| `minor-drift` \| `major-drift` \| `slop` \| `cannot-verify` \| `not-applicable` | Mandatory | Mandatory checker verdict. |
| `items[].fidelity.material_checks` | array<object> | Mandatory | Mandatory material fidelity findings. |
| `items[].fidelity.material_checks[].category` | string | Mandatory | Mandatory category. |
| `items[].fidelity.material_checks[].verdict` | enum: `match` \| `minor-drift` \| `major-drift` \| `slop` \| `cannot-verify` | Mandatory | Mandatory checker verdict. |
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
| `items[].rung_check.selected_rung` | enum `gate-common.rung` | Mandatory | Mandatory selected rung. |
| `items[].rung_check.highest_applicable` | enum `gate-common.rung` | Mandatory | Mandatory highest applicable. |
| `items[].rung_check.verdict` | enum `gate-common.accuracy_verdict` | Mandatory | Mandatory checker verdict. |
| `items[].rung_check.findings` | array<string> | Mandatory | Mandatory findings. |
| `items[].unsupported_claims` | array<string> | Mandatory | Mandatory unsupported claims. |
| `items[].required_repairs` | array<string> | Mandatory | Mandatory repairs required before acceptance. |
| `items[].confidence` | enum: `high` \| `medium` \| `low` | Mandatory | Mandatory checker confidence level. |
| `result_envelope_sha256` | string | Mandatory | Mandatory result envelope sha256. |
<!-- END GENERATED SCHEMA TABLE: gate.v2/verdict -->

## Closed vocabularies

Every named closed vocabulary referenced as ``enum `name` `` in a `Type` cell above, with
its exact members. Anonymous enums declared directly on a property are listed inline in
their own row instead.

<!-- BEGIN GENERATED SCHEMA TABLE: vocabularies/closed -->
- `attempt-common.reason_code` (80 members):
  `schema-invalid`, `stable-id-conflict`, `duplicate-revision-conflict`, `migration-invariant`,
  `identity-unresolved`, `missing-mandatory-link`, `source-model-mismatch`,
  `source-target-invalid`, `higher-rung-unresolved`, `unreachable`, `revision-missing`,
  `hash-mismatch`, `access-denied`, `artifact-missing`, `locator-missing`, `excerpt-mismatch`,
  `insufficient-detail`, `coverage-incomplete`, `search-incomplete`, `inaccurate-cap-exhausted`,
  `cannot-verify-cap-exhausted`, `identity-mismatch`, `checker-contract-invalid`,
  `solve-failed`, `lock-missing`, `artifact-hash-mismatch`, `build-failed`, `probe-failed`,
  `resolved-export-mismatch`, `island-cap`, `below-minimum-island-size`, `module-missing`,
  `symbol-missing`, `abi-load-failed`, `import-exception`, `exception`, `requires-checkpoint`,
  `requires-weight-asset`, `invalid-model-object`, `contract-invalid`, `source-invalid-shape`,
  `generation-exception`, `semantic-constraint`, `mode-run`, `incomplete-receipt`,
  `invalid-output-signature`, `confirmation-mismatch`, `major-drift-cap-exhausted`,
  `slop-cap-exhausted`, `timeout`, `oom`, `disk-floor`, `scratch-cap`, `rss-cap`,
  `network-attempt`, `checkpoint-read`, `write-outside-scratch`, `credentials-exposed`,
  `torchlens-import`, `opaque-code`, `native-crash`, `signal`, `missing-receipt`,
  `protocol-violation`, `ledger-corruption`, `internal-error`, `sandbox-unavailable-v1`,
  `effort-cap-exhausted`, `effort-exhausted:tool-calls`, `effort-exhausted:fetch-targets`,
  `effort-exhausted:wall-seconds`, `wall-exceeded`, `session-crashed`,
  `research-tools-unavailable`, `repair-exhausted`, `missing-material-source`,
  `needs-higher-tier`, `malformed-result`, `terminal-disposition-rejected`,
  `terminal-disposition-unverifiable`
- `attempt-common.rung_or_no_selection` (6 members):
  `R1_LIBRARY`, `R2_VENDOR`, `R3_PORT`, `R4_REIMPLEMENT`, `R5_SKIP`, `NO_RUNG_SELECTED`
- `attempt-common.stage` (15 members):
  `intake`, `source`, `fetch`, `author`, `evidence`, `accuracy-gate`, `environment`, `import`,
  `constructor`, `input`, `forward`, `fidelity`, `resource`, `policy`, `runner`
- `gate-common.accuracy_verdict` (3 members):
  `accurate`, `inaccurate`, `cannot-verify`
- `gate-common.rung` (5 members):
  `R1_LIBRARY`, `R2_VENDOR`, `R3_PORT`, `R4_REIMPLEMENT`, `R5_SKIP`
- `model-common.failure_stage` (15 members):
  `intake`, `source`, `fetch`, `author`, `evidence`, `accuracy-gate`, `environment`, `import`,
  `constructor`, `input`, `forward`, `fidelity`, `resource`, `policy`, `runner`
- `model-common.rung_or_no_selection` (6 members):
  `R1_LIBRARY`, `R2_VENDOR`, `R3_PORT`, `R4_REIMPLEMENT`, `R5_SKIP`, `NO_RUNG_SELECTED`
- `model-common.status_code` (23 members):
  `runs`, `deferred:needs-cuda`, `deferred:needs-x86`, `deferred:needs-opus-tier`,
  `deferred:needs-source-access`, `skipped:insufficient-description`, `skipped:no-description`,
  `skipped:not-a-real-NN`, `failed:intake`, `failed:source`, `failed:fetch`, `failed:author`,
  `failed:evidence`, `failed:accuracy-gate`, `failed:environment`, `failed:import`,
  `failed:constructor`, `failed:input`, `failed:forward`, `failed:fidelity`, `failed:resource`,
  `failed:policy`, `failed:runner`
<!-- END GENERATED SCHEMA TABLE: vocabularies/closed -->

## Schemas not tabulated here

The `author-proposal.v2` and `operational-event.v1` schemas are self-documenting: every
property carries its own executable JSON Schema description. Read them directly in
`menagerie/crawler/schemas/`.

# Multi-Industry State, Contract, and Method Reference

**Generated from:** `Autonomous_financial_analyst_Learners_Notebook copy.ipynb`  
**Scope:** F00–F16 notebook-local multi-industry implementation

This is the concise developer/agent contract index. Runtime types remain authoritative; 
the descriptions explain ownership and intended use. Private evidence values and secrets 
are deliberately excluded.

## 1. State and structured-contract fields

### `multiindustry_state_contracts`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `QueryPlan` | `query_type` | `QueryType` | Requested operation: fact, analyze, compare, or rank. |
| `QueryPlan` | `company_mentions` | `list[str]` | User-supplied company names before canonical resolution. |
| `QueryPlan` | `requested_dimensions` | `list[str]` | User-requested financial or sector research topics. |
| `QueryPlan` | `risk_profile` | `RiskProfile` | Weight policy used only by deterministic eligible scoring. |
| `QueryPlan` | `scoring_requested` | `bool` | Whether the user explicitly requested numeric ranking/scoring. |
| `QueryPlan` | `freshness_required` | `bool` | Whether current provider/news evidence is required. |
| `QueryPlan` | `time_horizon` | `str \| None` | Optional investment or analysis horizon stated by the user. |
| `ResolvedCompany` | `company_id` | `str` | Stable internal identifier independent of ticker changes. |
| `ResolvedCompany` | `ticker` | `str` | Canonical public-market symbol used by source tools. |
| `ResolvedCompany` | `company_name` | `str` | Display name from the supported-company registry. |
| `ResolvedCompany` | `aliases` | `list[str]` | Accepted normalized user mentions. |
| `ResolvedCompany` | `exchange` | `str \| None` | Listing exchange when known. |
| `ResolvedCompany` | `industry` | `str` | Broad registry industry. |
| `ResolvedCompany` | `sub_industry` | `str` | Narrow registry classification. |
| `ResolvedCompany` | `profile_id` | `str` | Versioned research profile selected by registry data. |
| `ResolvedCompany` | `resolution_status` | `ResolutionStatus` | Resolved, ambiguous, or unsupported outcome. |
| `ResolvedCompany` | `resolution_message` | `str \| None` | Optional explanation for non-resolved outcomes. |
| `EvidenceRecord` | `evidence_id` | `str` | Stable citation token unique within the run. |
| `EvidenceRecord` | `run_id` | `str` | Research run that owns this record. |
| `EvidenceRecord` | `company_id` | `str` | Canonical company owner. |
| `EvidenceRecord` | `ticker` | `str` | Canonical ticker owner. |
| `EvidenceRecord` | `profile_id` | `str` | Industry profile allowed to consume the record. |
| `EvidenceRecord` | `evidence_type` | `str` | Normalized category such as financial_metrics or technology_rag. |
| `EvidenceRecord` | `value` | `Any` | Raw normalized tool payload available to downstream reasoning. |
| `EvidenceRecord` | `source_name` | `str` | Exact tool/adapter that produced the record. |
| `EvidenceRecord` | `source_uri` | `str \| None` | Optional public source URL. |
| `EvidenceRecord` | `document_name` | `str \| None` | Optional local/public document label. |
| `EvidenceRecord` | `page` | `int \| None` | Optional one-based source page. |
| `EvidenceRecord` | `as_of` | `str \| None` | Optional effective date of the underlying data. |
| `EvidenceRecord` | `retrieved_at` | `str` | UTC retrieval timestamp. |
| `EvidenceRecord` | `freshness_status` | `FreshnessStatus` | Fresh, stale, or unknown evidence assessment. |
| `EvidenceRecord` | `cache_status` | `Literal['hit', 'miss', 'stale', 'unknown']` | Hit, miss, stale, or unknown cache outcome. |
| `EvidenceRecord` | `status` | `EvidenceStatus` | Success, missing, or failed source result. |
| `EvidenceRecord` | `source_metadata` | `dict[str, Any]` | Additional provenance excluding the primary value/error. |
| `EvidenceRecord` | `error` | `str \| None` | Explicit source failure, otherwise ``None``. |
| `CompanyTask` | `run_id` | `str` | Owning research run. |
| `CompanyTask` | `company` | `ResolvedCompany` | Canonical company the branch may research. |
| `CompanyTask` | `query_plan` | `QueryPlan` | Validated coordinator plan copied into the branch. |
| `CompanyTask` | `shared_dimensions` | `list[str]` | Financial dimensions this worker must cover. |
| `CompanyTask` | `industry_dimensions` | `list[str]` | Profile-specific dimensions this worker must cover. |
| `CompanyTask` | `unsupported_dimensions` | `list[str]` | Requested topics outside current profile support. |
| `CompanyTask` | `allowed_tools` | `list[str]` | Exact profile allowlist; the worker cannot expand it. |
| `CompanyResearchResult` | `run_id` | `str` | Owning research run. |
| `CompanyResearchResult` | `company` | `ResolvedCompany` | Canonical company identity. |
| `CompanyResearchResult` | `profile_id` | `str` | Profile used by the worker. |
| `CompanyResearchResult` | `financial_evidence` | `dict[str, Any]` | Optional derived financial view for compatibility. |
| `CompanyResearchResult` | `industry_signals` | `dict[str, Any]` | Evidence-linked profile signal dimensions. |
| `CompanyResearchResult` | `evidence` | `list[EvidenceRecord]` | Canonical records collected by allowed tools. |
| `CompanyResearchResult` | `missing_dimensions` | `list[str]` | Required topics without successful coverage. |
| `CompanyResearchResult` | `errors` | `list[str]` | Contained worker, tool, or normalization errors. |
| `CompanyResearchResult` | `status` | `CompanyResultStatus` | Success, partial, or failed terminal state. |
| `ScoringEligibility` | `eligible` | `bool` | Whether F13 may run for the normalized comparison. |
| `ScoringEligibility` | `rubric_id` | `str \| None` | Exact versioned rubric when eligible. |
| `ScoringEligibility` | `reason` | `str` | Human-readable authorization or rejection reason. |
| `ScoringEligibility` | `excluded_companies` | `list[str]` | Tickers that prevent scoring. |
| `ScoringEligibility` | `missing_requirements` | `dict[str, list[str]]` | Missing/failed requirements grouped by ticker. |
| `CompanyWorkerState` | `task` | `CompanyTask` | Single validated company assignment. |
| `CompanyWorkerState` | `messages` | `Annotated[Sequence[BaseMessage], add_messages]` | Reducer-managed worker conversation and tool messages. |
| `CompanyWorkerState` | `evidence` | `list[EvidenceRecord]` | Current-company canonical evidence accumulated from tools. |
| `CompanyWorkerState` | `industry_signals` | `dict[str, Any]` | Evidence-linked profile interpretation. |
| `CompanyWorkerState` | `missing_dimensions` | `list[str]` | Required dimensions still unsupported. |
| `CompanyWorkerState` | `evidence_gate_status` | `Literal['retry', 'complete', 'partial']` | Retry, complete, or bounded-partial decision. |
| `CompanyWorkerState` | `tool_round_count` | `int` | Source-tool loop count used by the hard ceiling. |
| `CompanyWorkerState` | `validation_retry_count` | `int` | Evidence-gate retry count. |
| `CompanyWorkerState` | `result` | `CompanyResearchResult \| None` | Terminal company result once assembled. |
| `CompanyWorkerState` | `errors` | `list[str]` | Contained branch errors visible at fan-in. |
| `OrchestratorState` | `messages` | `Annotated[Sequence[BaseMessage], add_messages]` | Conversation messages retained by the optional checkpointer. |
| `OrchestratorState` | `remembered_company_ids` | `list[str]` | Canonical companies available to follow-up references. |
| `OrchestratorState` | `last_profile_ids` | `list[str]` | Profiles used by the previous completed request. |
| `OrchestratorState` | `run_id` | `str` | Fresh identifier assigned to the current request. |
| `OrchestratorState` | `run_started_at` | `str` | UTC start timestamp. |
| `OrchestratorState` | `original_query` | `str` | Latest human question passed to planning. |
| `OrchestratorState` | `plan` | `QueryPlan \| None` | Validated free-text query plan. |
| `OrchestratorState` | `resolution_result` | `dict[str, Any] \| None` | Resolver output including unsupported/ambiguous mentions. |
| `OrchestratorState` | `resolution_gate_status` | `dict[str, Any] \| None` | Mandatory identity-gate verdict. |
| `OrchestratorState` | `resolved_companies` | `list[ResolvedCompany]` | Canonical companies authorized for this run. |
| `OrchestratorState` | `profile_selection` | `dict[str, Any] \| None` | Registry-backed profile mapping. |
| `OrchestratorState` | `profile_gate_status` | `dict[str, Any] \| None` | Mandatory profile-coverage verdict. |
| `OrchestratorState` | `company_tasks` | `list[CompanyTask]` | One guarded task per resolved company. |
| `OrchestratorState` | `task_gate_status` | `dict[str, Any] \| None` | Isolation, permission, and budget verdict. |
| `OrchestratorState` | `company_results` | `Annotated[CompanyResultMap, merge_company_results]` | Reducer-managed raw branch results keyed by ticker. |
| `OrchestratorState` | `normalized_company_results` | `CompanyResultMap` | F12 canonical fan-in results. |
| `OrchestratorState` | `fan_in_normalization` | `dict[str, Any] \| None` | F12 ordering/status/error summary. |
| `OrchestratorState` | `comparison_mode` | `ComparisonMode \| None` | Single, same-profile, or cross-profile route. |
| `OrchestratorState` | `comparison_route_status` | `dict[str, Any] \| None` | Mandatory comparison-route verdict. |
| `OrchestratorState` | `scoring_eligibility` | `ScoringEligibility \| None` | F12 permission for deterministic F13 scoring. |
| `OrchestratorState` | `scores` | `dict[str, Any] \| None` | Optional authoritative F13 table. |
| `OrchestratorState` | `final_answer` | `str \| None` | Validated answer or bounded-stop explanation. |
| `OrchestratorState` | `validation_retry_count` | `int` | Bounded final-validation retry count. |
| `OrchestratorState` | `validation_errors` | `list[str]` | Deterministic contract violations. |
| `OrchestratorState` | `run_errors` | `list[str]` | Non-blocking and blocking errors accumulated for the run. |

### `multiindustry_company_registry`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `CompanyRegistryEntry` | `company_id` | `str` | Stable internal identity. |
| `CompanyRegistryEntry` | `ticker` | `str` | Canonical listed symbol. |
| `CompanyRegistryEntry` | `company_name` | `str` | Developer/user display name. |
| `CompanyRegistryEntry` | `aliases` | `list[str]` | Accepted free-text aliases. |
| `CompanyRegistryEntry` | `exchange` | `str \| None` | Listing exchange when known. |
| `CompanyRegistryEntry` | `industry` | `str` | Broad deterministic industry. |
| `CompanyRegistryEntry` | `sub_industry` | `str` | Narrow deterministic classification. |
| `CompanyRegistryEntry` | `profile_id` | `str` | Versioned research profile assigned to the company. |

### `multiindustry_industry_profiles`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `IndustryProfile` | `profile_id` | `str` | Stable versioned identifier used by routing and evidence records. |
| `IndustryProfile` | `industry` | `str` | Broad supported industry name. |
| `IndustryProfile` | `sub_industry` | `str` | Narrow specialization supported by this playbook. |
| `IndustryProfile` | `worker_prompt` | `str` | Profile-specific constraints appended to the generic worker charter. |
| `IndustryProfile` | `allowed_tools` | `list[str]` | Tool-contract names the future worker may bind. |
| `IndustryProfile` | `shared_dimensions` | `list[str]` | Cross-profile financial dimensions. |
| `IndustryProfile` | `industry_dimensions` | `list[str]` | Profile-specific research dimensions. |
| `IndustryProfile` | `rag_tool_name` | `str` | Profile-owned retrieval contract. |
| `IndustryProfile` | `rag_collection` | `str` | Local vector collection name. |
| `IndustryProfile` | `corpus_version` | `str` | Version identifier for cache and provenance decisions. |
| `IndustryProfile` | `signal_extractor_name` | `str` | Deterministic extractor contract for normalized evidence. |
| `IndustryProfile` | `rubric_id` | `str \| None` | Validated scoring rubric, or ``None`` when scoring is disabled. |
| `IndustryProfile` | `scoring_function_name` | `str \| None` | Deterministic scoring function, or ``None`` when unavailable. |
| `IndustryProfile` | `scoring_enabled` | `bool` | Whether this profile currently permits numeric scoring. |
| `IndustryProfile` | `synthesis_prompt` | `str` | Profile-specific guidance for grounded narrative synthesis. |
| `ProfileSelection` | `status` | `Literal['ready', 'unsupported']` | ``ready`` or ``unsupported``. |
| `ProfileSelection` | `profiles_by_company` | `dict[str, IndustryProfile]` | Defensive profile copies keyed by canonical company ID. |
| `ProfileSelection` | `unknown_company_ids` | `list[str]` | Company IDs absent from the authoritative registry. |
| `ProfileSelection` | `message` | `str` | Human-readable selection outcome. |

### `multiindustry_company_tasks`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `TaskPlanningContext` | `run_id` | `str` | Current research-run identifier. |
| `TaskPlanningContext` | `plan` | `QueryPlan` | Deterministically validated free-text query plan. |
| `TaskPlanningContext` | `companies` | `list[ResolvedCompany]` | Successfully resolved and profile-validated companies. |
| `TaskPlanningContext` | `profile_selection` | `ProfileSelection` | Guarded profile selection that passed its mandatory gate. |

### `multiindustry_fan_in_normalization`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `FanInNormalization` | `run_id` | `str` | Run shared by every accepted task/result/evidence record. |
| `FanInNormalization` | `status` | `FanInStatus` | Complete, partial, or failed aggregate result. |
| `FanInNormalization` | `ready` | `bool` | Whether comparison routing may proceed. |
| `FanInNormalization` | `ordered_tickers` | `list[str]` | Deterministic task-order ticker list. |
| `FanInNormalization` | `results_by_ticker` | `CompanyResultMap` | Canonical normalized result map. |
| `FanInNormalization` | `ordered_results` | `list[CompanyResearchResult]` | Same results in task order for presentation. |
| `FanInNormalization` | `successful_tickers` | `list[str]` | Complete branches. |
| `FanInNormalization` | `partial_tickers` | `list[str]` | Usable branches with declared gaps. |
| `FanInNormalization` | `failed_tickers` | `list[str]` | Unusable/failed branches. |
| `FanInNormalization` | `blocking_errors` | `list[str]` | Identity/coverage errors that stop routing. |
| `FanInNormalization` | `errors` | `list[str]` | All contained fan-in and branch errors. |

### `multiindustry_mode_specific_synthesis`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `SynthesisContext` | `run_id` | `str` | Current research run used to enforce evidence ownership. |
| `SynthesisContext` | `original_query` | `str` | User question the answer must address. |
| `SynthesisContext` | `comparison_mode` | `ComparisonMode` | F12-selected single/same-profile/cross-profile policy. |
| `SynthesisContext` | `normalized_results` | `CompanyResultMap` | Current-run F12 evidence and signals keyed by ticker. |
| `SynthesisContext` | `scoring_eligibility` | `ScoringEligibility` | Deterministic F12 scoring authorization. |
| `SynthesisContext` | `scores` | `dict[str, Any] \| None` | Optional immutable authoritative F13 score table. |
| `SynthesisResult` | `mode` | `ComparisonMode` | Comparison mode actually used by the prompt policy. |
| `SynthesisResult` | `answer` | `str` | Candidate grounded prose containing explicit evidence citations. |
| `SynthesisResult` | `evidence_ids` | `list[str]` | Declared current-run citations used by the answer. |
| `SynthesisResult` | `scores_used` | `dict[str, Any]` | Exact copied F13 scores for eligible same-profile synthesis. |
| `SynthesisResult` | `limitations` | `list[str]` | Required and model-added limitations disclosed to the user. |

### `multiindustry_f15_evidence_validation`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `ValidationResult` | `valid` | `bool` | Aggregate pass required before returning an answer. |
| `ValidationResult` | `validated_evidence_ids` | `list[str]` | Current-run IDs that passed all ownership checks. |
| `ValidationResult` | `inline_evidence_ids` | `list[str]` | IDs parsed from explicit ``[EV-*]`` answer citations. |
| `ValidationResult` | `declared_evidence_ids` | `list[str]` | IDs listed by the structured synthesis result. |
| `ValidationResult` | `evidence_valid` | `bool` | Existence, ownership, status, duplicate, and consistency verdict. |
| `ValidationResult` | `score_fidelity_valid` | `bool` | Exact authoritative F13 score-use verdict. |
| `ValidationResult` | `mode_restrictions_valid` | `bool` | Single/same/cross-profile boundary verdict. |
| `ValidationResult` | `limitations_valid` | `bool` | Required limitation disclosure verdict. |
| `ValidationResult` | `errors` | `list[str]` | Deterministic failure explanations used by bounded correction. |

### `multiindustry_f15_local_traces`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `ResearchTrace` | `schema_version` | `str` | Trace contract version. |
| `ResearchTrace` | `run_id` | `str` | Unique filename-safe run identity. |
| `ResearchTrace` | `query` | `str` | Original user question after redaction. |
| `ResearchTrace` | `comparison_mode` | `str` | F12 routing mode. |
| `ResearchTrace` | `companies` | `list[dict[str, Any]]` | Canonical company summaries without private evidence bodies. |
| `ResearchTrace` | `profiles` | `list[str]` | Versioned profiles used by the run. |
| `ResearchTrace` | `evidence_provenance` | `list[dict[str, Any]]` | Redacted evidence identity/source metadata only. |
| `ResearchTrace` | `f13_scores` | `dict[str, Any]` | Optional authoritative deterministic scores. |
| `ResearchTrace` | `f14_synthesis` | `dict[str, Any]` | Latest redacted structured draft. |
| `ResearchTrace` | `validation_attempts` | `list[dict[str, Any]]` | Ordered F15 verdict summaries. |
| `ResearchTrace` | `started_at` | `str` | UTC trace creation timestamp. |
| `ResearchTrace` | `updated_at` | `str` | UTC last-write timestamp. |
| `ResearchTrace` | `completed_at` | `str \| None` | UTC terminal timestamp or ``None`` while in progress. |
| `ResearchTrace` | `final_status` | `str` | In-progress, success, failed, or interrupted. |
| `ResearchTrace` | `terminal_error` | `str \| None` | Redacted terminal failure message when present. |
| `TraceWriteResult` | `path` | `str` | Published trace path. |
| `TraceWriteResult` | `removed_paths` | `list[str]` | Older completed traces removed by bounded retention. |
| `TraceWriteResult` | `final_status` | `str` | Status contained in the published trace. |

### `multiindustry_f15_workflow`

| Contract | Field | Type | Meaning |
|---|---|---|---|
| `F15WorkflowResult` | `final_status` | `str` | Success, failed, or interrupted workflow outcome. |
| `F15WorkflowResult` | `final_answer` | `str` | Validated answer or visibly warned failed draft. |
| `F15WorkflowResult` | `synthesis` | `dict[str, Any]` | Last structured F14 candidate. |
| `F15WorkflowResult` | `validation` | `ValidationResult` | Deterministic evidence/score/mode/limitation verdict. |
| `F15WorkflowResult` | `attempts` | `int` | Total synthesis attempts including the first draft. |
| `F15WorkflowResult` | `correction_attempts` | `int` | Validation-driven correction count. |
| `F15WorkflowResult` | `warnings` | `list[str]` | Terminal warnings; empty on validated success. |
| `F15WorkflowResult` | `trace_path` | `str` | Local redacted JSON trace path. |

## 2. Classes and methods

Inputs show the callable signature; Output shows the declared return annotation. Nested 
LangGraph node functions are included because they are workflow contracts even though 
developers normally call the compiled graph rather than those nodes directly.

### `multiindustry_state_contracts`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| class `ResetCompanyResults` | — | — | Reducer update used by initialize_research_run to clear prior-run results. | Instantiate/use through the owning feature boundary. |
| function | `merge_company_results(current: CompanyResultMap \| None, update: CompanyResultMap \| ResetCompanyResults \| None)` | `CompanyResultMap` | Reset or merge parallel company results. | Reset or merge parallel company results. |
| function | `_latest_human_query(messages: Sequence[BaseMessage])` | `str` | Return the newest human question from reducer-managed conversation messages. | Internal helper; use through its owning public boundary. |
| function | `initialize_research_run(state: OrchestratorState)` | `dict[str, Any]` | Create a fresh request scope without overwriting conversation-lifetime fields. | Create a fresh request scope without overwriting conversation-lifetime fields. |

### `multiindustry_company_registry`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_company_entry(company_id: str, ticker: str, company_name: str, aliases: list[str], exchange: str \| None, industry: str, sub_industry: str, profile_id: str)` | `CompanyRegistryEntry` | Build one typed registry row from explicit authoritative values. | Internal helper; use through its owning public boundary. |
| function | `_normalize_company_text(value: str)` | `str` | Normalize a company mention for deterministic alias matching. | Internal helper; use through its owning public boundary. |
| function | `_build_alias_index()` | `dict[str, set[str]]` | Build normalized alias-to-ticker candidates from the registry. | Internal helper; use through its owning public boundary. |
| function | `_resolved_company(entry: CompanyRegistryEntry)` | `ResolvedCompany` | Project one registry row into a successful resolver result. | Internal helper; use through its owning public boundary. |
| function | `_unresolved_company(mention: str, status: Literal['ambiguous', 'unsupported'], message: str)` | `ResolvedCompany` | Create an identity-empty ambiguous or unsupported resolver result. | Internal helper; use through its owning public boundary. |
| function | `_ticker_tokens_in_mention(normalized_mention: str)` | `set[str]` | Find explicit supported ticker tokens inside a normalized mention. | Internal helper; use through its owning public boundary. |
| function | `_alias_candidates(normalized_mention: str)` | `set[str]` | Find registry tickers matching a normalized company alias. | Internal helper; use through its owning public boundary. |
| function | `resolve_company_mention(mention: str)` | `ResolvedCompany` | Resolve one planner-produced company mention without using an LLM or network call. | Resolve one planner-produced company mention without using an LLM or network call. |
| function | `resolve_company_mentions(company_mentions: list[str])` | `list[ResolvedCompany]` | Resolve mentions in order and collapse duplicate resolved company identities. | Resolve mentions in order and collapse duplicate resolved company identities. |
| function | `resolve_companies(plan: QueryPlan)` | `list[ResolvedCompany]` | Resolve the company mentions from a validated query plan. | Resolve the company mentions from a validated query plan. |
| function | `validate_resolution_gate(results: list[ResolvedCompany])` | `dict[str, Any]` | Return the non-bypassable routing decision for a resolution result. | Deterministic guard; call before the next workflow boundary. |
| function | `resolve_companies_tool(company_mentions: list[str])` | `dict[str, Any]` | Resolve user company mentions; this must succeed before research tools are used. | Guarded tool/node boundary; invoke with its declared schema. |
| function | `route_after_resolution(resolution: dict[str, Any])` | `str` | Mandatory graph router used after the coordinator's resolver tool call. | Mandatory graph router used after the coordinator's resolver tool call. |

### `multiindustry_query_planner`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| class `QueryPlanningError` | — | — | Raised when structured planner output violates the deterministic QueryPlan contract. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, errors: list[str])` | unspecified | Initialize a planning error from one or more contract violations. | Internal helper; use through its owning public boundary. |
| function | `_normalize_string_list(value: Any, field_name: str)` | `list[str]` | Validate, trim, deduplicate, and preserve order for a string list. | Internal helper; use through its owning public boundary. |
| function | `_normalize_dimension(value: str)` | `str` | Convert one requested research dimension to snake_case. | Internal helper; use through its owning public boundary. |
| function | `normalize_query_plan(raw_plan: Any)` | `QueryPlan` | Normalize structured output without silently accepting invalid types or enum values. | Normalize structured output without silently accepting invalid types or enum values. |
| function | `validate_query_plan(plan: QueryPlan)` | `list[str]` | Return every deterministic contract violation instead of accepting partial output. | Deterministic guard; call before the next workflow boundary. |
| function | `_query_uses_followup_reference(query: str)` | `bool` | Detect pronouns that may refer to previously remembered companies. | Internal helper; use through its owning public boundary. |
| function | `_conversation_excerpt(messages: Sequence[BaseMessage], limit: int=6)` | `str` | Render a bounded user/assistant conversation excerpt for planning context. | Internal helper; use through its owning public boundary. |
| function | `_default_query_planner_model()` | unspecified | Create the default deterministic-temperature chat model for query planning. | Internal helper; use through its owning public boundary. |
| function | `plan_query(query: str, conversation_context: Sequence[BaseMessage]=(), remembered_company_ids: Sequence[str]=(), model: Any \| None=None)` | `QueryPlan` | Use structured LLM output, then normalize and deterministically validate the result. | Use structured LLM output, then normalize and deterministically validate the result. |

### `multiindustry_f03_smoke`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| class `_F03FakeStructuredModel` | — | — | Local structured-output model double used by the F03 smoke test. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, response)` | unspecified | Store the predefined structured response returned by the fake model. | Internal helper; use through its owning public boundary. |
| function | `with_structured_output(self, schema, method='function_calling')` | unspecified | Record structured-output configuration and return this test double. | Record structured-output configuration and return this test double. |
| function | `invoke(self, messages)` | unspecified | Validate the planner call shape and return the predefined response. | Validate the planner call shape and return the predefined response. |

### `multiindustry_industry_profiles`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_build_company_profile_index()` | `dict[str, str]` | Build and validate the canonical company-to-profile mapping. | Internal helper; use through its owning public boundary. |
| function | `validate_industry_profile_registry()` | `list[str]` | Return all deterministic configuration errors in the profile registry. | Deterministic guard; call before the next workflow boundary. |
| function | `get_industry_profile(profile_id: str)` | `IndustryProfile` | Return a defensive copy of one supported industry profile. | Return a defensive copy of one supported industry profile. |
| function | `attach_industry_profiles(companies: list[ResolvedCompany])` | `list[ResolvedCompany]` | Validate resolved-company profile IDs against the authoritative registries. | Validate resolved-company profile IDs against the authoritative registries. |
| function | `select_industry_profiles(company_ids: list[str])` | `ProfileSelection` | Select authoritative profiles for canonical company IDs without using an LLM. | Select authoritative profiles for canonical company IDs without using an LLM. |
| function | `select_industry_profiles_tool(company_ids: list[str])` | `dict[str, Any]` | Select registry-backed profiles for resolved canonical company IDs. | Guarded tool/node boundary; invoke with its declared schema. |
| function | `validate_profile_gate(selection: ProfileSelection, expected_company_ids: list[str])` | `dict[str, Any]` | Validate profile selection against every company resolved for the current run. | Deterministic guard; call before the next workflow boundary. |
| function | `route_after_profile_selection(profile_gate: dict[str, Any])` | `str` | Route a validated profile selection to task construction or a bounded stop. | Route a validated profile selection to task construction or a bounded stop. |

### `multiindustry_company_tasks`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_ordered_unique(values: list[str])` | `list[str]` | Deduplicate strings while preserving their first-seen order. | Internal helper; use through its owning public boundary. |
| function | `_dimensions_for_profile(plan: QueryPlan, profile: IndustryProfile)` | `tuple[list[str], list[str], list[str]]` | Map requested dimensions into one profile without silent substitution. | Internal helper; use through its owning public boundary. |
| function | `register_task_planning_context(run_id: str, plan: QueryPlan, companies: list[ResolvedCompany], profile_selection: ProfileSelection)` | `TaskPlanningContext` | Register validated current-run inputs for guarded task construction. | Register validated current-run inputs for guarded task construction. |
| function | `clear_task_planning_context(run_id: str)` | `bool` | Remove one run-scoped task-planning context from local memory. | Remove one run-scoped task-planning context from local memory. |
| function | `build_company_tasks(plan: QueryPlan, companies: list[ResolvedCompany], run_id: str)` | `list[CompanyTask]` | Build one deterministic, profile-bounded research task per company. | Factory/builder; call during graph or contract setup. |
| function | `validate_task_gate(tasks: list[CompanyTask], companies: list[ResolvedCompany], max_companies: int=MAX_COMPANIES_PER_QUERY)` | `dict[str, Any]` | Validate task isolation, coverage, permissions, dimensions, and run consistency. | Deterministic guard; call before the next workflow boundary. |
| function | `build_company_tasks_tool(run_id: str)` | `dict[str, Any]` | Build guarded company tasks from validated state registered for one run. | Guarded tool/node boundary; invoke with its declared schema. |
| function | `route_after_task_gate(task_gate: dict[str, Any])` | `str` | Route validated tasks to LangGraph fan-out or a bounded planning stop. | Route validated tasks to LangGraph fan-out or a bounded planning stop. |

### `multiindustry_evidence_adapters`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_utc_now_iso()` | `str` | Return a timezone-aware UTC timestamp for evidence retrieval metadata. | Internal helper; use through its owning public boundary. |
| function | `_canonical_payload_hash(value: Any)` | `str` | Hash an arbitrary JSON-like evidence value deterministically. | Internal helper; use through its owning public boundary. |
| function | `_evidence_status(item: Any)` | `tuple[str, str | None]` | Normalize heterogeneous tool status fields into the evidence status contract. | Internal helper; use through its owning public boundary. |
| function | `_validate_evidence_identity(company: ResolvedCompany, item: Any)` | `None` | Reject a raw result that identifies a different company ticker. | Internal helper; use through its owning public boundary. |
| function | `to_evidence_record(run_id: str, company: ResolvedCompany, profile_id: str, evidence_type: str, tool_result: Any, source_name: str \| None=None)` | `list[EvidenceRecord]` | Convert one heterogeneous tool result into canonical evidence records. | Convert one heterogeneous tool result into canonical evidence records. |
| function | `_call_or_use(tool_callable: Callable[[], Any], tool_result: Any)` | `Any` | Use an injected deterministic result or invoke the real source capability. | Internal helper; use through its owning public boundary. |
| function | `fetch_price_evidence(task: CompanyTask, tool_result: Any=None)` | `list[EvidenceRecord]` | Fetch or adapt current-price evidence for one company task. | Fetch or adapt current-price evidence for one company task. |
| function | `fetch_history_evidence(task: CompanyTask, period: str='1y', tool_result: Any=None)` | `list[EvidenceRecord]` | Fetch or adapt historical-price evidence for one company task. | Fetch or adapt historical-price evidence for one company task. |
| function | `fetch_financial_metric_evidence(task: CompanyTask, tool_result: Any=None)` | `list[EvidenceRecord]` | Fetch or adapt the deterministic financial-metric snapshot for one task. | Fetch or adapt the deterministic financial-metric snapshot for one task. |
| function | `fetch_news_evidence(task: CompanyTask, query: str \| None=None, tool_result: Any=None)` | `list[EvidenceRecord]` | Fetch or adapt company-scoped financial-news evidence. | Fetch or adapt company-scoped financial-news evidence. |
| function | `fetch_sentiment_evidence(task: CompanyTask, text: str, tool_result: Any=None)` | `list[EvidenceRecord]` | Fetch or adapt sentiment evidence for company-scoped source text. | Fetch or adapt sentiment evidence for company-scoped source text. |

### `multiindustry_technology_profile`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_technology_company_from_ticker(ticker: str)` | `ResolvedCompany` | Resolve and validate one ticker as a supported Technology/AI company. | Internal helper; use through its owning public boundary. |
| function | `_invoke_legacy_technology_rag(query: str)` | `str` | Invoke the existing assignment RAG tool without changing its implementation. | Internal helper; use through its owning public boundary. |
| function | `query_technology_rag(ticker: str, query: str)` | `dict[str, Any]` | Retrieve Technology/AI evidence for one supported canonical company. | Retrieve Technology/AI evidence for one supported canonical company. |
| function | `query_private_database_compat(query: str)` | `str` | Call the legacy technology RAG implementation during migration. | Call the legacy technology RAG implementation during migration. |
| function | `query_technology_rag_evidence(task: CompanyTask, query: str, tool_result: Any=None)` | `list[EvidenceRecord]` | Retrieve or adapt technology RAG output into canonical evidence. | Retrieve or adapt technology RAG output into canonical evidence. |
| function | `_technology_evidence_ids(company: ResolvedCompany, records: list[EvidenceRecord])` | `list[str]` | Validate and collect successful current-company technology evidence IDs. | Internal helper; use through its owning public boundary. |
| function | `extract_technology_signals_with_evidence(companies: list[ResolvedCompany], evidence_by_company: dict[str, list[EvidenceRecord]], raw_signals: dict[str, dict[str, Any]] \| None=None)` | `dict[str, dict[str, Any]]` | Normalize existing AI signals and bind every non-missing signal to evidence IDs. | Normalize existing AI signals and bind every non-missing signal to evidence IDs. |
| function | `score_technology_companies(financial_metrics: dict[str, dict], technology_signals: dict[str, dict], sentiment_scores: dict[str, dict], risk_profile: str='balanced')` | `dict[str, dict]` | Apply the existing deterministic technology scoring function unchanged. | Apply the existing deterministic technology scoring function unchanged. |

### `multiindustry_biopharma_rag`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_biopharma_rag_log(message: str, verbose: bool=True)` | `None` | Print one immediately flushed notebook progress message when logging is enabled. | Internal helper; use through its owning public boundary. |
| function | `_safe_archive_members(archive: zipfile.ZipFile)` | `list[zipfile.ZipInfo]` | Validate archive members and reject absolute or parent-traversal paths. | Internal helper; use through its owning public boundary. |
| function | `prepare_biopharma_corpus(archive_path: Path=BIOPHARMA_ARCHIVE_PATH, target_dir: Path=BIOPHARMA_CORPUS_DIR, verbose: bool=True)` | `Path` | Safely extract the official-source archive into a stable local directory. | Safely extract the official-source archive into a stable local directory. |
| function | `load_biopharma_manifest(corpus_dir: Path=BIOPHARMA_CORPUS_DIR)` | `list[dict[str, Any]]` | Load and validate the official-source manifest. | Load and validate the official-source manifest. |
| function | `_manifest_metadata(record: dict[str, Any])` | `dict[str, Any]` | Convert one manifest record into canonical vector-document metadata. | Internal helper; use through its owning public boundary. |
| function | `load_biopharma_documents(corpus_dir: Path=BIOPHARMA_CORPUS_DIR, verbose: bool=False, tickers: Sequence[str] \| None=None)` | `list[Document]` | Load PDF pages and official-link text records with company metadata. | Load PDF pages and official-link text records with company metadata. |
| function | `_corpus_fingerprint(corpus_dir: Path, tickers: Sequence[str] \| None=None)` | `str` | Hash the manifest plus selected ticker scope to detect required index rebuilds. | Internal helper; use through its owning public boundary. |
| function | `biopharma_index_ready(persist_dir: Path=BIOPHARMA_VECTOR_DIR, corpus_version: str=BIOPHARMA_CORPUS_VERSION, corpus_fingerprint: str \| None=None)` | `bool` | Check that a completed index marker matches the requested corpus version and hash. | Check that a completed index marker matches the requested corpus version and hash. |
| function | `_completed_biopharma_index_dir(persist_dir: Path)` | `Path` | Resolve the immutable completed index directory recorded by the root marker. | Internal helper; use through its owning public boundary. |
| function | `build_biopharma_index(corpus_dir: Path=BIOPHARMA_CORPUS_DIR, persist_dir: Path=BIOPHARMA_VECTOR_DIR, embeddings: Any \| None=None, force: bool=False, verbose: bool=True, embedding_batch_size: int=100, tickers: Sequence[str] \| None=DEFAULT_BIOPHARMA_INDEX_TICKERS)` | `Any` | Build the isolated Chroma collection and write its marker only after success. | Factory/builder; call during graph or contract setup. |
| function | `_default_biopharma_embeddings()` | `Any` | Create embeddings with the notebook's configured API key and proxy base URL. | Internal helper; use through its owning public boundary. |
| function | `configure_biopharma_vectorstore(vectorstore: Any \| None)` | `None` | Set the notebook-local vector store used by the biopharma retrieval tool. | Set the notebook-local vector store used by the biopharma retrieval tool. |
| function | `_biopharma_company_from_ticker(ticker: str)` | `ResolvedCompany` | Resolve and validate one ticker as a supported biopharma company. | Internal helper; use through its owning public boundary. |
| function | `query_biopharma_rag(ticker: str, query: str)` | `dict[str, Any]` | Retrieve company-isolated official biopharma evidence. | Retrieve company-isolated official biopharma evidence. |
| function | `query_biopharma_rag_evidence(task: CompanyTask, query: str, tool_result: Any=None)` | `list[EvidenceRecord]` | Retrieve or adapt biopharma RAG output into canonical evidence records. | Retrieve or adapt biopharma RAG output into canonical evidence records. |

### `multiindustry_biopharma_signals`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `validate_pharma_signal_rubric()` | `list[str]` | Return deterministic coverage errors for the biopharma signal rubric. | Deterministic guard; call before the next workflow boundary. |
| function | `_validated_biopharma_evidence_ids(company: ResolvedCompany, records: list[EvidenceRecord])` | `list[str]` | Validate biopharma evidence identity and return successful evidence IDs. | Internal helper; use through its owning public boundary. |
| function | `_pharma_extraction_prompt(companies: list[ResolvedCompany], evidence_by_company: dict[str, list[EvidenceRecord]])` | `str` | Build a bounded JSON extraction prompt containing evidence IDs and official text. | Internal helper; use through its owning public boundary. |
| function | `_default_pharma_extractor_model()` | `Any` | Create the deterministic-temperature model used for structured pharma interpretation. | Internal helper; use through its owning public boundary. |
| function | `extract_pharma_signals(companies: list[ResolvedCompany], evidence_by_company: dict[str, list[EvidenceRecord]], raw_signals: dict[str, dict[str, Any]] \| None=None, model: Any \| None=None)` | `dict[str, dict[str, Any]]` | Extract and deterministically normalize five evidence-linked biopharma signals. | Extract and deterministically normalize five evidence-linked biopharma signals. |
| function | `check_biopharma_scoring_gate()` | `ScoringEligibility` | Return the configuration gate for the validated notebook-local baseline rubric. | Deterministic guard; call before the next workflow boundary. |

### `multiindustry_company_worker`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_default_worker_tools()` | `dict[str, Any]` | Return the current notebook's source-tool objects keyed by contract name. | Internal helper; use through its owning public boundary. |
| function | `_worker_missing_dimensions(task: CompanyTask, evidence: list[EvidenceRecord])` | `list[str]` | Determine which requested task dimensions lack successful source coverage. | Internal helper; use through its owning public boundary. |
| function | `_tool_result_to_worker_evidence(task: CompanyTask, tool_name: str, tool_args: dict[str, Any], raw_result: Any)` | `list[EvidenceRecord]` | Convert one worker tool result into canonical evidence records. | Internal helper; use through its owning public boundary. |
| function | `create_company_worker(profile: IndustryProfile, model: Any, tools_by_name: dict[str, Any] \| None=None, max_tool_rounds: int=4, signal_extractor: Callable[[ResolvedCompany, list[EvidenceRecord]], dict[str, Any]] \| None=None)` | unspecified | Create one profile-configured LangGraph worker for isolated company research. | Factory/builder; call during graph or contract setup. |
| function | `initialize_worker(state: CompanyWorkerState)` | `dict[str, Any]` | Validate one-company task identity and initialize branch-local fields. | Validate one-company task identity and initialize branch-local fields. |
| function | `worker_agent(state: CompanyWorkerState)` | `dict[str, Any]` | Invoke the autonomous worker with tools unless the round ceiling was reached. | Invoke the autonomous worker with tools unless the round ceiling was reached. |
| function | `execute_allowed_tools(state: CompanyWorkerState)` | `dict[str, Any]` | Execute requested tools with allowlist, ticker, error, and evidence controls. | Execution entry point; inspect its structured terminal result. |
| function | `route_after_agent(state: CompanyWorkerState)` | `Literal['tools', 'evidence_gate']` | Route tool calls to execution and final narratives to the evidence gate. | Route tool calls to execution and final narratives to the evidence gate. |
| function | `evidence_exit_gate(state: CompanyWorkerState)` | `dict[str, Any]` | Require grounded dimension coverage or terminate as bounded partial output. | Require grounded dimension coverage or terminate as bounded partial output. |
| function | `route_after_evidence_gate(state: CompanyWorkerState)` | `Literal['retry', 'extract']` | Route incomplete evidence back to the agent while budget remains. | Route incomplete evidence back to the agent while budget remains. |
| function | `extract_profile_signals(state: CompanyWorkerState)` | `dict[str, Any]` | Run the selected profile extractor over validated company-local evidence. | Run the selected profile extractor over validated company-local evidence. |
| function | `validate_company_result(state: CompanyWorkerState)` | `dict[str, Any]` | Assemble one identity-safe normalized result for the parent reducer. | Deterministic guard; call before the next workflow boundary. |

### `multiindustry_f10_smoke`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| class `_F10ScriptedModel` | — | — | Emit one source-tool call followed by a final answer. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, tool_name: str, arguments: dict[str, Any])` | unspecified | Store the single tool request and initialize call tracking. | Internal helper; use through its owning public boundary. |
| function | `bind_tools(self, tools)` | unspecified | Record the profile-selected tool names and return this model. | Record the profile-selected tool names and return this model. |
| function | `invoke(self, messages)` | unspecified | Return a tool call on the first turn and a narrative afterward. | Return a tool call on the first turn and a narrative afterward. |
| class `_F10FakeTool` | — | — | Minimal named tool double returning a predefined result. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, name: str, result: Any)` | unspecified | Store the LangChain-compatible name and deterministic result. | Internal helper; use through its owning public boundary. |
| function | `invoke(self, arguments)` | unspecified | Return the configured result without external calls. | Return the configured result without external calls. |

### `multiindustry_parent_orchestrator`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| class `NotebookOrchestrator` | — | — | Apply notebook-local execution limits to a compiled parent LangGraph. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, graph: Any, max_concurrency: int, recursion_limit: int)` | unspecified | Store the compiled graph and validated local execution ceilings. | Internal helper; use through its owning public boundary. |
| function | `_bounded_config(self, config: Mapping[str, Any] \| None)` | `dict[str, Any]` | Copy an invocation config and cap its concurrency and recursion values. | Internal helper; use through its owning public boundary. |
| function | `invoke(self, input_state: Mapping[str, Any], config: Mapping[str, Any] \| None=None, **kwargs: Any)` | `dict[str, Any]` | Invoke the parent graph under notebook-local execution ceilings. | Invoke the parent graph under notebook-local execution ceilings. |
| function | `stream(self, input_state: Mapping[str, Any], config: Mapping[str, Any] \| None=None, **kwargs: Any)` | unspecified | Stream parent-graph events under notebook-local execution ceilings. | Stream parent-graph events under notebook-local execution ceilings. |
| function | `get_graph(self, *args: Any, **kwargs: Any)` | unspecified | Return the compiled graph visualization object. | Return the compiled graph visualization object. |
| function | `_failed_company_result(task: CompanyTask, error: str)` | `CompanyResearchResult` | Create an identity-preserving failed result for one contained branch error. | Internal helper; use through its owning public boundary. |
| function | `_validate_fan_in_results(tasks: list[CompanyTask], results: CompanyResultMap, run_id: str)` | `list[str]` | Validate branch coverage, run identity, profile identity, and evidence isolation. | Internal helper; use through its owning public boundary. |
| function | `create_multi_company_orchestrator(planner_model: Any, worker_model_factory: Callable[[CompanyTask, IndustryProfile], Any], worker_tools_factory: Callable[[CompanyTask, IndustryProfile], dict[str, Any]], signal_extractor_factory: Callable[[CompanyTask, IndustryProfile], Callable[[ResolvedCompany, list[EvidenceRecord]], dict[str, Any]] \| None] \| None=None, max_companies: int=MAX_COMPANIES_PER_QUERY, max_concurrency: int=2, recursion_limit: int=50, worker_max_tool_rounds: int=4, enable_f12: bool=False)` | `NotebookOrchestrator` | Create the guarded parent graph for one or more isolated company workers. | Factory/builder; call during graph or contract setup. |
| function | `initialize_run(state: OrchestratorState)` | `dict[str, Any]` | Reset request-scoped state while preserving conversation-scoped memory. | Reset request-scoped state while preserving conversation-scoped memory. |
| function | `coordinator_plan(state: OrchestratorState)` | `dict[str, Any]` | Interpret the free-text request into a deterministically validated query plan. | Interpret the free-text request into a deterministically validated query plan. |
| function | `resolve_companies_node(state: OrchestratorState)` | `dict[str, Any]` | Invoke the guarded resolver only when planning produced an in-budget plan. | Invoke the guarded resolver only when planning produced an in-budget plan. |
| function | `mandatory_resolution_gate(state: OrchestratorState)` | `dict[str, Any]` | Recompute company-resolution validity before profile selection is reachable. | Recompute company-resolution validity before profile selection is reachable. |
| function | `route_resolution(state: OrchestratorState)` | `str` | Continue only when the mandatory resolution gate is ready. | Continue only when the mandatory resolution gate is ready. |
| function | `select_profiles_node(state: OrchestratorState)` | `dict[str, Any]` | Invoke registry-backed profile selection for canonical resolved identities. | Invoke registry-backed profile selection for canonical resolved identities. |
| function | `mandatory_profile_gate(state: OrchestratorState)` | `dict[str, Any]` | Require exact profile coverage for the companies resolved in this run. | Require exact profile coverage for the companies resolved in this run. |
| function | `route_profile(state: OrchestratorState)` | `str` | Continue only when mandatory profile validation is ready. | Continue only when mandatory profile validation is ready. |
| function | `build_tasks_node(state: OrchestratorState)` | `dict[str, Any]` | Build run-scoped company tasks through the guarded task tool. | Factory/builder; call during graph or contract setup. |
| function | `mandatory_task_gate(state: OrchestratorState)` | `dict[str, Any]` | Revalidate branch isolation, permissions, identity, and company budget. | Revalidate branch isolation, permissions, identity, and company budget. |
| function | `route_task_gate(state: OrchestratorState)` | `str` | Make the fan-out node unreachable until the mandatory task gate passes. | Make the fan-out node unreachable until the mandatory task gate passes. |
| function | `fan_out_company_tasks(state: OrchestratorState)` | `dict[str, Any]` | Mark the validated fan-out boundary without mutating branch inputs. | Mark the validated fan-out boundary without mutating branch inputs. |
| function | `dispatch_company_tasks(state: OrchestratorState)` | unspecified | Create exactly one LangGraph Send branch for every validated company task. | Create exactly one LangGraph Send branch for every validated company task. |
| function | `company_worker_node(state: OrchestratorState)` | `dict[str, Any]` | Execute one profile-configured F10 worker and contain branch failures. | Execute one profile-configured F10 worker and contain branch failures. |
| function | `collect_results(state: OrchestratorState)` | `dict[str, Any]` | Validate reducer output after all Send branches reach the fan-in barrier. | Validate reducer output after all Send branches reach the fan-in barrier. |
| function | `normalize_fan_in_node(state: OrchestratorState)` | `dict[str, Any]` | Normalize reducer output in task order and expose blocking boundary errors. | Normalize reducer output in task order and expose blocking boundary errors. |
| function | `route_after_fan_in_normalization(state: OrchestratorState)` | `str` | Continue only when normalized fan-in has usable, uncontaminated results. | Continue only when normalized fan-in has usable, uncontaminated results. |
| function | `bounded_stop(state: OrchestratorState)` | `dict[str, Any]` | Return a bounded explanation when a mandatory pre-research gate fails. | Return a bounded explanation when a mandatory pre-research gate fails. |

### `multiindustry_f11_smoke`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_f11_worker_model_factory(task, profile)` | unspecified | Return a fresh scripted model selecting the branch profile's RAG tool. | Internal helper; use through its owning public boundary. |
| function | `_f11_worker_tools_factory(task, profile)` | unspecified | Return profile-complete deterministic tools for one isolated branch. | Internal helper; use through its owning public boundary. |
| function | `_f11_signal_extractor_factory(task, profile)` | unspecified | Return a deterministic evidence-linked extractor for a smoke-test branch. | Internal helper; use through its owning public boundary. |
| function | `extract(company, evidence)` | unspecified | Build partial fixture signals linked only to successful branch evidence. | Build partial fixture signals linked only to successful branch evidence. |

### `multiindustry_fan_in_normalization`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_ordered_unique_strings(values: Iterable[Any])` | `list[str]` | Return stripped, non-empty strings once each while retaining input order. | Internal helper; use through its owning public boundary. |
| function | `_normalization_failed_result(task: CompanyTask, errors: Iterable[str])` | `CompanyResearchResult` | Create a failed result that keeps only the expected task identity. | Internal helper; use through its owning public boundary. |
| function | `_evidence_boundary_errors(task: CompanyTask, record: Mapping[str, Any], run_id: str)` | `list[str]` | Return hard boundary violations for one evidence record. | Internal helper; use through its owning public boundary. |
| function | `_normalize_evidence_records(task: CompanyTask, raw_records: Any, run_id: str)` | `tuple[list[EvidenceRecord], list[str], bool]` | Validate evidence shape, enums, uniqueness, and hard identity boundaries. | Internal helper; use through its owning public boundary. |
| function | `_acceptable_success_ids(task: CompanyTask, evidence: list[EvidenceRecord])` | `set[str]` | Return successful evidence IDs that satisfy the query's freshness requirement. | Internal helper; use through its owning public boundary. |
| function | `_normalize_industry_signals(raw_signals: Any, acceptable_ids: set[str])` | `tuple[dict[str, Any], list[str]]` | Keep only evidence-grounded signal references and downgrade ungrounded signals. | Internal helper; use through its owning public boundary. |
| function | `_required_dimension_gaps(task: CompanyTask, evidence: list[EvidenceRecord], signals: Mapping[str, Any])` | `list[str]` | Compute required dimensions that lack acceptable source or signal coverage. | Internal helper; use through its owning public boundary. |
| function | `normalize_company_result(task: CompanyTask, result: CompanyResearchResult \| Mapping[str, Any] \| None, run_id: str \| None=None)` | `CompanyResearchResult` | Normalize one worker result against its expected immutable task boundary. | Normalize one worker result against its expected immutable task boundary. |
| function | `_coerce_result_items(results: Mapping[str, CompanyResearchResult] \| Iterable[CompanyResearchResult])` | `tuple[dict[str, CompanyResearchResult], list[str]]` | Convert reducer maps or result sequences to one ticker map and detect duplicates. | Internal helper; use through its owning public boundary. |
| function | `normalize_all_results(tasks: list[CompanyTask], results: Mapping[str, CompanyResearchResult] \| Iterable[CompanyResearchResult], run_id: str \| None=None)` | `FanInNormalization` | Normalize fan-in results in expected task order, independent of completion order. | Normalize fan-in results in expected task order, independent of completion order. |

### `multiindustry_comparison_mode_routing`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_routing_text(value: Any)` | `str | None` | Return a stripped non-empty string, otherwise ``None``. | Internal helper; use through its owning public boundary. |
| function | `_unique_routing_errors(errors: Sequence[str])` | `list[str]` | Preserve deterministic error order while removing duplicate messages. | Internal helper; use through its owning public boundary. |
| function | `_validate_mode_result_map(results: Mapping[str, CompanyResearchResult] \| Any)` | `list[str]` | Validate the identity fields needed for safe comparison-mode selection. | Internal helper; use through its owning public boundary. |
| function | `select_comparison_mode(results: Mapping[str, CompanyResearchResult])` | `ComparisonMode` | Select exactly one deterministic mode from a structurally valid result map. | Select exactly one deterministic mode from a structurally valid result map. |
| function | `validate_comparison_routing(results: Mapping[str, CompanyResearchResult] \| Any, run_id: str, expected_tasks: Sequence[CompanyTask] \| Any)` | `dict[str, Any]` | Validate current-run fan-in coverage and return a fail-closed routing decision. | Deterministic guard; call before the next workflow boundary. |
| function | `check_scoring_eligibility(results: Mapping[str, CompanyResearchResult], mode: ComparisonMode, profile_lookup: Callable[[str], IndustryProfile] \| None=None)` | `ScoringEligibility` | Decide whether normalized results may enter deterministic sector scoring. | Deterministic guard; call before the next workflow boundary. |
| function | `register_scoring_eligibility_context(run_id: str, eligibility: ScoringEligibility)` | `None` | Store one defensive current-run eligibility decision for the guarded tool wrapper. | Store one defensive current-run eligibility decision for the guarded tool wrapper. |
| function | `clear_scoring_eligibility_context(run_id: str)` | `bool` | Remove a notebook-local eligibility context and report whether it existed. | Remove a notebook-local eligibility context and report whether it existed. |
| function | `check_scoring_eligibility_tool(run_id: str)` | `ScoringEligibility` | Return the mandatory scoring decision stored for a validated current run. | Guarded tool/node boundary; invoke with its declared schema. |
| function | `mandatory_comparison_mode_node(state: OrchestratorState)` | `dict[str, Any]` | LangGraph node that writes a mode only after current-run coverage validation. | LangGraph node that writes a mode only after current-run coverage validation. |
| function | `route_after_comparison_mode(state: OrchestratorState)` | `str` | Mandatory conditional-edge router that revalidates rather than trusting model state. | Mandatory conditional-edge router that revalidates rather than trusting model state. |

### `multiindustry_sector_scoring`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_f13_nonempty_text(value: Any, field_name: str)` | `str` | Return a required stripped string or raise a deterministic validation error. | Internal helper; use through its owning public boundary. |
| function | `_f13_finite_number(value: Any, field_name: str)` | `float` | Return one finite real number while rejecting booleans and non-numeric values. | Internal helper; use through its owning public boundary. |
| function | `_f13_validate_result_identity(ticker: str, result: Mapping[str, Any], expected_run_id: str \| None, expected_profile_id: str)` | `tuple[str, Mapping[str, Any], set[str]]` | Validate one normalized result and return its run, company, and valid evidence IDs. | Internal helper; use through its owning public boundary. |
| function | `_f13_financial_metrics(ticker: str, result: Mapping[str, Any])` | `dict[str, float]` | Extract all five finite metrics from one successful canonical evidence record. | Internal helper; use through its owning public boundary. |
| function | `_f13_technology_signals(ticker: str, result: Mapping[str, Any], successful_evidence_ids: set[str])` | `dict[str, dict[str, Any]]` | Rebuild four technology signals from levels and verified evidence references. | Internal helper; use through its owning public boundary. |
| function | `_f13_biopharma_signals(ticker: str, result: Mapping[str, Any], successful_evidence_ids: set[str])` | `dict[str, dict[str, Any]]` | Rebuild five grounded biopharma signals, including inverted sector risk. | Internal helper; use through its owning public boundary. |
| function | `_f13_research_band(total_score: float)` | `str` | Map a 0–100 research-strength score to a non-investment-action band. | Internal helper; use through its owning public boundary. |
| function | `score_biopharma_companies(financial_metrics: Mapping[str, Mapping[str, float]], biopharma_signals: Mapping[str, Mapping[str, Mapping[str, Any]]], risk_profile: str='balanced')` | `dict[str, dict[str, Any]]` | Compute the deterministic notebook-local biopharma research-strength score. | Compute the deterministic notebook-local biopharma research-strength score. |
| function | `compute_sector_scores(results: Mapping[str, CompanyResearchResult], comparison_mode: ComparisonMode, scoring_eligibility: ScoringEligibility, risk_profile: str)` | `dict[str, dict[str, Any]]` | Compute authoritative same-profile sector scores from canonical evidence. | Compute authoritative same-profile sector scores from canonical evidence. |
| function | `_freeze_f13_value(value: Any)` | `Any` | Recursively copy mutable containers into read-only scoring-context values. | Internal helper; use through its owning public boundary. |
| function | `_thaw_f13_value(value: Any)` | `Any` | Create ordinary defensive containers for one pure scoring invocation. | Internal helper; use through its owning public boundary. |
| class `SectorScoringContext` | — | — | Immutable notebook-local inputs authorized for one guarded score request. | Instantiate/use through the owning feature boundary. |
| function | `register_sector_scoring_context(run_id: str, normalized_results: Mapping[str, CompanyResearchResult], comparison_mode: ComparisonMode, scoring_eligibility: ScoringEligibility, risk_profile: str)` | `None` | Validate and defensively freeze the authoritative inputs for one current run. | Validate and defensively freeze the authoritative inputs for one current run. |
| function | `clear_sector_scoring_context(run_id: str)` | `bool` | Remove one notebook-local score context and report whether it existed. | Remove one notebook-local score context and report whether it existed. |
| function | `_f13_blocked_tool_result(run_id: str, error: str)` | `dict[str, Any]` | Build the stable fail-closed guarded-tool response envelope. | Internal helper; use through its owning public boundary. |
| function | `compute_sector_scores_tool(run_id: str)` | `dict[str, Any]` | Compute authoritative sector scores for a registered run using only ``run_id``. | Guarded tool/node boundary; invoke with its declared schema. |

### `multiindustry_f13_smoke`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_f13_smoke_result(company: ResolvedCompany, market_cap: float, level: str)` | `CompanyResearchResult` | Build one complete evidence-grounded technology result for the local smoke test. | Internal helper; use through its owning public boundary. |
| function | `_f13_biopharma_smoke_result(company: ResolvedCompany, market_cap: float, levels: dict[str, str])` | `CompanyResearchResult` | Build one grounded biopharma result for provisional-rubric verification. | Internal helper; use through its owning public boundary. |

### `multiindustry_mode_specific_synthesis`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_f14_text(value: Any, field_name: str)` | `str` | Return a stripped non-empty string or raise a deterministic validation error. | Internal helper; use through its owning public boundary. |
| function | `_f14_unique_text(values: Any, field_name: str)` | `list[str]` | Normalize a list of non-empty strings while preserving first-seen order. | Internal helper; use through its owning public boundary. |
| function | `_f14_required_limitations(results: Mapping[str, CompanyResearchResult], mode: ComparisonMode, eligibility: Mapping[str, Any], scores: Mapping[str, Any] \| None)` | `list[str]` | Derive limitations that the model is not permitted to omit. | Internal helper; use through its owning public boundary. |
| function | `_f14_validate_context(context: SynthesisContext \| Mapping[str, Any])` | `dict[str, Any]` | Validate run, mode, evidence, and score boundaries and return a defensive copy. | Internal helper; use through its owning public boundary. |
| function | `build_single_prompt(context: Mapping[str, Any])` | `str` | Build the one-company synthesis policy without comparison-score authority. | Factory/builder; call during graph or contract setup. |
| function | `build_same_profile_prompt(context: Mapping[str, Any])` | `str` | Build the like-for-like sector synthesis policy with immutable optional scores. | Factory/builder; call during graph or contract setup. |
| function | `build_cross_profile_prompt(context: Mapping[str, Any])` | `str` | Build the qualitative portfolio policy without a universal sector score. | Factory/builder; call during graph or contract setup. |
| function | `create_synthesizer(mode: ComparisonMode, profile: IndustryProfile \| None=None)` | `Callable[[Mapping[str, Any]], str]` | Select one prompt builder without binding any research or scoring tools. | Factory/builder; call during graph or contract setup. |
| function | `_f14_prompt_payload(context: Mapping[str, Any])` | `str` | Serialize the bounded synthesis inputs; omit score material outside same-profile mode. | Internal helper; use through its owning public boundary. |
| function | `_f14_parse_response(response: Any)` | `dict[str, Any]` | Parse a mapping or JSON message returned by an injected synthesis model. | Internal helper; use through its owning public boundary. |
| function | `synthesize_answer(context: SynthesisContext \| Mapping[str, Any], injected_model: Any)` | `SynthesisResult` | Generate and constrain one grounded mode-specific draft without research tools. | Generate and constrain one grounded mode-specific draft without research tools. |

### `multiindustry_f14_smoke`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| class `_F14SmokeModel` | — | — | Return one deterministic grounded synthesis payload and retain received messages. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, evidence_id: str)` | unspecified | Store the evidence ID cited by the deterministic response. | Internal helper; use through its owning public boundary. |
| function | `invoke(self, messages: list[Any])` | `dict[str, Any]` | Record prompt messages and return a deliberately score-free model payload. | Record prompt messages and return a deliberately score-free model payload. |

### `multiindustry_f15_evidence_validation`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_f15_nonempty_text(value: Any, field_name: str)` | `str | None` | Return stripped text or ``None``; validation callers add contextual errors. | Internal helper; use through its owning public boundary. |
| function | `_f15_unique_errors(errors: Sequence[str])` | `list[str]` | Remove repeated errors while preserving deterministic discovery order. | Internal helper; use through its owning public boundary. |
| function | `extract_inline_evidence_ids(answer: Any)` | `list[str]` | Extract ordered explicit ``[EV-*]`` citation tokens from answer prose. | Extract ordered explicit ``[EV-*]`` citation tokens from answer prose. |
| function | `_f15_validate_prose_score_claims(answer: Any, mode: str \| None, authoritative_scores: Mapping[str, Any])` | `list[str]` | Validate only explicit machine-recognizable score and rank claims in prose. | Internal helper; use through its owning public boundary. |
| function | `_f15_required_limitations(normalized_results: Mapping[str, Any], mode: str \| None, scoring_eligibility: Mapping[str, Any], authoritative_scores: Mapping[str, Any])` | `list[str]` | Rebuild F14's deterministic mandatory limitations without trusting model output. | Internal helper; use through its owning public boundary. |
| function | `_f15_catalog_evidence(run_id: str \| None, normalized_results: Any, errors: list[str])` | `tuple[dict[str, Mapping[str, Any]], dict[str, bool], str | None]` | Validate F12 evidence ownership and build an unambiguous ID catalog. | Internal helper; use through its owning public boundary. |
| function | `validate_synthesis_result(run_id: Any, normalized_results: Any, synthesis_result: Any, *, authoritative_scores: Mapping[str, Any] \| None=None, scoring_eligibility: Mapping[str, Any] \| None=None, required_limitations: Sequence[str] \| None=None)` | `ValidationResult` | Validate one F14 result against current-run F12/F13 deterministic authority. | Deterministic guard; call before the next workflow boundary. |

### `multiindustry_f15_local_traces`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_f15_trace_text(value: Any, field_name: str)` | `str` | Return a required stripped string or raise a deterministic validation error. | Internal helper; use through its owning public boundary. |
| function | `_f15_trace_timestamp(value: str \| datetime \| None=None)` | `str` | Normalize an injected or current timestamp to an ISO-8601 UTC string. | Internal helper; use through its owning public boundary. |
| function | `_f15_trace_is_sensitive_key(key: Any)` | `bool` | Return whether a mapping key names a credential-like value. | Internal helper; use through its owning public boundary. |
| function | `_f15_trace_sanitize_uri(value: Any)` | `Any` | Remove query parameters and fragments that commonly carry signed credentials. | Internal helper; use through its owning public boundary. |
| function | `redact_trace_value(value: Any, *, parent_key: str='')` | `Any` | Return a JSON-safe deep copy with credential fields and raw content redacted. | Return a JSON-safe deep copy with credential fields and raw content redacted. |
| function | `_f15_trace_company(ticker: str, result: Mapping[str, Any])` | `dict[str, Any]` | Extract non-secret canonical identity fields from one normalized company result. | Internal helper; use through its owning public boundary. |
| function | `_f15_trace_evidence_record(record: Mapping[str, Any], expected_run_id: str, ticker: str, profile_id: str)` | `dict[str, Any]` | Project one evidence item onto safe provenance metadata without its raw value. | Internal helper; use through its owning public boundary. |
| function | `create_research_trace(*, run_id: str, query: str, comparison_mode: str, normalized_results: Mapping[str, Mapping[str, Any]], f13_scores: Mapping[str, Any] \| None, f14_synthesis: Mapping[str, Any] \| None, timestamp: str \| datetime \| None=None)` | `ResearchTrace` | Create an in-memory redacted trace from already-computed F12/F13/F14 artifacts. | Factory/builder; call during graph or contract setup. |
| function | `record_validation_attempt(trace: Mapping[str, Any], validation_result: Mapping[str, Any], *, attempt_number: int, timestamp: str \| datetime \| None=None)` | `ResearchTrace` | Return a copied trace with one bounded deterministic validation attempt appended. | Return a copied trace with one bounded deterministic validation attempt appended. |
| function | `finalize_research_trace(trace: Mapping[str, Any], *, final_status: str, timestamp: str \| datetime \| None=None, terminal_error: str \| None=None)` | `ResearchTrace` | Return a copied trace finalized as success, failure, or interruption. | Return a copied trace finalized as success, failure, or interruption. |
| function | `_f15_trace_validate_for_write(trace: Mapping[str, Any])` | `dict[str, Any]` | Validate the minimum persisted trace contract and return a sanitized copy. | Internal helper; use through its owning public boundary. |
| function | `_f15_trace_apply_retention(trace_dir: Path, current_path: Path, retention_limit: int)` | `list[str]` | Remove oldest trace JSON files while always retaining the just-written record. | Internal helper; use through its owning public boundary. |
| function | `write_research_trace(trace: Mapping[str, Any], *, trace_dir: str \| Path='.research_runs', retention_limit: int=F15_TRACE_DEFAULT_RETENTION, replace_func: Callable[[str \| os.PathLike[str], str \| os.PathLike[str]], None]=os.replace)` | `TraceWriteResult` | Atomically publish one redacted JSON trace and enforce bounded local retention. | Atomically publish one redacted JSON trace and enforce bounded local retention. |

### `multiindustry_f15_workflow`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `_f15_workflow_now(timestamp_provider: Callable[[], str \| datetime] \| None)` | `str | datetime | None` | Return an injected timestamp or defer to the trace component's UTC clock. | Internal helper; use through its owning public boundary. |
| function | `_f15_failed_validation(error: Exception \| str)` | `ValidationResult` | Build a stable failed verdict when F14 cannot produce a structured candidate. | Internal helper; use through its owning public boundary. |
| class `_F15CorrectionModel` | — | — | Append deterministic validation feedback to the next tool-free F14 invocation. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, model: Any, previous_synthesis: Mapping[str, Any] \| None, validation_errors: list[str], correction_number: int)` | unspecified | Capture the underlying model and bounded correction context. | Internal helper; use through its owning public boundary. |
| function | `invoke(self, messages: list[Any])` | `Any` | Invoke the model with the original F14 messages plus correction-only feedback. | Invoke the model with the original F14 messages plus correction-only feedback. |
| function | `_f15_set_trace_synthesis(trace: Mapping[str, Any], synthesis: Mapping[str, Any] \| None)` | `ResearchTrace` | Return a copied trace containing the latest redacted F14 candidate. | Internal helper; use through its owning public boundary. |
| function | `run_f15_validated_synthesis(context: SynthesisContext \| Mapping[str, Any], injected_model: Any, *, trace_dir: str \| Path='.research_runs', retention_limit: int=F15_TRACE_DEFAULT_RETENTION, max_correction_attempts: int=F15_MAX_CORRECTION_ATTEMPTS, timestamp_provider: Callable[[], str \| datetime] \| None=None)` | `F15WorkflowResult` | Run F14, enforce F15, persist every attempt, and return a bounded terminal result. | Execution entry point; inspect its structured terminal result. |

### `multiindustry_f15_smoke`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| class `_F15SmokeModel` | — | — | Return grounded JSON and optionally force one citation-correction attempt. | Instantiate/use through the owning feature boundary. |
| function | `__init__(self, *, fail_first: bool=False)` | unspecified | Configure deterministic first-attempt behavior and call tracking. | Internal helper; use through its owning public boundary. |
| function | `invoke(self, messages: list[Any])` | `dict[str, Any]` | Read bounded F14 payload and return a deterministic structured candidate. | Read bounded F14 payload and return a deterministic structured candidate. |
| function | `_f15_smoke_company(ticker: str, profile_id: str)` | `dict[str, Any]` | Build one resolved canonical company for representative F15 examples. | Internal helper; use through its owning public boundary. |
| function | `_f15_smoke_result(run_id: str, ticker: str, profile_id: str)` | `dict[str, Any]` | Build one normalized result with a canonical current-run evidence record. | Internal helper; use through its owning public boundary. |

### `multiindustry_f16_interactive_query`

| Kind | Inputs | Output | Purpose | How to use |
|---|---|---|---|---|
| function | `ask_financial_analyst(query: str)` | unspecified | Run one arbitrary free-text question through the online F1-F15 workflow. | Execution entry point; inspect its structured terminal result. |

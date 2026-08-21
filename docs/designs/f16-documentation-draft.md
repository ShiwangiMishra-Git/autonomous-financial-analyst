# F16 Documentation Draft and Canonical-Document Review

This is an isolated review draft for the main agent. It records behavior observed in the current
F1–F15 implementation and separates that behavior from F16 work that is still being integrated.
It is not the canonical LLD or implementation plan.

## 1. Local setup

### Observed implementation

The supported implementation notebook is
`Autonomous_financial_analyst_Learners_Notebook copy.ipynb`. The merged, Part 1, Part 2, and
unsolved notebooks are reference artifacts and must not be edited.

1. Use the existing `.venv` and select the registered **Project 2** Jupyter kernel. If the local
   environment must be rebuilt, install `requirements.txt` and register an IPython kernel.
2. Start JupyterLab with `./run_jupyter.sh`, or open the notebook in VS Code/Cursor and select the
   **Project 2** kernel manually.
3. For live provider calls, create a local, uncommitted `config.json` containing `API_KEY`,
   `OPENAI_API_BASE`, and `TAVILY_API_KEY`. The course configuration uses the Great Learning
   OpenAI-compatible endpoint. Never print or commit this file.
4. Run the notebook configuration cell before constructing an embedding model or making live
   LLM/Tavily calls. The biopharma index builder specifically requires both `OPENAI_API_KEY` and
   `OPENAI_API_BASE` in the process environment.
5. Technology RAG uses `content/Companies-AI-Initiatives/`. Biopharma RAG uses
   `content/pharma_rag_official_sources/` (or its local archive) and persists versioned immutable
   index children below `content/vectorstore_biopharma/`.

The starter biopharma build is intentionally bounded to `PFE`, `MRK`, `LLY`, `JNJ`, and `AZN`.
Passing `tickers=None` selects the full manifest. Rebuilding the same unchanged corpus may reuse a
completed fingerprinted index; a successful rebuild publishes a new immutable child only when the
fingerprint or requested build changes.

### Offline tests available before F16

The deterministic suite requires no provider credentials:

```bash
.venv/bin/python -m pytest -q
```

Useful existing focused commands are:

```bash
.venv/bin/python -m pytest -q tests/test_comparison_mode_routing.py
.venv/bin/python -m pytest -q tests/test_sector_scoring.py
.venv/bin/python -m pytest -q tests/test_synthesis_modes.py
.venv/bin/python -m pytest -q tests/test_f15_evidence_validation.py tests/test_f15_local_traces.py tests/test_f15_workflow.py
```

### F16 target commands — planned until integration is reconciled

The delegated F16 ownership calls for these commands, but their exact CLI flags and skip messages
must be copied from the reconciled runner rather than inferred here:

```bash
.venv/bin/python -m pytest -q tests/test_f16_end_to_end_scenarios.py
.venv/bin/python -m pytest -q tests/test_f16_live_smoke.py
.venv/bin/python scripts/run_f16_scenarios.py
```

Live smoke tests must be explicit opt-in, must check configuration before making calls, and must
skip safely when required credentials or local corpora/indexes are unavailable. They must not emit
API keys, signed URLs, or full private-document text. The final canonical docs should record the
runner's actual opt-in flag/environment variable after Agent 2's implementation is reviewed.

## 2. Routing modes

### Observed implementation

F12 chooses the mode deterministically from normalized, canonically resolved results:

| Input | Mode | Scoring consequence |
|---|---|---|
| Exactly one resolved company | `single` | No comparison score |
| Two or more companies with the same exact `profile_id` | `same_profile` | F13 may score only if all eligibility checks pass |
| Two or more companies spanning exact profile IDs | `cross_profile` | No universal or sector-comparison score |

`validate_comparison_routing` additionally requires exact current-run task coverage. Missing,
unexpected, malformed, duplicated, mismatched, or cross-run results route to `bounded_stop` rather
than synthesis. Completion order is normalized back to task order before routing.

Company resolution occurs before fan-out. Aliases and explicit supported tickers resolve through
the local registry. Ambiguous mentions, such as bare `Roche`, require clarification. Unknown or
unsupported companies stop with `stop_unsupported`; there is no implemented shared-financial-only
fallback. Duplicate mentions resolving to the same `company_id` are collapsed.

Partial or failed company results can still take a narrative mode when the normalized expected
result map is complete, but they disable F13 scoring and produce mandatory limitations. A missing
branch is represented by F12 normalization as a failed placeholder; routing does not silently drop
the company.

## 3. Scoring rules

### Technology/AI (`technology.ai.score.v1`)

Observed F13 behavior:

- Scoring requires at least two complete `technology.ai.v1` companies in an eligible
  `same_profile` comparison.
- Every company must have finite `market_cap`, `total_revenue`, `pe_ratio`, `beta`, and
  `dividend_yield` values.
- The four evidence-grounded signals are `infrastructure_moat`, `product_deployment`,
  `research_depth`, and `strategic_commitment`.
- Signal levels are re-derived as `none=0`, `partial=0.5`, and `full=1`; any stored/model-proposed
  numeric signal score is ignored.
- Financial metrics are rank-scored against peers. Market cap, revenue, and dividend yield are
  higher-is-better; P/E and beta are lower-is-better. Each financial metric is capped at `0.40`.
- Financial weights by risk profile are:

| Profile | Market cap | Revenue | P/E | Beta | Dividend yield |
|---|---:|---:|---:|---:|---:|
| Conservative | 0.8 | 0.8 | 1.2 | 1.2 | 1.2 |
| Balanced | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Growth | 1.2 | 1.2 | 0.8 | 0.8 | 0.8 |

- Signal weights are:

| Profile | Infrastructure moat | Product deployment | Research depth | Strategic commitment |
|---|---:|---:|---:|---:|
| Conservative | 1.2 | 1.2 | 0.8 | 1.0 |
| Balanced | 1.0 | 1.0 | 1.0 | 1.0 |
| Growth | 1.2 | 0.8 | 1.2 | 1.0 |

- `total_score = financial rank score + AI signal score`. Recommendation thresholds scale to the
  achievable maximum for the selected risk profile: Buy at `3.25/4.5` of that maximum, Hold at
  `2.50/4.5`, otherwise Sell. Sentiment is retained in the output but is not included in
  `total_score`.

F13 rebuilds these inputs from current-run normalized evidence and takes `risk_profile` from the
validated query plan. The agent-callable wrapper accepts only `run_id`; it cannot accept metrics,
weights, signal numbers, risk profile, or a proposed score.

### Biopharma (`healthcare.biopharma.score.v1`)

Observed F13 behavior:

- Scoring requires at least two complete `healthcare.biopharma.v1` companies in an eligible
  `same_profile` comparison and the same five finite financial metrics.
- Positive signal levels map as `none=0`, `partial=0.5`, `full=1`. `sector_risks` is inverted as
  `none=1`, `partial=0.5`, `full=0`.
- Sector signal weights are:

| Signal | Conservative | Balanced | Growth |
|---|---:|---:|---:|
| Clinical pipeline | 15% | 25% | 35% |
| Regulatory progress | 20% | 20% | 25% |
| Exclusivity and patents | 25% | 20% | 10% |
| Commercialization | 20% | 20% | 20% |
| Sector risks | 20% | 15% | 10% |

- The normalized financial/pharma blend is `60/40` for conservative, `50/50` for balanced, and
  `35/65` for growth.
- The 0–100 result is `Strong research profile` at 70 or higher, `Moderate research profile` at
  50–69.999, and `Weak research profile` below 50. These labels are not Buy/Hold/Sell advice.
- Missing, non-finite, partial, failed, or ungrounded inputs fail closed; F13 does not impute them.

Neither rubric applies to `single` or `cross_profile` mode, and neither rubric may cross a profile
boundary.

## 4. `.research_runs` interpretation

### Observed implementation

`run_f15_validated_synthesis` writes `.research_runs/{run_id}.json` once F14/F15 processing begins
with a valid synthesis context. The default retention limit is 50 records. A record contains:

- schema version and run ID;
- original query and comparison mode;
- canonical company identities and profile IDs;
- provenance-only evidence fields;
- optional authoritative F13 scores;
- the latest F14 synthesis;
- every F15 validation attempt;
- UTC start/update/completion timestamps;
- `in_progress`, then terminal `success`, `failed`, or `interrupted` status;
- an optional terminal error.

Interpretation rules:

- `success`: a draft passed deterministic F15 checks and the finalized trace was published.
- `failed`: the initial draft and permitted corrections did not pass; inspect
  `validation_attempts[*].result.errors` and the warning returned in `F15WorkflowResult`.
- `interrupted`: a `KeyboardInterrupt` occurred inside the bounded F15 workflow; the interruption
  is recorded and re-raised.
- `in_progress`: the latest atomic publication happened before a terminal update, commonly because
  execution stopped outside the handled interruption path.

Writes use a same-directory temporary file, `fsync`, and `os.replace`. A failed replacement leaves
the prior final file intact. Retention always keeps the record just written and the newest prior
records.

Evidence is projected onto a provenance allowlist: raw `value`, chunk/body/page text, and arbitrary
source metadata are not copied from normalized evidence. Credential-like mapping keys are
redacted, and URL query strings/fragments are removed. This is a safety boundary, not a general
data-loss-prevention system: free-form query, answer, limitation, and error strings are not scanned
for secrets or private quotations. Callers and prompts must therefore avoid placing credentials or
full private-document content in those free-form fields.

Not every incoming user request necessarily creates a trace. Resolution/planning failures and
invalid F14 contexts can stop before `create_research_trace` is called. The directory records F15
synthesis/validation workflows, not the complete lifetime of every pre-F15 request.

## 5. Expected scenario outputs

### Observed F1–F15 contracts

| Scenario | Expected route/result |
|---|---|
| Single technology company | `single`; grounded narrative; empty `scores_used`; mandatory no-comparison-score limitation |
| Single biopharma company | `single`; same boundaries as technology, with biopharma evidence dimensions |
| Same-profile technology peers | `same_profile`; deterministic technology score table only when requested and eligible; F14 may explain but not modify it |
| Same-profile biopharma peers | `same_profile`; deterministic 0–100 biopharma research-strength table only when requested and eligible |
| Technology plus biopharma | `cross_profile`; shared-financial qualitative comparison, separate sector findings, no universal score |
| Supported alias | Resolves to one canonical registry identity before research |
| Ambiguous company | Stops before fan-out and requests clarification |
| Unknown company | Stops before fan-out as unsupported |
| Partial tool/RAG failure | Preserves successful evidence, produces partial/failed dimensions and explicit limitations, and disables scoring |
| Invalid evidence ID | F15 validation fails and sends deterministic correction feedback to F14 |
| Modified F13 score | F15 score-fidelity validation fails; F14 cannot replace the authoritative table |

For a successful terminal result, `F15WorkflowResult.final_status == "success"`,
`validation.valid is True`, warnings are empty, and the corresponding trace is finalized as
`success`. Offline scenario outputs should be compact summaries and deterministic fixtures, not
claims about live market conditions.

The exact names, ordering, and printed summary fields of F16 scenarios are planned until the
offline fixture suite and runner are centrally reconciled.

## 6. Correction and recovery behavior

### Observed implementation

The boundary is:

```text
F14 synthesis → F15 deterministic validation → atomic trace update
                     ↘ invalid: tool-free F14 correction
```

The initial F14 attempt may be followed by at most two correction attempts. Validation feedback
contains the prior structured synthesis and deterministic errors. It explicitly forbids research,
tool calls, score calculation, new evidence IDs, and removal of mandatory limitations.

A passing attempt returns immediately. Retry exhaustion returns the latest draft only with an
explicit validation warning and `final_status="failed"`; it is not presented as validated. Invalid
retry budgets are rejected before model invocation. A company-worker failure does not erase
successful siblings, but incomplete comparisons cannot score.

F15 deterministically checks:

- cited evidence existence, current `run_id`, company/ticker/profile ownership, usable status, and
  duplicate ambiguity;
- exact ordered consistency between inline `[EV-*]` citations and `evidence_ids`;
- exact `scores_used` equality with authoritative F13 scores;
- recognizable `TICKER score N` and `TICKER ranked N` claims;
- no numeric scoring/ranking in `single` or `cross_profile` mode;
- mandatory limitations reconstructed from normalized results and eligibility.

It does not research, calculate a new score, or prove semantic entailment for every prose claim.
Failures before F15, live provider retries, and RAG-index recovery remain the responsibility of
their earlier workflow layers.

## 7. Known limitations

### Observed limitations

- The supported universe is the local registry, not an open security master. Healthcare support is
  biopharma only.
- Bare Roche is intentionally ambiguous because both `RHHBY` and `ROG.SW` are registered.
- The notebook is one local process; it has no distributed execution, multi-user isolation,
  durable workflow service, or production observability.
- Cross-profile numeric ranking is deliberately unavailable. Cross-profile answers are qualitative
  and retain sector context.
- Technology financial scoring is peer-rank based and loses magnitude information; it was retained
  for assignment compatibility. Its Buy/Hold/Sell labels are deterministic rubric output, not
  independently calibrated investment advice.
- Biopharma scores measure research-profile strength and are not investment recommendations.
- F15 validates explicit provenance and narrow score/rank patterns, not the semantic truth of every
  sentence or every possible phrasing of a numeric claim.
- F15's inline citation parser is designed for `[EV-*]` tokens; alternative citation styles are not
  equivalent.
- Local traces are redacted by field/key policy, not by semantic content classification. Free-form
  strings can still contain sensitive material if an upstream component places it there.
- Provider-backed results remain dependent on API availability, credentials, source freshness,
  external schemas, and local index readiness. Offline fixtures demonstrate control flow, not live
  factual accuracy.
- The default biopharma index contains only five starter tickers until the full corpus is built.
- `MemorySaver`, local JSON/cache files, and Chroma persistence are development conveniences, not
  production-grade durable state or access-controlled audit storage.

### Planned/deferred F16 behavior

- A deterministic end-to-end fixture suite and reusable scenario runner are in parallel
  development and must be reviewed before their exact outputs are documented as observed.
- Live smoke testing remains optional and configuration-gated. It must never become part of the
  default deterministic suite.
- Universal cross-industry scoring, production secrets management, distributed execution,
  authentication/authorization, centralized traces, and semantic claim verification are deferred.

## 8. Canonical-document discrepancies for main-agent reconciliation

1. Both canonical documents still label themselves **Proposed** even though their own F00–F15
   sections say **Implementation status: Complete**. Their document-level status should distinguish
   implemented F00–F15 from in-progress F16 and deferred production work.
2. The implementation plan's F16 test section names `tests/test_multiindustry_e2e.py`. Delegated F16
   ownership instead requires `tests/test_f16_end_to_end_scenarios.py`,
   `tests/test_f16_live_smoke.py`, and `scripts/run_f16_scenarios.py`.
3. The canonical F16 scenario list differs from the approved delegated coverage. It currently lists
   an explicit cross-profile ranking request, ambiguous company, unsupported profile, failed
   branch, and follow-up-memory case, but omits the delegated explicit alias-resolution, invalid
   evidence-ID, and modified-F13-score cases. Reconcile the canonical list with the final tests;
   retain additional cases only if they are actually executed.
4. The LLD routing matrix says an unsupported company/profile may use a
   “shared-financial-only fallback.” F02 currently routes every unsupported registry result to
   `stop_unsupported`; no such fallback is implemented.
5. The LLD says `.research_runs/` contains one trace “per request,” and the implementation plan says
   every attempt is inspectable. Actual trace creation starts only after `_f14_validate_context`
   succeeds inside `run_f15_validated_synthesis`; planner/resolver failures and invalid F14
   contexts may leave no trace.
6. Canonical trace wording can be read as guaranteeing that private-document content never appears
   anywhere in a trace. The evidence projection does exclude raw evidence values/content, but
   free-form `query`, F14 `answer`/limitations, validation errors, and terminal errors are not
   semantically scanned. Narrow the guarantee to provenance projection and key-based redaction.
7. The canonical implementation plan's top-level principle says “Make changes only” in the working
   notebook, but F00–F15 also use versioned integration scripts and deterministic tests. Clarify
   that only the canonical notebook may be edited among notebook artifacts; companion scripts,
   tests, and design docs are expected implementation assets.
8. The implementation plan's F16 requirements mention an optional live-integration **cell** only.
   The delegated F16 plan assigns a configuration-gated live smoke suite and scenario runner.
   Document both only if the main integration actually retains both surfaces.
9. Neither canonical document currently gives runnable local setup, focused offline commands,
   explicit live opt-in behavior, compact expected scenario summaries, or operator guidance for
   interpreting `success`, `failed`, `interrupted`, and `in_progress` trace states. F16 should add
   these after the runner interface is frozen.
10. The canonical scoring sections accurately document biopharma weights and boundaries but do not
    expose the concrete technology financial/signal weight tables or scaled Buy/Hold/Sell
    thresholds. Add them if the intent is for F16 documentation to be sufficient to reproduce and
    interpret both sector rubrics.


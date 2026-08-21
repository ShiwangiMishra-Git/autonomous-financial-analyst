# SCALE-02: Large-N Synthesis Scaling — High-Level Design

Status: **Design only, not built.** Extends `SCALE-01` (per-company orchestration) in
`docs/archive/Financial_Agent_Master_Development_Plan.docx`. Scope: **moderate scale,
~25-100 companies.** True large scale (100s-1000s) is a different order of problem and
stays under the master plan's `PROD-05` (deferred, production fan-out).

## 1. Problem Statement

`SCALE-01` fixes the *research* side of multi-company scale: run the existing,
already-hardened single-company agent (`create_financial_agent`, with NB-02 dedup,
NB-03/NB-04 citation validation, NB-05 capped correction loop) once per company, each in
its own fresh conversation. That directly targets the notebook's own documented
22-company failure - cramming 22 companies into one shared conversation caused
`analyze_sentiment` to be skipped entirely while the report still fabricated ~44
`[Source: analyze_sentiment]` citations.

`SCALE-01` is silent on what happens next: once you have N independently-validated
per-company reports, how do you combine them into one comparison? A single flat
"synthesize all N reports in one LLM call" step has the *same shape of risk* NB-02
through NB-05 just fixed one layer down - cram too much into one context and the model
starts dropping or inventing things, just at the synthesis step instead of the
tool-calling step. Nothing in the master plan addresses this today. This document scopes
that gap.

## 2. Non-Goals

- **N > 100 companies.** That's `PROD-05` (production per-entity fan-out, with isolated
  execution records, bounded concurrency after load/rate-limit testing) - deferred, not
  this document.
- **Full claim-level grounding** (verifying the *specific numeric claim* in the synthesis
  matches the specific source, not just that the source name exists). That's `PROD-04` -
  deferred. This document proposes a much cheaper, deterministic proxy instead (see
  §3.5).
- **Persistent execution ledger, multi-turn/turn-scoped evidence.** Still `PROD-01`/`PROD-02`,
  untouched by this design.
- **Changing `SimpleAgentState`** or adding new nodes to `create_financial_agent`'s graph.
  Per `SCALE-01`'s own constraint, orchestration stays external.

## 3. Design Axes: Options and Trade-offs

### 3.1 Execution strategy (how the N per-company agent runs happen)

| Option | Description | Trade-off |
|---|---|---|
| **Sequential (recommended default)** | `for company in companies: run(company)`, one at a time. | Simplest, safest for rate limits, no new failure modes. Wall-clock scales linearly with N - see §5 for real numbers. |
| Bounded concurrency | Run K companies at a time (thread pool / asyncio, K small, e.g. 3-5). | Cuts wall-clock roughly K-fold, but needs rate-limit testing against the Great Learning proxy first (unknown throttling behavior), and failure isolation gets harder (one company's error shouldn't kill others' in-flight runs). `SCALE-01` itself says "start sequentially; add bounded concurrency only after rate-limit testing" - same principle applies here, just at a slightly larger N. |
| Batched concurrency | Sequential batches of size K, wait for each batch to finish before starting the next. | Middle ground: bounds concurrent load without full async plumbing. Reasonable first step *if* wall-clock at N~100 sequential proves too slow in practice (see §5) - but not the default until that's actually observed. |

**Recommendation for this range:** stay sequential by default. Document bounded/batched
concurrency as the next lever, gated on real observed latency (§5), not built pre-emptively.

### 3.2 Synthesis strategy (combining N validated reports into one comparison)

| Option | Description | Trade-off |
|---|---|---|
| Flat single-call | One LLM call gets all N validated reports and produces the final comparison. | Simplest, best fidelity at the low end (roughly N <= 20, where the combined report text still fits comfortably and the model doesn't need to juggle too much at once). Same failure shape as the original 22-company bug - re-introduces "too much in one context" risk as N grows within this very range. |
| **Single-level batched map-reduce (recommended)** | Split N reports into batches of ~10-15. One LLM call summarizes each batch into a compact intermediate summary ("map"). One final LLM call synthesizes the batch summaries into the final comparison ("reduce"). | Bounds context at every step regardless of N (as long as N <= 100, one level of batching is enough - no need for a deeper tree). Adds `ceil(N / batch_size)` extra LLM calls, e.g. ~9 extra calls at N=100, batch size 12 - small next to the ~700-900 research-side calls already needed for that many companies (see §5). This is architecturally the same lesson NB-02-05 already taught: keep any single context bounded, whether it's tool calls or report synthesis. |
| Rolling / incremental fold | Maintain a running aggregate summary; fold in one company's report at a time (`summary = combine(summary, next_report)`). | Never holds more than one report + the running summary in context, but inherently sequential (harder to parallelize than batched map-reduce), and prone to recency bias / earlier-company detail loss the more folds it goes through - a real risk once N approaches 100. Documented, not recommended, for this range. |

**Recommendation for this range:** single-level batched map-reduce, batch size ~10-15
(tunable), as the default rather than an escape hatch - it's barely more complex than
flat synthesis and removes the re-introduced context-overload risk entirely.

### 3.3 Failure handling per company

| Option | Description | Trade-off |
|---|---|---|
| Fail-fast | Abort the whole batch if any single company's agent run fails or exhausts its NB-05 retry cap. | Simplest, but throws away N-1 good results because of 1 bad one - a bad trade at N=25-100. |
| **Retry-then-skip-and-continue (recommended)** | Retry a failed company's run a small, fixed number of times (reusing the same capped-retry philosophy as NB-05, just at the company level rather than the citation-correction level); if still failing, exclude it from synthesis and record it in a "failed companies" list surfaced alongside the final report. | Matches the project's existing "fail closed but don't discard good work" pattern (same spirit as NB-05's fail-closed branch, which was itself just fixed to stop discarding the model's real report). Costs a bit of extra latency/API spend per retried failure, bounded by the cap. |

### 3.4 Architecture placement

Orchestration (the per-company loop) and synthesis (the map-reduce step) are new,
plain top-level functions/cells - **outside** `create_financial_agent`. `SimpleAgentState`
is untouched; each company's run is just a normal, independent
`create_financial_agent(...).invoke(...)` call with its own `thread_id`. This mirrors
`SCALE-01`'s own explicit constraint ("keep orchestration outside `create_financial_agent`
so `SimpleAgentState` stays small") and needs no new graph nodes.

### 3.5 Synthesis-level grounding check

| Option | Description | Trade-off |
|---|---|---|
| None | Trust the synthesis output as-is. | Cheapest, but re-opens exactly the risk this whole design is meant to close - a fabricated company name or number could slip into the final comparison undetected. |
| **Lightweight sanity check (recommended)** | After synthesis, extract every company name/ticker mentioned in the final output and check each appears (substring/fuzzy match) in at least one of the N validated input reports. Flag - don't hard-block - anything that doesn't match. | Deterministic, cheap (no extra LLM call), and directly analogous to NB-04's "does the cited tool actually appear in `ToolMessage` records" pattern, just one level up: "does the cited *company* actually appear in a validated input report." Doesn't catch a wrong *number* attributed to a real company - that's `PROD-04`'s job, deliberately out of scope here. |
| Full claim-level grounding | Structured claim extraction + verification against source data, entity/metric/value/unit/period-aware. | This is exactly `PROD-04`. Correctly deferred - disproportionate effort for a moderate-scale notebook feature. |

## 4. Recommended Design (Summary)

- **Execution:** sequential loop over companies, one `create_financial_agent` invocation
  per company, its own `thread_id`. Bounded/batched concurrency documented as the next
  lever, not built now.
- **Per-company failure handling:** capped retries, then skip-and-continue; failed
  companies are reported by name/reason alongside the final synthesis, never silently
  dropped.
- **Synthesis:** single-level batched map-reduce, batch size ~10-15 validated reports per
  intermediate summary, then one final reduce call over the batch summaries.
- **Grounding:** lightweight post-synthesis check - every company mentioned in the final
  output must appear in at least one validated input report; flag mismatches.
- **Architecture:** all of the above lives in new orchestration code outside
  `create_financial_agent`; `SimpleAgentState` and the existing graph are untouched.

## 5. Rough Cost/Latency Table (Moderate Scale)

Based on this session's real, live Test 3 runs against the actual model (not estimated):
a single company's full research pass (tool selection -> `get_stock_price` /
`get_stock_history` / `search_financial_news` -> 5x `analyze_sentiment` -> draft ->
citation validation, 0-2 correction rounds) took **~35-90 seconds** and **~7-9 LLM calls**
per company (1 tool-selection call + 5 `analyze_sentiment` calls, each itself an LLM
call + 1 final draft + 0-2 correction rounds).

| N | Sequential wall-clock (approx.) | Total LLM calls (research only) | Extra calls for batched synthesis (batch=12) |
|---|---|---|---|
| 25 | ~15-38 min | ~175-225 | 3 (2 batches + 1 reduce) |
| 50 | ~30-75 min | ~350-450 | 5 (4 batches + 1 reduce) |
| 100 | ~60-150 min | ~700-900 | 9 (8 batches + 1 reduce) |

Two things this table makes concrete rather than abstract:

1. **Synthesis overhead is small relative to research cost** at any N in this range - a
   handful of extra calls against hundreds already needed. Batched map-reduce is close to
   free next to the per-company research cost, which is the real cost driver.
2. **Sequential wall-clock at N=100 (up to ~2.5 hours) is the practical threshold** where
   bounded/batched concurrency stops being optional in practice, even though it's
   documented here only as a next lever, not built now. Below roughly N=40-50, sequential
   is likely fine for a notebook/interactive context; above that, expect this to become
   the next real ask.

## 6. Future Build Order (Documentation Only - Not Built Now)

Mirrors how `NB-02` through `NB-05` were sequenced (design -> tests -> implement,
one item at a time):

```
SCALE-02a  Per-company orchestration loop (sequential, retry-then-skip)
  -> SCALE-02b  Single-level batched map-reduce synthesis
  -> SCALE-02c  Failed-companies reporting alongside final synthesis
  -> SCALE-02d  Lightweight synthesis-level grounding check
  -> (only if real N pushes past ~40-50 companies in practice) bounded/batched concurrency
```

## 7. Cross-References

- `SCALE-01 — Optional per-company orchestration`,
  `docs/archive/Financial_Agent_Master_Development_Plan.docx` - the per-company research
  fix this design extends.
- `PROD-04 — Claim-level grounding` - the full version of §3.5's lightweight check,
  deliberately deferred here.
- `PROD-05 — Production per-entity fan-out` - the true large-scale (100s-1000s),
  production-hardened version of this same problem, deliberately out of scope here.

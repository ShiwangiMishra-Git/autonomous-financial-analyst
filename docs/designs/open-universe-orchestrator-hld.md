# Open-Universe Orchestrator — High-Level Design

Status: **Design only, discussed, not built.** This is a **separate, alternative** design to
what's already implemented (`create_scoring_agent` and everything built around it after the
TEST 7 cell, this session). Nothing here replaces that yet — the two are kept side by side so
their trade-offs can be compared before deciding which one (or whether both) move forward. See
§5 for the comparison framing.

**Update**: following `docs/designs/Recommendation.md`'s guidance (reviewed critically in
`docs/designs/complexity-review-baseline-and-final-hld.md`), this design is generalizing toward
an **industry-aware** pattern rather than staying technology-only - see §9. This was blocked
pending a real second-industry document corpus; `content/pharma_rag_official_sources.zip` (14
pharma companies, official 10-K/Annual Report/quarterly filings, real provenance) resolves that,
so the generalization is now buildable, not just theoretical.

## 1. Problem statement

The existing system (`route_financial_query`, `create_scoring_agent`, `_extract_mentioned_tickers`)
is deliberately scoped to a fixed 5-company universe (`COMPANY_ALIASES` = MSFT/GOOGL/NVDA/AMZN/IBM),
with a deterministic ticker-count router and a scoring agent that produces a specific,
deterministic Buy/Hold/Sell output via `score_companies()`.

This design targets a different goal: **any public company, any number of them, and a synthesis
step that isn't limited to ranking** — reusable for ranking today, and for other kinds of
cross-company synthesis later, without hardcoding "rank these" into the prompt the way the old
standalone TEST 7 (Concurrent) cell's `ranking_model.invoke(ranking_prompt)` did.

## 2. What's reused as-is vs. what's new

Inventory taken before designing anything new — most of the system does **not** need to change:

| Piece | Status |
|---|---|
| `AGENT_CHARTER_WITH_RAG` / `create_enhanced_financial_agent` | Reused, with one addition — see §3.3a. Still the per-company deep-dive agent otherwise. |
| `get_stock_price`, `get_stock_history`, `search_financial_news`, `analyze_sentiment` | Already fully general-purpose — none of these are scoped to the 5-company universe; they work for any ticker today. |
| `query_private_database` | The one genuine constraint — a RAG retriever over a fixed PDF corpus. Out-of-corpus companies just get that section noted as a gap (existing charter language already covers "if AI research data is unavailable, explicitly state this gap") — the tool itself is unchanged; its *calls* are now also captured into state as they happen, see §3.3a. |
| `validate_citations` / `validate_citation_authenticity` | Reused unchanged — general-purpose, works on any report text regardless of company. |
| `_extract_mentioned_tickers` / `COMPANY_ALIASES` | **Not reused** — hardcoded to 5 companies, replaced by `extract_companies` below. |
| `create_scoring_agent` / `compute_scores_tool` / `score_companies` | **Explicitly out of scope for this design** — see §5. |

## 3. Architecture

```
extract_companies  →  [dynamic fan-out, one Send per company]  →  deep_dive (×N, parallel)  →  synthesize  →  END
```

### 3.1 `extract_companies` (the orchestrator's only real decision)

One bounded LLM call with structured output: "list every real, publicly-traded company/ticker
mentioned in this query." This is the **one place in the whole redesign where LLM judgment
genuinely earns its keep** — a fixed alias dict can't recognize arbitrary company names,
nicknames, or indirect references the way `_extract_mentioned_tickers` could get away with for
exactly 5 known aliases. Deliberately scoped as a single classification call, not a tool-calling
agent — same shape as `analyze_sentiment`/`extract_ai_signals` (bounded LLM judgment feeding
straight into code), not a new agentic loop.

**Open item, not yet decided:** exact structured-output shape / prompt wording for this
extraction call.

### 3.2 Dynamic parallel fan-out

Uses LangGraph's `Send` API — the mechanism that actually replaces TEST 7 (Concurrent)'s
hardcoded `ThreadPoolExecutor` over a fixed list. A routing function returns a list of
`Send("deep_dive", {"ticker": t, "query": query})`, one per extracted company, however many
there are. LangGraph spawns each as an isolated sub-invocation, runs them concurrently (built
into the graph engine's own execution model), and automatically waits for all of them (fan-in)
before `synthesize` runs.

### 3.3 `deep_dive` node (×N, parallel, fully reused logic)

```python
def deep_dive_node(state):
    ticker = state["ticker"]
    try:
        agent = create_enhanced_financial_agent(with_rag=True, with_memory=False)
        query = f"Provide a comprehensive investment analysis for {ticker} including their AI research initiatives"
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        # result IS the graph's full final state, not just messages - no separate get_state()
        # call needed (and none would be possible anyway: with_memory=False means no
        # checkpointer exists to retrieve state from). average_sentiment/ai_signals
        # computed/validated by the agent's own graph - see §3.3a.
        narrative = result["messages"][-1].content
        status = "validation_failed" if "[VALIDATION FAILED" in narrative else "complete"

        financial_evidence = {
            "price": get_stock_price(ticker),         # current_price, day_high, day_low, volume, ...
            "history": get_stock_history(ticker),      # start_price, end_price, return_pct, ...
            "metrics": get_financial_metrics(ticker),  # market_cap, pe_ratio, beta, dividend_yield, ...
        }
        sentiment = result.get("average_sentiment")
        industry_signals = _wrap_industry_signals(ticker, "technology.ai.v1", result.get("ai_signals"))

        scoring_eligible = all([
            financial_evidence["price"].get("status") == "success",
            financial_evidence["history"].get("status") == "success",
            financial_evidence["metrics"].get("status") == "success",
            sentiment is not None,
            industry_signals is not None and industry_signals["status"] == "success",
        ])

        return {"results": {ticker: {
            "ticker": ticker,
            "report": narrative,
            "financial_evidence": financial_evidence,
            "sentiment": sentiment,
            "industry_signals": industry_signals,
            "status": status,
            "scoring_eligible": scoring_eligible,
        }}}
    except Exception as e:
        return {"results": {ticker: {
            "ticker": ticker, "report": f"[DEEP DIVE FAILED for {ticker}: {e}]",
            "financial_evidence": {}, "sentiment": None, "industry_signals": None,
            "status": "deep_dive_failed", "scoring_eligible": False,
        }}}
```

The narrative half (`create_enhanced_financial_agent`) is nearly identical to the existing
`_generate_enhanced_report` — reused, not reimplemented. Financial inputs
(`get_stock_price`/`get_stock_history`/`get_financial_metrics`) are called **directly by this
code, not offered to the agent's own LLM as optional tools** — deliberate: making them tools the
model *decides* whether to call would mean a missing scoring input could mean "the model judged
it unnecessary" instead of "genuinely unavailable," which breaks §6.3's whole "Not Scored"
honesty guarantee. They can *additionally* be added to the agent's own tool list if useful for
narrative grounding — harmless, since caching makes the direct call here a cache hit either way —
but this direct call is what guarantees the data exists regardless of what the agent chose to do.

Sentiment and AI signals are handled differently on purpose - **not** called directly by this
code the way financial data is, but computed *inside* the agent's own graph, from tools it's
already charter-mandated to call anyway (`analyze_sentiment`, `query_private_database`) - see
§3.3a for why and how. The `try/except` around the whole body is **new and load-bearing** — see
§3.5.

**Why `financial_evidence` is nested by source (`price`/`history`/`metrics`) instead of a
flat merge**: `get_stock_price`, `get_stock_history`, and `get_financial_metrics` share
overlapping key names - all three have `status`, two have `timestamp`, two have `market_cap`. A
flat `{**a, **b, **c}` merge would let whichever call is spread last silently win on those shared
keys - if, say, `get_stock_history` failed but the other two succeeded, the merged `status` field
would show whatever the last-spread call returned, hiding the partial failure and silently
dropping `get_stock_history`'s fields with no indication why. Nesting by source keeps one field to
look at (`results[ticker]["financial_evidence"]`) while keeping each sub-call's own status
independently checkable - no collision possible. This is also exactly what `scoring_eligible`
above checks explicitly (`status == "success"` on all three), rather than inferring completeness
from field presence.

Kept `price`/`history` even though `score_companies()`'s `METRIC_NAMES` (`market_cap`,
`total_revenue`, `pe_ratio`, `beta`, `dividend_yield`) only ever come from `get_financial_metrics`
- scoring only ever reads the `metrics` sub-key. `price`/`history` are captured anyway for
narrative/reference completeness and future non-ranking use cases (explicit decision - "we might
need it," not itself required by scoring).

**`_wrap_industry_signals(ticker, profile_id, raw_signals)`** - a small helper, not a new
mechanism: wraps whatever `extract_ai_signals` (or, later, the pharma extractor) actually
returned into the common `IndustrySignalResult` shape (§3.4) - `status="success"` if
`raw_signals` is a real non-empty dict, `status="missing"` with `industry_signals=None`
otherwise. This is structural alignment #2 (see the implementation-plan discussion this section
follows from) - every extractor produces the same envelope shape regardless of profile, so
`synthesize`/scoring code never needs a per-profile branch to read it.

### 3.3a `create_enhanced_financial_agent` additions - capture sentiment and AI signals as they're gathered, not again afterward

Sentiment and AI signals are both handled differently from financial data (deliberately kept
*outside* the agent's own tool-calling, §3.3), and for the same underlying reason: `analyze_sentiment`
and `query_private_database` are already the agent's existing, charter-mandated tools
(`AGENT_CHARTER_WITH_RAG` already requires calling `analyze_sentiment` once per article, and
calling `query_private_database` for AI research activity) - neither is a new optional tool being
added. Given that, doing a *separate* gathering pass afterward for either one (a standalone
`get_average_sentiment(ticker)` call, or an independent `extract_ai_signals([ticker])` retriever
query) has the same real downside in both cases: the separate pass can surface *different*
evidence than whatever the agent's own tool calls actually used for the narrative - `get_average_sentiment`
via its fixed `f"{ticker} AI news"` query, `extract_ai_signals` via its own independent retrieval
- meaning the written report and the score could silently end up grounded in different evidence.
Capturing both directly, as the agent's own calls happen, fixes this at the source for both and
avoids a redundant second gathering pass in either case.

**State additions to `SimpleAgentState`** (the state type `create_enhanced_financial_agent`
already uses):

```python
scored_articles: Annotated[Dict[str, Dict], _merge_by_ticker]  # article_hash -> {"text":..., "score":...}
average_sentiment: Optional[Dict]  # {"average": float, "articles": {...}} once computed
rag_queries: Annotated[Dict[str, Dict], _merge_by_ticker]  # query_hash -> {"query":..., "answer":...}
ai_signals: Optional[Dict]  # extract_ai_signals()'s per-signal shape, once computed
```

**Tool node - sentiment**: whenever an `analyze_sentiment` call succeeds, in addition to the
normal `ToolMessage`, write `{hash(text): {"text": text, "score": result["score"]}}` into
`scored_articles`. Deliberately **no `title`/`url` captured at this point** - `analyze_sentiment`'s
own signature only ever receives raw text, not the article it came from; reconstructing
`title`/`url` here would require matching that text back against an earlier
`search_financial_news` result by content, the same kind of fragile text-matching-against-
conversation-history that's caused repeat bugs this session. This field exists purely to support
the *average* calculation - citations continue to come from the narrative text itself, which the
agent writes with full access to real article metadata, unaffected by this change. The hash used
is the same one `analyze_sentiment`'s own `@cached_call` decorator already computes internally
(SHA-256 of the text) - not a new identity, just surfaced.

**Tool node - AI signals**: whenever a `query_private_database` call succeeds, write
`{hash(query): {"query": query, "answer": result}}` into `rag_queries`. Keyed by hash of the
*query* text this time (not the answer) - same dedup reasoning as `scored_articles`: if the agent
happens to ask the same question twice, it collapses to one entry instead of duplicating.

**New validator, folded into the existing `citation_validator_node`** (not two separate new
nodes - both checks run at the same point in the graph, after the agent stops calling tools, so
folding all of it into one node keeps the graph from growing extra nodes for what's really one
"is this answer complete and correct" gate with more conditions added):

1. Sentiment: if `state["average_sentiment"]` is already set, nothing to do. Otherwise, if
   `scored_articles` has real entries, compute the average **deterministically in code** -
   `sum(a["score"] for a in scored_articles.values()) / len(scored_articles)` - never ask the LLM
   to do this arithmetic. Store it, proceed. Otherwise (`scored_articles` is also empty - the
   agent never successfully called `analyze_sentiment` at all), capped retry: inject a correction
   message and route back to `agent`.
2. AI signals: if `state["ai_signals"]` is already set, nothing to do. Otherwise, if
   `rag_queries` has real entries, run `extract_ai_signals`'s existing rubric classification -
   this **does** need its own LLM call (unlike sentiment's arithmetic, "none/partial/full" is a
   genuine judgment task, not something code can compute) - grounded in the captured Q&A pairs
   specifically, not the whole narrative and not an independent retrieval: build a text blob from
   `rag_queries` ("Q: ...\nA: ...\n\nQ: ...\nA: ...") and pass it as `extract_ai_signals([ticker],
   prior_reports={ticker: qa_text})`, reusing the function's existing `prior_reports` grounding
   parameter rather than adding a new one. Tighter grounding than the whole narrative would be -
   `rag_queries` is already scoped to AI-research questions specifically, not diluted by
   unrelated financial/sentiment/risk content. Otherwise (`rag_queries` is also empty - the agent
   never called `query_private_database` at all), same capped retry shape.

Both retries use the same mechanism as `citation_validator_node`'s existing loop
(`MAX_CITATION_RETRIES`), not a new kind of mechanism - just added conditions on an already-proven
one. Doesn't fully solve §7.2 (the retriever could still return the wrong company's chunks for an
out-of-corpus company, if `query_private_database`'s own retrieval was already wrong) - it does
guarantee the narrative and the score can't independently disagree about what was found, which is
the part this mechanism actually targets.

`deep_dive_node` (§3.3) then reads `result.get("average_sentiment")` and `result.get("ai_signals")`
directly from the agent's own final state - no second `get_average_sentiment(ticker)` or
`extract_ai_signals([ticker])` call needed at all.

### 3.4 State schema

**Update — restructured for consistency with `open-universe-orchestrator-final-hld.md` as a
north star** (see the implementation-plan discussion, "structural alignment #1/#2"): the four
separate ticker-keyed dicts this section originally had (`reports`/`financial_metrics`/
`sentiment_scores`/`ai_signals`) are replaced by one `results` field holding a unified
`CompanyResult` per ticker, with an explicit `status` - the same shape the production HLD's
`ResearchState.worker_results`/`CompanyResult` use, adopted here not for consistency's own sake
but because it's strictly more robust: `synthesize` and the scoring step can now check
`result["status"]`/`result["scoring_eligible"]` directly instead of scanning report text for
`"[DEEP DIVE FAILED"`/`"[VALIDATION FAILED"` substrings (§3.5/§3.6 updated accordingly). Nothing
about the *behavior* designed in §3.5-§3.9 or §6 changes - only how a company's outcome is
represented in state.

```python
class IndustrySignalResult(TypedDict):
    ticker: str
    profile_id: str              # "technology.ai.v1" | "pharma.biopharma.v1" (§9)
    dimensions: Dict[str, Dict]  # {"infrastructure_moat": {"level":..., "reason":..., "sources":...}, ...}
    missing_dimensions: List[str]
    status: str                  # "success" | "partial" | "missing" | "failed"

class CompanyResult(TypedDict):
    ticker: str
    report: str                       # narrative text, or the failure detail if status != "complete"
    financial_evidence: Dict          # {"price":..., "history":..., "metrics":...}
    sentiment: Optional[Dict]         # average_sentiment (§3.3a), or None
    industry_signals: Optional[IndustrySignalResult]
    status: str                       # "complete" | "deep_dive_failed" | "validation_failed"
    scoring_eligible: bool            # true only if financial+sentiment+industry_signals all succeeded

class ResearchState(TypedDict):
    query: str
    companies: List[str]
    results: Annotated[Dict[str, CompanyResult], _merge_by_ticker]  # ticker -> CompanyResult
    synthesis: str

def _merge_by_ticker(existing: dict, new: dict) -> dict:
    return {**existing, **new}
```

One reducer, for the single ticker-keyed `results` field (generically named for reuse elsewhere,
not `_merge_reports` as first drafted). Dict-keyed (not `List[Dict] + operator.add`), chosen over
the plain-list version discussed first because `synthesize` and the scoring step both need to
look up data by ticker directly, and a list would need scanning each time. Requires a small
custom reducer since Python's `+` isn't defined between two dicts the way `operator.add` needs.

**Why a reducer is required at all** (established during discussion, worth keeping the reasoning
here): each `deep_dive` branch is a fully isolated invocation that only knows its own ticker —
it can't see its siblings' results or what's already accumulated. Without an explicit reducer,
LangGraph's default behavior for a state key is "last write wins," meaning N parallel branches
writing to the same key would silently overwrite each other, leaving only the last-finishing
company's contribution — silent data loss, not an error. `_merge_by_ticker`/`operator.add`-style
reducers are what make the fan-in half of map-reduce actually accumulate instead of racing.

**Why dedicated fields instead of extracting from `messages`**: the earlier alternative
(offering these three as agent tools and parsing their results back out of
`result["messages"]` after the fact) has a real extraction cost — scanning for the right
`ToolMessage`, checking its name, parsing JSON, handling it appearing zero or multiple times -
the same kind of "reconstruct reliably from conversation state" logic (`_find_tool_result`,
`get_successful_tool_names`) that's been a repeat bug source this session. Calling the functions
directly and returning their values as dedicated state keys sidesteps that entirely - plain dict
access, nothing to parse.

### 3.5 Failure handling — two distinct modes, treated the same way downstream

**Mode 1 — citation validator fails closed for one company.** `create_enhanced_financial_agent`'s
own `citation_validator_node` exhausts its 2 correction attempts → returns a report with an
embedded `[VALIDATION FAILED after 2 correction attempt(s) - ...]` tail. No exception; `deep_dive`
completes normally, and (§3.3's updated code) sets `status="validation_failed"` on the
`CompanyResult` by checking for that tail once, at the source - not left for every downstream
reader to re-detect via its own substring search.

**Mode 2 — an actual unhandled exception inside one `deep_dive` branch.** This is the mode that's
genuinely new and higher-stakes in a *parallel*, *open-ended-N* design specifically: LangGraph's
execution model runs all `Send`-dispatched branches in the same superstep, and if any single node
task raises uncaught, the whole superstep fails and the exception propagates through `.invoke()`
— losing every other company's already-completed work too, not just the one that broke. This risk
scales with N in a way it didn't in the old sequential, small-N (3-5 company) design, where a
crash had the same underlying problem but a much smaller blast radius. Fixed by the `try/except`
in §3.3 — a failure produces a `CompanyResult` with `status="deep_dive_failed"` and the exception
detail in `report`, never an exception that can take the batch down.

**Decision: both modes are treated identically by `synthesize`.** Tempting to think a
`validation_failed` report is still partially usable (the narrative text is there above the
tail) — rejected, because `validate_citation_authenticity` specifically catches *fabricated*
citations (a tag for a tool that was never actually called), so a validation-failed report could
contain claims that were never real. Given this session's rigor about never letting a fabricated
claim through, `synthesize` is instructed to use *no* content from any `CompanyResult` whose
`status != "complete"`, full stop — not attempt a confidence-adjusted partial use. This check is
now a field comparison, not text-marker detection - the marker still lives in `report`'s text
(useful for the verbatim reference section, §3.9), but nothing downstream needs to re-parse it.

### 3.6 Gap presentation in the final report

Explicit, required output section — same convention as the existing "Gaps & Limitations" sections
in both `AGENT_CHARTER_WITH_RAG` and `SCORING_AGENT_CHARTER`:

```
## Data Gaps
The following companies could not be included in this analysis:
- TICKER: [reason, taken from the failure marker]
```

Building this is now code, not a prompt-detection instruction: before calling the model,
`synthesize` partitions `results` by `status` -
`usable = {t: r for t, r in results.items() if r["status"] == "complete"}` and
`excluded = {t: r for t, r in results.items() if r["status"] != "complete"}` - and only passes
`usable` reports into the model's context at all. The model physically cannot use an excluded
company's claims, because they were never in its prompt - stronger than instructing it not to.
`excluded`'s tickers and `report` text (which still carries the `[DEEP DIVE FAILED ...]`/
`[VALIDATION FAILED ...]` detail) are used directly to build the "Data Gaps" section in code too.

**Verification layer (lightweight, deterministic):** for every ticker in `excluded`, confirm the
final synthesis text mentions that ticker at all (`ticker in final_text`). Doesn't verify the
*quality* of the acknowledgment — just catches total silent omission, which would be worse than
an explicit exclusion (the user has no idea their requested company just vanished). Not yet
built; agreed in discussion as worth adding once `synthesize` exists to test it against.

### 3.7 Single-company short-circuit

If `len(companies) <= 1`, skip `synthesize` entirely and return that one report directly — same
deterministic short-circuit `route_financial_query` already uses today. No LLM decision needed;
company count has exactly one correct answer here, same reasoning applied consistently throughout
this session.

### 3.8 Ranking vs. "just tell me about each separately"

The ambiguous case: 2+ companies extracted, but the query didn't actually ask for a
comparison/ranking (e.g. "tell me about MSFT and GOOGL separately" vs. "compare MSFT and GOOGL").

- **Layer 1**: `synthesize`'s prompt is *"address the user's original request using these
  reports"*, not a hardcoded "produce a ranking" template (unlike old TEST 7's
  `ranking_prompt`) — structurally already pointed at doing the right thing.
- **Layer 2**: made explicit rather than left to inference — *"If the user's request asks for a
  comparison/ranking, produce one. If it doesn't, present each company's findings separately and
  do not manufacture a ranking that wasn't requested."*
- **No deterministic verification for this one** — unlike the citation/recommendation checks
  built for the existing scoring agent (clean, binary, mechanically checkable facts), "did the
  model correctly avoid producing an unwanted opinion" is a soft, interpretive judgment with no
  clean substring check. Acknowledged as the one piece of this design that rests on prompt
  quality alone, without a deterministic backstop — a genuine gap relative to this session's
  usual standard, noted rather than resolved.

### 3.9 Output structure — grounded top, deterministic reference bottom

```
[TOP]    query-grounded response — format varies by request type (ranking format: §6)
[BOTTOM] reference section — full per-company reports, ALWAYS present, including failed ones
```

**The bottom section is assembled by plain code, never generated by the LLM:**

```python
response_top = model.invoke([...])  # the ONLY LLM call in synthesize - generates just the top,
                                     # from `usable` only (§3.6) - excluded companies never enter the prompt
reference_section = "\n\n---\n\n## Reference: Full Company Reports\n\n" + "\n\n".join(
    f"### {ticker}\n{r['report']}" for ticker, r in results.items()  # ALL companies, usable + excluded
)
final_output = response_top.content + reference_section
```

This changes the shape of the problem in a useful way:

- **The reference section can never be corrupted** — it's the exact, already-validated
  `deep_dive` output, concatenated verbatim, with zero LLM involvement in that half of the
  output. No risk of a number getting silently altered or a citation getting dropped while
  "restating" it.
- **Includes failed companies too, verbatim** (`[DEEP DIVE FAILED ...]` / `[VALIDATION FAILED
  ...]` and all) — full transparency over silently dropping them, and it's free since nothing
  is regenerated. The top section still gets its own separate, required Data Gaps
  acknowledgment (§3.6) — the bottom's inclusion of the raw failure text is a transparency/audit
  aid, not a substitute for that.
- **"Grounded" becomes partly structural, not just a prompt request** — the top section's job
  is to be consistent with what's verifiably sitting right below it, which a reader (or a future
  check) can cross-reference directly against the untouched source text.
- **Resolves the citation-preservation open item below**: since the LLM's own output is now only
  the top section, and the underlying facts are always available unmodified right below it, a
  separate "did citations survive synthesis unchanged" check is far less critical than it looked
  before this split — the source of truth is right there, unaltered, either way.

## 4. Open items (not yet decided)

- Exact structured-output shape/prompt for `extract_companies`.
- Whether to build the §3.6 verification check before or after seeing whether the prompt
  instruction alone holds up against real model output.
- The pharma rubric/dimensions (§9.3) and whether pharma reuses `extract_ai_signals`'s exact
  pipeline shape or needs adjustment for 10-K/Annual Report structure.

## 5. Trade-offs vs. the existing `create_scoring_agent` design — to discuss later

Framing only, not resolved here — flagged dimensions worth comparing once both are further along:

| Dimension | Existing (`create_scoring_agent` + router) | This design |
|---|---|---|
| Company scope | Fixed 5, zero-cost deterministic routing | Open-ended, needs an LLM extraction call |
| Scoring | Deterministic `score_companies()` — real Buy/Hold/Sell math, nothing fabricable, always available for all 5 companies (corpus covers exactly those 5) | Same `score_companies()` reused unchanged (§6.2) — but only for companies with complete data on all 3 inputs; out-of-corpus companies get explicit "Not Scored" (§6.3) instead of a number, and non-ranking requests get the qualitative-only mode (§6.1) with no score at all |
| Maturity | 8 real bugs found and fixed via live-kernel testing this session; currently stable | Unbuilt — bug surface unknown |
| Concurrency | Deep dives currently sequential (`{t: _generate_enhanced_report(t) for t in tickers}`) | Native parallel fan-out via `Send`, scales to arbitrary N |
| Extensibility | Adding a new capability means extending router dispatch logic | Adding a new synthesis *purpose* just means a different query, same graph shape |
| Verification rigor | Every fidelity dimension (recommendation match, citation tags, sentiment dual-citation) has a deterministic check | Most failure/gap handling has deterministic checks (§3.5, §3.6); the ranking-intent question (§3.8) does not |

## 6. Ranking format (top section, ranking use case)

### 6.1 Two ranking modes: qualitative (always available) and scored (conditional)

**Qualitative ranking** — always possible, uses only the deep-dive narrative, no new data needed:

```markdown
## Ranking

**Criterion**: [stated explicitly, derived from the user's actual query - e.g. "overall AI
investment attractiveness, weighing financial health, market sentiment, and AI research depth" -
not always the same axis, since the query could ask for "safest" or "highest growth potential"]

1. **TICKER** — [2-3 sentences grounding the ranking in specific claims from this company's own
   deep-dive report: which financial metric, sentiment signal, or AI-signal dimension is doing
   the work for this position]
2. **TICKER** — [...]

## Data Gaps
[unchanged from §3.6 - failed/excluded companies, reason stated]
```

Deliberately **no Buy/Hold/Sell-style labels or invented confidence numbers** here - the old
scoring agent's recommendation carries real weight because it's backed by actual deterministic
math (`score_companies()`). This ranking has no such backing; it's an LLM's comparative judgment
over prose reports. Using the same vocabulary would visually imply rigor this doesn't have, so
rank position + a stated criterion + grounded prose reasoning is the entire signal - nothing
numeric attached.

### 6.2 Scored ranking - deterministic, reusing the old design's math unchanged

The insight that makes this possible without reinventing anything: the deep-dive report's numbers
didn't come from nowhere - they came from real tool calls that already ran during `deep_dive`.
Rather than parsing figures back out of the narrative's prose (fragile - the same class of
text-matching problem this session hit repeatedly with citation regexes), capture the structured
data **at the source**, separately from the narrative, using functions the old design already
built and verified as general-purpose:

| Score component | Source (`results[ticker][...]`, §3.4) | Reusable for open universe? |
|---|---|---|
| Financial | `financial_evidence["metrics"]` (from `get_financial_metrics(ticker)`) | Yes - already ticker-agnostic |
| Sentiment | `sentiment` - computed by `create_enhanced_financial_agent`'s own graph, not a separate call (§3.3a) | Yes - ticker-agnostic, and grounded in the same articles as the narrative |
| AI signals | `industry_signals["dimensions"]` - computed by the agent's own graph, grounded in captured `rag_queries` (§3.3a), not a separate call | Only for in-corpus companies - depends on `query_private_database`'s fixed PDF set |
| The math itself | `score_companies(financial_metrics, ai_signals, sentiment_scores, risk_profile)` | Yes - pure function, doesn't care what universe the tickers came from |

`deep_dive_node` (§3.3) already populates all three directly into each `CompanyResult` - the same
way `fetch_financial_metrics_tool`/`gather_sentiment_tool`/`extract_ai_signals_tool` already do
for the old design, just as fields of one unified per-ticker object instead of three separate
ones. A plain deterministic function (not an LLM call) then re-shapes `results` into the three
dicts `score_companies()`'s existing signature expects (`{t: r["financial_evidence"]["metrics"]
for t, r in results.items()}` etc.) and runs it unchanged, once all N companies' data has landed.
Almost nothing new to build for the math; only the open-ended fan-out around it is new.

### 6.3 All-or-nothing per company - no reweighting, ever

**Considered and rejected**: if a company is missing one input (most commonly AI signals, for an
out-of-corpus company), renormalize `score_companies()`'s weights across only the available
dimensions so it still produces *a* number.

**Why this was rejected**: it doesn't fabricate a fake *value* for the missing piece, but it
fabricates *comparability*. A reweighted 2-dimension score sitting next to a real 3-dimension
score in the same ranked list looks like it means the same thing, but was computed by a genuinely
different formula for that one company - misleading even without inventing any single fake number,
since the reader has no way to tell the difference just by looking at the result.

**The rule, applied uniformly to all three inputs (financial, sentiment, AND AI signals - not
just AI signals specifically)**: a company only ever receives a numeric score if all three are
real and complete. This is the exact same all-or-nothing philosophy `compute_scores_tool` already
enforces in the existing design (`if fm is None: missing.append(...)` etc., refusing rather than
guessing) - just applied per-company here instead of once for a whole batch. Since §3.4's update,
this rule is no longer something the scoring step has to re-derive - `CompanyResult["scoring_eligible"]`
*is* this rule, computed once at the source in `deep_dive_node`: `## Scores` is built from
`{t: r for t, r in results.items() if r["scoring_eligible"]}`, `## Not Scored` from the rest.

**Financial specifically requires all three sub-calls to have succeeded** (`price`, `history`,
`metrics` - not just `metrics`, even though `metrics` is the only one `score_companies()` itself
reads) - exactly what `scoring_eligible`'s `all([...])` check in §3.3 enforces. Decided explicitly:
`price`/`history` are kept in state for future non-ranking use cases ("we might need it"), and
completeness there should mean the same thing it means everywhere else in this design - real data
or an honest gap, never a silently half-populated `financial_evidence` entry that *looks* complete
because `metrics` alone succeeded.

```markdown
## Scores
(financial + AI signals + sentiment - all three required)

1. **MSFT** — Score: 5.18 (Buy)
2. **GOOGL** — Score: 5.18 (Buy)

## Not Scored
The following companies have a full deep-dive report (see reference section) but could not
receive a numeric score:
- **TICKER**: AI signal data unavailable - not covered by the private research corpus
```

**"Not Scored" is distinct from "Data Gaps" (§3.6)** - a company can have a perfectly good
deep-dive report (real financial data, real sentiment, real narrative) and still not get a score,
purely because one dimension - commonly AI signals, but the same rule applies if financial or
sentiment data were ever unavailable too - requires something this company doesn't have. "Data
Gaps" means no usable report at all; "Not Scored" means a good report but an incomplete score
input. Keeping them separate matters because they mean different things to the reader.

## 7. Known gaps in the reused scoring-input functions — to fix at implementation time, in order

Surfaced while walking through §6.2's three scoring-input functions in detail. Both affect the
**existing** `get_average_sentiment`/`extract_ai_signals` functions directly - shared by the
already-built `create_scoring_agent` pipeline too, not just this design - so fixing them benefits
both, though they matter more here since an open-ended universe hits these edge cases far more
often than the fixed 5-company universe does. Not fixed yet - deferred to implementation. Fix
order: **#1 first, then #2.**

### 7.1 `get_average_sentiment` silently fabricates a neutral score on missing data

Currently, on *any* exception, or simply when zero articles come back, it returns
`{"average": 0.5, "articles": []}` - a fabricated default, not an explicit error marker. A
company with genuinely zero real news coverage becomes indistinguishable from one that actually
scored 0.5 from 3 real articles. This is exactly the "fabricated comparability" problem §6.3
already rejected for AI signals - it just already exists, silently, in a function both designs
depend on. Rarely bites the old design (the fixed 5 companies always have real news coverage);
would bite an open universe far more often (obscure companies with no coverage).

**Fix direction (not yet implemented):** add an explicit status field to the return so callers
can check it directly rather than inferring "no data" from an empty `articles` list (implicit
conventions like that are exactly what caused several bugs earlier this session - see
SESSION_NOTES.md). Checked `score_companies()`'s handling before deferring this: `_rank_scores()`
already treats a `None` metric value as "missing, scores zero on that metric" without crashing -
`valid = {t: v for t, v in values.items() if v is not None}` - so changing the `average` value to
`None` (instead of a fabricated 0.5) when there's genuinely no data would NOT break the existing
scoring math downstream. Useful confirmation for whenever this gets implemented.

### 7.2 `extract_ai_signals` can confidently ground itself in the wrong company's documents

For an out-of-corpus company, the retriever doesn't necessarily fail cleanly - vector retrievers
typically return the *closest* matches regardless of whether anything is actually relevant,
unless similarity-thresholded. The LLM classification step could then produce confident
"full/partial" levels grounded in some *other* company's PDF content that happened to be the
nearest match - a fabricated result, not an absent one, and nothing currently detects this.
**No fix direction decided yet** - candidates to evaluate at implementation time: checking
similarity scores if the retriever exposes them, or checking retrieved chunks' source-document
metadata against the target ticker before trusting the classification.

## 8. Follow-up support

**What this means concretely**: after an initial answer ("rank MSFT, GOOGL, NVDA"), a related
follow-up in the same conversation ("what about their debt levels?", "add AMZN to that") should
work without re-stating the companies - "their"/"that" resolved from actual conversation memory,
not treated as a cold start each time. Previously discussed as purely hypothetical (§ - the
MemorySaver mechanics turn); now being designed for real.

### 8.1 The MemorySaver mechanics (as discussed)

Same pattern as the old design's `route_financial_query`, applied to the top-level orchestrator
graph as a whole (not any single node inside it):

```python
def create_research_orchestrator(memory: MemorySaver = None):
    workflow = StateGraph(ResearchState)
    # ... extract_companies, deep_dive (via Send), synthesize ...
    return workflow.compile(checkpointer=memory or MemorySaver())

_ORCHESTRATOR_MEMORY_CACHE: Dict[str, MemorySaver] = {}

def handle_query(query: str, conversation_id: str):
    if conversation_id not in _ORCHESTRATOR_MEMORY_CACHE:
        _ORCHESTRATOR_MEMORY_CACHE[conversation_id] = MemorySaver()
    orchestrator = create_research_orchestrator(memory=_ORCHESTRATOR_MEMORY_CACHE[conversation_id])
    config = {"configurable": {"thread_id": conversation_id}}
    return orchestrator.invoke({"query": query}, config=config)
```

Graph structure rebuilds fresh every call (cheap, prompt changes take effect immediately); only
the conversation history inside the cached `MemorySaver` instance actually persists. Keyed by
`conversation_id` specifically, not by extracted companies - unlike the old design, the company
list here isn't known until *after* `extract_companies` runs *inside* the graph, so it can't be
the cache key the way `(sorted tickers, risk_profile)` was for `create_scoring_agent`.

### 8.2 The question this raises: does a follow-up reuse per-ticker state, or regenerate it?

This is the one genuinely tricky design decision, and it's what the staleness discussion (§ two
turns before this) was actually circling. Two options:

**(a) Reuse accumulated state** - if `results["MSFT"]` already exists from turn 1, skip
`deep_dive` for MSFT on a follow-up that implicitly still concerns it. Cheaper, but this is exactly
where the staleness bug lives: `sentiment`/`industry_signals` have no TTL of their own once
sitting in graph state (unlike the disk cache layer, which does) - a long-running conversation
would keep serving turn-1 data indefinitely with nothing to force a refresh.

**(b) Always regenerate** - every fresh top-level turn re-runs `extract_companies` and re-fans-out
`deep_dive` for whatever companies it identifies (whether newly named or inferred from context),
overwriting the accumulator fields rather than reusing them. **This is the recommended design.**
It sidesteps the staleness problem entirely rather than managing it - there's no stale per-ticker
data to accidentally serve, because per-ticker data is never carried forward as data. What
*does* carry forward is the conversation history itself (`messages`), which is what actually
matters for resolving "their"/"that" - `extract_companies` and `synthesize` both read prior
turns from the persisted thread to understand context, they just don't skip re-gathering because
of it. Freshness is still cheap in practice, since every underlying call
(`get_financial_metrics`, etc.) is already disk-cached with its own real TTL (§ api_cache_def) -
"always regenerate" at the orchestrator level mostly just means "always ask," not "always pay for
a live API call."

Mechanically, this means `results`/`companies` get reset to empty at the start of each fresh
top-level turn - same reset pattern already used for
`validation_retry_count`/`tool_round_count` elsewhere in this notebook (detected the same way: a
genuine new `HumanMessage`, not an internally-injected correction message). `messages` is the only
field that's actually allowed to accumulate across turns.

### 8.3 Why `extract_companies` needs conversation history, not just the current query

For a follow-up like "what about their debt levels?" with no company names in it,
`extract_companies` can't extract anything from the current query text alone - it needs to see
the prior turn(s) to resolve "their." This falls out for free once the graph has a real
checkpointer (§8.1): the node already receives the full accumulated `messages` list for the
thread, not just the latest one, so grounding the extraction prompt in "the whole conversation so
far," not just the newest message, is a prompt-shape decision, not a new mechanism.

## 9. Industry-aware generalization (per `Recommendation.md`)

### 9.1 `extract_ai_signals` was correct - it's being generalized, not replaced

Worth stating plainly before anything else in this section: `extract_ai_signals` (the real,
working, tested 4-dimension classifier - `infrastructure_moat`, `product_deployment`,
`research_depth`, `strategic_commitment`) was and remains the right design for the technology
profile. Nothing about it was wrong. What changed is the *framing* around it: instead of being
the one and only signal extractor this whole design revolves around, it becomes the **technology
industry's** instance of a more general pattern - "an industry has a document corpus, a rubric,
and a structured extractor that turns validated evidence into comparable dimensions." Pharma gets
its own extractor following the identical pattern, pointed at its own corpus, with its own
rubric. Neither replaces the other; they're siblings under the same shape.

This directly corrects the drift flagged in `complexity-review-baseline-and-final-hld.md` §2.4 -
the *other* two reviewed documents' 5-dimension `AISignalResult` schema
(`project_breadth`/`innovation_level`/`strategic_alignment`/`ai_governance_and_controls`/
`deployment_maturity`) doesn't match the real function and was never adopted. `Recommendation.md`
itself gets this right (§7: "For technology, it should reuse the real implemented AI
dimensions") - that's the version being followed here.

### 9.2 What's actually being adopted from `Recommendation.md`, and what isn't

Per the critical review in `complexity-review-baseline-and-final-hld.md`, this design adopts
`Recommendation.md`'s own **"Notebook scope"** section specifically - not the full
`open-universe-orchestrator-final-hld.md` production architecture. Concretely:

**Adopted:**
- A lightweight industry-awareness step alongside `extract_companies` (§3.1) - determining which
  profile (technology/pharma) applies per company, not a separate heavyweight planning stage.
- Two industry profiles: technology (existing, unchanged) and pharma (new - see §9.3).
- One additional, real signal extractor for pharma, following `extract_ai_signals`'s exact shape.
- Reuse of shared financial tools across both profiles - unchanged, per §3.3's existing "no
  duplicate tool layer" reasoning, which already matches what `Recommendation.md` §6 argues for.
- `MemorySaver` + disk cache + local ChromaDB, exactly as already designed in this document -
  `Recommendation.md` §9's own list explicitly confirms none of this needs to change.

**Not adopted** (per `complexity-review-baseline-and-final-hld.md`'s findings, and
`Recommendation.md`'s own "Do not implement" list, which agrees):
- Redis, PostgreSQL, distributed refresh locks - no multi-instance problem exists in a single
  Jupyter kernel.
- A versioned "Industry Profile Registry" as a standalone abstraction/component - two profiles
  don't need a registry; a plain dict or two `if`/`elif` branches at the point of use is enough
  until there's a real third profile to justify more structure.
- Full security-master-grade ticker/company validation.
- A general-purpose "structured planner" selecting from an open set of dimensions/tools per
  query - the bounded `extract_companies` + profile-lookup step covers what's actually needed at
  this scale.

### 9.3 Pharma extractor - rubric settled, grounded in real corpus text

`content/pharma_rag_official_sources.zip` (14 companies - AbbVie, Amgen, AstraZeneca,
Bristol-Myers Squibb, Eli Lilly, GSK, Johnson & Johnson, Merck, Novartis, Novo Nordisk, Pfizer,
Roche, Sanofi, Takeda - official 10-Ks/Annual Reports plus latest quarterly filings, real
SHA-256/source-URL provenance, no AI-generated substitutes) resolves the data prerequisite that
previously blocked this. Its own `metadata_schema.json` already anticipates
`therapeutic_area`/`drug_names`/`pipeline_phase`/`approval_status` as chunk-level fields, and its
`status_rule` ("do not infer marketed/approved/investigational status from context... only when
explicitly supported by the source text") matches this whole notebook's existing anti-fabrication
discipline directly.

#### Industry profile structure

Adopted from `Recommendation.md`'s `industry_profile` example, with one deliberate change: its
`industry_tools` list (`clinical_trials`, `regulatory_events`, `pipeline_rag`) implied three
separate external data sources, but only one actually exists in this project - the pharma PDF
corpus. `clinical_trials`/`regulatory_events` as standalone tools would need a ClinicalTrials.gov-
style registry API and an FDA/EMA regulatory-events feed, neither of which is sourced anywhere in
this project - building stubs for them would recreate the exact "no data behind this tool"
problem the pharma-corpus discovery just solved, one level down. Fixed by grounding all 5
dimensions in the *one* real tool, `pipeline_rag` (a retriever over the pharma corpus) - same
pattern as `extract_ai_signals` grounding all 4 of its dimensions in the one real
`query_private_database`/AI-corpus retriever, not one specialized tool per dimension.

```python
industry_profile = {
    "profile_id": "pharma.biopharma.v1",
    "shared_financial_dimensions": ["revenue_trend", "profitability_and_cash_generation",
                                     "liquidity_and_leverage", "valuation_relative_to_sector"],
    "industry_dimensions": ["clinical_pipeline", "regulatory_progress",
                             "exclusivity_and_patents", "commercialization", "sector_risks"],
    "shared_financial_tools": ["get_stock_price", "get_stock_history", "get_financial_metrics",
                                "search_financial_news", "analyze_sentiment"],
    "industry_tools": ["pipeline_rag"],  # one real retriever, not three unsourced tools
    "worker_prompt": "Use pharma terminology and preserve drug, indication, trial, and "
                      "regulatory identifiers.",
}
```

`shared_financial_dimensions` map directly onto existing `get_financial_metrics`/`get_stock_history`
fields (`revenue_trend` ≈ `total_revenue`, `valuation_relative_to_sector` ≈ `pe_ratio`, etc.) - no
new tools needed there either.

#### Verified against real corpus text before drafting level criteria

Extracted and searched the first 40 pages (where Item 1 Business/Item 1A Risk Factors live) of
two real 10-Ks (Merck, Pfizer) for each dimension, rather than assuming the topics would be there:

| Dimension | Real evidence found |
|---|---|
| `clinical_pipeline` | Merck's 10-K has an actual table of named drug candidates by clinical phase (e.g. "V181... Phase 3... 2029") |
| `regulatory_progress` | Both discuss concrete submission/approval events by name, not just generically |
| `exclusivity_and_patents` | Same Merck table gives specific U.S. patent expiration *years* per named candidate - unusually concrete |
| `commercialization` | Pfizer discusses specific product launches with detail (fewer raw mentions than other dimensions, but substantive where present) |
| `sector_risks` | Every 10-K has a dedicated, SEC-mandated "Item 1A. Risk Factors" section - guaranteed present for all 14 companies, not incidental |

All 5 dimensions confirmed classifiable from real text, not just plausible-sounding in the
abstract - same verification discipline `AI_SIGNAL_RUBRIC` was held to.

#### `PHARMA_SIGNAL_RUBRIC`

Same format as `AI_SIGNAL_RUBRIC` - none/partial/full per dimension, one-sentence reason,
`chunk_refs` citing which chunk(s) support the classification:

```
For EACH company below, classify these 5 signals using ONLY the provided context. Each signal
gets a level of "none", "partial", or "full", plus a one-sentence reason.

- clinical_pipeline: does the company disclose a substantive pipeline of drug candidates in
  active clinical development (not just already-marketed products)?
  none = no disclosed pipeline candidates. partial = pipeline mentioned without phase/candidate-
  level detail. full = named candidates with specific development phase (Phase 1/2/3) disclosed.

- regulatory_progress: has the company disclosed concrete regulatory milestones (submissions,
  approvals, responses to regulators) for its pipeline or recent products?
  none = no disclosed regulatory activity. partial = regulatory process described in general
  terms. full = specific submission/approval events named with product/indication and status.

- exclusivity_and_patents: does the company disclose concrete patent protection or exclusivity
  timelines for its products or pipeline candidates?
  none = no patent/exclusivity disclosure. partial = general patent risk discussed without
  specific dates. full = specific patent expiration years disclosed for named products/candidates.

- commercialization: is the company actively launching or scaling real-world availability of
  products, as opposed to only describing R&D?
  none = no disclosed launch/commercialization activity. partial = launch activity mentioned
  without specifics. full = specific product launches with market/geography/uptake detail.

- sector_risks: does the company disclose specific, named risks to its pharma business - trial
  failure, patent cliffs, regulatory setbacks - not generic corporate boilerplate?
  none = no meaningful pharma-specific risk disclosure found. partial = risk factors present but
  generic/non-pharma-specific. full = named, pharma-specific risks with concrete detail.

Each excerpt in the context below is tagged [Chunk N | document=..., page=...]. For each signal,
also list which chunk number(s) actually support your classification, as "chunk_refs": [N, ...] -
cite every chunk you drew on; use an empty list [] only if you found no supporting evidence at all.

Respond with ONLY a JSON object of this exact shape (no markdown fences, no extra text):
{
  "TICKER": {
    "clinical_pipeline": {"level": "none|partial|full", "reason": "...", "chunk_refs": [1, 3]},
    "regulatory_progress": {"level": "none|partial|full", "reason": "...", "chunk_refs": [2]},
    "exclusivity_and_patents": {"level": "none|partial|full", "reason": "...", "chunk_refs": []},
    "commercialization": {"level": "none|partial|full", "reason": "...", "chunk_refs": [1]},
    "sector_risks": {"level": "none|partial|full", "reason": "...", "chunk_refs": [4]}
  },
  ...
}
```

#### Still open before implementation

- Ingestion cost is real and worth planning for deliberately, not casually rerunning - 211MB
  total, with individual PDFs up to 76MB (Bristol-Myers Squibb's annual report) - meaningfully
  larger than the existing AI corpus.
- Whether pharma's extractor reuses the exact same chunk → LLM-classification pipeline shape as
  `extract_ai_signals` (`PyPDFDirectoryLoader` → `RecursiveCharacterTextSplitter` → `Chroma`), or
  needs adjustment given 10-K/Annual Report structure (much longer documents, more boilerplate
  between substantive sections) differs from the shorter AI-initiative documents the existing
  pipeline was built against.
- Edge-case check not yet done (§ plan step 3, deferred - not blocking, since all 5 dimensions
  are already verified present in 2 real 10-Ks): whether any of the 14 companies has a business
  shape meaningfully different from the "novel-pipeline biopharma" mold (e.g. more
  generics/biosimilars-weighted) that would need the same kind of deliberate handling
  `infrastructure_moat` got for NVIDIA, rather than being unfairly graded on `clinical_pipeline`.

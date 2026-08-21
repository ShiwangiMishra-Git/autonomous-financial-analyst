# Pharma implementation completion report — v75

## Outcome

The unified domain dispatcher and separate pharma policy router are implemented end to end for the approved scope. Pharma investment scoring remains intentionally disabled pending an approved and backtested rubric.

Latest notebook:

`/Users/shiwangimishra/Documents/Codex/2026-08-18/referenced-chatgpt-conversation-this-is-an/outputs/Merged-Autonomous_financial_analyst_Learners_Notebook-2026-08-18-v75-sentiment-three-year-window.ipynb`

## Sequential implementation details

1. Preserved the technology router and added a hybrid validated domain dispatcher. Strong entities and domain phrases route deterministically; a guarded LLM classifier is used only for genuinely unclear requests. Mixed-domain queries request clarification.
2. Added fail-closed provider controls. Permission, deadline, and remaining provider budget are checked before every specialized external boundary, with allowed/denied diagnostics returned in the result.
3. Reused the pinned five-company biopharma Chroma index for official-company RAG. Query embeddings are guarded and company filters remain isolated.
4. Added pharma evidence policies for company profile, commercial portfolio, pipeline, clinical results, safety, regulatory status, patents/exclusivity, financial profile, recent developments, and independent-news sentiment.
5. Added a guarded ClinicalTrials.gov v2 study tool for explicit NCT identifiers. It returns sponsor, phase, status, enrollment, primary endpoint, status date, and a trial-registry-primary claim.
6. Added jurisdiction-aware regulator routing. EMA/Europe requests search EMA primary evidence before FDA; U.S. patent requests expose the `US` jurisdiction.
7. Added source-bound claim normalization. Dates come from cited record metadata or explicit FDA primary-document URL years; clinical/safety labels reuse exact grounded claim text and source IDs.
8. Added pharma session state for named assets and company cohorts. Follow-ups such as “this drug” and “previously discussed pharma companies” reuse only the relevant pharma history.
9. Added output contracts for separate company sections, criterion-only rankings, broad comparisons, current developments, financial structured evidence, and long-term trade-off analysis.
10. Isolated sentiment evidence to dated, canonical, story-deduplicated independent editorial sources. The final recent window is 1,100 days, excluding older historical stories while maintaining stable publisher diversity.
11. Added the scoring stub requested by the user. Broad investment scores and Buy/Hold/Sell recommendations return `disabled_pending_approval`; criterion-only comparisons continue without numeric scoring.

## Verification

- Live core: 10/10 passed in a consolidated v62 run. Subsequent changes were additive; the two affected v75 live cases (sentiment and pipeline comparison) were rerun and passed.
- Offline: 4/4 passed with zero provider calls.
- Confirmed mocked: 9/9 passed with zero live provider calls. One prototype-scoring case is skipped because scoring is disabled.
- Chat: 5/5 enabled cases passed. Two scoring-dependent chat cases are skipped because scoring is disabled.
- Representative: three consecutive 8/8 passes.
- Stability: routing, cohort, evidence-completeness, and scoring-input fingerprints match across all three representative manifests.
- Technology/non-live regression: 264 passed, 1 intentionally skipped; one Chroma deprecation warning.

## Remaining work

Only pharma investment scoring is intentionally deferred. Before enabling it, define and approve the rubric, weights, missing-evidence policy, validation data, and backtesting thresholds; then enable the scoring acceptance cases and rerun all gates.

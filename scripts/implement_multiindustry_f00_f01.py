"""Idempotently add the F00/F01 multi-industry foundation cells to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_BEFORE_CELL_ID = "pZskPu1tn3Q-"


SECTION_INTRO = """# Section 3.0: Multi-Industry Orchestrator (Notebook-Local)

This section extends the existing technology workflow with a profile-driven, multi-company
LangGraph design. The first profiles are **Technology/AI** and
**Healthcare/Biopharma**. All execution remains local to this notebook.

Implementation order: **States → Tools → Agents → Workflow**.
"""


STATE_INTRO = """## Section 3.1: State and Contracts

The state model separates conversation context from one-request research data. A new user
request keeps message history but receives a fresh `run_id`, plan, company tasks, results,
scores, and validation fields.

Parallel company branches will merge results through a reset-aware reducer. This avoids
leaking prior-run results when the same `thread_id` continues into a new request.
"""


STATE_CODE = r'''from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Sequence, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


QueryType = Literal["fact", "analyze", "compare", "rank"]
RiskProfile = Literal["conservative", "balanced", "growth"]
ResolutionStatus = Literal["resolved", "ambiguous", "unsupported"]
EvidenceStatus = Literal["success", "missing", "failed"]
FreshnessStatus = Literal["fresh", "stale", "unknown"]
CompanyResultStatus = Literal["success", "partial", "failed"]
ComparisonMode = Literal["single", "same_profile", "cross_profile"]


class QueryPlan(TypedDict):
    """Normalized coordinator interpretation of one free-text question.

    Attributes:
        query_type: Requested operation: fact, analyze, compare, or rank.
        company_mentions: User-supplied company names before canonical resolution.
        requested_dimensions: User-requested financial or sector research topics.
        risk_profile: Weight policy used only by deterministic eligible scoring.
        scoring_requested: Whether the user explicitly requested numeric ranking/scoring.
        freshness_required: Whether current provider/news evidence is required.
        time_horizon: Optional investment or analysis horizon stated by the user.
    """
    query_type: QueryType
    company_mentions: list[str]
    requested_dimensions: list[str]
    risk_profile: RiskProfile
    scoring_requested: bool
    freshness_required: bool
    time_horizon: str | None


class ResolvedCompany(TypedDict):
    """Canonical registry identity produced by deterministic company resolution.

    Attributes:
        company_id: Stable internal identifier independent of ticker changes.
        ticker: Canonical public-market symbol used by source tools.
        company_name: Display name from the supported-company registry.
        aliases: Accepted normalized user mentions.
        exchange: Listing exchange when known.
        industry: Broad registry industry.
        sub_industry: Narrow registry classification.
        profile_id: Versioned research profile selected by registry data.
        resolution_status: Resolved, ambiguous, or unsupported outcome.
        resolution_message: Optional explanation for non-resolved outcomes.
    """
    company_id: str
    ticker: str
    company_name: str
    aliases: list[str]
    exchange: str | None
    industry: str
    sub_industry: str
    profile_id: str
    resolution_status: ResolutionStatus
    resolution_message: str | None


class EvidenceRecord(TypedDict):
    """Canonical current-run evidence envelope shared by every source tool.

    Attributes:
        evidence_id: Stable citation token unique within the run.
        run_id: Research run that owns this record.
        company_id: Canonical company owner.
        ticker: Canonical ticker owner.
        profile_id: Industry profile allowed to consume the record.
        evidence_type: Normalized category such as financial_metrics or technology_rag.
        value: Raw normalized tool payload available to downstream reasoning.
        source_name: Exact tool/adapter that produced the record.
        source_uri: Optional public source URL.
        document_name: Optional local/public document label.
        page: Optional one-based source page.
        as_of: Optional effective date of the underlying data.
        retrieved_at: UTC retrieval timestamp.
        freshness_status: Fresh, stale, or unknown evidence assessment.
        cache_status: Hit, miss, stale, or unknown cache outcome.
        status: Success, missing, or failed source result.
        source_metadata: Additional provenance excluding the primary value/error.
        error: Explicit source failure, otherwise ``None``.
    """
    evidence_id: str
    run_id: str
    company_id: str
    ticker: str
    profile_id: str
    evidence_type: str
    value: Any
    source_name: str
    source_uri: str | None
    document_name: str | None
    page: int | None
    as_of: str | None
    retrieved_at: str
    freshness_status: FreshnessStatus
    cache_status: Literal["hit", "miss", "stale", "unknown"]
    status: EvidenceStatus
    source_metadata: dict[str, Any]
    error: str | None


class CompanyTask(TypedDict):
    """Guarded immutable assignment sent to exactly one company worker.

    Attributes:
        run_id: Owning research run.
        company: Canonical company the branch may research.
        query_plan: Validated coordinator plan copied into the branch.
        shared_dimensions: Financial dimensions this worker must cover.
        industry_dimensions: Profile-specific dimensions this worker must cover.
        unsupported_dimensions: Requested topics outside current profile support.
        allowed_tools: Exact profile allowlist; the worker cannot expand it.
    """
    run_id: str
    company: ResolvedCompany
    query_plan: QueryPlan
    shared_dimensions: list[str]
    industry_dimensions: list[str]
    unsupported_dimensions: list[str]
    allowed_tools: list[str]


class CompanyResearchResult(TypedDict):
    """One worker's normalized terminal result before/after fan-in.

    Attributes:
        run_id: Owning research run.
        company: Canonical company identity.
        profile_id: Profile used by the worker.
        financial_evidence: Optional derived financial view for compatibility.
        industry_signals: Evidence-linked profile signal dimensions.
        evidence: Canonical records collected by allowed tools.
        missing_dimensions: Required topics without successful coverage.
        errors: Contained worker, tool, or normalization errors.
        status: Success, partial, or failed terminal state.
    """
    run_id: str
    company: ResolvedCompany
    profile_id: str
    financial_evidence: dict[str, Any]
    industry_signals: dict[str, Any]
    evidence: list[EvidenceRecord]
    missing_dimensions: list[str]
    errors: list[str]
    status: CompanyResultStatus


class ScoringEligibility(TypedDict):
    """Deterministic F12 authorization decision for F13 sector scoring.

    Attributes:
        eligible: Whether F13 may run for the normalized comparison.
        rubric_id: Exact versioned rubric when eligible.
        reason: Human-readable authorization or rejection reason.
        excluded_companies: Tickers that prevent scoring.
        missing_requirements: Missing/failed requirements grouped by ticker.
    """
    eligible: bool
    rubric_id: str | None
    reason: str
    excluded_companies: list[str]
    missing_requirements: dict[str, list[str]]


@dataclass(frozen=True)
class ResetCompanyResults:
    """Reducer update used by initialize_research_run to clear prior-run results."""


CompanyResultMap = dict[str, CompanyResearchResult]


def merge_company_results(
    current: CompanyResultMap | None,
    update: CompanyResultMap | ResetCompanyResults | None,
) -> CompanyResultMap:
    """Reset or merge parallel company results.

    Args:
        current: Reducer value already accumulated by LangGraph.
        update: Branch results, reset sentinel, or no update.

    Returns:
        Fresh ticker-keyed result map.

    Usage:
        Registered as the ``company_results`` reducer; callers normally do not invoke it directly.
    """
    if isinstance(update, ResetCompanyResults):
        return {}

    merged: CompanyResultMap = dict(current or {})
    if update:
        merged.update(update)
    return merged


class CompanyWorkerState(TypedDict):
    """Branch-local LangGraph state for one autonomous company worker.

    Attributes:
        task: Single validated company assignment.
        messages: Reducer-managed worker conversation and tool messages.
        evidence: Current-company canonical evidence accumulated from tools.
        industry_signals: Evidence-linked profile interpretation.
        missing_dimensions: Required dimensions still unsupported.
        evidence_gate_status: Retry, complete, or bounded-partial decision.
        tool_round_count: Source-tool loop count used by the hard ceiling.
        validation_retry_count: Evidence-gate retry count.
        result: Terminal company result once assembled.
        errors: Contained branch errors visible at fan-in.
    """
    task: CompanyTask
    messages: Annotated[Sequence[BaseMessage], add_messages]
    evidence: list[EvidenceRecord]
    industry_signals: dict[str, Any]
    missing_dimensions: list[str]
    evidence_gate_status: Literal["retry", "complete", "partial"]
    tool_round_count: int
    validation_retry_count: int
    result: CompanyResearchResult | None
    errors: list[str]


class OrchestratorState(TypedDict):
    """Parent LangGraph state spanning conversation and one research run.

    Attributes:
        messages: Conversation messages retained by the optional checkpointer.
        remembered_company_ids: Canonical companies available to follow-up references.
        last_profile_ids: Profiles used by the previous completed request.
        run_id: Fresh identifier assigned to the current request.
        run_started_at: UTC start timestamp.
        original_query: Latest human question passed to planning.
        plan: Validated free-text query plan.
        resolution_result: Resolver output including unsupported/ambiguous mentions.
        resolution_gate_status: Mandatory identity-gate verdict.
        resolved_companies: Canonical companies authorized for this run.
        profile_selection: Registry-backed profile mapping.
        profile_gate_status: Mandatory profile-coverage verdict.
        company_tasks: One guarded task per resolved company.
        task_gate_status: Isolation, permission, and budget verdict.
        company_results: Reducer-managed raw branch results keyed by ticker.
        normalized_company_results: F12 canonical fan-in results.
        fan_in_normalization: F12 ordering/status/error summary.
        comparison_mode: Single, same-profile, or cross-profile route.
        comparison_route_status: Mandatory comparison-route verdict.
        scoring_eligibility: F12 permission for deterministic F13 scoring.
        scores: Optional authoritative F13 table.
        final_answer: Validated answer or bounded-stop explanation.
        validation_retry_count: Bounded final-validation retry count.
        validation_errors: Deterministic contract violations.
        run_errors: Non-blocking and blocking errors accumulated for the run.
    """
    # Conversation lifetime: retained by MemorySaver for the configured thread_id.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remembered_company_ids: list[str]
    last_profile_ids: list[str]

    # Research-run lifetime: reset by initialize_research_run for every user request.
    run_id: str
    run_started_at: str
    original_query: str
    plan: QueryPlan | None
    resolution_result: dict[str, Any] | None
    resolution_gate_status: dict[str, Any] | None
    resolved_companies: list[ResolvedCompany]
    profile_selection: dict[str, Any] | None
    profile_gate_status: dict[str, Any] | None
    company_tasks: list[CompanyTask]
    task_gate_status: dict[str, Any] | None
    company_results: Annotated[CompanyResultMap, merge_company_results]
    normalized_company_results: CompanyResultMap
    fan_in_normalization: dict[str, Any] | None
    comparison_mode: ComparisonMode | None
    comparison_route_status: dict[str, Any] | None
    scoring_eligibility: ScoringEligibility | None
    scores: dict[str, Any] | None
    final_answer: str | None
    validation_retry_count: int
    validation_errors: list[str]
    run_errors: list[str]


# Resolve postponed annotations while this cell's complete contract namespace is available.
# LangGraph later calls get_type_hints through the class's module, which is unreliable when a
# notebook cell is deliberately executed inside an isolated test namespace.
CompanyWorkerState.__annotations__ = {
    "task": CompanyTask,
    "messages": Annotated[Sequence[BaseMessage], add_messages],
    "evidence": list[EvidenceRecord],
    "industry_signals": dict[str, Any],
    "missing_dimensions": list[str],
    "evidence_gate_status": Literal["retry", "complete", "partial"],
    "tool_round_count": int,
    "validation_retry_count": int,
    "result": CompanyResearchResult | None,
    "errors": list[str],
}
OrchestratorState.__annotations__ = {
    "messages": Annotated[Sequence[BaseMessage], add_messages],
    "remembered_company_ids": list[str],
    "last_profile_ids": list[str],
    "run_id": str,
    "run_started_at": str,
    "original_query": str,
    "plan": QueryPlan | None,
    "resolution_result": dict[str, Any] | None,
    "resolution_gate_status": dict[str, Any] | None,
    "resolved_companies": list[ResolvedCompany],
    "profile_selection": dict[str, Any] | None,
    "profile_gate_status": dict[str, Any] | None,
    "company_tasks": list[CompanyTask],
    "task_gate_status": dict[str, Any] | None,
    "company_results": Annotated[CompanyResultMap, merge_company_results],
    "normalized_company_results": CompanyResultMap,
    "fan_in_normalization": dict[str, Any] | None,
    "comparison_mode": ComparisonMode | None,
    "comparison_route_status": dict[str, Any] | None,
    "scoring_eligibility": ScoringEligibility | None,
    "scores": dict[str, Any] | None,
    "final_answer": str | None,
    "validation_retry_count": int,
    "validation_errors": list[str],
    "run_errors": list[str],
}


def _latest_human_query(messages: Sequence[BaseMessage]) -> str:
    """Return the newest human question from reducer-managed conversation messages.

    Args:
        messages: Ordered LangChain conversation messages.

    Returns:
        Latest human content as text, or an empty string when absent.

    Usage:
        Called only by run initialization to establish ``original_query``.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def initialize_research_run(state: OrchestratorState) -> dict[str, Any]:
    """Create a fresh request scope without overwriting conversation-lifetime fields."""
    messages = state.get("messages", ())
    return {
        "run_id": str(uuid4()),
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "original_query": _latest_human_query(messages),
        "plan": None,
        "resolution_result": None,
        "resolution_gate_status": None,
        "resolved_companies": [],
        "profile_selection": None,
        "profile_gate_status": None,
        "company_tasks": [],
        "task_gate_status": None,
        "company_results": ResetCompanyResults(),
        "normalized_company_results": {},
        "fan_in_normalization": None,
        "comparison_mode": None,
        "comparison_route_status": None,
        "scoring_eligibility": None,
        "scores": None,
        "final_answer": None,
        "validation_retry_count": 0,
        "validation_errors": [],
        "run_errors": [],
    }


print("✅ F01 state contracts and fresh-run initializer defined")
'''


STATE_SMOKE = r'''# F01 local smoke test: no network or API keys required.
_prior_result = {"MSFT": {"run_id": "old-run"}}
assert merge_company_results(_prior_result, ResetCompanyResults()) == {}

_f01_demo_state = {
    "messages": [HumanMessage(content="Compare Microsoft and Pfizer")],
    "remembered_company_ids": ["MSFT"],
    "last_profile_ids": ["technology.ai.v1"],
    "company_results": _prior_result,
}
_f01_update = initialize_research_run(_f01_demo_state)

assert _f01_update["original_query"] == "Compare Microsoft and Pfizer"
assert isinstance(_f01_update["company_results"], ResetCompanyResults)
assert _f01_update["plan"] is None
assert _f01_update["scores"] is None
assert "messages" not in _f01_update
assert "remembered_company_ids" not in _f01_update

print("✅ F01 smoke test passed: conversation retained; research-run fields reset")
'''


CELL_SPECS = [
    ("multiindustry_section3_intro", "markdown", SECTION_INTRO),
    ("multiindustry_f01_intro", "markdown", STATE_INTRO),
    ("multiindustry_state_contracts", "code", STATE_CODE),
    ("multiindustry_f01_smoke", "code", STATE_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    if cell_type == "markdown":
        cell = nbformat.v4.new_markdown_cell(source=source)
    else:
        cell = nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}

    for cell_id, cell_type, source in CELL_SPECS:
        existing = cells_by_id.get(cell_id)
        if existing is not None:
            existing["cell_type"] = cell_type
            existing["source"] = source
            if cell_type == "code":
                existing["execution_count"] = None
                existing["outputs"] = []

    missing_specs = [spec for spec in CELL_SPECS if spec[0] not in cells_by_id]
    if missing_specs:
        insertion_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.get("id") == INSERT_BEFORE_CELL_ID
        )
        new_cells = [_new_cell(*spec) for spec in missing_specs]
        notebook.cells[insertion_index:insertion_index] = new_cells

    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F00/F01 cells are present")


if __name__ == "__main__":
    main()

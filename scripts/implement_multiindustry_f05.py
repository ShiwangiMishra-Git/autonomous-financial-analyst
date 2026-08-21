"""Idempotently add the F05 guarded company-task builder to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f04_smoke"


F05_INTRO = """## Section 3.5: Guarded Research-Plan and Company-Task Builder

The coordinator calls `build_company_tasks_tool` with only a `run_id`. It cannot submit custom
companies, dimensions, profile IDs, or tool names. A deterministic planning-context registration
step first captures the validated query plan, resolved companies, and profile selection for that
run; the guarded tool then reads defensive copies of that state.

The pure builder produces exactly one isolated `CompanyTask` per company. Explicitly requested
but unsupported dimensions are recorded rather than silently replaced. `validate_task_gate`
checks company coverage, limits, identity, dimensions, allowlists, and run consistency before the
future LangGraph `Send` fan-out can execute.
"""


F05_CODE = r'''from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict

from langchain_core.tools import tool


MAX_COMPANIES_PER_QUERY = 5


class TaskPlanningContext(TypedDict):
    """Validated run-scoped inputs available to the guarded task-builder tool.

    Attributes:
        run_id: Current research-run identifier.
        plan: Deterministically validated free-text query plan.
        companies: Successfully resolved and profile-validated companies.
        profile_selection: Guarded profile selection that passed its mandatory gate.
    """

    run_id: str
    plan: QueryPlan
    companies: list[ResolvedCompany]
    profile_selection: ProfileSelection


_TASK_PLANNING_CONTEXTS: dict[str, TaskPlanningContext] = {}


SHARED_DIMENSION_ALIASES: dict[str, list[str]] = {
    "current_price": ["current_price"],
    "stock_price": ["current_price"],
    "price": ["current_price"],
    "market_cap": ["market_cap"],
    "revenue": ["total_revenue"],
    "total_revenue": ["total_revenue"],
    "valuation": ["market_cap", "pe_ratio"],
    "pe_ratio": ["pe_ratio"],
    "risk": ["beta"],
    "beta": ["beta"],
    "dividend": ["dividend_yield"],
    "dividend_yield": ["dividend_yield"],
    "price_history": ["price_history"],
    "price_performance": ["price_history"],
    "news": ["news_sentiment"],
    "sentiment": ["news_sentiment"],
    "news_sentiment": ["news_sentiment"],
    "financial_strength": [
        "market_cap", "total_revenue", "pe_ratio", "beta", "dividend_yield",
    ],
    "financial_health": [
        "market_cap", "total_revenue", "pe_ratio", "beta", "dividend_yield",
    ],
}


PROFILE_DIMENSION_ALIASES: dict[str, dict[str, list[str]]] = {
    "technology.ai.v1": {
        "ai": [
            "infrastructure_moat", "product_deployment", "research_depth",
            "strategic_commitment",
        ],
        "ai_strategy": ["product_deployment", "strategic_commitment"],
        "innovation": ["research_depth", "strategic_commitment"],
        "long_term_innovation": ["research_depth", "strategic_commitment"],
        "infrastructure": ["infrastructure_moat"],
        "products": ["product_deployment"],
    },
    "healthcare.biopharma.v1": {
        "pipeline": ["clinical_pipeline"],
        "clinical_pipeline": ["clinical_pipeline"],
        "regulatory": ["regulatory_progress"],
        "patents": ["exclusivity_and_patents"],
        "exclusivity": ["exclusivity_and_patents"],
        "commercialization": ["commercialization"],
        "sector_risk": ["sector_risks"],
        "long_term_innovation": ["clinical_pipeline", "exclusivity_and_patents"],
    },
}


def _ordered_unique(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving their first-seen order.

    Args:
        values: Ordered string values that may contain repeats.

    Returns:
        First occurrence of every unique value.
    """
    return list(dict.fromkeys(values))


def _dimensions_for_profile(
    plan: QueryPlan,
    profile: IndustryProfile,
) -> tuple[list[str], list[str], list[str]]:
    """Map requested dimensions into one profile without silent substitution.

    Args:
        plan: Validated user query plan.
        profile: Industry configuration for the assigned company.

    Returns:
        Tuple of shared dimensions, profile-specific dimensions, and unsupported requests.
    """
    supported_shared = list(profile["shared_dimensions"])
    supported_industry = list(profile["industry_dimensions"])
    requested = plan["requested_dimensions"]

    if not requested:
        return supported_shared, supported_industry, []

    selected_shared: list[str] = []
    selected_industry: list[str] = []
    unsupported: list[str] = []
    profile_aliases = PROFILE_DIMENSION_ALIASES.get(profile["profile_id"], {})

    for dimension in requested:
        matched = False
        if dimension in supported_shared:
            selected_shared.append(dimension)
            matched = True
        if dimension in supported_industry:
            selected_industry.append(dimension)
            matched = True

        shared_expansion = SHARED_DIMENSION_ALIASES.get(dimension, [])
        valid_shared_expansion = [
            item for item in shared_expansion if item in supported_shared
        ]
        if valid_shared_expansion:
            selected_shared.extend(valid_shared_expansion)
            matched = True

        industry_expansion = profile_aliases.get(dimension, [])
        valid_industry_expansion = [
            item for item in industry_expansion if item in supported_industry
        ]
        if valid_industry_expansion:
            selected_industry.extend(valid_industry_expansion)
            matched = True

        if not matched:
            unsupported.append(dimension)

    # Like-for-like comparison and scoring require the complete shared financial contract.
    if plan["query_type"] in {"compare", "rank"} or plan["scoring_requested"]:
        selected_shared = supported_shared + selected_shared
    if plan["scoring_requested"]:
        selected_industry = supported_industry + selected_industry

    return (
        _ordered_unique(selected_shared),
        _ordered_unique(selected_industry),
        _ordered_unique(unsupported),
    )


def register_task_planning_context(
    run_id: str,
    plan: QueryPlan,
    companies: list[ResolvedCompany],
    profile_selection: ProfileSelection,
) -> TaskPlanningContext:
    """Register validated current-run inputs for guarded task construction.

    Args:
        run_id: Non-empty identifier created by ``initialize_research_run``.
        plan: Normalized and validated query plan.
        companies: Successfully resolved companies for this run.
        profile_selection: Result of the guarded selector for those companies.

    Returns:
        Defensive copy of the stored run-scoped planning context.

    Raises:
        ValueError: If any prerequisite contract or mandatory profile gate is invalid.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    plan_errors = validate_query_plan(plan)
    if plan_errors:
        raise ValueError("Invalid query plan: " + "; ".join(plan_errors))
    validated_companies = attach_industry_profiles(companies)
    company_ids = [company["company_id"] for company in validated_companies]
    profile_gate = validate_profile_gate(profile_selection, company_ids)
    if not profile_gate["ready"]:
        raise ValueError("Profile gate failed: " + profile_gate["message"])
    if len(validated_companies) > MAX_COMPANIES_PER_QUERY:
        raise ValueError(
            f"Company limit exceeded: {len(validated_companies)} > {MAX_COMPANIES_PER_QUERY}"
        )

    context: TaskPlanningContext = {
        "run_id": run_id.strip(),
        "plan": deepcopy(plan),
        "companies": deepcopy(validated_companies),
        "profile_selection": deepcopy(profile_selection),
    }
    _TASK_PLANNING_CONTEXTS[context["run_id"]] = context
    return deepcopy(context)


def clear_task_planning_context(run_id: str) -> bool:
    """Remove one run-scoped task-planning context from local memory.

    Args:
        run_id: Research-run identifier to remove.

    Returns:
        ``True`` when a context existed and was removed.
    """
    return _TASK_PLANNING_CONTEXTS.pop(run_id, None) is not None


def build_company_tasks(
    plan: QueryPlan,
    companies: list[ResolvedCompany],
    run_id: str,
) -> list[CompanyTask]:
    """Build one deterministic, profile-bounded research task per company.

    Args:
        plan: Validated query plan shared by every company branch.
        companies: Unique, successfully resolved, profile-validated companies.
        run_id: Current research-run identifier copied into every task.

    Returns:
        Isolated tasks in company input order.

    Raises:
        ValueError: If inputs are invalid, duplicated, or exceed the local company limit.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if len(companies) > MAX_COMPANIES_PER_QUERY:
        raise ValueError(
            f"Company limit exceeded: {len(companies)} > {MAX_COMPANIES_PER_QUERY}"
        )
    plan_errors = validate_query_plan(plan)
    if plan_errors:
        raise ValueError("Invalid query plan: " + "; ".join(plan_errors))

    validated_companies = attach_industry_profiles(companies)
    seen_company_ids: set[str] = set()
    tasks: list[CompanyTask] = []
    for company in validated_companies:
        company_id = company["company_id"]
        if company_id in seen_company_ids:
            raise ValueError(f"Duplicate company in task input: {company_id!r}")
        seen_company_ids.add(company_id)

        profile = get_industry_profile(company["profile_id"])
        shared, industry, unsupported = _dimensions_for_profile(plan, profile)
        task: CompanyTask = {
            "run_id": run_id,
            "company": deepcopy(company),
            "query_plan": deepcopy(plan),
            "shared_dimensions": shared,
            "industry_dimensions": industry,
            "unsupported_dimensions": unsupported,
            "allowed_tools": list(profile["allowed_tools"]),
        }
        tasks.append(task)
    return tasks


def validate_task_gate(
    tasks: list[CompanyTask],
    companies: list[ResolvedCompany],
    max_companies: int = MAX_COMPANIES_PER_QUERY,
) -> dict[str, Any]:
    """Validate task isolation, coverage, permissions, dimensions, and run consistency.

    Args:
        tasks: Candidate tasks produced by the deterministic builder.
        companies: Validated companies expected in the current run.
        max_companies: Maximum branches permitted by the local notebook.

    Returns:
        Mandatory gate result with readiness, status, errors, and validated task count.
    """
    errors: list[str] = []
    expected_ids = [company["company_id"] for company in companies]
    task_ids = [task.get("company", {}).get("company_id", "") for task in tasks]

    if not companies:
        errors.append("No validated companies are available for task construction")
    if len(companies) > max_companies or len(tasks) > max_companies:
        errors.append(f"Company/task limit exceeded: maximum is {max_companies}")
    if len(expected_ids) != len(set(expected_ids)):
        errors.append("Validated company input contains duplicates")
    if len(task_ids) != len(set(task_ids)):
        errors.append("Task list contains duplicate companies")

    missing = sorted(set(expected_ids) - set(task_ids))
    unexpected = sorted(set(task_ids) - set(expected_ids))
    if missing:
        errors.append(f"Missing tasks for company IDs: {missing}")
    if unexpected:
        errors.append(f"Unexpected task companies: {unexpected}")

    run_ids = {task.get("run_id", "") for task in tasks}
    if tasks and ("" in run_ids or len(run_ids) != 1):
        errors.append("All tasks must share one non-empty run_id")

    for task in tasks:
        company = task.get("company", {})
        company_id = company.get("company_id", "")
        profile_id = company.get("profile_id", "")
        try:
            profile = get_industry_profile(profile_id)
        except KeyError:
            errors.append(f"Task for {company_id!r} references unsupported profile {profile_id!r}")
            continue

        authoritative_profile = COMPANY_PROFILE_INDEX.get(company_id)
        if authoritative_profile != profile_id:
            errors.append(f"Task for {company_id!r} has a swapped profile")
        disallowed_tools = sorted(set(task.get("allowed_tools", [])) - set(profile["allowed_tools"]))
        if disallowed_tools:
            errors.append(f"Task for {company_id!r} contains disallowed tools: {disallowed_tools}")
        invalid_shared = sorted(
            set(task.get("shared_dimensions", [])) - set(profile["shared_dimensions"])
        )
        invalid_industry = sorted(
            set(task.get("industry_dimensions", [])) - set(profile["industry_dimensions"])
        )
        if invalid_shared:
            errors.append(f"Task for {company_id!r} has invalid shared dimensions: {invalid_shared}")
        if invalid_industry:
            errors.append(f"Task for {company_id!r} has invalid industry dimensions: {invalid_industry}")
        plan_errors = validate_query_plan(task.get("query_plan", {}))
        if plan_errors:
            errors.append(f"Task for {company_id!r} has an invalid query plan: {plan_errors}")

    return {
        "ready": not errors,
        "status": "ready" if not errors else "invalid_tasks",
        "message": "Company tasks passed mandatory validation." if not errors else "; ".join(errors),
        "errors": errors,
        "task_count": len(tasks),
    }


@tool
def build_company_tasks_tool(run_id: str) -> dict[str, Any]:
    """Build guarded company tasks from validated state registered for one run.

    The coordinator supplies only ``run_id``. Company identity, profile, dimensions, and tool
    permissions are read from deterministic current-run context and cannot be injected by the LLM.

    Args:
        run_id: Current research-run identifier.

    Returns:
        Tasks plus the mandatory task-gate result, or an explicit missing-context error.
    """
    context = _TASK_PLANNING_CONTEXTS.get(run_id)
    if context is None:
        return {
            "status": "missing_context",
            "tasks": [],
            "task_gate": {
                "ready": False,
                "status": "invalid_tasks",
                "message": f"No validated task-planning context exists for run_id {run_id!r}.",
                "errors": ["Missing validated task-planning context"],
                "task_count": 0,
            },
            "unsupported_dimensions_by_company": {},
        }

    tasks = build_company_tasks(
        context["plan"],
        context["companies"],
        context["run_id"],
    )
    task_gate = validate_task_gate(tasks, context["companies"])
    unsupported = {
        task["company"]["company_id"]: list(task["unsupported_dimensions"])
        for task in tasks
        if task["unsupported_dimensions"]
    }
    return {
        "status": "ready" if task_gate["ready"] else "invalid_tasks",
        "tasks": tasks,
        "task_gate": task_gate,
        "unsupported_dimensions_by_company": unsupported,
    }


def route_after_task_gate(task_gate: dict[str, Any]) -> str:
    """Route validated tasks to LangGraph fan-out or a bounded planning stop.

    Args:
        task_gate: Output from ``validate_task_gate``.

    Returns:
        ``fan_out`` when ready; otherwise ``stop_invalid_tasks``.
    """
    return "fan_out" if task_gate.get("ready") is True else "stop_invalid_tasks"


print(
    "✅ F05 guarded task builder defined "
    f"(maximum {MAX_COMPANIES_PER_QUERY} companies per query)"
)
'''


F05_SMOKE = r'''# F05 local smoke test: no network, API keys, or research calls.
_f05_plan: QueryPlan = {
    "query_type": "compare",
    "company_mentions": ["Microsoft", "Pfizer"],
    "requested_dimensions": ["financial_strength", "long_term_innovation", "debt"],
    "risk_profile": "balanced",
    "scoring_requested": False,
    "freshness_required": True,
    "time_horizon": "long term",
}
_f05_companies = resolve_company_mentions(_f05_plan["company_mentions"])
_f05_selection = select_industry_profiles(
    [company["company_id"] for company in _f05_companies]
)
register_task_planning_context(
    "f05-smoke-run",
    _f05_plan,
    _f05_companies,
    _f05_selection,
)
_f05_result = build_company_tasks_tool.invoke({"run_id": "f05-smoke-run"})
assert _f05_result["status"] == "ready"
assert _f05_result["task_gate"]["ready"] is True
assert route_after_task_gate(_f05_result["task_gate"]) == "fan_out"
assert [task["company"]["ticker"] for task in _f05_result["tasks"]] == ["MSFT", "PFE"]
assert "research_depth" in _f05_result["tasks"][0]["industry_dimensions"]
assert "clinical_pipeline" in _f05_result["tasks"][1]["industry_dimensions"]
assert _f05_result["unsupported_dimensions_by_company"] == {
    "microsoft": ["debt"],
    "pfizer": ["debt"],
}
clear_task_planning_context("f05-smoke-run")

print("✅ F05 smoke test passed: isolated mixed-profile tasks and mandatory gate work")
'''


CELL_SPECS = [
    ("multiindustry_f05_intro", "markdown", F05_INTRO),
    ("multiindustry_company_tasks", "code", F05_CODE),
    ("multiindustry_f05_smoke", "code", F05_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create a notebook cell with a stable ID and source."""
    if cell_type == "markdown":
        cell = nbformat.v4.new_markdown_cell(source=source)
    else:
        cell = nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    """Insert or refresh F05 cells in the canonical working notebook."""
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
            if cell.get("id") == INSERT_AFTER_CELL_ID
        ) + 1
        notebook.cells[insertion_index:insertion_index] = [
            _new_cell(*spec) for spec in missing_specs
        ]

    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F05 cells are present")


if __name__ == "__main__":
    main()

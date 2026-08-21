"""Deterministic tests for the F05 guarded company-task builder."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


@lru_cache(maxsize=1)
def _task_namespace():
    """Execute only F01–F05 implementation cells without network access."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in (
            "multiindustry_state_contracts",
            "multiindustry_company_registry",
            "multiindustry_query_planner",
            "multiindustry_industry_profiles",
            "multiindustry_company_tasks",
        ):
            exec(cells[cell_id], namespace)
    return namespace


def _plan(*mentions, dimensions=None, query_type="compare", scoring=False):
    """Create a valid query-plan fixture for deterministic task tests."""
    return {
        "query_type": query_type,
        "company_mentions": list(mentions),
        "requested_dimensions": list(dimensions or []),
        "risk_profile": "balanced",
        "scoring_requested": scoring,
        "freshness_required": True,
        "time_horizon": "long term",
    }


def _register(namespace, run_id, plan):
    """Resolve companies, select profiles, and register one valid planning context."""
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    selection = namespace["select_industry_profiles"](
        [company["company_id"] for company in companies]
    )
    namespace["register_task_planning_context"](run_id, plan, companies, selection)
    return companies


def test_one_isolated_task_per_unique_company():
    """Require deterministic input order and exactly one company mapping per task."""
    namespace = _task_namespace()
    plan = _plan("Microsoft", "MSFT", "Pfizer")
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    tasks = namespace["build_company_tasks"](plan, companies, "run-one")

    assert [task["company"]["ticker"] for task in tasks] == ["MSFT", "PFE"]
    assert all(isinstance(task["company"], dict) for task in tasks)
    assert all(task["run_id"] == "run-one" for task in tasks)


def test_cross_industry_query_builds_profile_specific_dimensions_and_tools():
    """Build distinct technology and biopharma tasks from one shared plan."""
    namespace = _task_namespace()
    plan = _plan(
        "Microsoft",
        "Pfizer",
        dimensions=["financial_strength", "long_term_innovation"],
    )
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    tasks = namespace["build_company_tasks"](plan, companies, "run-cross")
    technology, biopharma = tasks

    assert "query_technology_rag" in technology["allowed_tools"]
    assert "query_biopharma_rag" not in technology["allowed_tools"]
    assert {"research_depth", "strategic_commitment"} <= set(technology["industry_dimensions"])
    assert "query_biopharma_rag" in biopharma["allowed_tools"]
    assert "query_technology_rag" not in biopharma["allowed_tools"]
    assert {"clinical_pipeline", "exclusivity_and_patents"} <= set(
        biopharma["industry_dimensions"]
    )
    assert technology["query_plan"]["risk_profile"] == "balanced"
    assert technology["query_plan"]["time_horizon"] == "long term"


def test_unsupported_dimensions_are_recorded_without_substitution():
    """Preserve unsupported user intent as an explicit task limitation."""
    namespace = _task_namespace()
    plan = _plan("Microsoft", dimensions=["debt", "quantum_patent_velocity"], query_type="analyze")
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    task = namespace["build_company_tasks"](plan, companies, "run-unsupported")[0]

    assert task["unsupported_dimensions"] == ["debt", "quantum_patent_velocity"]
    assert "debt" not in task["shared_dimensions"]
    assert "quantum_patent_velocity" not in task["industry_dimensions"]


def test_guarded_tool_accepts_only_run_id_and_reads_registered_context():
    """Prevent LLM-supplied identities, dimensions, or tool lists from entering task creation."""
    namespace = _task_namespace()
    plan = _plan("Microsoft", "Pfizer", dimensions=["long_term_innovation"])
    _register(namespace, "run-guarded", plan)

    guarded_tool = namespace["build_company_tasks_tool"]
    assert set(guarded_tool.args_schema.model_json_schema()["properties"]) == {"run_id"}
    result = guarded_tool.invoke({"run_id": "run-guarded"})

    assert result["status"] == "ready"
    assert result["task_gate"]["ready"] is True
    assert [task["company"]["ticker"] for task in result["tasks"]] == ["MSFT", "PFE"]


def test_missing_run_context_fails_closed():
    """Return a non-ready result when the agent supplies an unknown run identifier."""
    namespace = _task_namespace()
    result = namespace["build_company_tasks_tool"].invoke({"run_id": "unknown-run"})

    assert result["status"] == "missing_context"
    assert result["tasks"] == []
    assert result["task_gate"]["ready"] is False
    assert namespace["route_after_task_gate"](result["task_gate"]) == "stop_invalid_tasks"


def test_company_limit_is_enforced_before_task_construction():
    """Reject a validated context whose unique company count exceeds the notebook limit."""
    namespace = _task_namespace()
    plan = _plan("Microsoft", "Google", "NVIDIA", "Amazon", "IBM", "Pfizer")
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    selection = namespace["select_industry_profiles"](
        [company["company_id"] for company in companies]
    )

    with pytest.raises(ValueError, match="Company limit exceeded"):
        namespace["register_task_planning_context"](
            "run-too-many", plan, companies, selection
        )


def test_task_gate_rejects_missing_duplicate_and_injected_permissions():
    """Block task-list tampering before future LangGraph fan-out."""
    namespace = _task_namespace()
    plan = _plan("Microsoft", "Pfizer", dimensions=["financial_strength"])
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    tasks = namespace["build_company_tasks"](plan, companies, "run-gate")
    validate = namespace["validate_task_gate"]

    assert validate(tasks, companies)["ready"] is True

    missing = validate(tasks[:1], companies)
    assert missing["ready"] is False
    assert "Missing tasks" in missing["message"]

    duplicated = validate([tasks[0], deepcopy(tasks[0])], companies)
    assert duplicated["ready"] is False
    assert "duplicate companies" in duplicated["message"]

    injected = deepcopy(tasks)
    injected[0]["allowed_tools"].append("query_biopharma_rag")
    injected_result = validate(injected, companies)
    assert injected_result["ready"] is False
    assert "disallowed tools" in injected_result["message"]


def test_scoring_request_expands_to_complete_profile_contract():
    """Include every required shared and industry dimension when scoring is requested."""
    namespace = _task_namespace()
    plan = _plan("Microsoft", dimensions=["valuation"], query_type="rank", scoring=True)
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    task = namespace["build_company_tasks"](plan, companies, "run-score")[0]
    profile = namespace["get_industry_profile"]("technology.ai.v1")

    assert task["shared_dimensions"] == profile["shared_dimensions"]
    assert task["industry_dimensions"] == profile["industry_dimensions"]

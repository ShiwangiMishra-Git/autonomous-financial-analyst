"""Deterministic tests for F00/F01 in the designated working notebook."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from langchain_core.messages import HumanMessage


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _load_notebook_cells():
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    return notebook, {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}


def _state_namespace():
    _, cells = _load_notebook_cells()
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
    return namespace


def test_f00_cells_are_present_once_and_before_summary():
    notebook, _ = _load_notebook_cells()
    ids = [cell.get("id") for cell in notebook["cells"]]

    expected = [
        "multiindustry_section3_intro",
        "multiindustry_f01_intro",
        "multiindustry_state_contracts",
        "multiindustry_f01_smoke",
    ]
    for cell_id in expected:
        assert ids.count(cell_id) == 1

    assert ids.index("multiindustry_section3_intro") < ids.index("pZskPu1tn3Q-")


def test_initialize_research_run_preserves_conversation_fields_by_omission():
    namespace = _state_namespace()
    initialize = namespace["initialize_research_run"]
    reset_type = namespace["ResetCompanyResults"]

    prior_state = {
        "messages": [HumanMessage(content="Compare Microsoft and Pfizer")],
        "remembered_company_ids": ["MSFT"],
        "last_profile_ids": ["technology.ai.v1"],
        "company_results": {"MSFT": {"run_id": "old"}},
    }
    update = initialize(prior_state)

    assert update["original_query"] == "Compare Microsoft and Pfizer"
    assert isinstance(update["company_results"], reset_type)
    assert "messages" not in update
    assert "remembered_company_ids" not in update
    assert "last_profile_ids" not in update


def test_initialize_research_run_clears_all_request_fields():
    namespace = _state_namespace()
    update = namespace["initialize_research_run"]({"messages": []})

    assert update["plan"] is None
    assert update["resolved_companies"] == []
    assert update["company_tasks"] == []
    assert update["normalized_company_results"] == {}
    assert update["fan_in_normalization"] is None
    assert update["comparison_mode"] is None
    assert update["comparison_route_status"] is None
    assert update["scoring_eligibility"] is None
    assert update["scores"] is None
    assert update["final_answer"] is None
    assert update["validation_retry_count"] == 0
    assert update["validation_errors"] == []
    assert update["run_errors"] == []
    assert update["run_id"]
    assert update["run_started_at"].endswith("+00:00")


def test_each_initialization_gets_a_new_run_id():
    namespace = _state_namespace()
    initialize = namespace["initialize_research_run"]

    assert initialize({"messages": []})["run_id"] != initialize({"messages": []})["run_id"]


def test_company_result_reducer_can_reset_and_merge_parallel_updates():
    namespace = _state_namespace()
    merge = namespace["merge_company_results"]
    reset = namespace["ResetCompanyResults"]()

    old = {"IBM": {"run_id": "old"}}
    assert merge(old, reset) == {}

    msft = {"MSFT": {"run_id": "new", "status": "success"}}
    pfe = {"PFE": {"run_id": "new", "status": "partial"}}
    combined = merge(merge({}, msft), pfe)

    assert set(combined) == {"MSFT", "PFE"}
    assert combined["MSFT"]["status"] == "success"
    assert combined["PFE"]["status"] == "partial"


def test_worker_and_orchestrator_contracts_expose_required_fields():
    namespace = _state_namespace()
    worker_fields = namespace["CompanyWorkerState"].__annotations__
    orchestrator_fields = namespace["OrchestratorState"].__annotations__

    assert {
        "task",
        "messages",
        "evidence",
        "industry_signals",
        "tool_round_count",
        "validation_retry_count",
        "result",
        "errors",
    } <= set(worker_fields)

    assert {
        "messages",
        "run_id",
        "plan",
        "resolved_companies",
        "company_tasks",
        "company_results",
        "normalized_company_results",
        "fan_in_normalization",
        "comparison_mode",
        "comparison_route_status",
        "scoring_eligibility",
        "scores",
        "final_answer",
    } <= set(orchestrator_fields)

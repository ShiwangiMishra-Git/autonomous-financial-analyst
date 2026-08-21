"""Focused deterministic tests for F12 mandatory comparison-mode routing."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from scripts.implement_multiindustry_f12_routing import F12_ROUTING_CODE


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _namespace():
    """Execute the stable F01 contract and the isolated F12 routing source."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "get_industry_profile": lambda profile_id: {
            "profile_id": profile_id,
            "scoring_enabled": profile_id in {
                "technology.ai.v1", "healthcare.biopharma.v1",
            },
            "rubric_id": {
                "technology.ai.v1": "technology.ai.score.v1",
                "healthcare.biopharma.v1": "healthcare.biopharma.score.v1",
            }.get(profile_id),
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
        exec(F12_ROUTING_CODE, namespace)
    return namespace


def _company(ticker: str, company_id: str, profile_id: str):
    """Return a minimal resolved-company fixture satisfying the stable F01 contract."""
    return {
        "company_id": company_id,
        "ticker": ticker,
        "company_name": company_id.title(),
        "aliases": [],
        "exchange": "TEST",
        "industry": profile_id.split(".")[0],
        "sub_industry": profile_id.split(".")[1],
        "profile_id": profile_id,
        "resolution_status": "resolved",
        "resolution_message": None,
    }


def _task(run_id: str, ticker: str, company_id: str, profile_id: str):
    """Return a current-run company-task fixture used as routing authority."""
    return {
        "run_id": run_id,
        "company": _company(ticker, company_id, profile_id),
        "query_plan": {},
        "shared_dimensions": [],
        "industry_dimensions": [],
        "unsupported_dimensions": [],
        "allowed_tools": [],
    }


def _result(run_id: str, ticker: str, company_id: str, profile_id: str, status="success"):
    """Return a minimal normalized research-result fixture."""
    return {
        "run_id": run_id,
        "company": _company(ticker, company_id, profile_id),
        "profile_id": profile_id,
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": [],
        "missing_dimensions": [],
        "errors": [],
        "status": status,
    }


def _decision(entries):
    """Validate routing for ``(ticker, company_id, profile_id)`` fixture entries."""
    namespace = _namespace()
    run_id = "run-current"
    tasks = [_task(run_id, *entry) for entry in entries]
    results = {entry[0]: _result(run_id, *entry) for entry in entries}
    return namespace, namespace["validate_comparison_routing"](results, run_id, tasks), tasks, results


def test_one_company_routes_to_single():
    """Select single mode for exactly one validated company result."""
    namespace, decision, tasks, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
    ])
    assert decision["ready"] is True
    assert decision["comparison_mode"] == "single"
    state = {"run_id": "run-current", "company_tasks": tasks, "company_results": results}
    update = namespace["mandatory_comparison_mode_node"](state)
    assert update["comparison_mode"] == "single"
    assert update["comparison_route_status"]["ready"] is True
    assert update["scoring_eligibility"]["eligible"] is False
    assert namespace["route_after_comparison_mode"](state) == "single"


@pytest.mark.parametrize(
    "entries",
    [
        [
            ("MSFT", "microsoft", "technology.ai.v1"),
            ("NVDA", "nvidia", "technology.ai.v1"),
        ],
        [
            ("PFE", "pfizer", "healthcare.biopharma.v1"),
            ("MRK", "merck", "healthcare.biopharma.v1"),
        ],
    ],
)
def test_multiple_companies_with_one_exact_profile_route_to_same_profile(entries):
    """Use same_profile for both technology and biopharma peer comparisons."""
    namespace, decision, tasks, results = _decision(entries)
    assert decision["comparison_mode"] == "same_profile"
    assert namespace["select_comparison_mode"](results) == "same_profile"
    assert namespace["route_after_comparison_mode"]({
        "run_id": "run-current", "company_tasks": tasks, "company_results": results,
    }) == "same_profile"


@pytest.mark.parametrize(
    "entries",
    [
        [
            ("MSFT", "microsoft", "technology.ai.v1"),
            ("PFE", "pfizer", "healthcare.biopharma.v1"),
        ],
        [
            ("PFE", "pfizer", "healthcare.biopharma.v1"),
            ("UNH", "united_health", "healthcare.managed-care.v1"),
        ],
    ],
)
def test_multiple_exact_profiles_route_to_cross_profile(entries):
    """Use cross_profile across industries or across sub-industry profiles."""
    namespace, decision, tasks, results = _decision(entries)
    assert decision["comparison_mode"] == "cross_profile"
    assert namespace["select_comparison_mode"](results) == "cross_profile"
    assert namespace["route_after_comparison_mode"]({
        "run_id": "run-current", "company_tasks": tasks, "company_results": results,
    }) == "cross_profile"


def test_missing_and_unexpected_results_stop_at_mandatory_router():
    """Reject any fan-in map that does not exactly cover the authoritative tasks."""
    namespace, _, tasks, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
        ("PFE", "pfizer", "healthcare.biopharma.v1"),
    ])
    missing = {"MSFT": results["MSFT"]}
    missing_decision = namespace["validate_comparison_routing"](
        missing, "run-current", tasks,
    )
    assert missing_decision["ready"] is False
    assert "Missing company results" in missing_decision["message"]
    assert namespace["route_after_comparison_mode"]({
        "run_id": "run-current", "company_tasks": tasks, "company_results": missing,
    }) == "bounded_stop"

    unexpected = dict(results)
    unexpected["IBM"] = _result(
        "run-current", "IBM", "ibm", "technology.ai.v1",
    )
    unexpected_decision = namespace["validate_comparison_routing"](
        unexpected, "run-current", tasks,
    )
    assert unexpected_decision["ready"] is False
    assert "Unexpected company results" in unexpected_decision["message"]


def test_cross_run_results_and_tasks_stop():
    """Reject stale result or task state before comparison synthesis."""
    namespace, _, tasks, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
        ("PFE", "pfizer", "healthcare.biopharma.v1"),
    ])
    stale_results = dict(results)
    stale_results["PFE"] = _result(
        "run-old", "PFE", "pfizer", "healthcare.biopharma.v1",
    )
    decision = namespace["validate_comparison_routing"](
        stale_results, "run-current", tasks,
    )
    assert decision["ready"] is False
    assert any("wrong run_id" in error or "multiple run_ids" in error for error in decision["errors"])
    with pytest.raises(ValueError, match="multiple run_ids"):
        namespace["select_comparison_mode"](stale_results)

    stale_tasks = list(tasks)
    stale_tasks[0] = {**stale_tasks[0], "run_id": "run-old"}
    assert namespace["validate_comparison_routing"](
        results, "run-current", stale_tasks,
    )["ready"] is False


@pytest.mark.parametrize(
    "mutate, expected_error",
    [
        (lambda result: result.pop("profile_id"), "missing profile_id"),
        (lambda result: result.update(status="unknown"), "invalid status"),
        (lambda result: result["company"].update(ticker="WRONG"), "does not match"),
        (lambda result: result["company"].update(resolution_status="ambiguous"), "not canonically resolved"),
    ],
)
def test_invalid_result_identity_or_status_stops(mutate, expected_error):
    """Fail closed on malformed normalized-result routing fields."""
    namespace, _, tasks, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
    ])
    mutate(results["MSFT"])
    decision = namespace["validate_comparison_routing"](
        results, "run-current", tasks,
    )
    assert decision["ready"] is False
    assert expected_error in decision["message"]
    assert namespace["route_after_comparison_mode"]({
        "run_id": "run-current", "company_tasks": tasks, "company_results": results,
    }) == "bounded_stop"


def test_mode_selection_is_order_independent_and_allows_contained_failure_status():
    """Keep routing deterministic while leaving scoring eligibility to its later guard."""
    namespace, _, tasks, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
        ("NVDA", "nvidia", "technology.ai.v1"),
    ])
    results["NVDA"] = _result(
        "run-current", "NVDA", "nvidia", "technology.ai.v1", status="failed",
    )
    reversed_results = dict(reversed(list(results.items())))
    first = namespace["validate_comparison_routing"](results, "run-current", tasks)
    second = namespace["validate_comparison_routing"](reversed_results, "run-current", tasks)
    assert first["ready"] is second["ready"] is True
    assert first["comparison_mode"] == second["comparison_mode"] == "same_profile"


def test_empty_result_set_never_selects_a_mode():
    """Reject zero-company input rather than treating it as a comparison."""
    namespace = _namespace()
    with pytest.raises(ValueError, match="missing"):
        namespace["select_comparison_mode"]({})
    decision = namespace["validate_comparison_routing"]({}, "run-current", [])
    assert decision["comparison_mode"] is None
    assert decision["route"] == "bounded_stop"


def test_prior_normalization_error_cannot_be_bypassed_by_valid_routing_fields():
    """Honor an upstream normalization failure even when identities could select a mode."""
    namespace, _, tasks, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
    ])
    state = {
        "run_id": "run-current",
        "company_tasks": tasks,
        "company_results": results,
        "validation_errors": ["Evidence normalization failed"],
    }
    update = namespace["mandatory_comparison_mode_node"](state)
    assert update["comparison_mode"] is None
    assert update["comparison_route_status"]["ready"] is True
    assert update["validation_errors"] == ["Evidence normalization failed"]
    assert namespace["route_after_comparison_mode"](state) == "bounded_stop"


def test_complete_technology_peers_are_scoring_eligible():
    """Allow numeric scoring only for complete peers with an enabled versioned rubric."""
    namespace, decision, _, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
        ("NVDA", "nvidia", "technology.ai.v1"),
    ])
    eligibility = namespace["check_scoring_eligibility"](
        results, decision["comparison_mode"],
    )
    assert eligibility["eligible"] is True
    assert eligibility["rubric_id"] == "technology.ai.score.v1"


def test_partial_peers_disable_scoring_but_complete_biopharma_peers_are_eligible():
    """Block incomplete inputs while enabling the versioned biopharma baseline."""
    namespace, decision, _, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
        ("NVDA", "nvidia", "technology.ai.v1"),
    ])
    results["NVDA"]["status"] = "partial"
    results["NVDA"]["missing_dimensions"] = ["market_cap"]
    partial = namespace["check_scoring_eligibility"](
        results, decision["comparison_mode"],
    )
    assert partial["eligible"] is False
    assert partial["excluded_companies"] == ["NVDA"]

    _, bio_decision, _, bio_results = _decision([
        ("PFE", "pfizer", "healthcare.biopharma.v1"),
        ("MRK", "merck", "healthcare.biopharma.v1"),
    ])
    biopharma = namespace["check_scoring_eligibility"](
        bio_results, bio_decision["comparison_mode"],
    )
    assert biopharma["eligible"] is True
    assert biopharma["rubric_id"] == "healthcare.biopharma.score.v1"


def test_cross_profile_and_unknown_run_tool_context_disable_scoring():
    """Prevent universal cross-industry scores and reject missing guarded-tool context."""
    namespace, decision, _, results = _decision([
        ("MSFT", "microsoft", "technology.ai.v1"),
        ("PFE", "pfizer", "healthcare.biopharma.v1"),
    ])
    eligibility = namespace["check_scoring_eligibility"](
        results, decision["comparison_mode"],
    )
    assert eligibility["eligible"] is False
    assert "no validated universal" in eligibility["reason"]
    missing = namespace["check_scoring_eligibility_tool"].invoke({"run_id": "missing"})
    assert missing["eligible"] is False
    assert "No validated scoring context" in missing["reason"]

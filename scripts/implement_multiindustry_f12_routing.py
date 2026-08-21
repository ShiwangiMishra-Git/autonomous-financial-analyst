"""Define and idempotently integrate F12 deterministic comparison-mode routing cells."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f12_normalization_smoke"


F12_ROUTING_INTRO = """## Section 3.12: Mandatory Comparison-Mode Routing

After fan-in normalization, a deterministic gate verifies that the result set belongs to the
current run and exactly covers the expected company tasks. It then selects one of three modes:
`single`, `same_profile`, or `cross_profile`. Missing, extra, malformed, or cross-run results stop
at the guardrail instead of reaching a synthesis agent.
"""


F12_ROUTING_CODE = r'''from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from langchain_core.tools import tool


_VALID_COMPARISON_MODES = frozenset({"single", "same_profile", "cross_profile"})
_VALID_COMPANY_RESULT_STATUSES = frozenset({"success", "partial", "failed"})


def _routing_text(value: Any) -> str | None:
    """Return a stripped non-empty string, otherwise ``None``."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _unique_routing_errors(errors: Sequence[str]) -> list[str]:
    """Preserve deterministic error order while removing duplicate messages."""
    return list(dict.fromkeys(errors))


def _validate_mode_result_map(
    results: Mapping[str, CompanyResearchResult] | Any,
) -> list[str]:
    """Validate the identity fields needed for safe comparison-mode selection.

    This intentionally validates only the routing surface. F12 fan-in normalization owns
    evidence, freshness, dimension, and signal-reference validation.
    """
    if not isinstance(results, Mapping):
        return ["Company results must be a ticker-keyed mapping"]
    if not results:
        return ["Company results are missing"]

    errors: list[str] = []
    run_ids: set[str] = set()
    company_ids: set[str] = set()
    for result_key, result in sorted(results.items(), key=lambda item: str(item[0])):
        ticker = _routing_text(result_key)
        if ticker is None:
            errors.append("Company result contains an invalid ticker key")
            continue
        if not isinstance(result, Mapping):
            errors.append(f"Result {ticker} must be a mapping")
            continue

        run_id = _routing_text(result.get("run_id"))
        profile_id = _routing_text(result.get("profile_id"))
        company = result.get("company")
        status = result.get("status")
        if run_id is None:
            errors.append(f"Result {ticker} is missing run_id")
        else:
            run_ids.add(run_id)
        if profile_id is None:
            errors.append(f"Result {ticker} is missing profile_id")
        if status not in _VALID_COMPANY_RESULT_STATUSES:
            errors.append(f"Result {ticker} has invalid status {status!r}")
        if not isinstance(company, Mapping):
            errors.append(f"Result {ticker} is missing company identity")
            continue

        company_ticker = _routing_text(company.get("ticker"))
        company_id = _routing_text(company.get("company_id"))
        company_profile = _routing_text(company.get("profile_id"))
        if company_ticker != ticker:
            errors.append(f"Result key {ticker} does not match company ticker {company_ticker!r}")
        if company_id is None:
            errors.append(f"Result {ticker} is missing company_id")
        elif company_id in company_ids:
            errors.append(f"Duplicate company identity {company_id!r} in result set")
        else:
            company_ids.add(company_id)
        if company_profile != profile_id:
            errors.append(f"Result {ticker} profile does not match company profile")
        if company.get("resolution_status") != "resolved":
            errors.append(f"Result {ticker} company is not canonically resolved")

    if len(run_ids) > 1:
        errors.append("Company results span multiple run_ids")
    return _unique_routing_errors(errors)


def select_comparison_mode(
    results: Mapping[str, CompanyResearchResult],
) -> ComparisonMode:
    """Select exactly one deterministic mode from a structurally valid result map.

    One company selects ``single``. Multiple companies sharing one exact versioned profile
    select ``same_profile``. Multiple exact profile IDs select ``cross_profile`` regardless
    of whether those profiles belong to the same broad industry.

    Raises:
        ValueError: If the result set is empty, malformed, duplicated, or cross-run.
    """
    errors = _validate_mode_result_map(results)
    if errors:
        raise ValueError("Invalid comparison result set: " + "; ".join(errors))
    if len(results) == 1:
        return "single"
    profiles = {result["profile_id"] for result in results.values()}
    return "same_profile" if len(profiles) == 1 else "cross_profile"


def validate_comparison_routing(
    results: Mapping[str, CompanyResearchResult] | Any,
    run_id: str,
    expected_tasks: Sequence[CompanyTask] | Any,
) -> dict[str, Any]:
    """Validate current-run fan-in coverage and return a fail-closed routing decision.

    The expected task set is the authority for permitted tickers, company identities, and
    profile identities. The model cannot add or omit companies at this boundary.
    """
    errors = _validate_mode_result_map(results)
    current_run_id = _routing_text(run_id)
    if current_run_id is None:
        errors.append("Current run_id is missing")

    if (
        not isinstance(expected_tasks, Sequence)
        or isinstance(expected_tasks, (str, bytes))
        or not expected_tasks
    ):
        errors.append("Expected company tasks are missing")
        expected_tasks = []

    expected: dict[str, Mapping[str, Any]] = {}
    expected_company_ids: set[str] = set()
    for index, task in enumerate(expected_tasks):
        if not isinstance(task, Mapping):
            errors.append(f"Expected task {index} must be a mapping")
            continue
        task_run_id = _routing_text(task.get("run_id"))
        company = task.get("company")
        if task_run_id != current_run_id:
            errors.append(f"Expected task {index} has the wrong run_id")
        if not isinstance(company, Mapping):
            errors.append(f"Expected task {index} is missing company identity")
            continue
        ticker = _routing_text(company.get("ticker"))
        company_id = _routing_text(company.get("company_id"))
        profile_id = _routing_text(company.get("profile_id"))
        if ticker is None or company_id is None or profile_id is None:
            errors.append(f"Expected task {index} has incomplete company identity")
            continue
        if ticker in expected:
            errors.append(f"Duplicate expected ticker {ticker!r}")
            continue
        if company_id in expected_company_ids:
            errors.append(f"Duplicate expected company identity {company_id!r}")
            continue
        expected[ticker] = company
        expected_company_ids.add(company_id)

    result_keys = set(results) if isinstance(results, Mapping) else set()
    expected_keys = set(expected)
    missing = sorted(expected_keys - result_keys)
    unexpected = sorted(result_keys - expected_keys, key=str)
    if missing:
        errors.append(f"Missing company results: {missing}")
    if unexpected:
        errors.append(f"Unexpected company results: {unexpected}")

    if isinstance(results, Mapping):
        for ticker in sorted(result_keys & expected_keys):
            result = results[ticker]
            if not isinstance(result, Mapping):
                continue
            expected_company = expected[ticker]
            if result.get("run_id") != current_run_id:
                errors.append(f"Result {ticker} has the wrong run_id")
            company = result.get("company")
            if not isinstance(company, Mapping):
                continue
            if company.get("company_id") != expected_company["company_id"]:
                errors.append(f"Result {ticker} has the wrong company identity")
            if result.get("profile_id") != expected_company["profile_id"]:
                errors.append(f"Result {ticker} has the wrong expected profile")

    mode: ComparisonMode | None = None
    if not errors:
        try:
            mode = select_comparison_mode(results)
        except ValueError as exc:
            errors.append(str(exc))

    errors = _unique_routing_errors(errors)
    ready = not errors and mode in _VALID_COMPARISON_MODES
    return {
        "ready": ready,
        "status": "ready" if ready else "invalid_result_set",
        "comparison_mode": mode if ready else None,
        "route": mode if ready else "bounded_stop",
        "errors": errors,
        "message": (
            f"Comparison routing ready: {mode}"
            if ready
            else "Comparison routing blocked: " + "; ".join(errors)
        ),
    }


def check_scoring_eligibility(
    results: Mapping[str, CompanyResearchResult],
    mode: ComparisonMode,
    profile_lookup: Callable[[str], IndustryProfile] | None = None,
) -> ScoringEligibility:
    """Decide whether normalized results may enter deterministic sector scoring.

    Scoring is limited to complete same-profile comparisons whose registered profile explicitly
    enables a versioned rubric. Single-company, cross-profile, partial, failed, or incomplete
    comparisons remain valid narrative routes but cannot receive a numeric sector score.
    """
    lookup = profile_lookup or get_industry_profile
    excluded: list[str] = []
    missing: dict[str, list[str]] = {}
    for ticker, result in sorted(results.items()):
        requirements: list[str] = []
        if result.get("status") != "success":
            requirements.append(f"result_status:{result.get('status')}")
        requirements.extend(
            f"missing_dimension:{dimension}"
            for dimension in result.get("missing_dimensions", [])
        )
        if result.get("errors"):
            requirements.append("worker_or_normalization_errors")
        if requirements:
            excluded.append(ticker)
            missing[ticker] = list(dict.fromkeys(requirements))

    if mode == "single":
        return {
            "eligible": False, "rubric_id": None,
            "reason": "Numeric sector comparison requires at least two companies.",
            "excluded_companies": sorted(results),
            "missing_requirements": missing,
        }
    if mode == "cross_profile":
        return {
            "eligible": False, "rubric_id": None,
            "reason": "Cross-profile comparisons have no validated universal scoring rubric.",
            "excluded_companies": sorted(results),
            "missing_requirements": missing,
        }
    if mode != "same_profile" or len(results) < 2:
        return {
            "eligible": False, "rubric_id": None,
            "reason": "Comparison mode is not eligible for sector scoring.",
            "excluded_companies": sorted(results),
            "missing_requirements": missing,
        }
    if excluded:
        return {
            "eligible": False, "rubric_id": None,
            "reason": "All companies must have complete validated results before scoring.",
            "excluded_companies": excluded,
            "missing_requirements": missing,
        }

    profile_ids = {result["profile_id"] for result in results.values()}
    if len(profile_ids) != 1:
        return {
            "eligible": False, "rubric_id": None,
            "reason": "Sector scoring requires one exact versioned profile.",
            "excluded_companies": sorted(results),
            "missing_requirements": missing,
        }
    profile = lookup(next(iter(profile_ids)))
    if not profile.get("scoring_enabled") or not profile.get("rubric_id"):
        return {
            "eligible": False, "rubric_id": None,
            "reason": f"Profile {profile['profile_id']} does not yet enable validated scoring.",
            "excluded_companies": sorted(results),
            "missing_requirements": missing,
        }
    return {
        "eligible": True,
        "rubric_id": profile["rubric_id"],
        "reason": "Complete same-profile results passed deterministic scoring eligibility.",
        "excluded_companies": [],
        "missing_requirements": {},
    }


_F12_SCORING_CONTEXTS: dict[str, ScoringEligibility] = {}


def register_scoring_eligibility_context(
    run_id: str,
    eligibility: ScoringEligibility,
) -> None:
    """Store one defensive current-run eligibility decision for the guarded tool wrapper."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    _F12_SCORING_CONTEXTS[run_id] = deepcopy(eligibility)


def clear_scoring_eligibility_context(run_id: str) -> bool:
    """Remove a notebook-local eligibility context and report whether it existed."""
    return _F12_SCORING_CONTEXTS.pop(run_id, None) is not None


@tool
def check_scoring_eligibility_tool(run_id: str) -> ScoringEligibility:
    """Return the mandatory scoring decision stored for a validated current run.

    The caller supplies only ``run_id``; raw evidence, metrics, profiles, weights, and proposed
    scores cannot enter through this tool contract.
    """
    decision = _F12_SCORING_CONTEXTS.get(run_id)
    if decision is None:
        return {
            "eligible": False, "rubric_id": None,
            "reason": f"No validated scoring context exists for run_id {run_id!r}.",
            "excluded_companies": [],
            "missing_requirements": {"run": ["validated_f12_context"]},
        }
    return deepcopy(decision)


def mandatory_comparison_mode_node(state: OrchestratorState) -> dict[str, Any]:
    """LangGraph node that writes a mode only after current-run coverage validation."""
    prior_errors = list(state.get("validation_errors", []))
    normalized_results = state.get("normalized_company_results") or state.get(
        "company_results", {}
    )
    decision = validate_comparison_routing(
        normalized_results,
        state.get("run_id", ""),
        state.get("company_tasks", []),
    )
    if decision["ready"] and not prior_errors:
        eligibility = check_scoring_eligibility(
            normalized_results, decision["comparison_mode"],
        )
        register_scoring_eligibility_context(state["run_id"], eligibility)
        return {
            "comparison_mode": decision["comparison_mode"],
            "comparison_route_status": decision,
            "scoring_eligibility": eligibility,
        }
    return {
        "comparison_mode": None,
        "comparison_route_status": decision,
        "scoring_eligibility": None,
        "validation_errors": _unique_routing_errors(prior_errors + decision["errors"]),
    }


def route_after_comparison_mode(state: OrchestratorState) -> str:
    """Mandatory conditional-edge router that revalidates rather than trusting model state."""
    if state.get("validation_errors"):
        return "bounded_stop"
    normalized_results = state.get("normalized_company_results") or state.get(
        "company_results", {}
    )
    decision = validate_comparison_routing(
        normalized_results,
        state.get("run_id", ""),
        state.get("company_tasks", []),
    )
    if not decision["ready"]:
        return "bounded_stop"
    return decision["route"]


print("✅ F12 deterministic comparison-mode gate and router defined")
'''


F12_ROUTING_SMOKE = r'''# F12 routing smoke: current-run cross-profile results route deterministically.
_f12_routing_tasks = list(_f11_state["company_tasks"])
_f12_routing_results = dict(_f11_state["company_results"])
_f12_routing_decision = validate_comparison_routing(
    _f12_routing_results,
    _f11_state["run_id"],
    _f12_routing_tasks,
)
assert _f12_routing_decision["ready"] is True
assert _f12_routing_decision["comparison_mode"] == "cross_profile"
assert route_after_comparison_mode({
    **_f11_state,
    "company_tasks": _f12_routing_tasks,
    "company_results": _f12_routing_results,
}) == "cross_profile"

print("✅ F12 routing smoke passed: Microsoft + Pfizer selected cross_profile")
'''


CELL_SPECS = [
    ("multiindustry_f12_routing_intro", "markdown", F12_ROUTING_INTRO),
    ("multiindustry_comparison_mode_routing", "code", F12_ROUTING_CODE),
    ("multiindustry_f12_routing_smoke", "code", F12_ROUTING_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create a notebook cell with a stable identifier."""
    cell = (
        nbformat.v4.new_markdown_cell(source=source)
        if cell_type == "markdown"
        else nbformat.v4.new_code_cell(source=source)
    )
    cell["id"] = cell_id
    return cell


def integrate_f12_routing_cells(notebook_path: Path = NOTEBOOK_PATH) -> None:
    """Insert or refresh the F12 routing cells without duplicating notebook cells."""
    notebook = nbformat.read(notebook_path, as_version=4)
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}
    for cell_id, cell_type, source in CELL_SPECS:
        existing = cells_by_id.get(cell_id)
        if existing is not None:
            existing["cell_type"] = cell_type
            existing["source"] = source
            if cell_type == "code":
                existing["execution_count"] = None
                existing["outputs"] = []
    missing = [spec for spec in CELL_SPECS if spec[0] not in cells_by_id]
    if missing:
        index = next(
            i for i, cell in enumerate(notebook.cells)
            if cell.get("id") == INSERT_AFTER_CELL_ID
        ) + 1
        notebook.cells[index:index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")
    nbformat.write(notebook, notebook_path)


def main() -> None:
    """Integrate the F12 comparison-mode routing cells into the working notebook."""
    integrate_f12_routing_cells()
    print(f"Updated {NOTEBOOK_PATH.name}: F12 routing cells are present")


if __name__ == "__main__":
    main()

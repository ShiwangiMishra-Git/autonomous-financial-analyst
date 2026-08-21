"""Idempotently add the F11 parent Send-based multi-company orchestrator."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f10_smoke"


F11_INTRO = """## Section 3.11: Parent Multi-Company Fan-Out and Fan-In

F11 replaces notebook-side company loops with a parent LangGraph. Planning, company resolution,
profile selection, and task construction are separate guarded stages. Mandatory gates must pass
before LangGraph `Send` creates one isolated branch per company.

Each branch reuses the F10 profile-configured worker. A reset-aware reducer merges results by
ticker, so same-industry and cross-industry requests use the same orchestration path. Branch
failures become explicit failed company results and do not erase successful peers. Notebook-local
company, concurrency, recursion, and worker-round limits remain deterministic.
"""


F11_CODE = r'''from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from langchain_core.messages import HumanMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph


class NotebookOrchestrator:
    """Apply notebook-local execution limits to a compiled parent LangGraph.

    The wrapper deliberately exposes the familiar ``invoke``, ``stream``, and ``get_graph``
    methods while ensuring callers cannot raise concurrency or recursion above local limits.
    """

    def __init__(self, graph: Any, max_concurrency: int, recursion_limit: int):
        """Store the compiled graph and validated local execution ceilings."""
        self.graph = graph
        self.max_concurrency = max_concurrency
        self.recursion_limit = recursion_limit

    def _bounded_config(self, config: Mapping[str, Any] | None) -> dict[str, Any]:
        """Copy an invocation config and cap its concurrency and recursion values."""
        bounded = dict(config or {})
        requested_concurrency = int(bounded.get("max_concurrency", self.max_concurrency))
        requested_recursion = int(bounded.get("recursion_limit", self.recursion_limit))
        bounded["max_concurrency"] = min(max(1, requested_concurrency), self.max_concurrency)
        bounded["recursion_limit"] = min(max(1, requested_recursion), self.recursion_limit)
        return bounded

    def invoke(
        self,
        input_state: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the parent graph under notebook-local execution ceilings."""
        return self.graph.invoke(input_state, config=self._bounded_config(config), **kwargs)

    def stream(
        self,
        input_state: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Stream parent-graph events under notebook-local execution ceilings."""
        return self.graph.stream(input_state, config=self._bounded_config(config), **kwargs)

    def get_graph(self, *args: Any, **kwargs: Any):
        """Return the compiled graph visualization object."""
        return self.graph.get_graph(*args, **kwargs)


def _failed_company_result(task: CompanyTask, error: str) -> CompanyResearchResult:
    """Create an identity-preserving failed result for one contained branch error."""
    return {
        "run_id": task["run_id"],
        "company": deepcopy(task["company"]),
        "profile_id": task["company"]["profile_id"],
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": [],
        "missing_dimensions": list(dict.fromkeys(
            task["shared_dimensions"]
            + task["industry_dimensions"]
            + task["unsupported_dimensions"]
        )),
        "errors": [error],
        "status": "failed",
    }


def _validate_fan_in_results(
    tasks: list[CompanyTask],
    results: CompanyResultMap,
    run_id: str,
) -> list[str]:
    """Validate branch coverage, run identity, profile identity, and evidence isolation."""
    errors: list[str] = []
    expected = {task["company"]["ticker"]: task for task in tasks}
    missing = sorted(set(expected) - set(results))
    unexpected = sorted(set(results) - set(expected))
    if missing:
        errors.append(f"Missing company results: {missing}")
    if unexpected:
        errors.append(f"Unexpected company results: {unexpected}")

    for ticker, result in results.items():
        task = expected.get(ticker)
        if task is None:
            continue
        company = task["company"]
        if result.get("run_id") != run_id:
            errors.append(f"Result {ticker} has the wrong run_id")
        if result.get("company", {}).get("company_id") != company["company_id"]:
            errors.append(f"Result {ticker} has the wrong company identity")
        if result.get("profile_id") != company["profile_id"]:
            errors.append(f"Result {ticker} has the wrong profile identity")
        for evidence in result.get("evidence", []):
            if evidence.get("run_id") != run_id:
                errors.append(f"Evidence for {ticker} has the wrong run_id")
            if evidence.get("company_id") != company["company_id"]:
                errors.append(f"Evidence for {ticker} crossed the company boundary")
    return errors


def create_multi_company_orchestrator(
    planner_model: Any,
    worker_model_factory: Callable[[CompanyTask, IndustryProfile], Any],
    worker_tools_factory: Callable[[CompanyTask, IndustryProfile], dict[str, Any]],
    signal_extractor_factory: Callable[
        [CompanyTask, IndustryProfile],
        Callable[[ResolvedCompany, list[EvidenceRecord]], dict[str, Any]] | None,
    ] | None = None,
    max_companies: int = MAX_COMPANIES_PER_QUERY,
    max_concurrency: int = 2,
    recursion_limit: int = 50,
    worker_max_tool_rounds: int = 4,
    enable_f12: bool = False,
) -> NotebookOrchestrator:
    """Create the guarded parent graph for one or more isolated company workers.

    Args:
        planner_model: Structured-output model used by the F03 query planner.
        worker_model_factory: Creates a fresh worker model for one task and profile.
        worker_tools_factory: Supplies the complete allowed callable registry for one branch.
        signal_extractor_factory: Optionally supplies an injected profile signal extractor.
        max_companies: Maximum company branches permitted for a request.
        max_concurrency: Maximum worker branches executing concurrently in this notebook.
        recursion_limit: Maximum parent graph supersteps.
        worker_max_tool_rounds: Per-company autonomous source-tool round ceiling.
        enable_f12: Add mandatory fan-in normalization, comparison routing, and eligibility.

    Returns:
        A compiled parent graph wrapped with enforced local execution limits.
    """
    if not 1 <= max_companies <= MAX_COMPANIES_PER_QUERY:
        raise ValueError(
            f"max_companies must be between 1 and {MAX_COMPANIES_PER_QUERY}"
        )
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if recursion_limit < 10:
        raise ValueError("recursion_limit must be at least 10")
    if worker_max_tool_rounds < 1:
        raise ValueError("worker_max_tool_rounds must be at least 1")

    def initialize_run(state: OrchestratorState) -> dict[str, Any]:
        """Reset request-scoped state while preserving conversation-scoped memory."""
        return initialize_research_run(state)

    def coordinator_plan(state: OrchestratorState) -> dict[str, Any]:
        """Interpret the free-text request into a deterministically validated query plan."""
        plan = plan_query(
            state["original_query"],
            conversation_context=state.get("messages", ()),
            remembered_company_ids=state.get("remembered_company_ids", ()),
            model=planner_model,
        )
        if len(plan["company_mentions"]) > max_companies:
            return {
                "plan": plan,
                "validation_errors": [
                    f"Company limit exceeded: {len(plan['company_mentions'])} > {max_companies}"
                ],
            }
        return {"plan": plan}

    def resolve_companies_node(state: OrchestratorState) -> dict[str, Any]:
        """Invoke the guarded resolver only when planning produced an in-budget plan."""
        if state.get("validation_errors"):
            return {
                "resolution_result": {
                    "ready": False, "status": "company_limit_exceeded",
                    "resolved_companies": [], "ambiguous_companies": [],
                    "unsupported_companies": [],
                    "message": state["validation_errors"][-1],
                }
            }
        result = resolve_companies_tool.invoke({
            "company_mentions": state["plan"]["company_mentions"],
        })
        return {
            "resolution_result": result,
            "resolved_companies": list(result.get("resolved_companies", [])),
        }

    def mandatory_resolution_gate(state: OrchestratorState) -> dict[str, Any]:
        """Recompute company-resolution validity before profile selection is reachable."""
        result = state.get("resolution_result") or {}
        all_results = (
            list(result.get("resolved_companies", []))
            + list(result.get("ambiguous_companies", []))
            + list(result.get("unsupported_companies", []))
        )
        gate = validate_resolution_gate(all_results)
        if state.get("validation_errors"):
            gate = {
                **gate, "ready": False, "status": "company_limit_exceeded",
                "message": state["validation_errors"][-1],
            }
        return {"resolution_gate_status": gate}

    def route_resolution(state: OrchestratorState) -> str:
        """Continue only when the mandatory resolution gate is ready."""
        return "select_industry_profiles_tool" if (
            state.get("resolution_gate_status") or {}
        ).get("ready") else "bounded_stop"

    def select_profiles_node(state: OrchestratorState) -> dict[str, Any]:
        """Invoke registry-backed profile selection for canonical resolved identities."""
        company_ids = [item["company_id"] for item in state["resolved_companies"]]
        selection = select_industry_profiles_tool.invoke({"company_ids": company_ids})
        return {
            "profile_selection": selection,
            "last_profile_ids": sorted({
                profile["profile_id"]
                for profile in selection.get("profiles_by_company", {}).values()
            }),
        }

    def mandatory_profile_gate(state: OrchestratorState) -> dict[str, Any]:
        """Require exact profile coverage for the companies resolved in this run."""
        expected_ids = [item["company_id"] for item in state["resolved_companies"]]
        gate = validate_profile_gate(state.get("profile_selection") or {}, expected_ids)
        return {"profile_gate_status": gate}

    def route_profile(state: OrchestratorState) -> str:
        """Continue only when mandatory profile validation is ready."""
        return "build_company_tasks_tool" if (
            state.get("profile_gate_status") or {}
        ).get("ready") else "bounded_stop"

    def build_tasks_node(state: OrchestratorState) -> dict[str, Any]:
        """Build run-scoped company tasks through the guarded task tool."""
        register_task_planning_context(
            state["run_id"], state["plan"], state["resolved_companies"],
            state["profile_selection"],
        )
        try:
            result = build_company_tasks_tool.invoke({"run_id": state["run_id"]})
        finally:
            clear_task_planning_context(state["run_id"])
        return {
            "company_tasks": list(result.get("tasks", [])),
            "task_gate_status": result.get("task_gate"),
        }

    def mandatory_task_gate(state: OrchestratorState) -> dict[str, Any]:
        """Revalidate branch isolation, permissions, identity, and company budget."""
        gate = validate_task_gate(
            state.get("company_tasks", []), state.get("resolved_companies", []),
            max_companies=max_companies,
        )
        prior = state.get("task_gate_status") or {}
        if not prior.get("ready"):
            prior_message = prior.get("message", "Guarded task tool did not return ready tasks")
            gate = {
                **gate, "ready": False, "status": "invalid_tasks",
                "message": prior_message,
                "errors": list(gate.get("errors", [])) + [prior_message],
            }
        return {"task_gate_status": gate}

    def route_task_gate(state: OrchestratorState) -> str:
        """Make the fan-out node unreachable until the mandatory task gate passes."""
        return "fan_out_company_tasks" if (
            state.get("task_gate_status") or {}
        ).get("ready") else "bounded_stop"

    def fan_out_company_tasks(state: OrchestratorState) -> dict[str, Any]:
        """Mark the validated fan-out boundary without mutating branch inputs."""
        return {}

    def dispatch_company_tasks(state: OrchestratorState):
        """Create exactly one LangGraph Send branch for every validated company task."""
        gate = state.get("task_gate_status") or {}
        if not gate.get("ready"):
            return "bounded_stop"
        return [
            Send("company_worker", {
                "company_tasks": [deepcopy(task)],
                "run_id": state["run_id"],
                "original_query": state["original_query"],
            })
            for task in state["company_tasks"]
        ]

    def company_worker_node(state: OrchestratorState) -> dict[str, Any]:
        """Execute one profile-configured F10 worker and contain branch failures."""
        tasks = state.get("company_tasks", [])
        if len(tasks) != 1:
            raise ValueError("Each Send branch must contain exactly one company task")
        task = tasks[0]
        ticker = task["company"]["ticker"]
        try:
            profile = get_industry_profile(task["company"]["profile_id"])
            model = worker_model_factory(task, profile)
            tools = worker_tools_factory(task, profile)
            extractor = (
                signal_extractor_factory(task, profile)
                if signal_extractor_factory is not None else None
            )
            worker = create_company_worker(
                profile, model, tools,
                max_tool_rounds=worker_max_tool_rounds,
                signal_extractor=extractor,
            )
            worker_state = worker.invoke({
                "task": task,
                "messages": [HumanMessage(content=state["original_query"])],
            })
            result = worker_state.get("result")
            if result is None:
                raise ValueError("Company worker returned no result")
        except Exception as exc:
            result = _failed_company_result(task, f"Company worker failed: {exc}")
        return {"company_results": {ticker: result}}

    def collect_results(state: OrchestratorState) -> dict[str, Any]:
        """Validate reducer output after all Send branches reach the fan-in barrier."""
        results = state.get("company_results", {})
        validation_errors = _validate_fan_in_results(
            state["company_tasks"], results, state["run_id"],
        )
        branch_errors = [
            f"{ticker}: {'; '.join(result.get('errors', []))}"
            for ticker, result in sorted(results.items())
            if result.get("status") == "failed"
        ]
        return {"run_errors": validation_errors + branch_errors}

    def normalize_fan_in_node(state: OrchestratorState) -> dict[str, Any]:
        """Normalize reducer output in task order and expose blocking boundary errors."""
        summary = normalize_all_results(
            state.get("company_tasks", []), state.get("company_results", {}),
            state.get("run_id", ""),
        )
        return {
            "normalized_company_results": summary["results_by_ticker"],
            "fan_in_normalization": summary,
            "validation_errors": list(state.get("validation_errors", []))
            + list(summary.get("blocking_errors", [])),
            "run_errors": list(state.get("run_errors", []))
            + list(summary.get("errors", [])),
        }

    def route_after_fan_in_normalization(state: OrchestratorState) -> str:
        """Continue only when normalized fan-in has usable, uncontaminated results."""
        summary = state.get("fan_in_normalization") or {}
        return "mandatory_comparison_mode" if summary.get("ready") else "bounded_stop"

    def bounded_stop(state: OrchestratorState) -> dict[str, Any]:
        """Return a bounded explanation when a mandatory pre-research gate fails."""
        gates = (
            state.get("comparison_route_status"),
            state.get("task_gate_status"), state.get("profile_gate_status"),
            state.get("resolution_gate_status"),
        )
        message = next(
            (gate.get("message") for gate in gates if gate and not gate.get("ready")),
            "; ".join((state.get("fan_in_normalization") or {}).get("blocking_errors", []))
            or "Research stopped because a mandatory guardrail did not pass.",
        )
        return {"final_answer": message, "run_errors": [message]}

    workflow = StateGraph(OrchestratorState)
    workflow.add_node("initialize_run", initialize_run)
    workflow.add_node("coordinator_plan", coordinator_plan)
    workflow.add_node("resolve_companies_tool", resolve_companies_node)
    workflow.add_node("mandatory_resolution_gate", mandatory_resolution_gate)
    workflow.add_node("select_industry_profiles_tool", select_profiles_node)
    workflow.add_node("mandatory_profile_gate", mandatory_profile_gate)
    workflow.add_node("build_company_tasks_tool", build_tasks_node)
    workflow.add_node("mandatory_task_gate", mandatory_task_gate)
    workflow.add_node("fan_out_company_tasks", fan_out_company_tasks)
    workflow.add_node("company_worker", company_worker_node)
    workflow.add_node("collect_results", collect_results)
    if enable_f12:
        workflow.add_node("normalize_fan_in", normalize_fan_in_node)
        workflow.add_node("mandatory_comparison_mode", mandatory_comparison_mode_node)
    workflow.add_node("bounded_stop", bounded_stop)
    workflow.set_entry_point("initialize_run")
    workflow.add_edge("initialize_run", "coordinator_plan")
    workflow.add_edge("coordinator_plan", "resolve_companies_tool")
    workflow.add_edge("resolve_companies_tool", "mandatory_resolution_gate")
    workflow.add_conditional_edges(
        "mandatory_resolution_gate", route_resolution,
        ["select_industry_profiles_tool", "bounded_stop"],
    )
    workflow.add_edge("select_industry_profiles_tool", "mandatory_profile_gate")
    workflow.add_conditional_edges(
        "mandatory_profile_gate", route_profile,
        ["build_company_tasks_tool", "bounded_stop"],
    )
    workflow.add_edge("build_company_tasks_tool", "mandatory_task_gate")
    workflow.add_conditional_edges(
        "mandatory_task_gate", route_task_gate,
        ["fan_out_company_tasks", "bounded_stop"],
    )
    workflow.add_conditional_edges(
        "fan_out_company_tasks", dispatch_company_tasks,
        ["company_worker", "bounded_stop"],
    )
    workflow.add_edge("company_worker", "collect_results")
    if enable_f12:
        workflow.add_edge("collect_results", "normalize_fan_in")
        workflow.add_conditional_edges(
            "normalize_fan_in", route_after_fan_in_normalization,
            ["mandatory_comparison_mode", "bounded_stop"],
        )
        workflow.add_conditional_edges(
            "mandatory_comparison_mode", route_after_comparison_mode,
            {
                "single": END,
                "same_profile": END,
                "cross_profile": END,
                "bounded_stop": "bounded_stop",
            },
        )
    else:
        workflow.add_edge("collect_results", END)
    workflow.add_edge("bounded_stop", END)
    return NotebookOrchestrator(
        workflow.compile(), max_concurrency=max_concurrency,
        recursion_limit=recursion_limit,
    )


print("✅ F11 guarded parent Send fan-out/fan-in orchestrator defined")
'''


F11_SMOKE = r'''# F11 smoke: deterministic cross-profile Send fan-out with no provider calls.
_f11_planner = _F03FakeStructuredModel({
    "query_type": "compare",
    "company_mentions": ["Microsoft", "Pfizer"],
    "requested_dimensions": ["long_term_innovation"],
    "risk_profile": "balanced",
    "scoring_requested": False,
    "freshness_required": False,
    "time_horizon": "long term",
})


def _f11_worker_model_factory(task, profile):
    """Return a fresh scripted model selecting the branch profile's RAG tool."""
    return _F10ScriptedModel(
        profile["rag_tool_name"],
        {"ticker": task["company"]["ticker"], "query": "long-term innovation"},
    )


def _f11_worker_tools_factory(task, profile):
    """Return profile-complete deterministic tools for one isolated branch."""
    ticker = task["company"]["ticker"]
    tools = {
        name: _F10FakeTool(name, {"status": "missing", "ticker": ticker})
        for name in profile["allowed_tools"]
    }
    tools[profile["rag_tool_name"]] = _F10FakeTool(profile["rag_tool_name"], {
        "status": "success", "ticker": ticker,
        "data": f"Grounded {profile['industry']} evidence for {ticker}",
        "collection": profile["rag_collection"],
    })
    return tools


def _f11_signal_extractor_factory(task, profile):
    """Return a deterministic evidence-linked extractor for a smoke-test branch."""
    dimensions = (
        TECHNOLOGY_SIGNAL_NAMES
        if profile["profile_id"] == TECHNOLOGY_PROFILE_ID
        else BIOPHARMA_SIGNAL_NAMES
    )

    def extract(company, evidence):
        """Build partial fixture signals linked only to successful branch evidence."""
        evidence_ids = [
            item["evidence_id"] for item in evidence if item["status"] == "success"
        ]
        return {
            name: {
                "level": "partial", "score": 0.5, "reason": "F11 fixture",
                "evidence_ids": evidence_ids,
            }
            for name in dimensions
        }

    return extract


_f11_graph = create_multi_company_orchestrator(
    _f11_planner, _f11_worker_model_factory, _f11_worker_tools_factory,
    signal_extractor_factory=_f11_signal_extractor_factory,
    max_concurrency=2,
)
_f11_state = _f11_graph.invoke({
    "messages": [HumanMessage(content=(
        "Compare Microsoft and Pfizer on financial strength and long-term innovation"
    ))],
    "remembered_company_ids": [],
    "last_profile_ids": [],
})
assert _f11_state["resolution_gate_status"]["ready"] is True
assert _f11_state["profile_gate_status"]["ready"] is True
assert _f11_state["task_gate_status"]["ready"] is True
assert set(_f11_state["company_results"]) == {"MSFT", "PFE"}
assert _f11_state["company_results"]["MSFT"]["profile_id"] == TECHNOLOGY_PROFILE_ID
assert _f11_state["company_results"]["PFE"]["profile_id"] == BIOPHARMA_PROFILE_ID

print("✅ F11 smoke test passed: Send collected isolated Microsoft and Pfizer results")
'''


CELL_SPECS = [
    ("multiindustry_f11_intro", "markdown", F11_INTRO),
    ("multiindustry_parent_orchestrator", "code", F11_CODE),
    ("multiindustry_f11_smoke", "code", F11_SMOKE),
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


def main() -> None:
    """Insert or refresh F11 cells in the canonical working notebook."""
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
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F11 parent orchestrator cells are present")


if __name__ == "__main__":
    main()

"""Idempotently add the F10 generic company-worker graph and Section 3 integration test."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f09_smoke"


F10_INTRO = """## Section 3.10: Generic Profile-Configured Company Worker

F10 provides one LangGraph worker for both supported profiles. The profile—not scattered sector
conditionals—selects the prompt, dimensions, RAG capability, and tool allowlist. The LLM remains
autonomous over permitted tool order and follow-up research.

Software controls remain mandatory: one ticker per worker, fail-closed callable binding, tool
allowlist enforcement, argument identity checks, canonical evidence conversion, bounded tool
rounds, an evidence exit gate, profile-specific signal normalization, and one validated
`CompanyResearchResult`.
"""


F10_CODE = r'''from __future__ import annotations

import json
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph


WORKER_TOOL_EVIDENCE_TYPES = {
    "get_stock_price": "stock_price",
    "get_financial_metrics": "financial_metrics",
    "get_stock_history": "stock_history",
    "search_financial_news": "financial_news",
    "analyze_sentiment": "sentiment",
    "query_technology_rag": "technology_rag",
    "query_biopharma_rag": "biopharma_rag",
}

WORKER_TICKER_TOOLS = {
    "get_stock_price",
    "get_financial_metrics",
    "get_stock_history",
    "query_technology_rag",
    "query_biopharma_rag",
}


def _default_worker_tools() -> dict[str, Any]:
    """Return the current notebook's source-tool objects keyed by contract name."""
    candidates = {
        "get_stock_price": globals().get("get_stock_price"),
        "get_financial_metrics": globals().get("get_financial_metrics"),
        "get_stock_history": globals().get("get_stock_history"),
        "search_financial_news": globals().get("search_financial_news"),
        "analyze_sentiment": globals().get("analyze_sentiment"),
        "query_technology_rag": globals().get("query_technology_rag"),
        "query_biopharma_rag": globals().get("query_biopharma_rag"),
    }
    return {name: value for name, value in candidates.items() if value is not None}


def _worker_missing_dimensions(
    task: CompanyTask,
    evidence: list[EvidenceRecord],
) -> list[str]:
    """Determine which requested task dimensions lack successful source coverage.

    Args:
        task: One isolated company assignment.
        evidence: Canonical records collected by the worker.

    Returns:
        Required shared and industry dimensions not covered by successful evidence.
    """
    successful_types = {
        record["evidence_type"] for record in evidence if record["status"] == "success"
    }
    missing: list[str] = []
    for dimension in task["shared_dimensions"]:
        if dimension == "price_history":
            covered = "stock_history" in successful_types
        elif dimension == "news_sentiment":
            covered = bool({"financial_news", "sentiment"} & successful_types)
        elif dimension in {"current_price", "market_cap"}:
            covered = bool({"stock_price", "financial_metrics"} & successful_types)
        else:
            # Revenue, P/E, beta, and dividend yield must come from the canonical five-metric
            # contract. A price snapshot cannot silently authorize F13 scoring for these fields.
            covered = "financial_metrics" in successful_types
        if not covered:
            missing.append(dimension)
    required_rag = (
        "technology_rag"
        if task["company"]["profile_id"] == TECHNOLOGY_PROFILE_ID
        else "biopharma_rag"
    )
    if task["industry_dimensions"] and required_rag not in successful_types:
        missing.extend(task["industry_dimensions"])
    return list(dict.fromkeys(missing + task["unsupported_dimensions"]))


def _tool_result_to_worker_evidence(
    task: CompanyTask,
    tool_name: str,
    tool_args: dict[str, Any],
    raw_result: Any,
) -> list[EvidenceRecord]:
    """Convert one worker tool result into canonical evidence records."""
    if tool_name == "query_technology_rag":
        return query_technology_rag_evidence(
            task, str(tool_args.get("query", "technology research")), raw_result,
        )
    if tool_name == "query_biopharma_rag":
        return query_biopharma_rag_evidence(
            task, str(tool_args.get("query", "biopharma research")), raw_result,
        )
    evidence_type = WORKER_TOOL_EVIDENCE_TYPES[tool_name]
    return to_evidence_record(
        task["run_id"], task["company"], task["company"]["profile_id"],
        evidence_type, raw_result, tool_name,
    )


def create_company_worker(
    profile: IndustryProfile,
    model: Any,
    tools_by_name: dict[str, Any] | None = None,
    max_tool_rounds: int = 4,
    signal_extractor: Callable[[ResolvedCompany, list[EvidenceRecord]], dict[str, Any]] | None = None,
):
    """Create one profile-configured LangGraph worker for isolated company research.

    Args:
        profile: Versioned industry profile controlling prompt, dimensions, and allowlist.
        model: Chat model supporting ``bind_tools`` and ``invoke``.
        tools_by_name: Optional injected source-tool registry for tests.
        max_tool_rounds: Hard ceiling for autonomous tool execution.
        signal_extractor: Optional deterministic test extractor for one company and its evidence.

    Returns:
        Compiled LangGraph worker producing one ``CompanyResearchResult``.

    Raises:
        ValueError: If profile configuration is invalid or an allowed tool lacks a callable.
    """
    registered_profile = get_industry_profile(profile["profile_id"])
    if registered_profile != profile:
        raise ValueError("Worker profile must exactly match the registered versioned configuration")
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be at least 1")
    tool_registry = tools_by_name or _default_worker_tools()
    missing_tools = [name for name in profile["allowed_tools"] if name not in tool_registry]
    if missing_tools:
        raise ValueError(f"Profile tools are not callable: {missing_tools}")
    allowed_tool_map = {name: tool_registry[name] for name in profile["allowed_tools"]}
    allowed_tool_objects = list(allowed_tool_map.values())
    model_with_tools = model.bind_tools(allowed_tool_objects)

    def initialize_worker(state: CompanyWorkerState) -> dict[str, Any]:
        """Validate one-company task identity and initialize branch-local fields."""
        task = state["task"]
        company = task["company"]
        if company["profile_id"] != profile["profile_id"]:
            raise ValueError("Task profile does not match worker profile")
        task_gate = validate_task_gate([task], [company], max_companies=1)
        if not task_gate["ready"]:
            raise ValueError("Invalid company task: " + task_gate["message"])
        messages = list(state.get("messages", []))
        if not messages:
            messages = [HumanMessage(content=(
                f"Research {company['company_name']} ({company['ticker']}) for the requested "
                f"dimensions: {task['shared_dimensions'] + task['industry_dimensions']}."
            ))]
        return {
            "messages": messages,
            "evidence": [],
            "industry_signals": {},
            "missing_dimensions": [],
            "evidence_gate_status": "retry",
            "tool_round_count": 0,
            "validation_retry_count": 0,
            "result": None,
            "errors": [],
        }

    def worker_agent(state: CompanyWorkerState) -> dict[str, Any]:
        """Invoke the autonomous worker with tools unless the round ceiling was reached."""
        task = state["task"]
        company = task["company"]
        system = SystemMessage(content=(
            f"You research exactly one company: {company['company_name']} ({company['ticker']}). "
            f"Never request another ticker. Allowed tools: {', '.join(profile['allowed_tools'])}. "
            f"{profile['worker_prompt']} Choose the tool order autonomously. Stop when the "
            "requested dimensions are supported, and never invent missing evidence."
        ))
        messages = [system] + list(state["messages"])
        active_model = model if state.get("tool_round_count", 0) >= max_tool_rounds else model_with_tools
        return {"messages": [active_model.invoke(messages)]}

    def execute_allowed_tools(state: CompanyWorkerState) -> dict[str, Any]:
        """Execute requested tools with allowlist, ticker, error, and evidence controls."""
        task = state["task"]
        assigned_ticker = task["company"]["ticker"]
        last_message = state["messages"][-1]
        outputs: list[ToolMessage] = []
        new_evidence = list(state.get("evidence", []))
        errors = list(state.get("errors", []))
        for tool_call in getattr(last_message, "tool_calls", []) or []:
            name = tool_call["name"]
            arguments = dict(tool_call.get("args", {}))
            if name not in allowed_tool_map:
                message = f"Tool {name!r} is not allowed for profile {profile['profile_id']}"
                errors.append(message)
                outputs.append(ToolMessage(
                    content="Error: " + message, name=name, tool_call_id=tool_call["id"],
                ))
                continue
            if name in WORKER_TICKER_TOOLS:
                requested_ticker = str(arguments.get("ticker", "")).upper()
                if requested_ticker != assigned_ticker:
                    message = (
                        f"Tool {name!r} must use assigned ticker {assigned_ticker}; "
                        f"received {requested_ticker or '<missing>'}"
                    )
                    errors.append(message)
                    outputs.append(ToolMessage(
                        content="Error: " + message, name=name, tool_call_id=tool_call["id"],
                    ))
                    continue
            try:
                tool_object = allowed_tool_map[name]
                raw = tool_object.invoke(arguments) if hasattr(tool_object, "invoke") else tool_object(**arguments)
                records = _tool_result_to_worker_evidence(task, name, arguments, raw)
                new_evidence.extend(records)
                content = json.dumps(raw, default=str)
            except Exception as exc:
                content = f"Error: {exc}"
                errors.append(f"{name}: {exc}")
            outputs.append(ToolMessage(
                content=content, name=name, tool_call_id=tool_call["id"],
            ))
        return {
            "messages": outputs,
            "evidence": new_evidence,
            "tool_round_count": state.get("tool_round_count", 0) + 1,
            "errors": errors,
        }

    def route_after_agent(state: CompanyWorkerState) -> Literal["tools", "evidence_gate"]:
        """Route tool calls to execution and final narratives to the evidence gate."""
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else "evidence_gate"

    def evidence_exit_gate(state: CompanyWorkerState) -> dict[str, Any]:
        """Require grounded dimension coverage or terminate as bounded partial output."""
        missing = _worker_missing_dimensions(state["task"], state.get("evidence", []))
        if not missing:
            return {"missing_dimensions": [], "evidence_gate_status": "complete"}
        if (
            state.get("tool_round_count", 0) < max_tool_rounds
            and state.get("validation_retry_count", 0) < max_tool_rounds
        ):
            guidance = HumanMessage(content=(
                "[EVIDENCE_GATE] More grounded evidence is required for: "
                + ", ".join(missing)
                + ". Call only allowed tools for the assigned ticker, or state that evidence is unavailable."
            ))
            return {
                "messages": [guidance],
                "missing_dimensions": missing,
                "evidence_gate_status": "retry",
                "validation_retry_count": state.get("validation_retry_count", 0) + 1,
            }
        return {"missing_dimensions": missing, "evidence_gate_status": "partial"}

    def route_after_evidence_gate(state: CompanyWorkerState) -> Literal["retry", "extract"]:
        """Route incomplete evidence back to the agent while budget remains."""
        return "retry" if state["evidence_gate_status"] == "retry" else "extract"

    def extract_profile_signals(state: CompanyWorkerState) -> dict[str, Any]:
        """Run the selected profile extractor over validated company-local evidence."""
        company = state["task"]["company"]
        evidence = state.get("evidence", [])
        if signal_extractor is not None:
            signals = signal_extractor(company, evidence)
        elif profile["profile_id"] == TECHNOLOGY_PROFILE_ID:
            signals = extract_technology_signals_with_evidence(
                [company], {company["company_id"]: evidence},
            ).get(company["ticker"], {})
        elif profile["profile_id"] == BIOPHARMA_PROFILE_ID:
            signals = extract_pharma_signals(
                [company], {company["company_id"]: evidence},
            ).get(company["ticker"], {})
        else:
            raise ValueError(f"No extractor is configured for {profile['profile_id']}")
        return {"industry_signals": signals}

    def validate_company_result(state: CompanyWorkerState) -> dict[str, Any]:
        """Assemble one identity-safe normalized result for the parent reducer."""
        task = state["task"]
        evidence = list(state.get("evidence", []))
        successful = [record for record in evidence if record["status"] == "success"]
        missing = list(state.get("missing_dimensions", []))
        errors = list(state.get("errors", []))
        if not successful:
            status = "failed"
        elif missing or errors or any(record["status"] != "success" for record in evidence):
            status = "partial"
        else:
            status = "success"
        financial_evidence = {
            record["evidence_type"]: record["value"]
            for record in successful
            if record["evidence_type"] not in {"technology_rag", "biopharma_rag"}
        }
        result: CompanyResearchResult = {
            "run_id": task["run_id"],
            "company": task["company"],
            "profile_id": profile["profile_id"],
            "financial_evidence": financial_evidence,
            "industry_signals": state.get("industry_signals", {}),
            "evidence": evidence,
            "missing_dimensions": missing,
            "errors": errors,
            "status": status,
        }
        return {"result": result}

    workflow = StateGraph(CompanyWorkerState)
    workflow.add_node("initialize_worker", initialize_worker)
    workflow.add_node("worker_agent", worker_agent)
    workflow.add_node("execute_allowed_tools", execute_allowed_tools)
    workflow.add_node("evidence_exit_gate", evidence_exit_gate)
    workflow.add_node("extract_profile_signals", extract_profile_signals)
    workflow.add_node("validate_company_result", validate_company_result)
    workflow.set_entry_point("initialize_worker")
    workflow.add_edge("initialize_worker", "worker_agent")
    workflow.add_conditional_edges(
        "worker_agent", route_after_agent,
        {"tools": "execute_allowed_tools", "evidence_gate": "evidence_exit_gate"},
    )
    workflow.add_edge("execute_allowed_tools", "worker_agent")
    workflow.add_conditional_edges(
        "evidence_exit_gate", route_after_evidence_gate,
        {"retry": "worker_agent", "extract": "extract_profile_signals"},
    )
    workflow.add_edge("extract_profile_signals", "validate_company_result")
    workflow.add_edge("validate_company_result", END)
    return workflow.compile()


print("✅ F10 generic profile-configured company worker defined")
'''


F10_SMOKE = r'''# F10 smoke uses local scripted doubles; no provider or vector-store calls.
class _F10ScriptedModel:
    """Emit one source-tool call followed by a final answer."""

    def __init__(self, tool_name: str, arguments: dict[str, Any]):
        """Store the single tool request and initialize call tracking."""
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools):
        """Record the profile-selected tool names and return this model."""
        self.bound_tool_names = [item.name for item in tools]
        return self

    def invoke(self, messages):
        """Return a tool call on the first turn and a narrative afterward."""
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": self.tool_name, "args": self.arguments, "id": "f10-smoke-call",
            }])
        return AIMessage(content="Research complete from grounded evidence.")


class _F10FakeTool:
    """Minimal named tool double returning a predefined result."""

    def __init__(self, name: str, result: Any):
        """Store the LangChain-compatible name and deterministic result."""
        self.name = name
        self.result = result

    def invoke(self, arguments):
        """Return the configured result without external calls."""
        return self.result


_f10_company = resolve_company_mention("Microsoft")
_f10_plan: QueryPlan = {
    "query_type": "analyze", "company_mentions": ["Microsoft"],
    "requested_dimensions": ["ai_strategy"], "risk_profile": "balanced",
    "scoring_requested": False, "freshness_required": False, "time_horizon": None,
}
_f10_task = build_company_tasks(_f10_plan, [_f10_company], "f10-smoke-run")[0]
_f10_tools = {
    name: _F10FakeTool(name, {"status": "missing", "ticker": "MSFT"})
    for name in TECHNOLOGY_TOOL_NAMES
}
_f10_tools["query_technology_rag"] = _F10FakeTool("query_technology_rag", {
    "status": "success", "ticker": "MSFT", "data": "technology evidence",
    "collection": "AI_Initiatives",
})
_f10_model = _F10ScriptedModel(
    "query_technology_rag", {"ticker": "MSFT", "query": "AI strategy"},
)
_f10_worker = create_company_worker(
    get_industry_profile("technology.ai.v1"), _f10_model, _f10_tools,
    signal_extractor=lambda company, evidence: {
        name: {"level": "partial", "score": 0.5, "reason": "fixture",
               "evidence_ids": [item["evidence_id"] for item in evidence if item["status"] == "success"]}
        for name in TECHNOLOGY_SIGNAL_NAMES
    },
)
_f10_state = _f10_worker.invoke({"task": _f10_task, "messages": [HumanMessage(content="Analyze Microsoft")]})
assert _f10_state["result"]["company"]["ticker"] == "MSFT"
assert _f10_state["result"]["profile_id"] == "technology.ai.v1"
assert "query_biopharma_rag" not in _f10_model.bound_tool_names

print("✅ F10 smoke test passed: one generic worker completed an isolated technology task")
'''


SECTION3_INTEGRATION = r'''## Section 3 End-to-End Test Query

This deterministic integration executes the complete implemented Section 3 parent graph for:

> **Compare Microsoft and Pfizer on financial strength and long-term innovation.**

It exercises structured planning, guarded company resolution, profile selection, task building,
LangGraph `Send` fan-out, Technology/AI and Biopharma workers, reducer fan-in, canonical evidence,
profile-specific signals, and result identity validation. Scripted local models and tools make the
test reproducible and avoid provider cost; they exercise the same graph and guardrail code used by
live tools.
'''


SECTION3_INTEGRATION_CODE = r'''# Complete Section 3 deterministic parent-graph query.
_section3_query = "Compare Microsoft and Pfizer on financial strength and long-term innovation"
_section3_planner = _F03FakeStructuredModel({
    "query_type": "compare",
    "company_mentions": ["Microsoft", "Pfizer"],
    "requested_dimensions": ["financial_strength", "long_term_innovation"],
    "risk_profile": "balanced",
    "scoring_requested": False,
    "freshness_required": False,
    "time_horizon": "long term",
})
_section3_graph = create_multi_company_orchestrator(
    _section3_planner,
    _f11_worker_model_factory,
    _f11_worker_tools_factory,
    signal_extractor_factory=_f11_signal_extractor_factory,
    max_concurrency=2,
    enable_f12=True,
)
_section3_state = _section3_graph.invoke({
    "messages": [HumanMessage(content=_section3_query)],
    "remembered_company_ids": [],
    "last_profile_ids": [],
})
_section3_results = _section3_state["normalized_company_results"]

assert _section3_state["resolution_gate_status"]["ready"] is True
assert _section3_state["profile_gate_status"]["ready"] is True
assert _section3_state["task_gate_status"]["ready"] is True
assert _section3_state["fan_in_normalization"]["ready"] is True
assert _section3_state["comparison_mode"] == "cross_profile"
assert _section3_state["scoring_eligibility"]["eligible"] is False
assert set(_section3_results) == {"MSFT", "PFE"}
assert _section3_results["MSFT"]["profile_id"] == "technology.ai.v1"
assert _section3_results["PFE"]["profile_id"] == "healthcare.biopharma.v1"
assert all(
    evidence["ticker"] == ticker
    for ticker, result in _section3_results.items()
    for evidence in result["evidence"]
)
assert check_biopharma_scoring_gate()["eligible"] is True
assert check_biopharma_scoring_gate()["rubric_id"] == "healthcare.biopharma.score.v1"

print(
    "✅ Full Section 3 test query passed: parent Send graph normalized and routed "
    "Microsoft + Pfizer to cross_profile"
)
'''


CELL_SPECS = [
    ("multiindustry_f10_intro", "markdown", F10_INTRO),
    ("multiindustry_company_worker", "code", F10_CODE),
    ("multiindustry_f10_smoke", "code", F10_SMOKE),
    ("multiindustry_section3_integration_intro", "markdown", SECTION3_INTEGRATION),
    ("multiindustry_section3_integration_test", "code", SECTION3_INTEGRATION_CODE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create a notebook cell with a stable identifier."""
    cell = nbformat.v4.new_markdown_cell(source=source) if cell_type == "markdown" else nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    """Insert or refresh F10 and Section 3 integration cells in the working notebook."""
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
        index = next(i for i, cell in enumerate(notebook.cells) if cell.get("id") == INSERT_AFTER_CELL_ID) + 1
        notebook.cells[index:index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F10 and Section 3 integration cells are present")


if __name__ == "__main__":
    main()

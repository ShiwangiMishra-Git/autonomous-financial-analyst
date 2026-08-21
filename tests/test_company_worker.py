"""Deterministic tests for the F10 generic profile-configured worker."""

from __future__ import annotations

import contextlib
from functools import lru_cache
import io
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


class _FakeTool:
    """Named tool double returning a deterministic result."""

    def __init__(self, name, result):
        """Store tool identity and result."""
        self.name = name
        self.result = result
        self.calls = []

    def invoke(self, arguments):
        """Record arguments and return the configured result."""
        self.calls.append(arguments)
        return self.result


class _ScriptedModel:
    """Return a configured sequence of AI messages and record bound tools."""

    def __init__(self, responses):
        """Store scripted responses and initialize counters."""
        self.responses = list(responses)
        self.index = 0
        self.bound_tool_names = []

    def bind_tools(self, tools):
        """Record profile-specific bindings and return this model."""
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        """Return the next response, repeating the final response if exhausted."""
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response


@lru_cache(maxsize=1)
def _worker_namespace():
    """Execute F01–F10 cells with stubbed legacy technology functions."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "query_private_database": lambda query: "legacy", "extract_ai_signals": lambda *a, **k: {},
        "score_companies": lambda *a, **k: {},
    }
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in (
            "multiindustry_state_contracts", "multiindustry_company_registry",
            "multiindustry_query_planner", "multiindustry_industry_profiles",
            "multiindustry_company_tasks", "multiindustry_evidence_adapters",
            "multiindustry_technology_profile", "multiindustry_biopharma_rag",
            "multiindustry_biopharma_signals", "multiindustry_company_worker",
        ):
            exec(cells[cell_id], namespace)
    return namespace


def _task(namespace, company_name, dimensions):
    """Build one analyze task for a supported company."""
    company = namespace["resolve_company_mention"](company_name)
    plan = {
        "query_type": "analyze", "company_mentions": [company_name],
        "requested_dimensions": dimensions, "risk_profile": "balanced",
        "scoring_requested": False, "freshness_required": False, "time_horizon": None,
    }
    return namespace["build_company_tasks"](plan, [company], f"run-{company['ticker']}")[0]


def _tools_for_profile(profile, ticker, rag_result):
    """Create a complete fake callable registry for one profile."""
    tools = {
        name: _FakeTool(name, {"status": "missing", "ticker": ticker})
        for name in profile["allowed_tools"]
    }
    tools[profile["rag_tool_name"]] = _FakeTool(profile["rag_tool_name"], rag_result)
    return tools


def _extractor(company, evidence):
    """Return a minimal evidence-linked fixture signal mapping."""
    return {"fixture_signal": {
        "level": "partial", "score": 0.5,
        "evidence_ids": [item["evidence_id"] for item in evidence if item["status"] == "success"],
    }}


def test_technology_and_biopharma_bind_only_profile_tools():
    """Use one factory while preserving separate source-tool allowlists."""
    namespace = _worker_namespace()
    for company_name, profile_id, rag_name in (
        ("Microsoft", "technology.ai.v1", "query_technology_rag"),
        ("Pfizer", "healthcare.biopharma.v1", "query_biopharma_rag"),
    ):
        task = _task(namespace, company_name, ["long_term_innovation"])
        profile = namespace["get_industry_profile"](profile_id)
        ticker = task["company"]["ticker"]
        model = _ScriptedModel([AIMessage(content="done")])
        namespace["create_company_worker"](
            profile, model,
            _tools_for_profile(profile, ticker, {"status": "missing", "ticker": ticker}),
            signal_extractor=_extractor,
        )
        assert set(model.bound_tool_names) == set(profile["allowed_tools"])
        assert rag_name in model.bound_tool_names
        other = "query_biopharma_rag" if "technology" in profile_id else "query_technology_rag"
        assert other not in model.bound_tool_names


def test_worker_produces_one_identity_isolated_result():
    """Collect canonical evidence and return only the assigned company identity."""
    namespace = _worker_namespace()
    task = _task(namespace, "Microsoft", ["ai_strategy"])
    profile = namespace["get_industry_profile"]("technology.ai.v1")
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[{
            "name": "query_technology_rag", "args": {"ticker": "MSFT", "query": "AI"}, "id": "c1",
        }]),
        AIMessage(content="done"),
    ])
    tools = _tools_for_profile(profile, "MSFT", {
        "status": "success", "ticker": "MSFT", "data": "grounded technology evidence",
    })
    graph = namespace["create_company_worker"](
        profile, model, tools, signal_extractor=_extractor,
    )
    state = graph.invoke({"task": task, "messages": [HumanMessage(content="Analyze Microsoft")]})
    result = state["result"]

    assert result["company"]["ticker"] == "MSFT"
    assert result["profile_id"] == "technology.ai.v1"
    assert all(item["ticker"] == "MSFT" for item in result["evidence"])
    assert result["status"] in {"success", "partial"}


def test_disallowed_tool_and_wrong_ticker_are_rejected():
    """Contain model attempts to escape the profile or assigned-company boundary."""
    namespace = _worker_namespace()
    task = _task(namespace, "Microsoft", ["ai_strategy"])
    profile = namespace["get_industry_profile"]("technology.ai.v1")
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "query_biopharma_rag", "args": {"ticker": "PFE", "query": "pipeline"}, "id": "bad1"},
            {"name": "get_stock_price", "args": {"ticker": "PFE"}, "id": "bad2"},
        ]),
        AIMessage(content="done"),
    ])
    tools = _tools_for_profile(profile, "MSFT", {"status": "missing", "ticker": "MSFT"})
    graph = namespace["create_company_worker"](
        profile, model, tools, max_tool_rounds=1, signal_extractor=_extractor,
    )
    result = graph.invoke({"task": task, "messages": [HumanMessage(content="Analyze")]})["result"]

    assert any("not allowed" in error for error in result["errors"])
    assert any("assigned ticker MSFT" in error for error in result["errors"])
    assert result["evidence"] == []
    assert result["status"] == "failed"


def test_failed_tool_does_not_erase_successful_evidence():
    """Continue as partial when one source fails but another succeeds."""
    namespace = _worker_namespace()
    task = _task(namespace, "Pfizer", ["pipeline"])
    profile = namespace["get_industry_profile"]("healthcare.biopharma.v1")
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "get_stock_price", "args": {"ticker": "PFE"}, "id": "p1"},
            {"name": "query_biopharma_rag", "args": {"ticker": "PFE", "query": "pipeline"}, "id": "p2"},
        ]),
        AIMessage(content="done"),
    ])
    tools = _tools_for_profile(profile, "PFE", {
        "status": "success", "ticker": "PFE", "data": [{
            "data": "official pipeline evidence", "ticker": "PFE", "document_name": "PFE.pdf",
        }],
    })
    tools["get_stock_price"] = _FakeTool(
        "get_stock_price", {"status": "error", "ticker": "PFE", "error": "provider down"}
    )
    graph = namespace["create_company_worker"](
        profile, model, tools, signal_extractor=_extractor,
    )
    result = graph.invoke({"task": task, "messages": [HumanMessage(content="Analyze")]})["result"]

    assert any(item["status"] == "failed" for item in result["evidence"])
    assert any(item["status"] == "success" for item in result["evidence"])
    assert result["status"] == "partial"


def test_tool_round_ceiling_terminates_with_bounded_partial_result():
    """Prevent an agent from looping forever when required evidence remains unavailable."""
    namespace = _worker_namespace()
    task = _task(namespace, "Microsoft", ["ai_strategy"])
    profile = namespace["get_industry_profile"]("technology.ai.v1")
    repeated = AIMessage(content="", tool_calls=[{
        "name": "query_technology_rag", "args": {"ticker": "MSFT", "query": "AI"}, "id": "repeat",
    }])
    model = _ScriptedModel([repeated, AIMessage(content="no more evidence")])
    tools = _tools_for_profile(profile, "MSFT", {"status": "missing", "ticker": "MSFT"})
    graph = namespace["create_company_worker"](
        profile, model, tools, max_tool_rounds=1, signal_extractor=_extractor,
    )
    state = graph.invoke({"task": task, "messages": [HumanMessage(content="Analyze")]})

    assert state["tool_round_count"] == 1
    assert state["evidence_gate_status"] == "partial"
    assert state["result"]["status"] == "failed"

"""Deterministic tests for the F11 parent Send fan-out/fan-in graph."""

from __future__ import annotations

import contextlib
from functools import lru_cache
import io
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


class _FakePlanner:
    """Return one deterministic structured query plan."""

    def __init__(self, mentions):
        """Store company mentions used in the structured planner response."""
        self.mentions = mentions

    def with_structured_output(self, schema, method="function_calling"):
        """Accept the notebook planner's structured-output binding contract."""
        return self

    def invoke(self, messages):
        """Return a valid comparison plan for the configured companies."""
        return {
            "query_type": "compare" if len(self.mentions) > 1 else "analyze",
            "company_mentions": self.mentions,
            "requested_dimensions": ["long_term_innovation"],
            "risk_profile": "balanced",
            "scoring_requested": False,
            "freshness_required": False,
            "time_horizon": "long term",
        }


class _FakeTool:
    """Named deterministic source tool used by company workers."""

    def __init__(self, name, result):
        """Store the tool name and static result."""
        self.name = name
        self.result = result

    def invoke(self, arguments):
        """Return the static result without external I/O."""
        return self.result


class _WorkerModel:
    """Call the profile RAG tool once, then finish the branch."""

    def __init__(self, tool_name, ticker):
        """Store the branch identity and initialize invocation tracking."""
        self.tool_name = tool_name
        self.ticker = ticker
        self.calls = 0

    def bind_tools(self, tools):
        """Accept F10 profile tool binding and return this model."""
        return self

    def invoke(self, messages):
        """Emit one identity-safe tool request followed by a final response."""
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": self.tool_name,
                "args": {"ticker": self.ticker, "query": "long-term innovation"},
                "id": f"call-{self.ticker}",
            }])
        return AIMessage(content="grounded research complete")


@lru_cache(maxsize=1)
def _namespace():
    """Execute the F01–F11 implementation cells in one cached test namespace."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "query_private_database": lambda query: "legacy",
        "extract_ai_signals": lambda *args, **kwargs: {},
        "score_companies": lambda *args, **kwargs: {},
    }
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in (
            "multiindustry_state_contracts", "multiindustry_company_registry",
            "multiindustry_query_planner", "multiindustry_industry_profiles",
            "multiindustry_company_tasks", "multiindustry_evidence_adapters",
            "multiindustry_technology_profile", "multiindustry_biopharma_rag",
            "multiindustry_biopharma_signals", "multiindustry_company_worker",
            "multiindustry_parent_orchestrator",
        ):
            exec(cells[cell_id], namespace)
    return namespace


def _tools_factory(task, profile):
    """Return all profile tools with successful RAG evidence for one company."""
    ticker = task["company"]["ticker"]
    tools = {
        name: _FakeTool(name, {"status": "missing", "ticker": ticker})
        for name in profile["allowed_tools"]
    }
    tools[profile["rag_tool_name"]] = _FakeTool(profile["rag_tool_name"], {
        "status": "success", "ticker": ticker,
        "data": f"official evidence for {ticker}",
        "collection": profile["rag_collection"],
    })
    return tools


def _model_factory(task, profile):
    """Create a fresh deterministic model for one company branch."""
    return _WorkerModel(profile["rag_tool_name"], task["company"]["ticker"])


def _extractor_factory(task, profile):
    """Create an evidence-linked deterministic signal extractor."""
    def extract(company, evidence):
        """Return one fixture signal grounded in successful branch evidence."""
        return {"fixture": {
            "level": "partial", "score": 0.5,
            "evidence_ids": [
                item["evidence_id"] for item in evidence if item["status"] == "success"
            ],
        }}

    return extract


def _invoke(mentions, **factory_options):
    """Build and run a deterministic parent graph for the supplied mentions."""
    namespace = _namespace()
    graph = namespace["create_multi_company_orchestrator"](
        _FakePlanner(mentions), _model_factory, _tools_factory,
        signal_extractor_factory=_extractor_factory,
        **factory_options,
    )
    return graph, graph.invoke({
        "messages": [HumanMessage(content="Compare " + " and ".join(mentions))],
        "remembered_company_ids": [],
        "last_profile_ids": [],
    })


def test_one_company_creates_one_isolated_result_branch():
    """Return exactly one result when the plan contains one supported company."""
    _, state = _invoke(["Microsoft"])
    assert set(state["company_results"]) == {"MSFT"}
    assert len(state["company_tasks"]) == 1
    assert state["task_gate_status"]["ready"] is True


def test_four_companies_merge_without_identity_overwrite():
    """Collect four same- and cross-profile branches through the result reducer."""
    _, state = _invoke(["Microsoft", "NVIDIA", "Pfizer", "Merck"])
    assert set(state["company_results"]) == {"MSFT", "NVDA", "PFE", "MRK"}
    for ticker, result in state["company_results"].items():
        assert result["company"]["ticker"] == ticker
        assert all(item["ticker"] == ticker for item in result["evidence"])


def test_one_failed_branch_does_not_erase_successful_peer():
    """Contain a branch factory failure while preserving the successful company result."""
    namespace = _namespace()

    def partially_failing_tools(task, profile):
        """Fail only the Pfizer branch before its F10 worker is created."""
        if task["company"]["ticker"] == "PFE":
            raise RuntimeError("fixture branch failure")
        return _tools_factory(task, profile)

    graph = namespace["create_multi_company_orchestrator"](
        _FakePlanner(["Microsoft", "Pfizer"]), _model_factory,
        partially_failing_tools, signal_extractor_factory=_extractor_factory,
    )
    state = graph.invoke({
        "messages": [HumanMessage(content="Compare Microsoft and Pfizer")],
        "remembered_company_ids": [], "last_profile_ids": [],
    })
    assert state["company_results"]["MSFT"]["status"] in {"success", "partial"}
    assert state["company_results"]["PFE"]["status"] == "failed"
    assert any("PFE:" in error for error in state["run_errors"])


def test_company_limit_blocks_fan_out_before_worker_factory():
    """Stop at a mandatory gate when the configured branch budget is exceeded."""
    namespace = _namespace()
    calls = []

    def recording_model_factory(task, profile):
        """Record any erroneous attempt to enter a worker branch."""
        calls.append(task["company"]["ticker"])
        return _model_factory(task, profile)

    graph = namespace["create_multi_company_orchestrator"](
        _FakePlanner(["Microsoft", "NVIDIA"]), recording_model_factory,
        _tools_factory, signal_extractor_factory=_extractor_factory,
        max_companies=1,
    )
    state = graph.invoke({
        "messages": [HumanMessage(content="Compare Microsoft and NVIDIA")],
        "remembered_company_ids": [], "last_profile_ids": [],
    })
    assert calls == []
    assert state["company_results"] == {}
    assert state["resolution_gate_status"]["status"] == "company_limit_exceeded"


def test_ambiguous_resolution_cannot_skip_to_research():
    """Block fan-out when canonical company resolution requires clarification."""
    _, state = _invoke(["Roche"])
    assert state["resolution_gate_status"]["status"] == "needs_clarification"
    assert state["company_tasks"] == []
    assert state["company_results"] == {}
    assert state["final_answer"]


def test_local_concurrency_and_recursion_ceilings_are_enforced():
    """Cap caller-provided execution values at factory-configured notebook limits."""
    namespace = _namespace()
    graph = namespace["create_multi_company_orchestrator"](
        _FakePlanner(["Microsoft"]), _model_factory, _tools_factory,
        signal_extractor_factory=_extractor_factory,
        max_concurrency=2, recursion_limit=40,
    )
    bounded = graph._bounded_config({"max_concurrency": 99, "recursion_limit": 999})
    assert bounded["max_concurrency"] == 2
    assert bounded["recursion_limit"] == 40


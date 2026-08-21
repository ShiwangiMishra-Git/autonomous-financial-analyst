"""Focused deterministic tests for the explicitly gated F16 notebook live adapter."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.f16_live_adapter import (
    LIVE_CONTRACT_NAMES,
    LIVE_TOOL_NAMES,
    create_notebook_live_executor,
    notebook_live_readiness,
)
from scripts.run_f16_scenarios import PRIMARY_SCENARIOS


CONFIGURED = {
    "F16_ENABLE_LIVE_TESTS": "1",
    "OPENAI_API_KEY": "top-secret-key",
    "OPENAI_API_BASE": "https://private.example.invalid/v1",
    "TAVILY_API_KEY": "top-secret-tavily",
}


def _evidence_result(run_id: str, ticker: str, company_id: str, profile_id: str) -> dict[str, Any]:
    """Build the smallest normalized current-run result needed by the adapter boundary."""
    return {
        "run_id": run_id,
        "company": {
            "company_id": company_id,
            "ticker": ticker,
            "company_name": company_id.title(),
            "profile_id": profile_id,
            "resolution_status": "resolved",
        },
        "profile_id": profile_id,
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": [{
            "evidence_id": f"EV-{run_id}-{ticker}",
            "run_id": run_id,
            "company_id": company_id,
            "ticker": ticker,
            "profile_id": profile_id,
            "evidence_type": "live-test",
            "source_name": "fake-provider",
            "status": "success",
        }],
        "missing_dimensions": [],
        "errors": [],
        "status": "success",
    }


class _FakeGraph:
    """Return an injected terminal F12 state and retain invocation input."""

    def __init__(self, state: dict[str, Any]):
        self.state = deepcopy(state)
        self.inputs: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(deepcopy(state))
        return deepcopy(self.state)


def _workflow_result(context: dict[str, Any]) -> dict[str, Any]:
    """Build a compact successful F15-shaped result for adapter isolation tests."""
    evidence_id = next(iter(context["normalized_results"].values()))["evidence"][0][
        "evidence_id"
    ]
    return {
        "final_status": "success",
        "final_answer": f"Grounded answer [{evidence_id}].",
        "synthesis": {
            "mode": context["comparison_mode"],
            "answer": f"Grounded answer [{evidence_id}].",
            "evidence_ids": [evidence_id],
            "scores_used": deepcopy(context["scores"] or {}),
            "limitations": [],
        },
        "validation": {
            "valid": True,
            "validated_evidence_ids": [evidence_id],
            "errors": [],
        },
        "attempts": 1,
        "correction_attempts": 0,
        "warnings": [],
        "trace_path": ".research_runs/fake-live.json",
    }


def _namespace(state: dict[str, Any]) -> tuple[dict[str, Any], _FakeGraph, list[Any]]:
    """Create a complete fake notebook namespace and capture F15 calls."""
    graph = _FakeGraph(state)
    f15_calls: list[Any] = []

    def build_graph(*args: Any, **kwargs: Any) -> _FakeGraph:
        build_graph.args = args
        build_graph.kwargs = kwargs
        return graph

    def run_f15(context: dict[str, Any], model: Any, **kwargs: Any) -> dict[str, Any]:
        f15_calls.append((deepcopy(context), model, deepcopy(kwargs)))
        return _workflow_result(context)

    namespace: dict[str, Any] = {
        "ChatOpenAI": lambda **_kwargs: object(),
        "create_multi_company_orchestrator": build_graph,
        "run_f15_validated_synthesis": run_f15,
        "retriever": object(),
        "_BIOPHARMA_VECTORSTORE": object(),
    }
    namespace.update({name: (lambda **_kwargs: {}) for name in LIVE_TOOL_NAMES})
    return namespace, graph, f15_calls


def test_readiness_reports_names_and_booleans_without_secret_values() -> None:
    """Presence checks must never echo keys, endpoints, or document content."""
    namespace, _, _ = _namespace({})
    status = notebook_live_readiness(namespace, CONFIGURED)
    rendered = json.dumps(status, sort_keys=True)

    assert status["ready"] is True
    assert status["rag_ready"] == {"technology": True, "biopharma": True}
    assert "top-secret" not in rendered
    assert "private.example" not in rendered


def test_readiness_distinguishes_missing_indexes_from_hard_contracts() -> None:
    """An absent local RAG index is visible but remains a bounded-partial condition."""
    namespace, _, _ = _namespace({})
    namespace["retriever"] = None
    namespace["_BIOPHARMA_VECTORSTORE"] = None

    status = notebook_live_readiness(namespace, CONFIGURED)

    assert status["ready"] is True
    assert status["rag_ready"] == {"technology": False, "biopharma": False}


def test_gate_fails_before_constructing_provider_or_graph() -> None:
    """No model or orchestrator may be created without explicit opt-in."""
    calls: list[str] = []
    namespace = {
        name: (lambda *args, _name=name, **kwargs: calls.append(_name))
        for name in (*LIVE_CONTRACT_NAMES, *LIVE_TOOL_NAMES)
    }

    with pytest.raises(RuntimeError, match="explicit F16_ENABLE_LIVE_TESTS=1"):
        create_notebook_live_executor(
            namespace,
            environ={key: value for key, value in CONFIGURED.items() if key != "F16_ENABLE_LIVE_TESTS"},
            model_factory=lambda role: calls.append(role),
        )

    assert calls == []


def test_single_live_scenario_reaches_f15_with_current_graph_state(tmp_path: Path) -> None:
    """A live single-company result must pass the graph's actual F12 mode into F15."""
    spec = PRIMARY_SCENARIOS[0]
    run_id = "live-single"
    state = {
        "run_id": run_id,
        "original_query": spec.query,
        "plan": {"scoring_requested": False, "risk_profile": "balanced"},
        "comparison_mode": "single",
        "normalized_company_results": {
            "MSFT": _evidence_result(run_id, "MSFT", "microsoft", "technology.ai.v1"),
        },
        "scoring_eligibility": {"eligible": False, "reason": "single mode"},
    }
    namespace, graph, f15_calls = _namespace(state)
    progress: list[str] = []
    executor = create_notebook_live_executor(
        namespace,
        environ=CONFIGURED,
        trace_dir=tmp_path,
        model_factory=lambda role: f"model:{role}",
        progress=progress.append,
    )

    result = executor(spec)

    assert result["final_status"] == "success"
    assert len(graph.inputs) == 1
    assert graph.inputs[0]["messages"][0].content == spec.query
    context, model, options = f15_calls[0]
    assert context["comparison_mode"] == "single"
    assert context["scores"] is None
    assert model == "model:synthesis"
    assert options["trace_dir"] == tmp_path
    assert any("normalized MSFT" in message for message in progress)


def test_arbitrary_query_string_uses_graph_selected_mode() -> None:
    """Interactive free text must accept the graph's guarded mode without a predefined scenario."""
    query = "Please analyze Microsoft using current financial and AI evidence."
    run_id = "live-interactive"
    state = {
        "run_id": run_id,
        "original_query": query,
        "plan": {"scoring_requested": False, "risk_profile": "balanced"},
        "comparison_mode": "single",
        "normalized_company_results": {
            "MSFT": _evidence_result(run_id, "MSFT", "microsoft", "technology.ai.v1"),
        },
        "scoring_eligibility": {"eligible": False, "reason": "single mode"},
    }
    namespace, graph, f15_calls = _namespace(state)
    executor = create_notebook_live_executor(
        namespace, environ=CONFIGURED, model_factory=lambda role: role,
    )

    result = executor(query)

    assert result["final_status"] == "success"
    assert graph.inputs[0]["messages"][0].content == query
    assert f15_calls[0][0]["comparison_mode"] == "single"


def test_arbitrary_query_rejects_blank_text_before_graph_invocation() -> None:
    """Interactive execution must reject a blank question locally."""
    namespace, graph, _ = _namespace({})
    executor = create_notebook_live_executor(
        namespace, environ=CONFIGURED, model_factory=lambda role: role,
    )

    with pytest.raises(ValueError, match="non-empty"):
        executor("   ")

    assert graph.inputs == []


def test_same_profile_scores_only_when_plan_requests_and_f12_authorizes() -> None:
    """F13 remains deterministic and cannot be entered solely because the mode is comparable."""
    spec = PRIMARY_SCENARIOS[2]
    run_id = "live-same-profile"
    results = {
        "MSFT": _evidence_result(run_id, "MSFT", "microsoft", "technology.ai.v1"),
        "NVDA": _evidence_result(run_id, "NVDA", "nvidia", "technology.ai.v1"),
    }
    eligibility = {"eligible": True, "rubric_id": "technology.ai.score.v1"}
    state = {
        "run_id": run_id,
        "original_query": spec.query,
        "plan": {"scoring_requested": True, "risk_profile": "growth"},
        "comparison_mode": "same_profile",
        "normalized_company_results": results,
        "scoring_eligibility": eligibility,
    }
    namespace, _, f15_calls = _namespace(state)
    score_calls: list[Any] = []

    def score(*args: Any) -> dict[str, Any]:
        score_calls.append(args)
        return {"MSFT": {"total_score": 80}, "NVDA": {"total_score": 75}}

    namespace["compute_sector_scores"] = score
    executor = create_notebook_live_executor(
        namespace, environ=CONFIGURED, model_factory=lambda role: role,
    )

    executor(spec)

    assert len(score_calls) == 1
    assert score_calls[0][1:] == ("same_profile", eligibility, "growth")
    assert f15_calls[0][0]["scores"]["MSFT"]["total_score"] == 80


def test_route_mismatch_fails_closed_before_synthesis() -> None:
    """The declared demonstration scenario cannot silently produce a different comparison mode."""
    spec = PRIMARY_SCENARIOS[0]
    run_id = "live-route-mismatch"
    state = {
        "run_id": run_id,
        "original_query": spec.query,
        "plan": {"scoring_requested": False},
        "comparison_mode": "same_profile",
        "normalized_company_results": {
            "MSFT": _evidence_result(run_id, "MSFT", "microsoft", "technology.ai.v1"),
            "NVDA": _evidence_result(run_id, "NVDA", "nvidia", "technology.ai.v1"),
        },
        "scoring_eligibility": {"eligible": True},
    }
    namespace, _, f15_calls = _namespace(state)
    executor = create_notebook_live_executor(
        namespace, environ=CONFIGURED, model_factory=lambda role: role,
    )

    with pytest.raises(RuntimeError, match="routing mismatch"):
        executor(spec)

    assert f15_calls == []


def test_guardrail_stop_never_claims_validation_or_creates_trace() -> None:
    """A pre-research stop remains visibly distinct from an F15-validated answer."""
    spec = PRIMARY_SCENARIOS[1]
    namespace, _, f15_calls = _namespace({
        "final_answer": "Company resolution was ambiguous.",
        "run_errors": ["Company resolution was ambiguous."],
    })
    executor = create_notebook_live_executor(
        namespace, environ=CONFIGURED, model_factory=lambda role: role,
    )

    result = executor(spec)

    assert result["final_status"] == "bounded_stop"
    assert result["validation"]["valid"] is False
    assert result["trace_path"] == ""
    assert f15_calls == []

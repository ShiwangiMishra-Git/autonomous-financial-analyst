"""Explicitly gated provider-backed adapter for the F16 notebook demonstration.

The adapter deliberately depends on an already initialized notebook namespace.  It does not
load ``config.json``, build a vector index, import notebook cells, or make calls at import time.
This keeps credentials and expensive setup under the learner's control while allowing the same
F1--F15 graph used by the notebook to be exercised online.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain_core.messages import HumanMessage

from scripts.run_f16_scenarios import (
    LIVE_OPT_IN_ENV,
    LIVE_REQUIRED_ENV,
    ScenarioSpec,
)


LIVE_TOOL_NAMES = (
    "get_stock_price",
    "get_financial_metrics",
    "get_stock_history",
    "search_financial_news",
    "analyze_sentiment",
    "query_technology_rag",
    "query_biopharma_rag",
)

LIVE_CONTRACT_NAMES = (
    "ChatOpenAI",
    "create_multi_company_orchestrator",
    "run_f15_validated_synthesis",
)


def _enabled(value: Any) -> bool:
    """Return whether a conventional environment flag explicitly enables a feature."""
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _is_tool(value: Any) -> bool:
    """Return whether a notebook value can be invoked as a LangChain or Python tool."""
    return callable(value) or callable(getattr(value, "invoke", None))


def notebook_live_readiness(
    namespace: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Report presence-only readiness without returning secrets, endpoints, or document text.

    ``ready`` means the live call boundary may be entered.  RAG readiness is reported separately
    because the graph can still return a bounded partial answer when one local index is absent.
    """
    if not isinstance(namespace, Mapping):
        raise TypeError("namespace must be a mapping")
    environment = os.environ if environ is None else environ
    missing_environment = [name for name in LIVE_REQUIRED_ENV if not environment.get(name)]
    missing_contracts = [
        name for name in LIVE_CONTRACT_NAMES if not callable(namespace.get(name))
    ]
    missing_tools = [name for name in LIVE_TOOL_NAMES if not _is_tool(namespace.get(name))]
    rag_readiness = {
        "technology": namespace.get("retriever") is not None,
        "biopharma": namespace.get("_BIOPHARMA_VECTORSTORE") is not None,
    }
    opted_in = _enabled(environment.get(LIVE_OPT_IN_ENV))
    return {
        "opted_in": opted_in,
        "configured": not missing_environment,
        "contracts_ready": not missing_contracts and not missing_tools,
        "rag_ready": rag_readiness,
        "ready": opted_in and not missing_environment and not missing_contracts and not missing_tools,
        "missing_environment_variable_names": missing_environment,
        "missing_contract_names": missing_contracts,
        "missing_tool_names": missing_tools,
    }


def _default_model_factory(
    namespace: Mapping[str, Any],
    model_name: str,
    environment: Mapping[str, str],
) -> Callable[[str], Any]:
    """Create role-labelled zero-temperature clients using notebook environment configuration."""
    chat_model = namespace["ChatOpenAI"]

    def create_model(role: str) -> Any:
        # ``role`` is retained for injected factories and progress diagnostics. All current live
        # roles use the same bounded model configuration.
        del role
        return chat_model(
            model=model_name,
            temperature=0,
            openai_api_key=environment.get("OPENAI_API_KEY"),
            openai_api_base=environment.get("OPENAI_API_BASE"),
        )

    return create_model


def _bounded_stop_result(state: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a pre-F15 guardrail stop into the compact runner's terminal contract."""
    message = str(state.get("final_answer") or "A mandatory research guardrail stopped the run.")
    return {
        "final_status": "bounded_stop",
        "final_answer": message,
        "synthesis": {},
        "validation": {
            "valid": False,
            "validated_evidence_ids": [],
            "errors": [message],
        },
        "attempts": 0,
        "correction_attempts": 0,
        "warnings": [message],
        "trace_path": "",
    }


def create_notebook_live_executor(
    namespace: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    trace_dir: str | Path = ".research_runs",
    model_name: str = "gpt-4o-mini",
    max_concurrency: int = 2,
    worker_max_tool_rounds: int = 4,
    model_factory: Callable[[str], Any] | None = None,
    orchestrator_factory: Callable[..., Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Callable[[ScenarioSpec | str], Mapping[str, Any]]:
    """Build a live scenario executor from contracts already loaded in the notebook.

    The returned callable performs real planning, guarded fan-out, allowed tool calls, F12
    normalization/routing, optional F13 scoring, F14 synthesis, and F15 validation/tracing.
    Creation fails before any provider client or graph is constructed unless explicit opt-in,
    environment presence, and notebook contracts all pass.
    """
    environment = os.environ if environ is None else environ
    status = notebook_live_readiness(namespace, environment)
    if not status["opted_in"]:
        raise RuntimeError(f"Live execution requires explicit {LIVE_OPT_IN_ENV}=1 opt-in")
    if status["missing_environment_variable_names"]:
        missing = ", ".join(status["missing_environment_variable_names"])
        raise RuntimeError(f"Live execution is not configured; missing variables: {missing}")
    if status["missing_contract_names"] or status["missing_tool_names"]:
        missing = status["missing_contract_names"] + status["missing_tool_names"]
        raise RuntimeError("Notebook live contracts are not loaded: " + ", ".join(missing))
    if not 1 <= max_concurrency <= 4:
        raise ValueError("max_concurrency must be between 1 and 4 for the notebook demo")
    if not 1 <= worker_max_tool_rounds <= 6:
        raise ValueError("worker_max_tool_rounds must be between 1 and 6")

    emit = progress or (lambda _message: None)
    make_model = model_factory or _default_model_factory(namespace, model_name, environment)
    build_orchestrator = orchestrator_factory or namespace["create_multi_company_orchestrator"]
    tools = {name: namespace[name] for name in LIVE_TOOL_NAMES}

    def worker_model_factory(_task: Mapping[str, Any], _profile: Mapping[str, Any]) -> Any:
        """Return a fresh model so concurrent company branches do not share mutable state."""
        return make_model("company_worker")

    def worker_tools_factory(
        _task: Mapping[str, Any], profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Expose only the profile allowlist from the already loaded notebook tools."""
        return {name: tools[name] for name in profile["allowed_tools"]}

    graph = build_orchestrator(
        make_model("planner"),
        worker_model_factory,
        worker_tools_factory,
        max_concurrency=max_concurrency,
        worker_max_tool_rounds=worker_max_tool_rounds,
        enable_f12=True,
    )

    def execute(spec: ScenarioSpec | str) -> Mapping[str, Any]:
        """Execute a predefined scenario or arbitrary free-text query through the real graph."""
        if isinstance(spec, str):
            query = spec.strip()
            if not query:
                raise ValueError("live query must be a non-empty string")
            scenario_name = "interactive_query"
            expected_mode = None
            expected_tickers = "resolved from free text"
            mode_label = "auto"
        elif isinstance(spec, ScenarioSpec):
            query = spec.query
            scenario_name = spec.name
            expected_mode = spec.mode
            expected_tickers = ", ".join(company[1] for company in spec.companies) or "unresolved"
            mode_label = spec.mode
        else:
            raise TypeError("live executor expects a ScenarioSpec or query string")

        emit(f"[F16 live] starting {scenario_name} ({mode_label}): {expected_tickers}")
        state = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "remembered_company_ids": [],
            "last_profile_ids": [],
        })
        results = state.get("normalized_company_results") or {}
        actual_mode = state.get("comparison_mode")
        if not results or actual_mode not in {"single", "same_profile", "cross_profile"}:
            emit(f"[F16 live] {scenario_name}: stopped at a mandatory guardrail")
            return _bounded_stop_result(state)
        if expected_mode is not None and actual_mode != expected_mode:
            raise RuntimeError(
                f"Live routing mismatch for {scenario_name}: expected {expected_mode}, got {actual_mode}"
            )

        tickers = sorted(results)
        emit(f"[F16 live] {scenario_name}: normalized {', '.join(tickers)} as {actual_mode}")
        eligibility = deepcopy(dict(state.get("scoring_eligibility") or {}))
        plan = state.get("plan") or {}
        scores = None
        if plan.get("scoring_requested") is True and eligibility.get("eligible") is True:
            compute_scores = namespace.get("compute_sector_scores")
            if not callable(compute_scores):
                raise RuntimeError("Scoring was requested but compute_sector_scores is not loaded")
            emit(f"[F16 live] {scenario_name}: applying deterministic F13 sector scoring")
            scores = compute_scores(
                results,
                actual_mode,
                eligibility,
                plan.get("risk_profile") or "balanced",
            )

        context = {
            "run_id": state["run_id"],
            "original_query": state.get("original_query") or query,
            "comparison_mode": actual_mode,
            "normalized_results": results,
            "scoring_eligibility": eligibility,
            "scores": scores,
        }
        result = namespace["run_f15_validated_synthesis"](
            context,
            make_model("synthesis"),
            trace_dir=trace_dir,
        )
        emit(
            f"[F16 live] {scenario_name}: {result.get('final_status')} after "
            f"{result.get('attempts')} synthesis attempt(s)"
        )
        return result

    return execute

"""Deterministic tests for the F07 Technology/AI profile adapters."""

from __future__ import annotations

import contextlib
from functools import lru_cache
import io
import json
from pathlib import Path
import warnings


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


class _FakeLegacyRag:
    """Record legacy RAG invocations and return a fixed grounded answer."""

    def __init__(self, answer="legacy grounded answer"):
        """Store the answer returned by every fake invocation."""
        self.answer = answer
        self.calls = []

    def invoke(self, arguments):
        """Record tool arguments and return the configured answer."""
        self.calls.append(arguments)
        return self.answer


@lru_cache(maxsize=1)
def _technology_namespace():
    """Execute F01–F07 cells with stubbed legacy technology functions."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "query_private_database": _FakeLegacyRag(),
        "extract_ai_signals": lambda companies, prior_reports=None: {},
        "score_companies": lambda financial, signals, sentiment, risk_profile="balanced": {
            "risk_profile": risk_profile,
            "tickers": list(financial),
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in (
            "multiindustry_state_contracts", "multiindustry_company_registry",
            "multiindustry_query_planner", "multiindustry_industry_profiles",
            "multiindustry_company_tasks", "multiindustry_evidence_adapters",
            "multiindustry_technology_profile",
        ):
            exec(cells[cell_id], namespace)
    return namespace


def _technology_task(namespace, run_id="run-tech"):
    """Build one Microsoft technology task fixture."""
    company = namespace["resolve_company_mention"]("Microsoft")
    plan = {
        "query_type": "analyze", "company_mentions": ["Microsoft"],
        "requested_dimensions": ["ai_strategy"], "risk_profile": "balanced",
        "scoring_requested": False, "freshness_required": False, "time_horizon": None,
    }
    task = namespace["build_company_tasks"](plan, [company], run_id)[0]
    return company, task


def test_explicit_technology_rag_wrapper_uses_only_technology_collection():
    """Validate profile identity and prevent biopharma collection access."""
    namespace = _technology_namespace()
    result = namespace["query_technology_rag"].invoke(
        {"ticker": "MSFT", "query": "AI infrastructure"}
    )

    assert result["status"] == "success"
    assert result["profile_id"] == "technology.ai.v1"
    assert result["collection"] == "AI_Initiatives"
    assert "Biopharma" not in result["collection"]
    legacy = namespace["query_private_database"]
    assert "Microsoft Corporation (MSFT)" in legacy.calls[0]["query"]


def test_technology_rag_rejects_biopharma_company_before_retrieval():
    """Ensure a Pfizer call cannot reach the technology retriever."""
    namespace = _technology_namespace()
    legacy = namespace["query_private_database"]
    legacy.calls.clear()
    result = namespace["query_technology_rag"].invoke(
        {"ticker": "PFE", "query": "pipeline"}
    )

    assert result["status"] == "error"
    assert legacy.calls == []


def test_compatibility_access_returns_the_underlying_legacy_result():
    """Preserve assignment behavior while emitting a migration warning."""
    namespace = _technology_namespace()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = namespace["query_private_database_compat"]("legacy question")

    assert result == "legacy grounded answer"
    assert any(item.category is DeprecationWarning for item in caught)


def test_technology_signals_keep_four_dimensions_and_evidence_ids():
    """Bind every non-missing technology signal to current-company evidence."""
    namespace = _technology_namespace()
    company, task = _technology_task(namespace)
    evidence = namespace["query_technology_rag_evidence"](
        task, "AI strategy",
        {"status": "success", "ticker": "MSFT", "data": "grounded", "page": 1},
    )
    raw = {"MSFT": {
        name: {"level": "partial", "reason": "grounded"}
        for name in namespace["TECHNOLOGY_SIGNAL_NAMES"]
    }}
    signals = namespace["extract_technology_signals_with_evidence"](
        [company], {"microsoft": evidence}, raw_signals=raw,
    )

    assert set(signals["MSFT"]) == set(namespace["TECHNOLOGY_SIGNAL_NAMES"])
    assert all(signal["evidence_ids"] for signal in signals["MSFT"].values())
    assert all(signal["score"] == 0.5 for signal in signals["MSFT"].values())


def test_ungrounded_non_missing_signal_is_downgraded_to_missing():
    """Reject LLM signal claims that have no valid current-run evidence reference."""
    namespace = _technology_namespace()
    company, _ = _technology_task(namespace)
    raw = {"MSFT": {
        name: {"level": "full", "reason": "unsupported"}
        for name in namespace["TECHNOLOGY_SIGNAL_NAMES"]
    }}
    signals = namespace["extract_technology_signals_with_evidence"](
        [company], {"microsoft": []}, raw_signals=raw,
    )

    assert all(signal["level"] == "missing" for signal in signals["MSFT"].values())


def test_score_wrapper_delegates_without_changing_inputs_or_risk_profile():
    """Prove the profile-specific score name preserves the existing pure-function path."""
    namespace = _technology_namespace()
    result = namespace["score_technology_companies"](
        {"MSFT": {"market_cap": 1}}, {"MSFT": {}}, {"MSFT": {}}, "growth"
    )

    assert result == {"risk_profile": "growth", "tickers": ["MSFT"]}

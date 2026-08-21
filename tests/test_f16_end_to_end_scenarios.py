"""Offline F16 scenarios spanning the real deterministic F1-F15 boundaries.

The fixtures in this module replace market, news, RAG, and synthesis providers with local
deterministic doubles.  They intentionally exercise provenance and contract validation, not
semantic entailment of answer prose.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts.implement_multiindustry_f12_normalization import F12_NORMALIZATION_CODE
from scripts.implement_multiindustry_f12_routing import F12_ROUTING_CODE
from scripts.implement_multiindustry_f13 import F13_CODE
from scripts.implement_multiindustry_f14 import F14_CODE
from scripts.implement_multiindustry_f15 import F15_WORKFLOW_CODE
from scripts.implement_multiindustry_f15_evidence import F15_EVIDENCE_CODE
from scripts.implement_multiindustry_f15_traces import F15_TRACES_CODE


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
TECH_PROFILE = "technology.ai.v1"
BIOPHARMA_PROFILE = "healthcare.biopharma.v1"
TECH_SIGNAL_LEVELS = {"none": 0.0, "partial": 0.5, "full": 1.0, "missing": None}


@lru_cache(maxsize=1)
def _namespace() -> dict[str, Any]:
    """Load the implemented notebook contracts and isolated F12-F15 source once."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace: dict[str, Any] = {
        "Dict": Dict,
        "List": List,
        "TECHNOLOGY_PROFILE_ID": TECH_PROFILE,
        "TECHNOLOGY_SIGNAL_LEVEL_SCORES": TECH_SIGNAL_LEVELS,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in (
            "multiindustry_state_contracts",
            "multiindustry_company_registry",
            "multiindustry_query_planner",
            "multiindustry_industry_profiles",
            "multiindustry_company_tasks",
            "score_companies_def",
        ):
            exec(cells[cell_id], namespace)

        def score_technology_companies(
            financial_metrics: dict[str, Any],
            technology_signals: dict[str, Any],
            sentiment_scores: dict[str, Any],
            risk_profile: str = "balanced",
        ) -> dict[str, Any]:
            """Adapt the assignment's existing scorer to the F13 profile interface."""
            return namespace["score_companies"](
                financial_metrics,
                technology_signals,
                sentiment_scores,
                risk_profile=risk_profile,
            )

        namespace["score_technology_companies"] = score_technology_companies
        for source in (
            F12_NORMALIZATION_CODE,
            F12_ROUTING_CODE,
            F13_CODE,
            F14_CODE,
            F15_EVIDENCE_CODE,
            F15_TRACES_CODE,
            F15_WORKFLOW_CODE,
        ):
            exec(source, namespace)
    return namespace


def _plan(mentions: list[str]) -> dict[str, Any]:
    """Build a valid offline plan that requests scoring only for comparisons."""
    comparing = len(mentions) > 1
    return {
        "query_type": "compare" if comparing else "analyze",
        "company_mentions": list(mentions),
        "requested_dimensions": [],
        "risk_profile": "balanced",
        "scoring_requested": comparing,
        "freshness_required": True,
        "time_horizon": None,
    }


def _record(
    task: dict[str, Any],
    evidence_type: str,
    value: Any,
    source_name: str,
    *,
    status: str = "success",
    error: str | None = None,
) -> dict[str, Any]:
    """Create one canonical fake-provider evidence record for the assigned company."""
    company = task["company"]
    suffix = evidence_type.replace("_", "-").upper()
    return {
        "evidence_id": f"EV-{task['run_id']}-{company['ticker']}-{suffix}",
        "run_id": task["run_id"],
        "company_id": company["company_id"],
        "ticker": company["ticker"],
        "profile_id": company["profile_id"],
        "evidence_type": evidence_type,
        "value": deepcopy(value),
        "source_name": source_name,
        "source_uri": None,
        "document_name": None,
        "page": None,
        "as_of": "2026-08-06",
        "retrieved_at": "2026-08-06T12:00:00+00:00",
        "freshness_status": "fresh",
        "cache_status": "miss",
        "status": status,
        "source_metadata": {"fixture": "offline-f16"},
        "error": error,
    }


class FakeOfflineResearchProvider:
    """Replace market, history, news/sentiment, and profile RAG with local fixtures."""

    def __init__(self, *, failed_rag_tickers: set[str] | None = None):
        """Configure contained RAG failures and initialize an observable call log."""
        self.failed_rag_tickers = set(failed_rag_tickers or set())
        self.calls: list[tuple[str, str]] = []

    def market_metrics(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic five-metric market data required by F13."""
        ticker = task["company"]["ticker"]
        self.calls.append(("market", ticker))
        ordinal = sum(ord(character) for character in ticker) % 7
        return _record(
            task,
            "financial_metrics",
            {
                "ticker": ticker,
                "current_price": 100.0 + ordinal,
                "market_cap": 1_000.0 + ordinal * 100.0,
                "total_revenue": 200.0 + ordinal * 10.0,
                "pe_ratio": 18.0 + ordinal,
                "beta": 0.8 + ordinal / 10.0,
                "dividend_yield": 0.01 + ordinal / 1_000.0,
            },
            "get_financial_metrics",
        )

    def price_history(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic local price history without calling yfinance."""
        ticker = task["company"]["ticker"]
        self.calls.append(("history", ticker))
        return _record(task, "stock_history", [{"close": 100.0}], "get_stock_history")

    def news_sentiment(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic news sentiment without Tavily or an LLM call."""
        ticker = task["company"]["ticker"]
        self.calls.append(("sentiment", ticker))
        return _record(
            task,
            "sentiment",
            {"label": "neutral", "average": None, "articles": []},
            "fake_news_sentiment",
        )

    def profile_rag(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return bounded local RAG provenance or one explicit failed record."""
        ticker = task["company"]["ticker"]
        self.calls.append(("rag", ticker))
        evidence_type = (
            "technology_rag"
            if task["company"]["profile_id"] == TECH_PROFILE
            else "biopharma_rag"
        )
        if ticker in self.failed_rag_tickers:
            return _record(
                task,
                evidence_type,
                None,
                "fake_local_rag",
                status="failed",
                error="Injected offline RAG failure",
            )
        return _record(
            task,
            evidence_type,
            {"summary": "Bounded fixture evidence; no private document content."},
            "fake_local_rag",
        )

    def research(self, task: dict[str, Any]) -> dict[str, Any]:
        """Assemble a worker-shaped result from the four fake provider surfaces."""
        market = self.market_metrics(task)
        history = self.price_history(task)
        sentiment = self.news_sentiment(task)
        rag = self.profile_rag(task)
        rag_ok = rag["status"] == "success"
        signals = {
            dimension: {
                "level": "full" if rag_ok else "missing",
                "score": 999_999.0,
                "reason": "Grounded in the local RAG fixture." if rag_ok else "RAG unavailable.",
                "evidence_ids": [rag["evidence_id"]] if rag_ok else [],
            }
            for dimension in task["industry_dimensions"]
        }
        return {
            "run_id": task["run_id"],
            "company": deepcopy(task["company"]),
            "profile_id": task["company"]["profile_id"],
            "financial_evidence": {},
            "industry_signals": signals,
            "evidence": [market, history, sentiment, rag],
            "missing_dimensions": [] if rag_ok else list(task["industry_dimensions"]),
            "errors": [] if rag_ok else ["Injected offline RAG failure"],
            "status": "success" if rag_ok else "partial",
        }


class FakeSynthesisModel:
    """Produce grounded structured F14 output from only the supplied bounded payload."""

    def __init__(self):
        """Initialize model and tool-binding observations."""
        self.calls: list[list[Any]] = []
        self.bind_tools_called = False

    def bind_tools(self, tools: Any) -> Any:
        """Fail if the answer-writing path exposes any research or scoring tools."""
        self.bind_tools_called = True
        raise AssertionError(f"F14/F15 must remain tool-free: {tools!r}")

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        """Cite every supplied usable evidence ID in declared and inline order."""
        self.calls.append(list(messages))
        payload = next(
            json.loads(message.content)
            for message in messages
            if '"available_evidence_ids"' in getattr(message, "content", "")
        )
        evidence_ids = list(payload["available_evidence_ids"])
        citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
        return {
            "answer": f"Offline grounded synthesis. {citations}".strip(),
            "evidence_ids": evidence_ids,
            "limitations": payload["required_limitations"],
        }


def _run_offline_scenario(
    tmp_path: Path,
    mentions: list[str],
    *,
    run_id: str,
    failed_rag_tickers: set[str] | None = None,
) -> dict[str, Any]:
    """Run resolution through F15 using only deterministic local substitutes."""
    namespace = _namespace()
    plan = _plan(mentions)
    companies = namespace["resolve_companies"](plan)
    resolution = namespace["validate_resolution_gate"](companies)
    provider = FakeOfflineResearchProvider(failed_rag_tickers=failed_rag_tickers)
    model = FakeSynthesisModel()
    if not resolution["ready"]:
        return {
            "plan": plan,
            "companies": companies,
            "resolution": resolution,
            "provider": provider,
            "model": model,
        }

    tasks = namespace["build_company_tasks"](plan, companies, run_id)
    raw_results = {
        task["company"]["ticker"]: provider.research(task)
        for task in reversed(tasks)
    }
    fan_in = namespace["normalize_all_results"](tasks, raw_results, run_id)
    normalized = fan_in["results_by_ticker"]
    routing = namespace["validate_comparison_routing"](normalized, run_id, tasks)
    assert routing["ready"] is True
    mode = routing["comparison_mode"]
    eligibility = namespace["check_scoring_eligibility"](normalized, mode)
    scores = (
        namespace["compute_sector_scores"](
            normalized, mode, eligibility, plan["risk_profile"],
        )
        if eligibility["eligible"]
        else None
    )
    context = {
        "run_id": run_id,
        "original_query": "Offline F16 scenario for " + ", ".join(mentions),
        "comparison_mode": mode,
        "normalized_results": normalized,
        "scoring_eligibility": eligibility,
        "scores": scores,
    }
    workflow = namespace["run_f15_validated_synthesis"](
        context,
        model,
        trace_dir=tmp_path,
        timestamp_provider=lambda: "2026-08-06T12:00:00+00:00",
    )
    return {
        "plan": plan,
        "companies": companies,
        "resolution": resolution,
        "tasks": tasks,
        "provider": provider,
        "fan_in": fan_in,
        "routing": routing,
        "eligibility": eligibility,
        "scores": scores,
        "context": context,
        "model": model,
        "workflow": workflow,
    }


def _assert_success(scenario: dict[str, Any], mode: str) -> None:
    """Assert the common successful offline F16 terminal contract."""
    workflow = scenario["workflow"]
    assert scenario["routing"]["comparison_mode"] == mode
    assert workflow["final_status"] == "success"
    assert workflow["validation"]["valid"] is True
    assert workflow["validation"]["validated_evidence_ids"]
    assert scenario["model"].bind_tools_called is False
    trace = json.loads(Path(workflow["trace_path"]).read_text(encoding="utf-8"))
    assert trace["final_status"] == "success"
    assert trace["comparison_mode"] == mode


def test_single_technology_runs_without_scores_or_credentials(tmp_path: Path) -> None:
    """Run one technology company through grounded narrative-only validation."""
    scenario = _run_offline_scenario(tmp_path, ["Microsoft"], run_id="f16-single-tech")

    _assert_success(scenario, "single")
    assert [company["ticker"] for company in scenario["companies"]] == ["MSFT"]
    assert scenario["eligibility"]["eligible"] is False
    assert scenario["scores"] is None


def test_single_biopharma_runs_without_scores_or_credentials(tmp_path: Path) -> None:
    """Run one biopharma company while retaining its exact healthcare profile."""
    scenario = _run_offline_scenario(tmp_path, ["Pfizer"], run_id="f16-single-bio")

    _assert_success(scenario, "single")
    assert scenario["companies"][0]["ticker"] == "PFE"
    assert scenario["companies"][0]["profile_id"] == BIOPHARMA_PROFILE
    assert scenario["scores"] is None


def test_same_profile_technology_uses_authoritative_f13_scores(tmp_path: Path) -> None:
    """Score technology peers once and preserve that table through F14 and F15."""
    scenario = _run_offline_scenario(
        tmp_path, ["Microsoft", "NVIDIA"], run_id="f16-same-tech",
    )

    _assert_success(scenario, "same_profile")
    assert scenario["eligibility"]["rubric_id"] == "technology.ai.score.v1"
    assert set(scenario["scores"]) == {"MSFT", "NVDA"}
    assert scenario["workflow"]["synthesis"]["scores_used"] == scenario["scores"]


def test_same_profile_biopharma_uses_authoritative_f13_scores(tmp_path: Path) -> None:
    """Score biopharma peers with the fixed pharma rubric and validate immutable use."""
    scenario = _run_offline_scenario(
        tmp_path, ["Pfizer", "Merck"], run_id="f16-same-bio",
    )

    _assert_success(scenario, "same_profile")
    assert scenario["eligibility"]["rubric_id"] == "healthcare.biopharma.score.v1"
    assert set(scenario["scores"]) == {"PFE", "MRK"}
    assert all("pharma_score" in score for score in scenario["scores"].values())
    assert scenario["workflow"]["synthesis"]["scores_used"] == scenario["scores"]


def test_cross_profile_is_qualitative_and_has_no_universal_score(tmp_path: Path) -> None:
    """Route technology plus biopharma to a score-free portfolio comparison."""
    scenario = _run_offline_scenario(
        tmp_path, ["Microsoft", "Pfizer"], run_id="f16-cross-profile",
    )

    _assert_success(scenario, "cross_profile")
    assert scenario["eligibility"]["eligible"] is False
    assert scenario["scores"] is None
    assert scenario["workflow"]["synthesis"]["scores_used"] == {}
    assert any(
        "No universal numeric score" in limitation
        for limitation in scenario["workflow"]["synthesis"]["limitations"]
    )


def test_alias_resolution_precedes_research_and_preserves_canonical_identity(
    tmp_path: Path,
) -> None:
    """Resolve a spaced AstraZeneca alias before any fake provider is called."""
    scenario = _run_offline_scenario(
        tmp_path, ["ASTRA ZENECA"], run_id="f16-alias-astrazeneca",
    )

    _assert_success(scenario, "single")
    assert scenario["companies"][0]["ticker"] == "AZN"
    assert {ticker for _, ticker in scenario["provider"].calls} == {"AZN"}


def test_unknown_company_stops_before_tasks_tools_or_synthesis(tmp_path: Path) -> None:
    """Fail closed at deterministic resolution for an unsupported company."""
    scenario = _run_offline_scenario(
        tmp_path, ["Unknown Example Company"], run_id="f16-unknown",
    )

    assert scenario["resolution"]["ready"] is False
    assert scenario["resolution"]["status"] == "unsupported"
    assert scenario["companies"][0]["resolution_status"] == "unsupported"
    assert scenario["provider"].calls == []
    assert scenario["model"].calls == []
    assert not list(tmp_path.glob("*.json"))


def test_partial_rag_failure_is_contained_and_disclosed(tmp_path: Path) -> None:
    """Keep successful market evidence while explicitly limiting failed profile RAG."""
    scenario = _run_offline_scenario(
        tmp_path,
        ["Pfizer"],
        run_id="f16-partial-rag",
        failed_rag_tickers={"PFE"},
    )

    _assert_success(scenario, "single")
    normalized = scenario["fan_in"]["results_by_ticker"]["PFE"]
    assert normalized["status"] == "partial"
    assert scenario["fan_in"]["partial_tickers"] == ["PFE"]
    assert ("rag", "PFE") in scenario["provider"].calls
    assert any(
        "Injected offline RAG failure" in limitation
        for limitation in scenario["workflow"]["synthesis"]["limitations"]
    )
    assert all(
        not evidence_id.endswith("BIOPHARMA-RAG")
        for evidence_id in scenario["workflow"]["validation"]["validated_evidence_ids"]
    )


def test_invalid_evidence_id_is_rejected_by_explicit_provenance_validation(
    tmp_path: Path,
) -> None:
    """Reject a post-synthesis citation mutation without asserting semantic entailment."""
    namespace = _namespace()
    scenario = _run_offline_scenario(
        tmp_path, ["Microsoft"], run_id="f16-invalid-evidence",
    )
    _assert_success(scenario, "single")
    synthesis = deepcopy(scenario["workflow"]["synthesis"])
    synthesis["answer"] = "A claim with explicit but nonexistent provenance [EV-NOT-IN-RUN]."
    synthesis["evidence_ids"] = ["EV-NOT-IN-RUN"]

    validation = namespace["validate_synthesis_result"](
        scenario["context"]["run_id"],
        scenario["context"]["normalized_results"],
        synthesis,
        authoritative_scores=scenario["context"]["scores"],
        scoring_eligibility=scenario["context"]["scoring_eligibility"],
        required_limitations=namespace["_f14_validate_context"](scenario["context"])[
            "required_limitations"
        ],
    )

    assert validation["valid"] is False
    assert validation["validated_evidence_ids"] == []
    assert any("does not exist in current-run evidence" in error for error in validation["errors"])
    assert not any("semantic" in error.casefold() for error in validation["errors"])


def test_modified_f13_score_is_rejected_without_recalculation(tmp_path: Path) -> None:
    """Detect score-table drift at F15 while leaving authoritative F13 arithmetic untouched."""
    namespace = _namespace()
    scenario = _run_offline_scenario(
        tmp_path, ["Microsoft", "NVIDIA"], run_id="f16-modified-score",
    )
    _assert_success(scenario, "same_profile")
    authoritative = deepcopy(scenario["scores"])
    synthesis = deepcopy(scenario["workflow"]["synthesis"])
    synthesis["scores_used"]["MSFT"]["total_score"] += 1.0

    validation = namespace["validate_synthesis_result"](
        scenario["context"]["run_id"],
        scenario["context"]["normalized_results"],
        synthesis,
        authoritative_scores=authoritative,
        scoring_eligibility=scenario["context"]["scoring_eligibility"],
        required_limitations=namespace["_f14_validate_context"](scenario["context"])[
            "required_limitations"
        ],
    )

    assert validation["valid"] is False
    assert validation["score_fidelity_valid"] is False
    assert any("does not exactly match authoritative F13" in error for error in validation["errors"])
    assert scenario["scores"] == authoritative

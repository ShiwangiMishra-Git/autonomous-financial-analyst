"""Define isolated F13 deterministic sector-scoring notebook cells.

This module intentionally does not edit the canonical notebook when imported or executed.  The
main integration agent can call :func:`integrate_f13_cells` after reviewing this isolated change.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f12_routing_smoke"


F13_INTRO = """## Section 3.13: Guarded Deterministic Sector Scoring

F13 makes technology and biopharma sector scores available only after F12 has approved a complete
`same_profile` comparison. Both scorers rebuild their inputs from canonical current-run evidence,
derive signal numbers from fixed level rubrics, and take the risk profile from validated run
context. An agent may request the score with `run_id`, but cannot supply metrics, weights, signals,
risk profile, or a proposed result.

Technology uses the assignment's existing deterministic arithmetic unchanged. Biopharma uses a
transparent notebook-local research-strength rubric with conservative, balanced, and growth
weights; `sector_risks` is inverted and its bands are not Buy/Hold/Sell recommendations. Numeric
single-company, cross-profile, partial, failed, and otherwise ineligible scoring remain disabled.
"""


F13_CODE = r'''from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

from langchain_core.tools import tool


F13_TECHNOLOGY_PROFILE_ID = "technology.ai.v1"
F13_TECHNOLOGY_RUBRIC_ID = "technology.ai.score.v1"
F13_BIOPHARMA_PROFILE_ID = "healthcare.biopharma.v1"
F13_BIOPHARMA_RUBRIC_ID = "healthcare.biopharma.score.v1"
F13_FINANCIAL_METRIC_NAMES = (
    "market_cap",
    "total_revenue",
    "pe_ratio",
    "beta",
    "dividend_yield",
)
F13_TECHNOLOGY_SIGNAL_NAMES = (
    "infrastructure_moat",
    "product_deployment",
    "research_depth",
    "strategic_commitment",
)
F13_TECHNOLOGY_SIGNAL_LEVEL_SCORES = MappingProxyType({
    "none": 0.0,
    "partial": 0.5,
    "full": 1.0,
})
F13_BIOPHARMA_SIGNAL_NAMES = (
    "clinical_pipeline",
    "regulatory_progress",
    "exclusivity_and_patents",
    "commercialization",
    "sector_risks",
)
F13_BIOPHARMA_POSITIVE_LEVEL_SCORES = MappingProxyType({
    "none": 0.0,
    "partial": 0.5,
    "full": 1.0,
})
F13_BIOPHARMA_RISK_LEVEL_SCORES = MappingProxyType({
    "none": 1.0,
    "partial": 0.5,
    "full": 0.0,
})
F13_BIOPHARMA_SIGNAL_WEIGHTS = MappingProxyType({
    "conservative": MappingProxyType({
        "clinical_pipeline": 0.15,
        "regulatory_progress": 0.20,
        "exclusivity_and_patents": 0.25,
        "commercialization": 0.20,
        "sector_risks": 0.20,
    }),
    "balanced": MappingProxyType({
        "clinical_pipeline": 0.25,
        "regulatory_progress": 0.20,
        "exclusivity_and_patents": 0.20,
        "commercialization": 0.20,
        "sector_risks": 0.15,
    }),
    "growth": MappingProxyType({
        "clinical_pipeline": 0.35,
        "regulatory_progress": 0.25,
        "exclusivity_and_patents": 0.10,
        "commercialization": 0.20,
        "sector_risks": 0.10,
    }),
})
F13_BIOPHARMA_COMPONENT_WEIGHTS = MappingProxyType({
    "conservative": MappingProxyType({"financial": 0.60, "pharma": 0.40}),
    "balanced": MappingProxyType({"financial": 0.50, "pharma": 0.50}),
    "growth": MappingProxyType({"financial": 0.35, "pharma": 0.65}),
})
F13_FINANCIAL_MAX_SCORE = 2.0
F13_RISK_PROFILES = frozenset({"conservative", "balanced", "growth"})


def _f13_nonempty_text(value: Any, field_name: str) -> str:
    """Return a required stripped string or raise a deterministic validation error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _f13_finite_number(value: Any, field_name: str) -> float:
    """Return one finite real number while rejecting booleans and non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _f13_validate_result_identity(
    ticker: str,
    result: Mapping[str, Any],
    expected_run_id: str | None,
    expected_profile_id: str,
) -> tuple[str, Mapping[str, Any], set[str]]:
    """Validate one normalized result and return its run, company, and valid evidence IDs."""
    if result.get("status") != "success":
        raise ValueError(f"{ticker} is not a complete successful result")
    if result.get("errors"):
        raise ValueError(f"{ticker} contains worker or normalization errors")
    if result.get("missing_dimensions"):
        raise ValueError(f"{ticker} has missing required dimensions")

    run_id = _f13_nonempty_text(result.get("run_id"), f"{ticker}.run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("Sector-scoring results span multiple run_ids")
    if result.get("profile_id") != expected_profile_id:
        raise ValueError(f"{ticker} does not match scoring profile {expected_profile_id}")

    company = result.get("company")
    if not isinstance(company, Mapping):
        raise ValueError(f"{ticker} is missing canonical company identity")
    if company.get("resolution_status") != "resolved":
        raise ValueError(f"{ticker} is not canonically resolved")
    company_ticker = _f13_nonempty_text(company.get("ticker"), f"{ticker}.company.ticker")
    if company_ticker != ticker:
        raise ValueError(f"Result key {ticker} does not match company ticker {company_ticker}")
    _f13_nonempty_text(company.get("company_id"), f"{ticker}.company.company_id")
    if company.get("profile_id") != expected_profile_id:
        raise ValueError(f"{ticker} company profile does not match the selected rubric")

    records = result.get("evidence")
    if not isinstance(records, list):
        raise ValueError(f"{ticker}.evidence must be a list")
    successful_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{ticker}.evidence[{index}] must be a mapping")
        evidence_id = _f13_nonempty_text(
            record.get("evidence_id"), f"{ticker}.evidence[{index}].evidence_id",
        )
        if evidence_id in successful_ids:
            raise ValueError(f"{ticker} has duplicate evidence_id {evidence_id}")
        if record.get("run_id") != run_id:
            raise ValueError(f"{ticker} evidence {evidence_id} has the wrong run_id")
        if record.get("company_id") != company["company_id"]:
            raise ValueError(f"{ticker} evidence {evidence_id} crossed the company boundary")
        if record.get("ticker") != ticker:
            raise ValueError(f"{ticker} evidence {evidence_id} has the wrong ticker")
        if record.get("profile_id") != expected_profile_id:
            raise ValueError(f"{ticker} evidence {evidence_id} crossed the profile boundary")
        if record.get("status") == "success":
            successful_ids.add(evidence_id)
    return run_id, company, successful_ids


def _f13_financial_metrics(
    ticker: str,
    result: Mapping[str, Any],
) -> dict[str, float]:
    """Extract all five finite metrics from one successful canonical evidence record."""
    records = [
        record
        for record in result["evidence"]
        if isinstance(record, Mapping) and record.get("evidence_type") == "financial_metrics"
    ]
    if len(records) != 1:
        raise ValueError(f"{ticker} requires exactly one canonical financial_metrics record")
    record = records[0]
    if record.get("status") != "success":
        raise ValueError(f"{ticker} financial_metrics evidence is not successful")
    if record.get("source_name") != "get_financial_metrics":
        raise ValueError(f"{ticker} financial_metrics evidence has a non-canonical source")
    payload = record.get("value")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{ticker} financial_metrics evidence value must be a mapping")
    payload_ticker = payload.get("ticker")
    if payload_ticker is not None and str(payload_ticker).upper() != ticker:
        raise ValueError(f"{ticker} financial_metrics payload has the wrong ticker")
    return {
        metric: _f13_finite_number(payload.get(metric), f"{ticker}.{metric}")
        for metric in F13_FINANCIAL_METRIC_NAMES
    }


def _f13_technology_signals(
    ticker: str,
    result: Mapping[str, Any],
    successful_evidence_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Rebuild four technology signals from levels and verified evidence references."""
    raw_signals = result.get("industry_signals")
    if not isinstance(raw_signals, Mapping):
        raise ValueError(f"{ticker}.industry_signals must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for dimension in F13_TECHNOLOGY_SIGNAL_NAMES:
        signal = raw_signals.get(dimension)
        if not isinstance(signal, Mapping):
            raise ValueError(f"{ticker} is missing technology signal {dimension}")
        level = signal.get("level")
        if level not in F13_TECHNOLOGY_SIGNAL_LEVEL_SCORES:
            raise ValueError(f"{ticker}.{dimension} has an incomplete or invalid level")
        evidence_ids = signal.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError(f"{ticker}.{dimension} is not grounded in evidence")
        if any(
            not isinstance(evidence_id, str)
            or not evidence_id.strip()
            or evidence_id not in successful_evidence_ids
            for evidence_id in evidence_ids
        ):
            raise ValueError(f"{ticker}.{dimension} references invalid current-run evidence")
        normalized[dimension] = {
            "level": level,
            # Deliberately ignore signal["score"] and derive the authoritative value here.
            "score": F13_TECHNOLOGY_SIGNAL_LEVEL_SCORES[level],
            "reason": str(signal.get("reason", "")),
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
        }
    return normalized


def _f13_biopharma_signals(
    ticker: str,
    result: Mapping[str, Any],
    successful_evidence_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Rebuild five grounded biopharma signals, including inverted sector risk."""
    raw_signals = result.get("industry_signals")
    if not isinstance(raw_signals, Mapping):
        raise ValueError(f"{ticker}.industry_signals must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for dimension in F13_BIOPHARMA_SIGNAL_NAMES:
        signal = raw_signals.get(dimension)
        if not isinstance(signal, Mapping):
            raise ValueError(f"{ticker} is missing biopharma signal {dimension}")
        level = signal.get("level")
        score_map = (
            F13_BIOPHARMA_RISK_LEVEL_SCORES
            if dimension == "sector_risks"
            else F13_BIOPHARMA_POSITIVE_LEVEL_SCORES
        )
        if level not in score_map:
            raise ValueError(f"{ticker}.{dimension} has an incomplete or invalid level")
        evidence_ids = signal.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError(f"{ticker}.{dimension} is not grounded in evidence")
        if any(
            not isinstance(evidence_id, str)
            or not evidence_id.strip()
            or evidence_id not in successful_evidence_ids
            for evidence_id in evidence_ids
        ):
            raise ValueError(f"{ticker}.{dimension} references invalid current-run evidence")
        normalized[dimension] = {
            "level": level,
            # Ignore any proposed numeric value and apply the fixed rubric, including risk inversion.
            "score": score_map[level],
            "reason": str(signal.get("reason", "")),
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
        }
    return normalized


def _f13_research_band(total_score: float) -> str:
    """Map a 0–100 research-strength score to a non-investment-action band."""
    if total_score >= 70.0:
        return "Strong research profile"
    if total_score >= 50.0:
        return "Moderate research profile"
    return "Weak research profile"


def score_biopharma_companies(
    financial_metrics: Mapping[str, Mapping[str, float]],
    biopharma_signals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    risk_profile: str = "balanced",
) -> dict[str, dict[str, Any]]:
    """Compute the deterministic notebook-local biopharma research-strength score.

    The existing five-metric rank arithmetic supplies the normalized financial component. Fixed
    biopharma signal weights supply the sector component. Output bands communicate evidence-backed
    research strength and are intentionally not Buy/Hold/Sell recommendations.
    """
    if risk_profile not in F13_RISK_PROFILES:
        raise ValueError(f"risk_profile must be one of {sorted(F13_RISK_PROFILES)}")
    if not isinstance(financial_metrics, Mapping) or len(financial_metrics) < 2:
        raise ValueError("Biopharma scoring requires at least two companies")
    if set(financial_metrics) != set(biopharma_signals):
        raise ValueError("Financial and biopharma signal company sets must match")

    zero_technology_signals = {
        ticker: {
            dimension: {"score": 0.0}
            for dimension in F13_TECHNOLOGY_SIGNAL_NAMES
        }
        for ticker in financial_metrics
    }
    sentiment_scores = {
        ticker: {"average": None, "articles": []}
        for ticker in financial_metrics
    }
    legacy_financial = score_technology_companies(
        dict(financial_metrics),
        zero_technology_signals,
        sentiment_scores,
        risk_profile=risk_profile,
    )
    signal_weights = F13_BIOPHARMA_SIGNAL_WEIGHTS[risk_profile]
    component_weights = F13_BIOPHARMA_COMPONENT_WEIGHTS[risk_profile]
    output: dict[str, dict[str, Any]] = {}
    for ticker in financial_metrics:
        legacy_ticker = legacy_financial.get(ticker)
        if not isinstance(legacy_ticker, Mapping):
            raise ValueError(f"Legacy financial scorer omitted {ticker}")
        raw_financial = _f13_finite_number(
            legacy_ticker.get("fin_score"), f"{ticker}.fin_score",
        )
        normalized_financial = min(max(raw_financial / F13_FINANCIAL_MAX_SCORE, 0.0), 1.0)
        pharma_score = 0.0
        for dimension in F13_BIOPHARMA_SIGNAL_NAMES:
            signal = biopharma_signals[ticker].get(dimension)
            if not isinstance(signal, Mapping):
                raise ValueError(f"{ticker} is missing biopharma signal {dimension}")
            score = _f13_finite_number(signal.get("score"), f"{ticker}.{dimension}.score")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{ticker}.{dimension}.score must be between 0 and 1")
            pharma_score += score * signal_weights[dimension]
        total_score = 100.0 * (
            normalized_financial * component_weights["financial"]
            + pharma_score * component_weights["pharma"]
        )
        output[ticker] = {
            "fin_score": round(raw_financial, 3),
            "financial_score": round(normalized_financial * 100.0, 3),
            "pharma_score": round(pharma_score * 100.0, 3),
            "total_score": round(total_score, 3),
            "research_band": _f13_research_band(total_score),
            "risk_profile": risk_profile,
        }
    return output


def compute_sector_scores(
    results: Mapping[str, CompanyResearchResult],
    comparison_mode: ComparisonMode,
    scoring_eligibility: ScoringEligibility,
    risk_profile: str,
) -> dict[str, dict[str, Any]]:
    """Compute authoritative same-profile sector scores from canonical evidence.

    This pure boundary accepts no LLM-proposed metrics, weights, signal numbers, or scores. It
    raises ``ValueError`` for every ineligible or incomplete input, preserves legacy technology
    arithmetic, and dispatches biopharma peers to the fixed research-strength rubric.
    """
    if comparison_mode != "same_profile":
        raise ValueError("Sector scoring requires same_profile comparison mode")
    if not isinstance(results, Mapping) or len(results) < 2:
        raise ValueError("Sector scoring requires at least two normalized company results")
    if not isinstance(scoring_eligibility, Mapping) or scoring_eligibility.get("eligible") is not True:
        raise ValueError("F12 scoring eligibility did not authorize this comparison")
    rubric_id = scoring_eligibility.get("rubric_id")
    profile_by_rubric = {
        F13_TECHNOLOGY_RUBRIC_ID: F13_TECHNOLOGY_PROFILE_ID,
        F13_BIOPHARMA_RUBRIC_ID: F13_BIOPHARMA_PROFILE_ID,
    }
    expected_profile_id = profile_by_rubric.get(rubric_id)
    if expected_profile_id is None:
        raise ValueError(f"F12 authorized an unsupported sector rubric: {rubric_id!r}")
    if scoring_eligibility.get("excluded_companies"):
        raise ValueError("F12 scoring eligibility excludes one or more companies")
    if scoring_eligibility.get("missing_requirements"):
        raise ValueError("F12 scoring eligibility reports missing requirements")
    if risk_profile not in F13_RISK_PROFILES:
        raise ValueError(f"risk_profile must be one of {sorted(F13_RISK_PROFILES)}")

    recomputed = check_scoring_eligibility(results, comparison_mode)
    if not recomputed.get("eligible") or recomputed.get("rubric_id") != rubric_id:
        raise ValueError("Current normalized results fail deterministic scoring revalidation")

    financial_metrics: dict[str, dict[str, float]] = {}
    sector_signals: dict[str, dict[str, Any]] = {}
    expected_run_id: str | None = None
    company_ids: set[str] = set()
    for raw_ticker, result in results.items():
        ticker = _f13_nonempty_text(raw_ticker, "result ticker")
        if ticker != ticker.upper():
            raise ValueError(f"Result ticker {ticker!r} must be canonical uppercase")
        if not isinstance(result, Mapping):
            raise ValueError(f"Result {ticker} must be a mapping")
        run_id, company, successful_ids = _f13_validate_result_identity(
            ticker, result, expected_run_id, expected_profile_id,
        )
        expected_run_id = expected_run_id or run_id
        if company["company_id"] in company_ids:
            raise ValueError(f"Duplicate company identity {company['company_id']!r}")
        company_ids.add(company["company_id"])
        financial_metrics[ticker] = _f13_financial_metrics(ticker, result)
        if rubric_id == F13_TECHNOLOGY_RUBRIC_ID:
            sector_signals[ticker] = _f13_technology_signals(
                ticker, result, successful_ids,
            )
        else:
            sector_signals[ticker] = _f13_biopharma_signals(
                ticker, result, successful_ids,
            )

    if rubric_id == F13_TECHNOLOGY_RUBRIC_ID:
        # Sentiment is intentionally absent from v1 arithmetic. Preserve the legacy output field
        # without turning an LLM classification into a numeric scoring input.
        sentiment_scores = {
            ticker: {"average": None, "articles": []}
            for ticker in financial_metrics
        }
        scores = score_technology_companies(
            financial_metrics,
            sector_signals,
            sentiment_scores,
            risk_profile=risk_profile,
        )
    else:
        scores = score_biopharma_companies(
            financial_metrics,
            sector_signals,
            risk_profile=risk_profile,
        )
    if not isinstance(scores, Mapping) or set(scores) != set(results):
        raise ValueError("Sector scorer returned an invalid company set")
    return deepcopy(dict(scores))


def _freeze_f13_value(value: Any) -> Any:
    """Recursively copy mutable containers into read-only scoring-context values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_f13_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_f13_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_f13_value(item) for item in value)
    return deepcopy(value)


def _thaw_f13_value(value: Any) -> Any:
    """Create ordinary defensive containers for one pure scoring invocation."""
    if isinstance(value, Mapping):
        return {key: _thaw_f13_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_f13_value(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw_f13_value(item) for item in value}
    return deepcopy(value)


@dataclass(frozen=True)
class SectorScoringContext:
    """Immutable notebook-local inputs authorized for one guarded score request."""

    run_id: str
    normalized_results: Mapping[str, Any]
    comparison_mode: str
    scoring_eligibility: Mapping[str, Any]
    risk_profile: str


_F13_SCORING_CONTEXTS: dict[str, SectorScoringContext] = {}


def register_sector_scoring_context(
    run_id: str,
    normalized_results: Mapping[str, CompanyResearchResult],
    comparison_mode: ComparisonMode,
    scoring_eligibility: ScoringEligibility,
    risk_profile: str,
) -> None:
    """Validate and defensively freeze the authoritative inputs for one current run."""
    validated_run_id = _f13_nonempty_text(run_id, "run_id")
    # Preflight through the same pure boundary so an invalid context is never published.
    compute_sector_scores(
        normalized_results, comparison_mode, scoring_eligibility, risk_profile,
    )
    result_run_ids = {
        result.get("run_id") for result in normalized_results.values()
        if isinstance(result, Mapping)
    }
    if result_run_ids != {validated_run_id}:
        raise ValueError("Scoring context run_id does not match normalized results")
    _F13_SCORING_CONTEXTS[validated_run_id] = SectorScoringContext(
        run_id=validated_run_id,
        normalized_results=_freeze_f13_value(normalized_results),
        comparison_mode=comparison_mode,
        scoring_eligibility=_freeze_f13_value(scoring_eligibility),
        risk_profile=risk_profile,
    )


def clear_sector_scoring_context(run_id: str) -> bool:
    """Remove one notebook-local score context and report whether it existed."""
    return _F13_SCORING_CONTEXTS.pop(run_id, None) is not None


def _f13_blocked_tool_result(run_id: str, error: str) -> dict[str, Any]:
    """Build the stable fail-closed guarded-tool response envelope."""
    return {
        "status": "blocked",
        "run_id": run_id,
        "rubric_id": None,
        "risk_profile": None,
        "scores": {},
        "errors": [error],
    }


@tool
def compute_sector_scores_tool(run_id: str) -> dict[str, Any]:
    """Compute authoritative sector scores for a registered run using only ``run_id``.

    Metrics, signals, weights, risk profile, eligibility, and proposed scores are intentionally
    absent from this agent-callable contract.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        return _f13_blocked_tool_result(str(run_id), "run_id must be a non-empty string")
    context = _F13_SCORING_CONTEXTS.get(run_id)
    if context is None:
        return _f13_blocked_tool_result(
            run_id, f"No validated sector-scoring context exists for run_id {run_id!r}.",
        )
    try:
        results = _thaw_f13_value(context.normalized_results)
        eligibility = _thaw_f13_value(context.scoring_eligibility)
        if context.run_id != run_id:
            raise ValueError("Registered sector-scoring context is stale or mismatched")
        scores = compute_sector_scores(
            results,
            context.comparison_mode,
            eligibility,
            context.risk_profile,
        )
    except (TypeError, ValueError) as exc:
        return _f13_blocked_tool_result(run_id, str(exc))
    return {
        "status": "success",
        "run_id": run_id,
        "rubric_id": eligibility["rubric_id"],
        "risk_profile": context.risk_profile,
        "scores": scores,
        "errors": [],
    }


print("✅ F13 guarded deterministic technology and biopharma sector scoring defined")
'''


F13_SMOKE = r'''# F13 local deterministic smoke; no model, provider, or live RAG call.
def _f13_smoke_result(
    company: ResolvedCompany,
    market_cap: float,
    level: str,
) -> CompanyResearchResult:
    """Build one complete evidence-grounded technology result for the local smoke test."""
    ticker = company["ticker"]
    financial_id = f"f13-financial-{ticker}"
    rag_id = f"f13-rag-{ticker}"
    identity = {
        "run_id": "f13-smoke-run", "company_id": company["company_id"],
        "ticker": ticker, "profile_id": company["profile_id"],
    }
    evidence = [
        {
            **identity, "evidence_id": financial_id, "evidence_type": "financial_metrics",
            "source_name": "get_financial_metrics", "status": "success",
            "value": {
                "ticker": ticker, "market_cap": market_cap, "total_revenue": market_cap / 2,
                "pe_ratio": 20.0, "beta": 1.0, "dividend_yield": 0.01,
            },
        },
        {
            **identity, "evidence_id": rag_id, "evidence_type": "technology_rag",
            "source_name": "query_technology_rag", "status": "success", "value": "Supported.",
        },
    ]
    signals = {
        name: {
            "level": level, "score": 9999.0, "reason": "Supported.",
            "evidence_ids": [rag_id],
        }
        for name in F13_TECHNOLOGY_SIGNAL_NAMES
    }
    return {
        "run_id": "f13-smoke-run", "company": company,
        "profile_id": F13_TECHNOLOGY_PROFILE_ID, "financial_evidence": {},
        "industry_signals": signals, "evidence": evidence,
        "missing_dimensions": [], "errors": [], "status": "success",
    }


_f13_smoke_results = {
    "MSFT": _f13_smoke_result(resolve_company_mention("Microsoft"), 300.0, "full"),
    "GOOGL": _f13_smoke_result(resolve_company_mention("Google"), 200.0, "partial"),
}
_f13_smoke_eligibility: ScoringEligibility = {
    "eligible": True, "rubric_id": F13_TECHNOLOGY_RUBRIC_ID,
    "reason": "Complete same-profile technology comparison.",
    "excluded_companies": [], "missing_requirements": {},
}
_f13_scores_1 = compute_sector_scores(
    _f13_smoke_results, "same_profile", _f13_smoke_eligibility, "balanced",
)
_f13_scores_2 = compute_sector_scores(
    _f13_smoke_results, "same_profile", _f13_smoke_eligibility, "balanced",
)
assert _f13_scores_1 == _f13_scores_2
register_sector_scoring_context(
    "f13-smoke-run", _f13_smoke_results, "same_profile",
    _f13_smoke_eligibility, "balanced",
)
_f13_guarded = compute_sector_scores_tool.invoke({"run_id": "f13-smoke-run"})
assert _f13_guarded["status"] == "success"
assert _f13_guarded["scores"] == _f13_scores_1
clear_sector_scoring_context("f13-smoke-run")


def _f13_biopharma_smoke_result(
    company: ResolvedCompany,
    market_cap: float,
    levels: dict[str, str],
) -> CompanyResearchResult:
    """Build one grounded biopharma result for provisional-rubric verification."""
    ticker = company["ticker"]
    financial_id = f"f13-bio-financial-{ticker}"
    rag_id = f"f13-bio-rag-{ticker}"
    identity = {
        "run_id": "f13-bio-smoke-run", "company_id": company["company_id"],
        "ticker": ticker, "profile_id": company["profile_id"],
    }
    evidence = [
        {
            **identity, "evidence_id": financial_id, "evidence_type": "financial_metrics",
            "source_name": "get_financial_metrics", "status": "success",
            "value": {
                "ticker": ticker, "market_cap": market_cap, "total_revenue": market_cap / 2,
                "pe_ratio": 18.0, "beta": 0.8, "dividend_yield": 0.02,
            },
        },
        {
            **identity, "evidence_id": rag_id, "evidence_type": "biopharma_rag",
            "source_name": "query_biopharma_rag", "status": "success", "value": "Supported.",
        },
    ]
    signals = {
        name: {
            "level": levels[name], "score": 9999.0, "reason": "Supported.",
            "evidence_ids": [rag_id],
        }
        for name in F13_BIOPHARMA_SIGNAL_NAMES
    }
    return {
        "run_id": "f13-bio-smoke-run", "company": company,
        "profile_id": F13_BIOPHARMA_PROFILE_ID, "financial_evidence": {},
        "industry_signals": signals, "evidence": evidence,
        "missing_dimensions": [], "errors": [], "status": "success",
    }


_f13_bio_results = {
    "PFE": _f13_biopharma_smoke_result(resolve_company_mention("Pfizer"), 250.0, {
        "clinical_pipeline": "full", "regulatory_progress": "full",
        "exclusivity_and_patents": "partial", "commercialization": "partial",
        "sector_risks": "full",
    }),
    "MRK": _f13_biopharma_smoke_result(resolve_company_mention("Merck"), 220.0, {
        "clinical_pipeline": "partial", "regulatory_progress": "partial",
        "exclusivity_and_patents": "full", "commercialization": "full",
        "sector_risks": "none",
    }),
}
_f13_bio_eligibility: ScoringEligibility = {
    "eligible": True, "rubric_id": F13_BIOPHARMA_RUBRIC_ID,
    "reason": "Complete same-profile biopharma comparison.",
    "excluded_companies": [], "missing_requirements": {},
}
_f13_bio_scores = compute_sector_scores(
    _f13_bio_results, "same_profile", _f13_bio_eligibility, "balanced",
)
assert set(_f13_bio_scores) == {"PFE", "MRK"}
assert all("research_band" in score for score in _f13_bio_scores.values())
register_sector_scoring_context(
    "f13-bio-smoke-run", _f13_bio_results, "same_profile",
    _f13_bio_eligibility, "balanced",
)
_f13_bio_guarded = compute_sector_scores_tool.invoke({"run_id": "f13-bio-smoke-run"})
assert _f13_bio_guarded["status"] == "success"
assert _f13_bio_guarded["scores"] == _f13_bio_scores
clear_sector_scoring_context("f13-bio-smoke-run")

print("✅ F13 smoke passed: technology and biopharma scores match guarded run contexts")
'''


CELL_SPECS = [
    ("multiindustry_f13_intro", "markdown", F13_INTRO),
    ("multiindustry_sector_scoring", "code", F13_CODE),
    ("multiindustry_f13_smoke", "code", F13_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create one notebook cell with a stable identifier."""
    cell = (
        nbformat.v4.new_markdown_cell(source=source)
        if cell_type == "markdown"
        else nbformat.v4.new_code_cell(source=source)
    )
    cell["id"] = cell_id
    return cell


def integrate_f13_cells(notebook_path: Path = NOTEBOOK_PATH) -> None:
    """Insert or refresh F13 immediately after the F12 routing smoke, idempotently."""
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
        try:
            insertion_index = next(
                index for index, cell in enumerate(notebook.cells)
                if cell.get("id") == INSERT_AFTER_CELL_ID
            ) + 1
        except StopIteration as exc:
            raise ValueError(f"Notebook is missing insertion anchor {INSERT_AFTER_CELL_ID!r}") from exc
        notebook.cells[insertion_index:insertion_index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")
    nbformat.write(notebook, notebook_path)

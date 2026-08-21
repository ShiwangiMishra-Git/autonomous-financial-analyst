"""Focused deterministic tests for F13 guarded sector scoring."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path
from typing import Dict, List

import pytest

from scripts.implement_multiindustry_f13 import F13_CODE, F13_SMOKE
from scripts.implement_multiindustry_f12_routing import F12_ROUTING_CODE


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
TECHNOLOGY_PROFILE_ID = "technology.ai.v1"
BIOPHARMA_PROFILE_ID = "healthcare.biopharma.v1"
TECHNOLOGY_RUBRIC_ID = "technology.ai.score.v1"
BIOPHARMA_RUBRIC_ID = "healthcare.biopharma.score.v1"
TECHNOLOGY_SIGNAL_LEVEL_SCORES = {
    "none": 0.0,
    "partial": 0.5,
    "full": 1.0,
    "missing": None,
}


@lru_cache(maxsize=1)
def _namespace():
    """Execute the legacy scorer, stable contracts, and isolated F13 source."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "Dict": Dict,
        "List": List,
        "TECHNOLOGY_PROFILE_ID": TECHNOLOGY_PROFILE_ID,
        "TECHNOLOGY_SIGNAL_LEVEL_SCORES": TECHNOLOGY_SIGNAL_LEVEL_SCORES,
        "get_industry_profile": lambda profile_id: {
            "profile_id": profile_id,
            "scoring_enabled": profile_id in {
                TECHNOLOGY_PROFILE_ID, BIOPHARMA_PROFILE_ID,
            },
            "rubric_id": {
                TECHNOLOGY_PROFILE_ID: TECHNOLOGY_RUBRIC_ID,
                BIOPHARMA_PROFILE_ID: BIOPHARMA_RUBRIC_ID,
            }.get(profile_id),
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
        exec(cells["multiindustry_company_registry"], namespace)
        exec(cells["score_companies_def"], namespace)

        def score_technology_companies(
            financial_metrics, technology_signals, sentiment_scores, risk_profile="balanced",
        ):
            """Delegate to the actual assignment scorer, matching the F07 wrapper."""
            return namespace["score_companies"](
                financial_metrics,
                technology_signals,
                sentiment_scores,
                risk_profile=risk_profile,
            )

        namespace["score_technology_companies"] = score_technology_companies
        exec(F12_ROUTING_CODE, namespace)
        exec(F13_CODE, namespace)
    return namespace


def _company(ticker: str, profile_id: str = TECHNOLOGY_PROFILE_ID):
    """Return a canonical resolved-company fixture."""
    names = {
        "MSFT": ("microsoft", "Microsoft Corporation"),
        "NVDA": ("nvidia", "NVIDIA Corporation"),
        "PFE": ("pfizer", "Pfizer Inc."),
        "MRK": ("merck", "Merck & Co."),
    }
    company_id, company_name = names[ticker]
    industry, sub_industry, _ = profile_id.split(".")
    return {
        "company_id": company_id,
        "ticker": ticker,
        "company_name": company_name,
        "aliases": [],
        "exchange": "NASDAQ" if profile_id == TECHNOLOGY_PROFILE_ID else "NYSE",
        "industry": industry,
        "sub_industry": sub_industry,
        "profile_id": profile_id,
        "resolution_status": "resolved",
        "resolution_message": None,
    }


def _evidence_record(
    run_id: str,
    ticker: str,
    profile_id: str,
    evidence_type: str,
    value,
):
    """Build one complete canonical evidence record for a score fixture."""
    company = _company(ticker, profile_id)
    return {
        "evidence_id": f"ev-{run_id}-{ticker}-{evidence_type}",
        "run_id": run_id,
        "company_id": company["company_id"],
        "ticker": ticker,
        "profile_id": profile_id,
        "evidence_type": evidence_type,
        "value": deepcopy(value),
        "source_name": (
            "get_financial_metrics"
            if evidence_type == "financial_metrics"
            else "deterministic-test-fixture"
        ),
        "source_uri": None,
        "document_name": None,
        "page": None,
        "as_of": "2026-01-01",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "freshness_status": "fresh",
        "cache_status": "miss",
        "status": "success",
        "source_metadata": {},
        "error": None,
    }


def _technology_result(run_id: str, ticker: str, metrics: dict, levels: list[str]):
    """Build one complete normalized technology result accepted by F13."""
    company = _company(ticker)
    financial_record = _evidence_record(
        run_id, ticker, TECHNOLOGY_PROFILE_ID, "financial_metrics", metrics,
    )
    rag_record = _evidence_record(
        run_id, ticker, TECHNOLOGY_PROFILE_ID, "technology_rag", "grounded report",
    )
    dimensions = [
        "infrastructure_moat",
        "product_deployment",
        "research_depth",
        "strategic_commitment",
    ]
    signals = {
        dimension: {
            "level": level,
            "score": TECHNOLOGY_SIGNAL_LEVEL_SCORES[level],
            "reason": "Grounded fixture signal.",
            "evidence_ids": [rag_record["evidence_id"]],
        }
        for dimension, level in zip(dimensions, levels, strict=True)
    }
    return {
        "run_id": run_id,
        "company": company,
        "profile_id": TECHNOLOGY_PROFILE_ID,
        "financial_evidence": {"financial_metrics": deepcopy(metrics)},
        "industry_signals": signals,
        "evidence": [financial_record, rag_record],
        "missing_dimensions": [],
        "errors": [],
        "status": "success",
    }


def _technology_results(run_id="run-f13"):
    """Return two complete same-profile results with deliberately different ranks."""
    return {
        "MSFT": _technology_result(
            run_id,
            "MSFT",
            {
                "market_cap": 3_000.0,
                "total_revenue": 240.0,
                "pe_ratio": 30.0,
                "beta": 0.9,
                "dividend_yield": 0.8,
            },
            ["full", "partial", "full", "partial"],
        ),
        "NVDA": _technology_result(
            run_id,
            "NVDA",
            {
                "market_cap": 2_000.0,
                "total_revenue": 130.0,
                "pe_ratio": 45.0,
                "beta": 1.4,
                "dividend_yield": 0.1,
            },
            ["partial", "full", "partial", "full"],
        ),
    }


def _biopharma_result(run_id: str, ticker: str, metrics: dict, levels: dict[str, str]):
    """Build one complete normalized biopharma result accepted by F13."""
    company = _company(ticker, BIOPHARMA_PROFILE_ID)
    financial_record = _evidence_record(
        run_id, ticker, BIOPHARMA_PROFILE_ID, "financial_metrics", metrics,
    )
    rag_record = _evidence_record(
        run_id, ticker, BIOPHARMA_PROFILE_ID, "biopharma_rag", "official evidence",
    )
    signals = {
        dimension: {
            "level": level,
            "score": 999_999.0,
            "reason": "Grounded fixture signal.",
            "evidence_ids": [rag_record["evidence_id"]],
        }
        for dimension, level in levels.items()
    }
    return {
        "run_id": run_id,
        "company": company,
        "profile_id": BIOPHARMA_PROFILE_ID,
        "financial_evidence": {"financial_metrics": deepcopy(metrics)},
        "industry_signals": signals,
        "evidence": [financial_record, rag_record],
        "missing_dimensions": [],
        "errors": [],
        "status": "success",
    }


def _biopharma_results(run_id="run-f13-biopharma"):
    """Return pipeline-led and mature biopharma fixtures for profile calibration."""
    shared_metrics = {
        "market_cap": 200.0,
        "total_revenue": 60.0,
        "pe_ratio": 18.0,
        "beta": 0.8,
        "dividend_yield": 0.02,
    }
    return {
        "PFE": _biopharma_result(run_id, "PFE", shared_metrics, {
            "clinical_pipeline": "full",
            "regulatory_progress": "full",
            "exclusivity_and_patents": "none",
            "commercialization": "partial",
            "sector_risks": "full",
        }),
        "MRK": _biopharma_result(run_id, "MRK", shared_metrics, {
            "clinical_pipeline": "none",
            "regulatory_progress": "partial",
            "exclusivity_and_patents": "full",
            "commercialization": "full",
            "sector_risks": "none",
        }),
    }


def _eligible(rubric_id=TECHNOLOGY_RUBRIC_ID):
    """Return the exact successful F12 eligibility contract consumed by F13."""
    return {
        "eligible": True,
        "rubric_id": rubric_id,
        "reason": "Complete same-profile results passed deterministic scoring eligibility.",
        "excluded_companies": [],
        "missing_requirements": {},
    }


def _legacy_inputs(results):
    """Derive the legacy scorer inputs expected from the trusted normalized fixtures."""
    financial = {
        ticker: deepcopy(result["financial_evidence"]["financial_metrics"])
        for ticker, result in results.items()
    }
    signals = {
        ticker: {
            dimension: {
                **deepcopy(signal),
                "score": TECHNOLOGY_SIGNAL_LEVEL_SCORES[signal["level"]],
            }
            for dimension, signal in result["industry_signals"].items()
        }
        for ticker, result in results.items()
    }
    sentiment = {ticker: {} for ticker in results}
    return financial, signals, sentiment


def test_same_inputs_produce_identical_scores_without_mutation():
    """Keep F13 pure and repeatable for one validated input snapshot."""
    namespace = _namespace()
    results = _technology_results()
    original = deepcopy(results)

    first = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(), "balanced",
    )
    second = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(), "balanced",
    )

    assert first == second
    assert results == original


@pytest.mark.parametrize("risk_profile", ["conservative", "balanced", "growth"])
def test_technology_output_exactly_preserves_legacy_scorer(risk_profile):
    """Prevent the F13 adapter from changing the established technology score table."""
    namespace = _namespace()
    results = _technology_results()
    financial, signals, sentiment = _legacy_inputs(results)
    expected = namespace["score_technology_companies"](
        financial, signals, sentiment, risk_profile=risk_profile,
    )

    actual = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(), risk_profile,
    )

    assert actual == expected


def test_risk_profile_comes_from_registered_plan_context_not_result_payload():
    """Use the validated caller-selected profile; the score tool exposes no override."""
    namespace = _namespace()
    run_id = "run-plan-growth"
    results = _technology_results(run_id)
    for result in results.values():
        result["financial_evidence"]["risk_profile"] = "conservative"
    expected = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(), "growth",
    )
    namespace["register_sector_scoring_context"](
        run_id, results, "same_profile", _eligible(), "growth",
    )
    try:
        tool_result = namespace["compute_sector_scores_tool"].invoke({"run_id": run_id})
    finally:
        namespace["clear_sector_scoring_context"](run_id)

    assert tool_result["status"] == "success"
    assert tool_result["risk_profile"] == "growth"
    assert tool_result["scores"] == expected
    schema = namespace["compute_sector_scores_tool"].args_schema.model_json_schema()
    assert set(schema["properties"]) == {"run_id"}


@pytest.mark.parametrize("failure_kind", ["partial", "missing_metric", "missing_dimension"])
def test_incomplete_or_partial_result_is_rejected(failure_kind):
    """Refuse arithmetic when F12 completeness invariants no longer hold."""
    namespace = _namespace()
    results = _technology_results()
    if failure_kind == "partial":
        results["NVDA"]["status"] = "partial"
    elif failure_kind == "missing_metric":
        del results["NVDA"]["evidence"][0]["value"]["beta"]
    else:
        results["NVDA"]["missing_dimensions"] = ["beta"]

    with pytest.raises(ValueError):
        namespace["compute_sector_scores"](
            results, "same_profile", _eligible(), "balanced",
        )


def test_mixed_profile_results_are_rejected_even_with_forged_eligibility():
    """Prevent a universal score from crossing an industry-profile boundary."""
    namespace = _namespace()
    results = _technology_results()
    results["NVDA"]["profile_id"] = BIOPHARMA_PROFILE_ID
    results["NVDA"]["company"]["profile_id"] = BIOPHARMA_PROFILE_ID
    for record in results["NVDA"]["evidence"]:
        record["profile_id"] = BIOPHARMA_PROFILE_ID

    with pytest.raises(ValueError):
        namespace["compute_sector_scores"](
            results, "same_profile", _eligible(), "balanced",
        )


def test_missing_or_ineligible_context_never_scores():
    """Fail closed both before registration and for an explicitly ineligible run."""
    namespace = _namespace()
    missing = namespace["compute_sector_scores_tool"].invoke({"run_id": "run-not-registered"})
    assert missing["status"] == "blocked"
    assert missing["scores"] == {}
    assert missing["errors"]

    eligibility = {
        **_eligible(),
        "eligible": False,
        "rubric_id": None,
        "reason": "Incomplete comparison.",
    }
    with pytest.raises(ValueError):
        namespace["compute_sector_scores"](
            _technology_results(), "same_profile", eligibility, "balanced",
        )

    run_id = "run-ineligible"
    with pytest.raises(ValueError):
        namespace["register_sector_scoring_context"](
            run_id, _technology_results(run_id), "same_profile", eligibility, "balanced",
        )
    blocked = namespace["compute_sector_scores_tool"].invoke({"run_id": run_id})
    assert blocked["status"] == "blocked"
    assert blocked["scores"] == {}
    assert blocked["errors"]


def test_registered_context_is_defensive_and_tool_matches_pure_function():
    """Read only the frozen run snapshot and return the authoritative pure-function table."""
    namespace = _namespace()
    run_id = "run-defensive-context"
    results = _technology_results(run_id)
    expected = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(), "balanced",
    )
    namespace["register_sector_scoring_context"](
        run_id, results, "same_profile", _eligible(), "balanced",
    )
    results["MSFT"]["status"] = "partial"
    results["MSFT"]["evidence"][0]["value"]["market_cap"] = -999_999
    try:
        response = namespace["compute_sector_scores_tool"].invoke({"run_id": run_id})
    finally:
        assert namespace["clear_sector_scoring_context"](run_id) is True

    assert response == {
        "status": "success",
        "run_id": run_id,
        "rubric_id": TECHNOLOGY_RUBRIC_ID,
        "risk_profile": "balanced",
        "scores": expected,
        "errors": [],
    }
    after_clear = namespace["compute_sector_scores_tool"].invoke({"run_id": run_id})
    assert after_clear["status"] == "blocked"


@pytest.mark.parametrize("risk_profile", ["conservative", "balanced", "growth"])
def test_biopharma_baseline_scores_all_risk_profiles_with_research_bands(risk_profile):
    """Produce deterministic 0–100 research-strength output for complete pharma peers."""
    namespace = _namespace()
    first = namespace["compute_sector_scores"](
        _biopharma_results(), "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), risk_profile,
    )
    second = namespace["compute_sector_scores"](
        _biopharma_results(), "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), risk_profile,
    )

    assert first == second
    assert set(first) == {"PFE", "MRK"}
    for score in first.values():
        assert 0.0 <= score["financial_score"] <= 100.0
        assert 0.0 <= score["pharma_score"] <= 100.0
        assert 0.0 <= score["total_score"] <= 100.0
        assert score["research_band"] in {
            "Strong research profile", "Moderate research profile", "Weak research profile",
        }
        assert "recommendation" not in score


def test_biopharma_calibration_changes_sector_emphasis_by_risk_profile():
    """Favor stability conservatively and pipeline/regulatory progress for growth."""
    namespace = _namespace()
    results = _biopharma_results()
    conservative = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), "conservative",
    )
    growth = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), "growth",
    )

    assert conservative["PFE"]["pharma_score"] == 45.0
    assert conservative["MRK"]["pharma_score"] == 75.0
    assert growth["PFE"]["pharma_score"] == 70.0
    assert growth["MRK"]["pharma_score"] == 52.5


def test_biopharma_sector_risk_is_inverted():
    """Reward no evidenced material risk and penalize a full material-risk classification."""
    namespace = _namespace()
    results = _biopharma_results("run-risk-inversion")
    for result in results.values():
        for dimension, signal in result["industry_signals"].items():
            signal["level"] = "partial"
    results["PFE"]["industry_signals"]["sector_risks"]["level"] = "none"
    results["MRK"]["industry_signals"]["sector_risks"]["level"] = "full"

    scores = namespace["compute_sector_scores"](
        results, "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), "balanced",
    )

    assert scores["PFE"]["pharma_score"] == 57.5
    assert scores["MRK"]["pharma_score"] == 42.5


def test_biopharma_guarded_tool_matches_direct_score_and_freezes_context():
    """Expose pharma scoring through run_id only while protecting the registered snapshot."""
    namespace = _namespace()
    run_id = "run-biopharma-context"
    results = _biopharma_results(run_id)
    eligibility = _eligible(BIOPHARMA_RUBRIC_ID)
    expected = namespace["compute_sector_scores"](
        results, "same_profile", eligibility, "balanced",
    )
    namespace["register_sector_scoring_context"](
        run_id, results, "same_profile", eligibility, "balanced",
    )
    results["PFE"]["industry_signals"]["clinical_pipeline"]["level"] = "none"
    try:
        response = namespace["compute_sector_scores_tool"].invoke({"run_id": run_id})
    finally:
        namespace["clear_sector_scoring_context"](run_id)

    assert response["status"] == "success"
    assert response["rubric_id"] == BIOPHARMA_RUBRIC_ID
    assert response["scores"] == expected


def test_llm_supplied_signal_scores_and_proposed_totals_never_enter_arithmetic():
    """Recompute fixed signal values from levels and ignore proposed score fields."""
    namespace = _namespace()
    trusted = _technology_results()
    adversarial = deepcopy(trusted)
    for result in adversarial.values():
        result["financial_evidence"]["total_score"] = 1_000_000
        result["financial_evidence"]["recommendation"] = "Buy"
        for signal in result["industry_signals"].values():
            signal["score"] = 1_000_000
            signal["proposed_weight"] = 1_000_000

    expected = namespace["compute_sector_scores"](
        trusted, "same_profile", _eligible(), "balanced",
    )
    actual = namespace["compute_sector_scores"](
        adversarial, "same_profile", _eligible(), "balanced",
    )

    assert actual == expected


def test_biopharma_proposed_signal_scores_and_weights_are_ignored():
    """Re-derive pharma values from levels instead of accepting model-proposed arithmetic."""
    namespace = _namespace()
    trusted = _biopharma_results()
    adversarial = deepcopy(trusted)
    for result in adversarial.values():
        result["financial_evidence"]["proposed_total"] = 999_999
        for signal in result["industry_signals"].values():
            signal["score"] = 999_999
            signal["proposed_weight"] = 999_999

    expected = namespace["compute_sector_scores"](
        trusted, "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), "balanced",
    )
    actual = namespace["compute_sector_scores"](
        adversarial, "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), "balanced",
    )

    assert actual == expected


@pytest.mark.parametrize("failure_kind", ["missing_signal", "ungrounded_signal", "nonfinite_metric"])
def test_incomplete_or_invalid_biopharma_inputs_are_rejected(failure_kind):
    """Apply strict no-imputation rules to every pharma score input."""
    namespace = _namespace()
    results = _biopharma_results()
    if failure_kind == "missing_signal":
        del results["PFE"]["industry_signals"]["clinical_pipeline"]
    elif failure_kind == "ungrounded_signal":
        results["PFE"]["industry_signals"]["clinical_pipeline"]["evidence_ids"] = ["invented"]
    else:
        results["PFE"]["evidence"][0]["value"]["pe_ratio"] = float("nan")

    with pytest.raises(ValueError):
        namespace["compute_sector_scores"](
            results, "same_profile", _eligible(BIOPHARMA_RUBRIC_ID), "balanced",
        )


def test_integrated_notebook_cells_match_reviewed_sources_and_smoke_executes():
    """Keep the canonical notebook synchronized and verify its no-provider F13 example."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}

    assert cells["multiindustry_sector_scoring"] == F13_CODE
    assert cells["multiindustry_f13_smoke"] == F13_SMOKE
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_f13_smoke"], _namespace())

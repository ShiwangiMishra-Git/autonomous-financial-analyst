"""Focused tests for F15 deterministic evidence and output validation."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import io
import contextlib

import pytest

from scripts.implement_multiindustry_f15_evidence import F15_EVIDENCE_CODE


TECH_PROFILE = "technology.ai.v1"
BIO_PROFILE = "healthcare.biopharma.v1"


@lru_cache(maxsize=1)
def _namespace():
    """Execute the notebook-injectable F15 source in an isolated namespace."""
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(F15_EVIDENCE_CODE, namespace)
    return namespace


def _company(ticker: str, profile_id: str):
    """Return a minimal canonical company identity."""
    return {
        "company_id": ticker.casefold(),
        "ticker": ticker,
        "profile_id": profile_id,
        "resolution_status": "resolved",
    }


def _evidence(run_id: str, ticker: str, profile_id: str, suffix: str = "fact"):
    """Return one current-run evidence record with valid ownership."""
    return {
        "evidence_id": f"EV-{run_id}-{ticker}-{suffix}",
        "run_id": run_id,
        "company_id": ticker.casefold(),
        "ticker": ticker,
        "profile_id": profile_id,
        "evidence_type": "financial_metrics",
        "status": "success",
        "source_name": "test",
    }


def _result(run_id: str, ticker: str, profile_id: str, *, evidence=None, **overrides):
    """Return one F12-shaped normalized company result."""
    result = {
        "run_id": run_id,
        "company": _company(ticker, profile_id),
        "profile_id": profile_id,
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": (
            list(evidence) if evidence is not None
            else [_evidence(run_id, ticker, profile_id)]
        ),
        "missing_dimensions": [],
        "errors": [],
        "status": "success",
    }
    result.update(overrides)
    return result


def _results(mode: str, run_id: str = "run15"):
    """Return a result map selecting single, same-profile, or cross-profile mode."""
    msft = _result(run_id, "MSFT", TECH_PROFILE)
    if mode == "single":
        return {"MSFT": msft}
    if mode == "same_profile":
        return {
            "MSFT": msft,
            "NVDA": _result(run_id, "NVDA", TECH_PROFILE),
        }
    if mode == "cross_profile":
        return {
            "MSFT": msft,
            "PFE": _result(run_id, "PFE", BIO_PROFILE),
        }
    raise ValueError(mode)


def _limitations(mode: str):
    """Return the mandatory clean-result limitation for a non-scored mode."""
    if mode == "single":
        return ["Single-company analysis does not use a comparison score."]
    if mode == "cross_profile":
        return ["No universal numeric score was applied across industry profiles."]
    return []


def _synthesis(mode: str, evidence_ids, *, scores=None, limitations=None, answer=None):
    """Return one F14-shaped structured synthesis fixture."""
    ids = list(evidence_ids)
    return {
        "mode": mode,
        "answer": answer or " ".join(f"Supported [{item}]." for item in ids),
        "evidence_ids": ids,
        "scores_used": deepcopy(scores or {}),
        "limitations": list(_limitations(mode) if limitations is None else limitations),
    }


def _validate(mode: str, synthesis=None, **kwargs):
    """Validate a fixture using matching current-run normalized results."""
    results = _results(mode)
    evidence_id = results[next(iter(results))]["evidence"][0]["evidence_id"]
    return _namespace()["validate_synthesis_result"](
        "run15",
        results,
        synthesis or _synthesis(mode, [evidence_id]),
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["single", "cross_profile"])
def test_valid_non_scored_modes_return_structured_provenance_verdict(mode):
    """Accept exact current-run citations and expose deterministic validation fields."""
    result = _validate(mode)

    assert result["valid"] is True
    assert result["evidence_valid"] is True
    assert result["score_fidelity_valid"] is True
    assert result["mode_restrictions_valid"] is True
    assert result["limitations_valid"] is True
    assert result["validated_evidence_ids"] == result["declared_evidence_ids"]
    assert result["inline_evidence_ids"] == result["declared_evidence_ids"]
    assert result["errors"] == []


def test_citation_must_exist_in_current_run():
    """Reject an invented evidence ID even when inline and declared lists agree."""
    synthesis = _synthesis("single", ["EV-run15-MSFT-invented"])

    result = _validate("single", synthesis)

    assert result["valid"] is False
    assert result["validated_evidence_ids"] == []
    assert any("does not exist" in error for error in result["errors"])


@pytest.mark.parametrize(
    "field,value,error_fragment",
    [
        ("run_id", "old-run", "run_id ownership"),
        ("ticker", "PFE", "ticker ownership"),
        ("company_id", "pfizer", "company_id ownership"),
        ("profile_id", BIO_PROFILE, "profile ownership"),
    ],
)
def test_cited_record_must_match_run_company_and_profile(field, value, error_fragment):
    """Fail closed when a cited record crosses any canonical ownership boundary."""
    results = _results("single")
    record = results["MSFT"]["evidence"][0]
    evidence_id = record["evidence_id"]
    record[field] = value

    result = _namespace()["validate_synthesis_result"](
        "run15", results, _synthesis("single", [evidence_id])
    )

    assert result["valid"] is False
    assert result["validated_evidence_ids"] == []
    assert any(error_fragment in error for error in result["errors"])


def test_only_successful_evidence_is_usable():
    """Reject a known record whose adapter status is not successful."""
    results = _results("single")
    record = results["MSFT"]["evidence"][0]
    record["status"] = "failed"

    result = _namespace()["validate_synthesis_result"](
        "run15", results, _synthesis("single", [record["evidence_id"]])
    )

    assert result["valid"] is False
    assert any("unusable status" in error for error in result["errors"])


def test_duplicate_corpus_id_is_ambiguous_even_with_matching_ownership():
    """Reject duplicate normalized evidence IDs rather than accepting first-seen data."""
    results = _results("single")
    record = results["MSFT"]["evidence"][0]
    results["MSFT"]["evidence"].append(deepcopy(record))

    result = _namespace()["validate_synthesis_result"](
        "run15", results, _synthesis("single", [record["evidence_id"]])
    )

    assert result["valid"] is False
    assert result["validated_evidence_ids"] == []
    assert any("Duplicate evidence_id" in error for error in result["errors"])


@pytest.mark.parametrize(
    "answer,declared,error_fragment",
    [
        ("Supported [EV-run15-MSFT-fact].", [], "exactly match"),
        ("No inline citation.", ["EV-run15-MSFT-fact"], "exactly match"),
        (
            "First [EV-run15-MSFT-fact], again [EV-run15-MSFT-fact].",
            ["EV-run15-MSFT-fact", "EV-run15-MSFT-fact"],
            "duplicate",
        ),
    ],
)
def test_inline_and_declared_ids_must_match_exactly_without_duplicates(
    answer, declared, error_fragment,
):
    """Treat prose/list drift and repeated IDs as deterministic correction failures."""
    synthesis = _synthesis("single", declared, answer=answer)

    result = _validate("single", synthesis)

    assert result["valid"] is False
    assert any(error_fragment in error.casefold() for error in result["errors"])


def test_same_profile_scores_must_exactly_equal_f13_authority_without_mutation():
    """Compare opaque score tables exactly and never recalculate their values."""
    scores = {
        "MSFT": {"total_score": 78.0, "rank": 1},
        "NVDA": {"total_score": 71.0, "rank": 2},
    }
    original = deepcopy(scores)
    results = _results("same_profile")
    evidence_id = results["MSFT"]["evidence"][0]["evidence_id"]
    synthesis = _synthesis("same_profile", [evidence_id], scores=scores)

    result = _namespace()["validate_synthesis_result"](
        "run15",
        results,
        synthesis,
        authoritative_scores=scores,
        scoring_eligibility={"eligible": True},
    )

    assert result["valid"] is True
    assert result["score_fidelity_valid"] is True
    assert scores == original

    tampered = deepcopy(synthesis)
    tampered["scores_used"]["MSFT"]["total_score"] = 99.0
    rejected = _namespace()["validate_synthesis_result"](
        "run15",
        results,
        tampered,
        authoritative_scores=scores,
        scoring_eligibility={"eligible": True},
    )
    assert rejected["valid"] is False
    assert rejected["score_fidelity_valid"] is False
    assert any("exactly match" in error for error in rejected["errors"])


def test_explicit_prose_score_and_rank_claims_must_match_f13_authority():
    """Check recognizable score/rank statements without claiming general semantic proof."""
    scores = {
        "MSFT": {"total_score": 78.0, "rank": 1},
        "NVDA": {"total_score": 71.0, "rank": 2},
    }
    results = _results("same_profile")
    evidence_id = results["MSFT"]["evidence"][0]["evidence_id"]
    valid = _synthesis(
        "same_profile",
        [evidence_id],
        scores=scores,
        answer=f"MSFT total score is 78 and MSFT ranked 1 [{evidence_id}].",
    )
    assert _namespace()["validate_synthesis_result"](
        "run15",
        results,
        valid,
        authoritative_scores=scores,
        scoring_eligibility={"eligible": True},
    )["valid"] is True

    wrong = deepcopy(valid)
    wrong["answer"] = f"MSFT total score is 99 and MSFT ranked 2 [{evidence_id}]."
    rejected = _namespace()["validate_synthesis_result"](
        "run15",
        results,
        wrong,
        authoritative_scores=scores,
        scoring_eligibility={"eligible": True},
    )
    assert rejected["valid"] is False
    assert rejected["score_fidelity_valid"] is False
    assert any("Prose score claim" in error for error in rejected["errors"])
    assert any("Prose rank claim" in error for error in rejected["errors"])


@pytest.mark.parametrize("mode", ["single", "cross_profile"])
def test_non_scored_modes_reject_explicit_numeric_score_or_rank_prose(mode):
    """Block obvious numeric scoring/ranking prose even when scores_used is empty."""
    results = _results(mode)
    ticker = next(iter(results))
    evidence_id = results[ticker]["evidence"][0]["evidence_id"]
    synthesis = _synthesis(
        mode,
        [evidence_id],
        answer=f"{ticker} score is 88 and {ticker} ranked 1 [{evidence_id}].",
    )
    rejected = _namespace()["validate_synthesis_result"]("run15", results, synthesis)
    assert rejected["valid"] is False
    assert rejected["mode_restrictions_valid"] is False
    assert any("prohibited numeric score" in error for error in rejected["errors"])
    assert any("prohibited numeric ranking" in error for error in rejected["errors"])


@pytest.mark.parametrize("mode", ["single", "cross_profile"])
def test_single_and_cross_profile_cannot_use_any_comparison_score(mode):
    """Enforce the no-score boundary for non-like-for-like synthesis modes."""
    results = _results(mode)
    evidence_id = results[next(iter(results))]["evidence"][0]["evidence_id"]
    synthesis = _synthesis(mode, [evidence_id], scores={"MSFT": {"total_score": 100}})

    result = _namespace()["validate_synthesis_result"](
        "run15", results, synthesis,
        authoritative_scores={"MSFT": {"total_score": 100}},
    )

    assert result["valid"] is False
    assert result["mode_restrictions_valid"] is False
    assert any("cannot use comparison scores" in error for error in result["errors"])


def test_required_limitations_are_rebuilt_from_normalized_results():
    """Require partial-result, missing-dimension, and caller-mandated disclosures."""
    results = _results("same_profile")
    results["NVDA"].update({
        "status": "partial",
        "missing_dimensions": ["research_depth"],
        "errors": ["private source unavailable"],
    })
    evidence_id = results["MSFT"]["evidence"][0]["evidence_id"]
    synthesis = _synthesis("same_profile", [evidence_id], limitations=[])

    result = _namespace()["validate_synthesis_result"](
        "run15",
        results,
        synthesis,
        scoring_eligibility={"eligible": False, "reason": "Incomplete company result."},
        required_limitations=["User-visible methodology caveat."],
    )

    assert result["valid"] is False
    assert result["limitations_valid"] is False
    joined = " ".join(result["errors"])
    assert "result status is partial" in joined
    assert "missing dimensions: research_depth" in joined
    assert "private source unavailable" in joined
    assert "Numeric sector scoring was not applied" in joined
    assert "User-visible methodology caveat" in joined


def test_validator_only_checks_explicit_contract_not_semantic_entailment():
    """Document the boundary: valid provenance is not a semantic proof of answer prose."""
    results = _results("single")
    evidence_id = results["MSFT"]["evidence"][0]["evidence_id"]
    synthesis = _synthesis(
        "single",
        [evidence_id],
        answer=f"A deterministic validator does not judge this sentence [{evidence_id}].",
    )

    result = _namespace()["validate_synthesis_result"]("run15", results, synthesis)

    assert result["valid"] is True
    assert result["validated_evidence_ids"] == [evidence_id]

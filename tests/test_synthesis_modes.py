"""Focused deterministic tests for F14 mode-specific grounded synthesis."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path

import pytest

from scripts.implement_multiindustry_f12_routing import F12_ROUTING_CODE
from scripts.implement_multiindustry_f14 import F14_CODE, F14_SMOKE


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
TECHNOLOGY_PROFILE_ID = "technology.ai.v1"
BIOPHARMA_PROFILE_ID = "healthcare.biopharma.v1"


@lru_cache(maxsize=1)
def _namespace():
    """Execute stable contracts, F12 routing, and the isolated F14 implementation."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "get_industry_profile": lambda profile_id: {
            "profile_id": profile_id,
            "scoring_enabled": True,
            "rubric_id": (
                "technology.ai.score.v1"
                if profile_id == TECHNOLOGY_PROFILE_ID
                else "healthcare.biopharma.score.v1"
            ),
        },
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
        exec(F12_ROUTING_CODE, namespace)
        exec(F14_CODE, namespace)
    return namespace


def _company(ticker: str, profile_id: str):
    """Return one canonical resolved-company fixture."""
    names = {
        "MSFT": ("microsoft", "Microsoft Corporation"),
        "NVDA": ("nvidia", "NVIDIA Corporation"),
        "PFE": ("pfizer", "Pfizer Inc."),
    }
    company_id, company_name = names[ticker]
    industry, sub_industry, _ = profile_id.split(".")
    return {
        "company_id": company_id,
        "ticker": ticker,
        "company_name": company_name,
        "aliases": [],
        "exchange": "TEST",
        "industry": industry,
        "sub_industry": sub_industry,
        "profile_id": profile_id,
        "resolution_status": "resolved",
        "resolution_message": None,
    }


def _result(
    run_id: str,
    ticker: str,
    profile_id: str,
    *,
    status: str = "success",
    missing_dimensions: list[str] | None = None,
    errors: list[str] | None = None,
):
    """Build one normalized result with a current-run canonical evidence record."""
    company = _company(ticker, profile_id)
    evidence_id = f"ev-{run_id}-{ticker}"
    return {
        "run_id": run_id,
        "company": company,
        "profile_id": profile_id,
        "financial_evidence": {"financial_metrics": {"market_cap": 1_000.0}},
        "industry_signals": {},
        "evidence": [
            {
                "evidence_id": evidence_id,
                "run_id": run_id,
                "company_id": company["company_id"],
                "ticker": ticker,
                "profile_id": profile_id,
                "evidence_type": "financial_metrics",
                "value": {"market_cap": 1_000.0},
                "source_name": "deterministic-test-fixture",
                "source_uri": None,
                "document_name": None,
                "page": None,
                "as_of": "2026-08-06",
                "retrieved_at": "2026-08-06T00:00:00+00:00",
                "freshness_status": "fresh",
                "cache_status": "miss",
                "status": "success",
                "source_metadata": {},
                "error": None,
            }
        ],
        "missing_dimensions": list(missing_dimensions or []),
        "errors": list(errors or []),
        "status": status,
    }


def _results(mode: str, run_id: str = "run-f14"):
    """Return normalized results whose identities deterministically select ``mode``."""
    msft = _result(run_id, "MSFT", TECHNOLOGY_PROFILE_ID)
    if mode == "single":
        return {"MSFT": msft}
    if mode == "same_profile":
        return {
            "MSFT": msft,
            "NVDA": _result(run_id, "NVDA", TECHNOLOGY_PROFILE_ID),
        }
    if mode == "cross_profile":
        return {
            "MSFT": msft,
            "PFE": _result(run_id, "PFE", BIOPHARMA_PROFILE_ID),
        }
    raise ValueError(mode)


def _eligibility(mode: str):
    """Return a small F12-style eligibility fixture for one comparison mode."""
    eligible = mode == "same_profile"
    return {
        "eligible": eligible,
        "mode": mode,
        "profile_id": TECHNOLOGY_PROFILE_ID if eligible else None,
        "rubric_id": "technology.ai.score.v1" if eligible else None,
        "reason": "Eligible." if eligible else f"Scoring disabled for {mode} mode.",
    }


def _scores():
    """Return a minimal authoritative F13 score table."""
    return {
        "MSFT": {"total_score": 78.0, "rank": 1, "assessment": "Strong"},
        "NVDA": {"total_score": 71.0, "rank": 2, "assessment": "Strong"},
    }


def _context(mode: str, **overrides):
    """Build one valid synthesis context and apply explicit test overrides."""
    context = {
        "run_id": "run-f14",
        "original_query": "Compare the requested companies.",
        "comparison_mode": mode,
        "normalized_results": _results(mode),
        "scoring_eligibility": _eligibility(mode),
        "scores": _scores() if mode == "same_profile" else None,
    }
    context.update(overrides)
    return context


class FakeSynthesisModel:
    """Return a deterministic structured draft without exposing research tools."""

    def __init__(self, evidence_ids, *, limitations=None, scores_used=None):
        """Configure the response payload and initialize call tracking."""
        self.evidence_ids = list(evidence_ids)
        self.limitations = list(limitations or [])
        self.scores_used = deepcopy(scores_used or {})
        self.messages = None
        self.bind_tools_called = False

    def bind_tools(self, tools):
        """Fail if F14 ever attempts to expose tools to the synthesis model."""
        self.bind_tools_called = True
        raise AssertionError(f"F14 must not bind tools: {tools!r}")

    def invoke(self, messages):
        """Record messages and return the configured structured draft."""
        self.messages = list(messages)
        return {
            "answer": "Grounded answer using supplied evidence.",
            "evidence_ids": self.evidence_ids,
            "limitations": self.limitations,
            "scores_used": self.scores_used,
        }


@pytest.mark.parametrize(
    "mode, marker",
    [
        ("single", "single-company synthesis agent"),
        ("same_profile", "same-profile synthesis agent"),
        ("cross_profile", "cross-profile portfolio synthesis agent"),
    ],
)
def test_correct_prompt_selected_for_every_mode(mode, marker):
    """Select the requested mode policy and pass only system plus bounded-data messages."""
    namespace = _namespace()
    results = _results(mode)
    first_id = next(iter(results.values()))["evidence"][0]["evidence_id"]
    model = FakeSynthesisModel([first_id])
    result = namespace["synthesize_answer"](_context(mode), model)

    assert result["mode"] == mode
    assert marker in model.messages[0].content
    assert len(model.messages) == 2
    assert model.bind_tools_called is False


def test_same_profile_uses_only_immutable_authoritative_scores():
    """Ignore model-proposed arithmetic and return an unmodified F13 score copy."""
    namespace = _namespace()
    context = _context("same_profile")
    original = deepcopy(context)
    evidence_id = context["normalized_results"]["MSFT"]["evidence"][0]["evidence_id"]
    model = FakeSynthesisModel([evidence_id], scores_used={"tampered": 999})

    result = namespace["synthesize_answer"](context, model)

    assert result["scores_used"] == _scores()
    assert result["scores_used"] is not context["scores"]
    assert context == original
    payload = json.loads(model.messages[1].content)
    assert payload["authoritative_f13_scores"] == _scores()


@pytest.mark.parametrize("mode", ["single", "cross_profile"])
def test_non_sector_modes_reject_score_context(mode):
    """Prevent universal or comparison scores from entering prohibited modes."""
    namespace = _namespace()
    context = _context(mode, scores={"MSFT": {"total_score": 100}})
    model = FakeSynthesisModel([])
    with pytest.raises(ValueError, match="cannot receive a sector comparison score"):
        namespace["synthesize_answer"](context, model)
    assert model.messages is None


def test_cross_profile_payload_excludes_sector_rubrics_and_scores():
    """Keep technology and biopharma arithmetic out of portfolio synthesis context."""
    namespace = _namespace()
    context = _context("cross_profile")
    evidence_id = context["normalized_results"]["PFE"]["evidence"][0]["evidence_id"]
    model = FakeSynthesisModel([evidence_id])

    result = namespace["synthesize_answer"](context, model)

    prompt_text = "\n".join(message.content for message in model.messages)
    assert "authoritative_f13_scores" not in prompt_text
    assert "technology.ai.score.v1" not in prompt_text
    assert "healthcare.biopharma.score.v1" not in prompt_text
    assert result["scores_used"] == {}
    assert any("No universal numeric score" in item for item in result["limitations"])


def test_partial_and_missing_evidence_limitations_are_mandatory():
    """Add deterministic data-quality limitations even if the model omits them."""
    namespace = _namespace()
    results = _results("same_profile")
    results["NVDA"] = _result(
        "run-f14",
        "NVDA",
        TECHNOLOGY_PROFILE_ID,
        status="partial",
        missing_dimensions=["research_depth"],
        errors=["private source unavailable"],
    )
    context = _context(
        "same_profile",
        normalized_results=results,
        scoring_eligibility={
            **_eligibility("same_profile"),
            "eligible": False,
            "reason": "Incomplete company result.",
        },
        scores=None,
    )
    evidence_id = results["MSFT"]["evidence"][0]["evidence_id"]
    model = FakeSynthesisModel([evidence_id], limitations=["Model-stated caveat."])

    result = namespace["synthesize_answer"](context, model)

    joined = " ".join(result["limitations"])
    assert "NVDA: result status is partial." in joined
    assert "missing dimensions: research_depth" in joined
    assert "private source unavailable" in joined
    assert "Numeric sector scoring was not applied" in joined
    assert "Model-stated caveat." in result["limitations"]


def test_unknown_or_cross_run_evidence_is_rejected():
    """Reject fabricated citations and stale evidence before returning an F14 result."""
    namespace = _namespace()
    with pytest.raises(ValueError, match="unavailable evidence IDs"):
        namespace["synthesize_answer"](
            _context("single"), FakeSynthesisModel(["ev-fabricated"]),
        )

    stale = _results("single")
    stale["MSFT"]["evidence"][0]["run_id"] = "run-old"
    with pytest.raises(ValueError, match="crossed the run or company boundary"):
        namespace["synthesize_answer"](
            _context("single", normalized_results=stale), FakeSynthesisModel([]),
        )


def test_mode_mismatch_and_missing_citation_fail_closed():
    """Require routing consistency and at least one citation when evidence is available."""
    namespace = _namespace()
    with pytest.raises(ValueError, match="does not match normalized results mode"):
        namespace["synthesize_answer"](
            _context("single", normalized_results=_results("same_profile")),
            FakeSynthesisModel([]),
        )
    with pytest.raises(ValueError, match="must cite at least one"):
        namespace["synthesize_answer"](_context("single"), FakeSynthesisModel([]))


def test_notebook_cells_match_script_and_local_smoke_passes():
    """Keep the canonical notebook synchronized and execute its deterministic F14 smoke."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    assert cells["multiindustry_mode_specific_synthesis"] == F14_CODE
    assert cells["multiindustry_f14_smoke"] == F14_SMOKE


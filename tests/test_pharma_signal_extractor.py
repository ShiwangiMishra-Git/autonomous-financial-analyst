"""Deterministic tests for F09 biopharma signal extraction and scoring gate."""

from __future__ import annotations

import contextlib
from functools import lru_cache
import io
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


@lru_cache(maxsize=1)
def _signal_namespace():
    """Execute F01–F09 cells with stubbed technology dependencies."""
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
            "multiindustry_biopharma_signals",
        ):
            exec(cells[cell_id], namespace)
    return namespace


def _pfizer_evidence(namespace, run_id="run-pharma"):
    """Create one canonical Pfizer official-source evidence record."""
    company = namespace["resolve_company_mention"]("Pfizer")
    record = namespace["to_evidence_record"](
        run_id, company, company["profile_id"], "biopharma_rag",
        {"status": "success", "ticker": "PFE", "data": "official evidence", "page": 1},
    )[0]
    return company, record


def test_rubric_covers_every_dimension_and_level():
    """Require complete definitions for none, partial, full, and missing."""
    namespace = _signal_namespace()
    assert namespace["validate_pharma_signal_rubric"]() == []
    assert set(namespace["PHARMA_SIGNAL_RUBRIC"]) == set(namespace["BIOPHARMA_SIGNAL_NAMES"])


def test_output_has_same_schema_and_evidence_ids_for_every_dimension():
    """Normalize structured signals into the stable five-dimension contract."""
    namespace = _signal_namespace()
    company, record = _pfizer_evidence(namespace)
    raw = {"PFE": {
        name: {"level": "partial", "reason": "supported", "evidence_ids": [record["evidence_id"]]}
        for name in namespace["BIOPHARMA_SIGNAL_NAMES"]
    }}
    signals = namespace["extract_pharma_signals"](
        [company], {"pfizer": [record]}, raw_signals=raw,
    )

    assert set(signals["PFE"]) == set(namespace["BIOPHARMA_SIGNAL_NAMES"])
    assert all(signal["score"] == 0.5 for signal in signals["PFE"].values())
    assert all(signal["evidence_ids"] == [record["evidence_id"]] for signal in signals["PFE"].values())


def test_missing_evidence_produces_missing_not_a_guess():
    """Downgrade apparently confident structured output when no evidence is available."""
    namespace = _signal_namespace()
    company = namespace["resolve_company_mention"]("Pfizer")
    raw = {"PFE": {
        name: {"level": "full", "reason": "ungrounded", "evidence_ids": ["invented"]}
        for name in namespace["BIOPHARMA_SIGNAL_NAMES"]
    }}
    signals = namespace["extract_pharma_signals"](
        [company], {"pfizer": []}, raw_signals=raw,
    )

    assert all(signal["level"] == "missing" for signal in signals["PFE"].values())
    assert all(signal["score"] is None for signal in signals["PFE"].values())


def test_wrong_company_or_technology_evidence_is_rejected():
    """Prevent cross-company and cross-profile evidence contamination."""
    namespace = _signal_namespace()
    company, record = _pfizer_evidence(namespace)
    raw = {"PFE": {}}

    wrong_company = dict(record, company_id="merck", ticker="MRK")
    with pytest.raises(ValueError, match="identity mismatch"):
        namespace["extract_pharma_signals"](
            [company], {"pfizer": [wrong_company]}, raw_signals=raw,
        )
    technology = dict(record, profile_id="technology.ai.v1")
    with pytest.raises(ValueError, match="cannot use technology evidence"):
        namespace["extract_pharma_signals"](
            [company], {"pfizer": [technology]}, raw_signals=raw,
        )


def test_biopharma_rubric_configuration_is_enabled_after_f13_calibration():
    """Expose the versioned baseline while leaving per-run completeness to F12/F13."""
    namespace = _signal_namespace()
    eligibility = namespace["check_biopharma_scoring_gate"]()

    assert eligibility["eligible"] is True
    assert eligibility["rubric_id"] == "healthcare.biopharma.score.v1"
    assert eligibility["missing_requirements"] == {}

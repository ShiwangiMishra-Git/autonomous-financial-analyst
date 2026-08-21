"""Deterministic tests for F06 canonical evidence adapters."""

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
def _evidence_namespace():
    """Execute F01–F06 cells without making live provider calls."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in (
            "multiindustry_state_contracts",
            "multiindustry_company_registry",
            "multiindustry_query_planner",
            "multiindustry_industry_profiles",
            "multiindustry_company_tasks",
            "multiindustry_evidence_adapters",
        ):
            exec(cells[cell_id], namespace)
    return namespace


def _company(namespace, name="Microsoft"):
    """Resolve one supported company for evidence fixtures."""
    return namespace["resolve_company_mention"](name)


def test_successful_result_converts_with_identity_and_provenance():
    """Normalize a successful source result without losing dates or cache metadata."""
    namespace = _evidence_namespace()
    company = _company(namespace)
    records = namespace["to_evidence_record"](
        "run-evidence", company, company["profile_id"], "stock_price",
        {
            "ticker": "MSFT", "status": "success", "current_price": 500,
            "timestamp": "2026-08-06T10:00:00Z",
            "retrieved_at": "2026-08-06T10:01:00Z", "cache_status": "hit",
        },
    )
    record = records[0]

    assert record["company_id"] == "microsoft"
    assert record["ticker"] == "MSFT"
    assert record["status"] == "success"
    assert record["as_of"] == "2026-08-06T10:00:00Z"
    assert record["retrieved_at"] == "2026-08-06T10:01:00Z"
    assert record["cache_status"] == "hit"


def test_error_and_missing_results_are_explicit_non_success_evidence():
    """Retain failures for observability without treating them as successful evidence."""
    namespace = _evidence_namespace()
    company = _company(namespace)

    failed = namespace["to_evidence_record"](
        "run-failed", company, company["profile_id"], "stock_price",
        {"ticker": "MSFT", "status": "error", "error": "provider unavailable"},
    )[0]
    missing = namespace["to_evidence_record"](
        "run-missing", company, company["profile_id"], "technology_rag",
        {"status": "missing", "data": None},
    )[0]

    assert failed["status"] == "failed"
    assert failed["error"] == "provider unavailable"
    assert missing["status"] == "missing"


def test_company_and_profile_mismatches_are_rejected():
    """Prevent evidence from crossing company or profile boundaries."""
    namespace = _evidence_namespace()
    company = _company(namespace)
    convert = namespace["to_evidence_record"]

    with pytest.raises(ValueError, match="ticker mismatch"):
        convert(
            "run-wrong", company, company["profile_id"], "stock_price",
            {"ticker": "PFE", "status": "success"},
        )
    with pytest.raises(ValueError, match="profile mismatch"):
        convert(
            "run-wrong", company, "healthcare.biopharma.v1", "stock_price",
            {"ticker": "MSFT", "status": "success"},
        )


def test_evidence_ids_are_stable_and_distinct_for_distinct_items():
    """Generate stable run-scoped IDs without collisions across source items."""
    namespace = _evidence_namespace()
    company = _company(namespace)
    result = [
        {"title": "One", "url": "https://example.test/1", "status": "success"},
        {"title": "Two", "url": "https://example.test/2", "status": "success"},
    ]
    first = namespace["to_evidence_record"](
        "run-stable", company, company["profile_id"], "financial_news", result,
    )
    second = namespace["to_evidence_record"](
        "run-stable", company, company["profile_id"], "financial_news", result,
    )

    assert [item["evidence_id"] for item in first] == [item["evidence_id"] for item in second]
    assert len({item["evidence_id"] for item in first}) == 2


def test_adapter_uses_injected_result_without_calling_live_tool():
    """Support deterministic adapter testing through explicit source-result injection."""
    namespace = _evidence_namespace()
    company = _company(namespace)
    plan = {
        "query_type": "fact", "company_mentions": ["Microsoft"],
        "requested_dimensions": ["current_price"], "risk_profile": "balanced",
        "scoring_requested": False, "freshness_required": True, "time_horizon": None,
    }
    task = namespace["build_company_tasks"](plan, [company], "run-adapter")[0]
    record = namespace["fetch_price_evidence"](
        task, {"ticker": "MSFT", "status": "success", "current_price": 500}
    )[0]

    assert record["source_name"] == "get_stock_price"
    assert record["status"] == "success"

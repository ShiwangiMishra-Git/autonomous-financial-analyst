"""Deterministic tests for the F12 fan-in normalization boundary."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path

from scripts.implement_multiindustry_f12_normalization import F12_NORMALIZATION_CODE


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


@lru_cache(maxsize=1)
def _namespace():
    """Execute the prerequisite notebook contracts and F12 source in one namespace."""
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
        exec(F12_NORMALIZATION_CODE, namespace)
    return namespace


def _plan(*mentions, freshness=False):
    """Create a minimal current-price plan for normalization fixtures."""
    return {
        # Keep the fixture focused on one required dimension. F05 intentionally expands a
        # compare plan to the complete shared financial contract, which is tested elsewhere.
        "query_type": "analyze",
        "company_mentions": list(mentions),
        "requested_dimensions": ["current_price"],
        "risk_profile": "balanced",
        "scoring_requested": False,
        "freshness_required": freshness,
        "time_horizon": None,
    }


def _tasks(namespace, run_id="run-f12", mentions=("Microsoft", "Pfizer"), freshness=False):
    """Build validated tasks in the user's requested company order."""
    plan = _plan(*mentions, freshness=freshness)
    companies = namespace["resolve_company_mentions"](plan["company_mentions"])
    return namespace["build_company_tasks"](plan, companies, run_id)


def _evidence(namespace, task, *, ticker=None, run_id=None, freshness="fresh"):
    """Build one canonical current-price record, optionally with a boundary mismatch."""
    record = namespace["to_evidence_record"](
        run_id or task["run_id"],
        task["company"],
        task["company"]["profile_id"],
        "stock_price",
        {
            "ticker": task["company"]["ticker"],
            "status": "success",
            "data": {"price": 100},
            "freshness_status": freshness,
        },
        "get_stock_price",
    )[0]
    if ticker is not None:
        record["ticker"] = ticker
    return record


def _result(namespace, task, *, evidence=None, status="success", errors=None, missing=None):
    """Create a worker-result fixture bound to one expected task."""
    records = evidence if evidence is not None else [_evidence(namespace, task)]
    return {
        "run_id": task["run_id"],
        "company": deepcopy(task["company"]),
        "profile_id": task["company"]["profile_id"],
        "financial_evidence": {"untrusted": "worker summary"},
        "industry_signals": {},
        "evidence": records,
        "missing_dimensions": list(missing or []),
        "errors": list(errors or []),
        "status": status,
    }


def test_completion_order_cannot_change_canonical_task_order():
    """Restore task order even when the reducer map reflects reverse completion order."""
    namespace = _namespace()
    tasks = _tasks(namespace)
    microsoft, pfizer = tasks
    results = {
        "PFE": _result(namespace, pfizer),
        "MSFT": _result(namespace, microsoft),
    }

    normalized = namespace["normalize_all_results"](tasks, results, "run-f12")

    assert normalized["ordered_tickers"] == ["MSFT", "PFE"]
    assert [item["company"]["ticker"] for item in normalized["ordered_results"]] == [
        "MSFT", "PFE",
    ]
    assert normalized["status"] == "complete"


def test_missing_branch_becomes_failed_placeholder_without_erasing_peer():
    """Preserve exactly one result per task when a worker update never arrives."""
    namespace = _namespace()
    tasks = _tasks(namespace)
    microsoft, _ = tasks

    normalized = namespace["normalize_all_results"](
        tasks, {"MSFT": _result(namespace, microsoft)}, "run-f12",
    )

    assert normalized["ordered_tickers"] == ["MSFT", "PFE"]
    assert normalized["results_by_ticker"]["MSFT"]["status"] == "success"
    assert normalized["results_by_ticker"]["PFE"]["status"] == "failed"
    assert normalized["failed_tickers"] == ["PFE"]
    assert normalized["status"] == "partial"
    assert normalized["ready"] is True


def test_explicit_failed_branch_remains_contained():
    """Retain branch failure semantics while allowing a successful sibling to proceed."""
    namespace = _namespace()
    tasks = _tasks(namespace)
    microsoft, pfizer = tasks
    failed = _result(
        namespace, pfizer, evidence=[], status="failed", errors=["provider unavailable"],
    )

    normalized = namespace["normalize_all_results"](
        tasks, {"MSFT": _result(namespace, microsoft), "PFE": failed}, "run-f12",
    )

    assert normalized["successful_tickers"] == ["MSFT"]
    assert normalized["failed_tickers"] == ["PFE"]
    assert any("PFE: provider unavailable" in error for error in normalized["errors"])


def test_result_identity_mismatch_fails_closed_to_expected_identity():
    """Replace a cross-company result rather than propagating its data."""
    namespace = _namespace()
    microsoft, pfizer = _tasks(namespace)
    contaminated = _result(namespace, microsoft)
    contaminated["company"] = deepcopy(pfizer["company"])

    normalized = namespace["normalize_company_result"](
        microsoft, contaminated, "run-f12",
    )

    assert normalized["status"] == "failed"
    assert normalized["company"]["ticker"] == "MSFT"
    assert normalized["evidence"] == []
    assert any("company boundary" in error for error in normalized["errors"])


def test_evidence_identity_mismatch_fails_entire_branch_boundary():
    """Reject all branch evidence when one record crosses the ticker boundary."""
    namespace = _namespace()
    microsoft = _tasks(namespace, mentions=("Microsoft",))[0]
    good = _evidence(namespace, microsoft)
    crossed = _evidence(namespace, microsoft, ticker="PFE")

    normalized = namespace["normalize_company_result"](
        microsoft, _result(namespace, microsoft, evidence=[good, crossed]), "run-f12",
    )

    assert normalized["status"] == "failed"
    assert normalized["evidence"] == []
    assert any("wrong ticker" in error for error in normalized["errors"])


def test_duplicate_evidence_is_deduplicated_and_marks_result_partial():
    """Keep the first canonical evidence ID and surface duplicate provenance."""
    namespace = _namespace()
    task = _tasks(namespace, mentions=("Microsoft",))[0]
    evidence = _evidence(namespace, task)

    normalized = namespace["normalize_company_result"](
        task,
        _result(namespace, task, evidence=[evidence, deepcopy(evidence)]),
        "run-f12",
    )

    assert normalized["status"] == "partial"
    assert len(normalized["evidence"]) == 1
    assert any("Duplicate evidence_id" in error for error in normalized["errors"])


def test_ungrounded_signal_is_downgraded_and_cannot_remain_successful():
    """Remove invented signal references while retaining independently valid evidence."""
    namespace = _namespace()
    task = _tasks(namespace, mentions=("Microsoft",))[0]
    result = _result(namespace, task)
    result["industry_signals"] = {
        "strategic_commitment": {
            "level": "full", "score": 1.0, "reason": "invented",
            "evidence_ids": ["ev-not-current-run"],
        }
    }

    normalized = namespace["normalize_company_result"](task, result, "run-f12")

    signal = normalized["industry_signals"]["strategic_commitment"]
    assert normalized["status"] == "partial"
    assert signal["level"] == "missing"
    assert signal["score"] is None
    assert signal["evidence_ids"] == []


def test_freshness_required_rejects_unknown_evidence_for_coverage():
    """Do not count successful-but-unknown evidence when the query requires freshness."""
    namespace = _namespace()
    task = _tasks(namespace, mentions=("Microsoft",), freshness=True)[0]
    unknown = _evidence(namespace, task, freshness="unknown")

    normalized = namespace["normalize_company_result"](
        task, _result(namespace, task, evidence=[unknown]), "run-f12",
    )

    assert normalized["status"] == "failed"
    assert "current_price" in normalized["missing_dimensions"]
    assert any("Fresh evidence is required" in error for error in normalized["errors"])


def test_unexpected_result_is_excluded_and_reported():
    """Prevent reducer pollution from adding a company absent from expected tasks."""
    namespace = _namespace()
    microsoft, pfizer = _tasks(namespace)
    normalized = namespace["normalize_all_results"](
        [microsoft],
        {
            "MSFT": _result(namespace, microsoft),
            "PFE": _result(namespace, pfizer),
        },
        "run-f12",
    )

    assert normalized["ordered_tickers"] == ["MSFT"]
    assert "PFE" not in normalized["results_by_ticker"]
    assert normalized["ready"] is False
    assert normalized["blocking_errors"]
    assert any("Unexpected company results" in error for error in normalized["errors"])


def test_result_sequence_detects_duplicate_company_outputs():
    """Detect duplicate branch outputs before constructing the canonical ticker map."""
    namespace = _namespace()
    task = _tasks(namespace, mentions=("Microsoft",))[0]
    result = _result(namespace, task)

    normalized = namespace["normalize_all_results"](
        [task], [result, deepcopy(result)], "run-f12",
    )

    assert normalized["ordered_tickers"] == ["MSFT"]
    assert normalized["ready"] is False
    assert normalized["blocking_errors"]
    assert any("Duplicate company result" in error for error in normalized["errors"])


def test_financial_summary_is_rebuilt_only_from_canonical_evidence():
    """Discard worker-provided summary fields not supported by normalized records."""
    namespace = _namespace()
    task = _tasks(namespace, mentions=("Microsoft",))[0]

    normalized = namespace["normalize_company_result"](
        task, _result(namespace, task), "run-f12",
    )

    assert "untrusted" not in normalized["financial_evidence"]
    assert normalized["financial_evidence"] == {"stock_price": {"price": 100}}


def test_malformed_evidence_fields_are_contained_without_late_key_errors():
    """Normalize missing payload metadata into an observable partial branch result."""
    namespace = _namespace()
    task = _tasks(namespace, mentions=("Microsoft",))[0]
    record = _evidence(namespace, task)
    del record["evidence_type"]
    del record["value"]
    del record["source_name"]

    normalized = namespace["normalize_company_result"](
        task, _result(namespace, task, evidence=[record]), "run-f12",
    )

    assert normalized["status"] == "partial"
    assert normalized["evidence"][0]["evidence_type"] == "unknown"
    assert normalized["evidence"][0]["value"] is None
    assert any("has no evidence_type" in error for error in normalized["errors"])

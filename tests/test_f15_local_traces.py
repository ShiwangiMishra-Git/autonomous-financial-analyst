"""Focused deterministic tests for F15 local redacted research traces."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.implement_multiindustry_f15_traces import F15_TRACES_CODE


@pytest.fixture()
def trace_api():
    """Execute the notebook-injectable trace implementation in an isolated namespace."""
    namespace: dict[str, object] = {}
    exec(F15_TRACES_CODE, namespace)
    return namespace


def _normalized_results(run_id: str = "run-trace") -> dict:
    """Return two compact normalized F12 results with private document values."""
    return {
        "MSFT": {
            "run_id": run_id,
            "profile_id": "technology.ai.v1",
            "company": {
                "company_id": "microsoft",
                "ticker": "MSFT",
                "company_name": "Microsoft Corporation",
            },
            "evidence": [{
                "evidence_id": "EV-MSFT-1",
                "run_id": run_id,
                "company_id": "microsoft",
                "ticker": "MSFT",
                "profile_id": "technology.ai.v1",
                "evidence_type": "private_rag",
                "source_name": "private filing",
                "source_uri": "https://example.test/filing.pdf?token=secret#page=4",
                "document_name": "filing.pdf",
                "page": 4,
                "retrieved_at": "2026-08-06T10:00:00+00:00",
                "status": "success",
                "value": "FULL PRIVATE DOCUMENT CONTENT MUST NOT BE WRITTEN",
                "page_content": "A second raw content field",
                "source_metadata": {"OPENAI_API_KEY": "sk-private"},
            }],
        },
        "PFE": {
            "run_id": run_id,
            "profile_id": "healthcare.biopharma.v1",
            "company": {
                "company_id": "pfizer",
                "ticker": "PFE",
                "company_name": "Pfizer Inc.",
            },
            "evidence": [{
                "evidence_id": "EV-PFE-1",
                "run_id": run_id,
                "company_id": "pfizer",
                "ticker": "PFE",
                "profile_id": "healthcare.biopharma.v1",
                "evidence_type": "financial_metrics",
                "source_name": "market adapter",
                "source_uri": None,
                "retrieved_at": "2026-08-06T10:01:00+00:00",
                "status": "success",
                "value": {"market_cap": 1.0},
            }],
        },
    }


def _new_trace(api, *, run_id: str = "run-trace", mode: str = "cross_profile") -> dict:
    """Build a deterministic in-progress trace through the public API."""
    return api["create_research_trace"](
        run_id=run_id,
        query="Compare Microsoft and Pfizer.",
        comparison_mode=mode,
        normalized_results=_normalized_results(run_id),
        f13_scores={},
        f14_synthesis={
            "mode": mode,
            "answer": "Grounded answer [EV-MSFT-1] [EV-PFE-1].",
            "evidence_ids": ["EV-MSFT-1", "EV-PFE-1"],
            "scores_used": {},
            "limitations": ["No universal numeric score was applied."],
        },
        timestamp="2026-08-06T10:02:00+00:00",
    )


def test_successful_trace_contains_bounded_workflow_artifacts(trace_api, tmp_path):
    """Persist query, routing, identities, provenance, scores, synthesis, and validation."""
    trace = _new_trace(trace_api)
    trace = trace_api["record_validation_attempt"](
        trace,
        {"valid": True, "validated_evidence_ids": ["EV-MSFT-1", "EV-PFE-1"], "errors": []},
        attempt_number=1,
        timestamp="2026-08-06T10:03:00+00:00",
    )
    trace = trace_api["finalize_research_trace"](
        trace, final_status="success", timestamp="2026-08-06T10:04:00+00:00",
    )
    outcome = trace_api["write_research_trace"](
        trace, trace_dir=tmp_path / ".research_runs", retention_limit=10,
    )

    stored = json.loads(Path(outcome["path"]).read_text(encoding="utf-8"))
    assert stored["query"] == "Compare Microsoft and Pfizer."
    assert stored["comparison_mode"] == "cross_profile"
    assert [company["ticker"] for company in stored["companies"]] == ["MSFT", "PFE"]
    assert stored["profiles"] == ["technology.ai.v1", "healthcare.biopharma.v1"]
    assert [item["evidence_id"] for item in stored["evidence_provenance"]] == [
        "EV-MSFT-1", "EV-PFE-1",
    ]
    assert stored["f13_scores"] == {}
    assert stored["f14_synthesis"]["mode"] == "cross_profile"
    assert stored["validation_attempts"][0]["result"]["valid"] is True
    assert stored["final_status"] == "success"
    assert stored["completed_at"] == "2026-08-06T10:04:00+00:00"


@pytest.mark.parametrize(
    "status,error",
    [
        ("failed", "Validation retries exhausted"),
        ("interrupted", "Notebook kernel interrupted"),
    ],
)
def test_failed_and_interrupted_records_are_explicit(trace_api, tmp_path, status, error):
    """Retain deterministic terminal status and a bounded diagnostic for non-success runs."""
    trace = _new_trace(trace_api, run_id=f"run-{status}")
    trace = trace_api["record_validation_attempt"](
        trace,
        {"valid": False, "validated_evidence_ids": [], "errors": [error]},
        attempt_number=1,
        timestamp="2026-08-06T10:03:00Z",
    )
    trace = trace_api["finalize_research_trace"](
        trace,
        final_status=status,
        terminal_error=error,
        timestamp="2026-08-06T10:04:00Z",
    )
    outcome = trace_api["write_research_trace"](trace, trace_dir=tmp_path)
    stored = json.loads(Path(outcome["path"]).read_text(encoding="utf-8"))

    assert stored["final_status"] == status
    assert stored["terminal_error"] == error
    assert stored["validation_attempts"][0]["result"]["valid"] is False


def test_sensitive_fields_and_private_document_content_are_not_persisted(trace_api, tmp_path):
    """Redact secret-like fields and retain evidence provenance without raw RAG content."""
    trace = _new_trace(trace_api)
    trace["f13_scores"]["debug"] = {
        "API_KEY": "super-secret-api-key",
        "password": "do-not-write",
        "safe_field": "kept",
    }
    trace["f14_synthesis"]["client_secret"] = "synthesis-secret"
    trace = trace_api["record_validation_attempt"](
        trace,
        {"valid": False, "errors": [], "authorization": "Bearer private"},
        attempt_number=1,
        timestamp="2026-08-06T10:03:00Z",
    )
    outcome = trace_api["write_research_trace"](trace, trace_dir=tmp_path)
    raw = Path(outcome["path"]).read_text(encoding="utf-8")
    stored = json.loads(raw)

    assert "super-secret-api-key" not in raw
    assert "do-not-write" not in raw
    assert "synthesis-secret" not in raw
    assert "Bearer private" not in raw
    assert "FULL PRIVATE DOCUMENT CONTENT" not in raw
    assert "source_metadata" not in raw
    assert stored["f13_scores"]["debug"]["API_KEY"] == "[REDACTED]"
    assert stored["f13_scores"]["debug"]["safe_field"] == "kept"
    assert stored["evidence_provenance"][0]["source_uri"] == "https://example.test/filing.pdf"


def test_bounded_retention_keeps_current_and_newest_prior_record(trace_api, tmp_path):
    """Remove the oldest local record after an atomic publish exceeds retention."""
    trace_dir = tmp_path / ".research_runs"
    written: list[Path] = []
    for index, run_id in enumerate(("run-old", "run-middle", "run-current"), start=1):
        trace = _new_trace(trace_api, run_id=run_id)
        outcome = trace_api["write_research_trace"](
            trace, trace_dir=trace_dir, retention_limit=10,
        )
        path = Path(outcome["path"])
        os.utime(path, ns=(index, index))
        written.append(path)

    current_trace = _new_trace(trace_api, run_id="run-current")
    outcome = trace_api["write_research_trace"](
        current_trace, trace_dir=trace_dir, retention_limit=2,
    )

    assert Path(outcome["path"]).exists()
    assert not written[0].exists()
    assert written[1].exists()
    assert written[2].exists()
    assert outcome["removed_paths"] == [str(written[0])]


def test_failed_atomic_replace_preserves_previous_file_and_cleans_temp(trace_api, tmp_path):
    """Leave the prior final JSON intact and remove temporary files when replace fails."""
    trace = _new_trace(trace_api)
    first = trace_api["write_research_trace"](trace, trace_dir=tmp_path)
    final_path = Path(first["path"])
    previous = final_path.read_text(encoding="utf-8")
    changed = trace_api["finalize_research_trace"](
        trace,
        final_status="failed",
        terminal_error="replacement should fail",
        timestamp="2026-08-06T11:00:00Z",
    )

    def fail_replace(source, destination):
        """Raise an injected write failure before publishing the temporary record."""
        raise OSError(f"injected replace failure: {source} -> {destination}")

    with pytest.raises(OSError, match="injected replace failure"):
        trace_api["write_research_trace"](
            changed, trace_dir=tmp_path, replace_func=fail_replace,
        )

    assert final_path.read_text(encoding="utf-8") == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_rejects_unsafe_run_id_and_duplicate_validation_attempt(trace_api):
    """Prevent path traversal and ambiguous duplicate attempt numbering."""
    with pytest.raises(ValueError, match="unsafe filename"):
        _new_trace(trace_api, run_id="../escape")

    trace = _new_trace(trace_api)
    trace = trace_api["record_validation_attempt"](
        trace, {"valid": True}, attempt_number=1, timestamp="2026-08-06T10:03:00Z",
    )
    with pytest.raises(ValueError, match="already exists"):
        trace_api["record_validation_attempt"](
            trace, {"valid": True}, attempt_number=1, timestamp="2026-08-06T10:04:00Z",
        )

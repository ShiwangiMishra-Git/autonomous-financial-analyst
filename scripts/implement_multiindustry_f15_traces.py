"""Provide notebook-injectable F15 local research-trace support.

This module deliberately does not edit the working notebook.  The integration agent can review
and inject :data:`F15_TRACES_CODE` into a canonical F15 cell.  The injected implementation only
records already-produced workflow artifacts; it performs no research, synthesis, or scoring.
"""

from __future__ import annotations


F15_TRACES_INTRO = """### F15 local research traces

Each request may write one redacted JSON record to `.research_runs/<run_id>.json`. The record
captures bounded provenance and workflow decisions without copying private-document bodies or
credentials. Writes use a temporary file in the same directory followed by `os.replace`, and
retention removes the oldest completed files after a successful publish.
"""


F15_TRACES_CODE = r'''from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, TypedDict
from urllib.parse import urlsplit, urlunsplit


class ResearchTrace(TypedDict):
    """Serializable local audit record for one bounded research workflow run.

    Attributes:
        schema_version: Trace contract version.
        run_id: Unique filename-safe run identity.
        query: Original user question after redaction.
        comparison_mode: F12 routing mode.
        companies: Canonical company summaries without private evidence bodies.
        profiles: Versioned profiles used by the run.
        evidence_provenance: Redacted evidence identity/source metadata only.
        f13_scores: Optional authoritative deterministic scores.
        f14_synthesis: Latest redacted structured draft.
        validation_attempts: Ordered F15 verdict summaries.
        started_at: UTC trace creation timestamp.
        updated_at: UTC last-write timestamp.
        completed_at: UTC terminal timestamp or ``None`` while in progress.
        final_status: In-progress, success, failed, or interrupted.
        terminal_error: Redacted terminal failure message when present.
    """

    schema_version: str
    run_id: str
    query: str
    comparison_mode: str
    companies: list[dict[str, Any]]
    profiles: list[str]
    evidence_provenance: list[dict[str, Any]]
    f13_scores: dict[str, Any]
    f14_synthesis: dict[str, Any]
    validation_attempts: list[dict[str, Any]]
    started_at: str
    updated_at: str
    completed_at: str | None
    final_status: str
    terminal_error: str | None


class TraceWriteResult(TypedDict):
    """Outcome of one atomic trace publication and retention pass.

    Attributes:
        path: Published trace path.
        removed_paths: Older completed traces removed by bounded retention.
        final_status: Status contained in the published trace.
    """

    path: str
    removed_paths: list[str]
    final_status: str


F15_TRACE_SCHEMA_VERSION = "f15.research_trace.v1"
F15_TRACE_FINAL_STATUSES = frozenset({"success", "failed", "interrupted"})
F15_TRACE_ALL_STATUSES = F15_TRACE_FINAL_STATUSES | {"in_progress"}
F15_TRACE_MODES = frozenset({"single", "same_profile", "cross_profile"})
F15_TRACE_DEFAULT_RETENTION = 50
F15_TRACE_REDACTED = "[REDACTED]"
F15_TRACE_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
)
F15_TRACE_PRIVATE_CONTENT_KEYS = frozenset({
    "body",
    "chunk",
    "chunk_text",
    "content",
    "document_content",
    "full_text",
    "page_content",
    "raw_content",
    "text",
    "value",
})
F15_TRACE_EVIDENCE_FIELDS = (
    "evidence_id",
    "run_id",
    "company_id",
    "ticker",
    "profile_id",
    "evidence_type",
    "source_name",
    "source_uri",
    "document_name",
    "page",
    "as_of",
    "retrieved_at",
    "freshness_status",
    "cache_status",
    "status",
    "error",
)
F15_TRACE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _f15_trace_text(value: Any, field_name: str) -> str:
    """Return a required stripped string or raise a deterministic validation error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _f15_trace_timestamp(value: str | datetime | None = None) -> str:
    """Normalize an injected or current timestamp to an ISO-8601 UTC string."""
    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("timestamp must be valid ISO-8601") from exc
    else:
        raise ValueError("timestamp must be an ISO-8601 string, datetime, or None")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def _f15_trace_is_sensitive_key(key: Any) -> bool:
    """Return whether a mapping key names a credential-like value."""
    normalized = str(key).strip().casefold().replace("-", "_")
    return any(part in normalized for part in F15_TRACE_SENSITIVE_KEY_PARTS)


def _f15_trace_sanitize_uri(value: Any) -> Any:
    """Remove query parameters and fragments that commonly carry signed credentials."""
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme:
        return value
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def redact_trace_value(value: Any, *, parent_key: str = "") -> Any:
    """Return a JSON-safe deep copy with credential fields and raw content redacted.

    Content-key redaction applies to arbitrary supplemental mappings. Evidence records are also
    projected onto a provenance allowlist before this function is called, ensuring that full
    private-document values and source metadata never enter a trace.
    """
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized_key = key.strip().casefold().replace("-", "_")
            if _f15_trace_is_sensitive_key(key):
                cleaned[key] = F15_TRACE_REDACTED
            elif normalized_key in F15_TRACE_PRIVATE_CONTENT_KEYS:
                cleaned[key] = F15_TRACE_REDACTED
            elif normalized_key in {"source_uri", "url", "uri"}:
                cleaned[key] = _f15_trace_sanitize_uri(raw_value)
            else:
                cleaned[key] = redact_trace_value(raw_value, parent_key=key)
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_trace_value(item, parent_key=parent_key) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _f15_trace_timestamp(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _f15_trace_company(ticker: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract non-secret canonical identity fields from one normalized company result."""
    company = result.get("company")
    if not isinstance(company, Mapping):
        raise ValueError(f"{ticker} is missing canonical company identity")
    result_ticker = _f15_trace_text(company.get("ticker"), f"{ticker}.company.ticker")
    if result_ticker != ticker:
        raise ValueError(f"Result key {ticker} does not match company ticker {result_ticker}")
    return {
        "company_id": _f15_trace_text(company.get("company_id"), f"{ticker}.company.company_id"),
        "ticker": result_ticker,
        "company_name": _f15_trace_text(
            company.get("company_name"), f"{ticker}.company.company_name",
        ),
    }


def _f15_trace_evidence_record(
    record: Mapping[str, Any], expected_run_id: str, ticker: str, profile_id: str,
) -> dict[str, Any]:
    """Project one evidence item onto safe provenance metadata without its raw value."""
    evidence_id = _f15_trace_text(record.get("evidence_id"), f"{ticker}.evidence_id")
    if record.get("run_id") != expected_run_id:
        raise ValueError(f"Evidence {evidence_id} does not belong to run {expected_run_id}")
    if record.get("ticker") != ticker:
        raise ValueError(f"Evidence {evidence_id} does not belong to company {ticker}")
    if record.get("profile_id") != profile_id:
        raise ValueError(f"Evidence {evidence_id} does not belong to profile {profile_id}")
    projected = {
        field: record.get(field)
        for field in F15_TRACE_EVIDENCE_FIELDS
        if field in record
    }
    projected["source_uri"] = _f15_trace_sanitize_uri(projected.get("source_uri"))
    return redact_trace_value(projected)


def create_research_trace(
    *,
    run_id: str,
    query: str,
    comparison_mode: str,
    normalized_results: Mapping[str, Mapping[str, Any]],
    f13_scores: Mapping[str, Any] | None,
    f14_synthesis: Mapping[str, Any] | None,
    timestamp: str | datetime | None = None,
) -> ResearchTrace:
    """Create an in-memory redacted trace from already-computed F12/F13/F14 artifacts."""
    safe_run_id = _f15_trace_text(run_id, "run_id")
    if not F15_TRACE_RUN_ID_PATTERN.fullmatch(safe_run_id):
        raise ValueError("run_id contains unsafe filename characters")
    safe_query = _f15_trace_text(query, "query")
    if comparison_mode not in F15_TRACE_MODES:
        raise ValueError(f"Unsupported comparison_mode: {comparison_mode!r}")
    if not isinstance(normalized_results, Mapping) or not normalized_results:
        raise ValueError("normalized_results must be a non-empty ticker mapping")
    if f13_scores is not None and not isinstance(f13_scores, Mapping):
        raise ValueError("f13_scores must be a mapping or None")
    if f14_synthesis is not None and not isinstance(f14_synthesis, Mapping):
        raise ValueError("f14_synthesis must be a mapping or None")

    companies: list[dict[str, Any]] = []
    profiles: list[str] = []
    evidence_provenance: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    for ticker, raw_result in normalized_results.items():
        safe_ticker = _f15_trace_text(ticker, "result ticker")
        if not isinstance(raw_result, Mapping):
            raise ValueError(f"Result {safe_ticker} must be a mapping")
        if raw_result.get("run_id") != safe_run_id:
            raise ValueError(f"Result {safe_ticker} does not belong to run {safe_run_id}")
        profile_id = _f15_trace_text(raw_result.get("profile_id"), f"{safe_ticker}.profile_id")
        companies.append(_f15_trace_company(safe_ticker, raw_result))
        if profile_id not in profiles:
            profiles.append(profile_id)
        records = raw_result.get("evidence")
        if not isinstance(records, list):
            raise ValueError(f"{safe_ticker}.evidence must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"{safe_ticker} contains malformed evidence")
            projected = _f15_trace_evidence_record(
                record, safe_run_id, safe_ticker, profile_id,
            )
            evidence_id = projected["evidence_id"]
            if evidence_id in seen_evidence_ids:
                raise ValueError(f"Duplicate evidence_id across trace: {evidence_id}")
            seen_evidence_ids.add(evidence_id)
            evidence_provenance.append(projected)

    started_at = _f15_trace_timestamp(timestamp)
    return {
        "schema_version": F15_TRACE_SCHEMA_VERSION,
        "run_id": safe_run_id,
        "query": safe_query,
        "comparison_mode": comparison_mode,
        "companies": companies,
        "profiles": profiles,
        "evidence_provenance": evidence_provenance,
        "f13_scores": redact_trace_value(deepcopy(dict(f13_scores or {}))),
        "f14_synthesis": redact_trace_value(deepcopy(dict(f14_synthesis or {}))),
        "validation_attempts": [],
        "started_at": started_at,
        "updated_at": started_at,
        "completed_at": None,
        "final_status": "in_progress",
        "terminal_error": None,
    }


def record_validation_attempt(
    trace: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    *,
    attempt_number: int,
    timestamp: str | datetime | None = None,
) -> ResearchTrace:
    """Return a copied trace with one bounded deterministic validation attempt appended."""
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int) or attempt_number < 1:
        raise ValueError("attempt_number must be a positive integer")
    if not isinstance(trace, Mapping) or not isinstance(validation_result, Mapping):
        raise ValueError("trace and validation_result must be mappings")
    updated = deepcopy(dict(trace))
    attempts = updated.get("validation_attempts")
    if not isinstance(attempts, list):
        raise ValueError("trace.validation_attempts must be a list")
    if any(attempt.get("attempt_number") == attempt_number for attempt in attempts):
        raise ValueError(f"Validation attempt {attempt_number} already exists")
    moment = _f15_trace_timestamp(timestamp)
    attempts.append({
        "attempt_number": attempt_number,
        "timestamp": moment,
        "result": redact_trace_value(deepcopy(dict(validation_result))),
    })
    updated["updated_at"] = moment
    return updated


def finalize_research_trace(
    trace: Mapping[str, Any],
    *,
    final_status: str,
    timestamp: str | datetime | None = None,
    terminal_error: str | None = None,
) -> ResearchTrace:
    """Return a copied trace finalized as success, failure, or interruption."""
    if final_status not in F15_TRACE_FINAL_STATUSES:
        raise ValueError(f"Unsupported final_status: {final_status!r}")
    if not isinstance(trace, Mapping):
        raise ValueError("trace must be a mapping")
    if terminal_error is not None and not isinstance(terminal_error, str):
        raise ValueError("terminal_error must be a string or None")
    updated = deepcopy(dict(trace))
    moment = _f15_trace_timestamp(timestamp)
    updated["updated_at"] = moment
    updated["completed_at"] = moment
    updated["final_status"] = final_status
    updated["terminal_error"] = (
        redact_trace_value({"error": terminal_error})["error"]
        if terminal_error is not None
        else None
    )
    return updated


def _f15_trace_validate_for_write(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum persisted trace contract and return a sanitized copy."""
    if not isinstance(trace, Mapping):
        raise ValueError("trace must be a mapping")
    safe = redact_trace_value(deepcopy(dict(trace)))
    run_id = _f15_trace_text(safe.get("run_id"), "trace.run_id")
    if not F15_TRACE_RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("trace.run_id contains unsafe filename characters")
    status = safe.get("final_status")
    if status not in F15_TRACE_ALL_STATUSES:
        raise ValueError(f"Unsupported trace final_status: {status!r}")
    if safe.get("schema_version") != F15_TRACE_SCHEMA_VERSION:
        raise ValueError("Unsupported trace schema_version")
    for field in ("companies", "profiles", "evidence_provenance", "validation_attempts"):
        if not isinstance(safe.get(field), list):
            raise ValueError(f"trace.{field} must be a list")
    json.dumps(safe, sort_keys=True, allow_nan=False)
    return safe


def _f15_trace_apply_retention(
    trace_dir: Path, current_path: Path, retention_limit: int,
) -> list[str]:
    """Remove oldest trace JSON files while always retaining the just-written record."""
    candidates = [path for path in trace_dir.glob("*.json") if path.is_file()]
    if len(candidates) <= retention_limit:
        return []
    others = sorted(
        (path for path in candidates if path != current_path),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    keep = {current_path, *others[: max(retention_limit - 1, 0)]}
    removed: list[str] = []
    for path in sorted(candidates, key=lambda item: item.name):
        if path not in keep:
            path.unlink()
            removed.append(str(path))
    return removed


def write_research_trace(
    trace: Mapping[str, Any],
    *,
    trace_dir: str | Path = ".research_runs",
    retention_limit: int = F15_TRACE_DEFAULT_RETENTION,
    replace_func: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace,
) -> TraceWriteResult:
    """Atomically publish one redacted JSON trace and enforce bounded local retention.

    ``replace_func`` is injectable solely for deterministic failure testing. A failed replacement
    leaves any previous final record untouched and removes the temporary file.
    """
    if isinstance(retention_limit, bool) or not isinstance(retention_limit, int):
        raise ValueError("retention_limit must be an integer")
    if retention_limit < 1:
        raise ValueError("retention_limit must be at least 1")
    safe = _f15_trace_validate_for_write(trace)
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / f"{safe['run_id']}.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{safe['run_id']}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(safe, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_func(temporary_path, final_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    removed = _f15_trace_apply_retention(directory, final_path, retention_limit)
    return {
        "path": str(final_path),
        "removed_paths": removed,
        "final_status": safe["final_status"],
    }
'''

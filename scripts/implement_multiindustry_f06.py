"""Idempotently add F06 canonical evidence adapters to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f05_smoke"


F06_INTRO = """## Section 3.6: Canonical Evidence Adapters

Source tools return different shapes, so downstream agents must not reason directly over raw
tool dictionaries. F06 converts every successful, missing, or failed source result into the same
run- and company-bound `EvidenceRecord` contract with stable IDs, provenance, freshness, cache
status, and explicit errors.

Adapters accept an injected `tool_result` for deterministic tests. When omitted, they call the
existing notebook capability. Failed results remain observable but cannot count as successful
evidence in later gates.
"""


F06_CODE = r'''from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5


EVIDENCE_SOURCE_BY_TYPE = {
    "stock_price": "get_stock_price",
    "stock_history": "get_stock_history",
    "financial_metrics": "get_financial_metrics",
    "financial_news": "search_financial_news",
    "sentiment": "analyze_sentiment",
    "technology_rag": "query_technology_rag",
    "biopharma_rag": "query_biopharma_rag",
}


def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp for evidence retrieval metadata."""
    return datetime.now(timezone.utc).isoformat()


def _canonical_payload_hash(value: Any) -> str:
    """Hash an arbitrary JSON-like evidence value deterministically.

    Args:
        value: Tool result or evidence payload.

    Returns:
        SHA-256 digest of a canonical JSON representation.
    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_status(item: Any) -> tuple[str, str | None]:
    """Normalize heterogeneous tool status fields into the evidence status contract.

    Args:
        item: One raw source item.

    Returns:
        Pair of ``success``, ``missing``, or ``failed`` and an optional error message.
    """
    if isinstance(item, str):
        lowered = item.strip().casefold()
        if lowered.startswith("error"):
            return "failed", item
        if "i don't know" in lowered or "not available" in lowered:
            return "missing", None
        return "success", None
    if not isinstance(item, dict):
        return "success", None

    raw_status = str(item.get("status", "success")).casefold()
    error = item.get("error")
    if raw_status in {"error", "failed", "failure"} or error:
        return "failed", str(error or raw_status)
    if raw_status in {"missing", "not_found", "unavailable"}:
        return "missing", None
    return "success", None


def _validate_evidence_identity(company: ResolvedCompany, item: Any) -> None:
    """Reject a raw result that identifies a different company ticker.

    Args:
        company: Canonical company assigned to the current task.
        item: Raw result that may contain a ticker field.

    Raises:
        ValueError: If the company is unresolved or a result ticker does not match.
    """
    if company["resolution_status"] != "resolved":
        raise ValueError("Evidence requires a successfully resolved company")
    if isinstance(item, dict) and item.get("ticker"):
        actual = str(item["ticker"]).upper()
        expected = company["ticker"].upper()
        if actual != expected:
            raise ValueError(f"Evidence ticker mismatch: expected {expected}, received {actual}")


def to_evidence_record(
    run_id: str,
    company: ResolvedCompany,
    profile_id: str,
    evidence_type: str,
    tool_result: Any,
    source_name: str | None = None,
) -> list[EvidenceRecord]:
    """Convert one heterogeneous tool result into canonical evidence records.

    Args:
        run_id: Current research-run identifier.
        company: Canonical company assigned to the source call.
        profile_id: Versioned profile owning the evidence.
        evidence_type: Stable source category used by downstream validation.
        tool_result: Dictionary, list, scalar, or structured source result.
        source_name: Optional tool name; inferred from ``evidence_type`` when omitted.

    Returns:
        One record per source item, including explicit missing or failed records.

    Raises:
        ValueError: If run, profile, or company identity is invalid.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if profile_id != company["profile_id"]:
        raise ValueError(
            f"Evidence profile mismatch: company requires {company['profile_id']!r}, got {profile_id!r}"
        )

    items = tool_result if isinstance(tool_result, list) else [tool_result]
    if not items:
        items = [{"status": "missing", "error": None}]
    resolved_source_name = source_name or EVIDENCE_SOURCE_BY_TYPE.get(evidence_type, evidence_type)
    records: list[EvidenceRecord] = []

    for index, item in enumerate(items):
        _validate_evidence_identity(company, item)
        status, error = _evidence_status(item)
        mapping = item if isinstance(item, dict) else {}
        value = mapping.get("data", item) if isinstance(item, dict) else item
        retrieved_at = str(mapping.get("retrieved_at") or _utc_now_iso())
        as_of_value = (
            mapping.get("as_of")
            or mapping.get("timestamp")
            or mapping.get("end_date")
            or mapping.get("published_date")
        )
        source_uri = mapping.get("source_uri") or mapping.get("url")
        document_name = (
            mapping.get("document_name") or mapping.get("document") or mapping.get("title")
        )
        page = mapping.get("page")
        cache_status = str(mapping.get("cache_status", "unknown")).casefold()
        if cache_status not in {"hit", "miss", "stale", "unknown"}:
            cache_status = "unknown"
        source_metadata = {
            key: deepcopy_value
            for key, deepcopy_value in mapping.items()
            if key not in {"data", "value", "error"}
        }
        identity_seed = {
            "run_id": run_id,
            "company_id": company["company_id"],
            "profile_id": profile_id,
            "evidence_type": evidence_type,
            "source_name": resolved_source_name,
            "index": index,
            "value_hash": _canonical_payload_hash(value),
            "source_uri": source_uri,
            "document_name": document_name,
            "page": page,
        }
        evidence_id = "ev-" + str(uuid5(NAMESPACE_URL, json.dumps(identity_seed, sort_keys=True)))
        record: EvidenceRecord = {
            "evidence_id": evidence_id,
            "run_id": run_id,
            "company_id": company["company_id"],
            "ticker": company["ticker"],
            "profile_id": profile_id,
            "evidence_type": evidence_type,
            "value": value,
            "source_name": resolved_source_name,
            "source_uri": str(source_uri) if source_uri is not None else None,
            "document_name": str(document_name) if document_name is not None else None,
            "page": int(page) if page is not None else None,
            "as_of": str(as_of_value) if as_of_value is not None else None,
            "retrieved_at": retrieved_at,
            "freshness_status": str(mapping.get("freshness_status", "unknown")),
            "cache_status": cache_status,
            "status": status,
            "source_metadata": source_metadata,
            "error": error,
        }
        records.append(record)
    return records


def _call_or_use(tool_callable: Callable[[], Any], tool_result: Any) -> Any:
    """Use an injected deterministic result or invoke the real source capability.

    Args:
        tool_callable: Zero-argument closure that performs the real call.
        tool_result: Injected result, with ``None`` meaning no injection.

    Returns:
        Injected or live source result.
    """
    return tool_callable() if tool_result is None else tool_result


def fetch_price_evidence(task: CompanyTask, tool_result: Any = None) -> list[EvidenceRecord]:
    """Fetch or adapt current-price evidence for one company task."""
    raw = _call_or_use(
        lambda: get_stock_price.invoke({"ticker": task["company"]["ticker"]}),
        tool_result,
    )
    return to_evidence_record(
        task["run_id"], task["company"], task["company"]["profile_id"],
        "stock_price", raw, "get_stock_price",
    )


def fetch_history_evidence(
    task: CompanyTask,
    period: str = "1y",
    tool_result: Any = None,
) -> list[EvidenceRecord]:
    """Fetch or adapt historical-price evidence for one company task."""
    raw = _call_or_use(
        lambda: get_stock_history.invoke({"ticker": task["company"]["ticker"], "period": period}),
        tool_result,
    )
    return to_evidence_record(
        task["run_id"], task["company"], task["company"]["profile_id"],
        "stock_history", raw, "get_stock_history",
    )


def fetch_financial_metric_evidence(
    task: CompanyTask,
    tool_result: Any = None,
) -> list[EvidenceRecord]:
    """Fetch or adapt the deterministic financial-metric snapshot for one task."""
    raw = _call_or_use(
        lambda: get_financial_metrics(task["company"]["ticker"]),
        tool_result,
    )
    return to_evidence_record(
        task["run_id"], task["company"], task["company"]["profile_id"],
        "financial_metrics", raw, "get_financial_metrics",
    )


def fetch_news_evidence(
    task: CompanyTask,
    query: str | None = None,
    tool_result: Any = None,
) -> list[EvidenceRecord]:
    """Fetch or adapt company-scoped financial-news evidence."""
    scoped_query = query or f"{task['company']['company_name']} financial news"
    raw = _call_or_use(
        lambda: search_financial_news.invoke({"query": scoped_query}),
        tool_result,
    )
    return to_evidence_record(
        task["run_id"], task["company"], task["company"]["profile_id"],
        "financial_news", raw, "search_financial_news",
    )


def fetch_sentiment_evidence(
    task: CompanyTask,
    text: str,
    tool_result: Any = None,
) -> list[EvidenceRecord]:
    """Fetch or adapt sentiment evidence for company-scoped source text."""
    raw = _call_or_use(
        lambda: analyze_sentiment.invoke({"text": text}),
        tool_result,
    )
    return to_evidence_record(
        task["run_id"], task["company"], task["company"]["profile_id"],
        "sentiment", raw, "analyze_sentiment",
    )


print("✅ F06 canonical evidence adapters defined")
'''


F06_SMOKE = r'''# F06 local smoke test with injected source results; no network or API keys.
_f06_company = resolve_company_mention("Microsoft")
_f06_plan: QueryPlan = {
    "query_type": "fact",
    "company_mentions": ["Microsoft"],
    "requested_dimensions": ["current_price"],
    "risk_profile": "balanced",
    "scoring_requested": False,
    "freshness_required": True,
    "time_horizon": None,
}
_f06_task = build_company_tasks(_f06_plan, [_f06_company], "f06-smoke-run")[0]
_f06_records = fetch_price_evidence(
    _f06_task,
    tool_result={
        "ticker": "MSFT",
        "current_price": 500.0,
        "status": "success",
        "timestamp": "2026-08-06T14:00:00+00:00",
        "retrieved_at": "2026-08-06T14:01:00+00:00",
        "cache_status": "hit",
    },
)
assert len(_f06_records) == 1
assert _f06_records[0]["status"] == "success"
assert _f06_records[0]["as_of"] != _f06_records[0]["retrieved_at"]
assert _f06_records[0]["cache_status"] == "hit"

print("✅ F06 smoke test passed: source result normalized with identity and provenance")
'''


CELL_SPECS = [
    ("multiindustry_f06_intro", "markdown", F06_INTRO),
    ("multiindustry_evidence_adapters", "code", F06_CODE),
    ("multiindustry_f06_smoke", "code", F06_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create a notebook cell with a stable identifier."""
    cell = (
        nbformat.v4.new_markdown_cell(source=source)
        if cell_type == "markdown"
        else nbformat.v4.new_code_cell(source=source)
    )
    cell["id"] = cell_id
    return cell


def main() -> None:
    """Insert or refresh F06 cells in the working notebook."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
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
        index = next(
            i for i, cell in enumerate(notebook.cells)
            if cell.get("id") == INSERT_AFTER_CELL_ID
        ) + 1
        notebook.cells[index:index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F06 cells are present")


if __name__ == "__main__":
    main()

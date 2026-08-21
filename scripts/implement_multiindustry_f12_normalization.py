"""Idempotently add the F12 deterministic fan-in normalization cells."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f11_smoke"


F12_NORMALIZATION_INTRO = """## Section 3.12: Deterministic Fan-In Normalization

F12 begins with a mandatory deterministic boundary after F11 reducer fan-in. It restores the
original task order regardless of worker completion order, creates an explicit failed result for
every missing branch, and excludes unexpected branches.

Run, company, ticker, profile, and evidence identity cannot cross this boundary. Invalid branch
data is replaced with an identity-preserving failed placeholder. Valid source gaps remain explicit
as partial or failed results, allowing successful sibling branches to continue to comparison and
synthesis without treating incomplete evidence as complete.
"""


F12_NORMALIZATION_CODE = r'''from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Literal, Mapping, TypedDict


FanInStatus = Literal["complete", "partial", "failed"]


class FanInNormalization(TypedDict):
    """Validated, canonically ordered output of the company-result fan-in boundary.

    Attributes:
        run_id: Run shared by every accepted task/result/evidence record.
        status: Complete, partial, or failed aggregate result.
        ready: Whether comparison routing may proceed.
        ordered_tickers: Deterministic task-order ticker list.
        results_by_ticker: Canonical normalized result map.
        ordered_results: Same results in task order for presentation.
        successful_tickers: Complete branches.
        partial_tickers: Usable branches with declared gaps.
        failed_tickers: Unusable/failed branches.
        blocking_errors: Identity/coverage errors that stop routing.
        errors: All contained fan-in and branch errors.
    """

    run_id: str
    status: FanInStatus
    ready: bool
    ordered_tickers: list[str]
    results_by_ticker: CompanyResultMap
    ordered_results: list[CompanyResearchResult]
    successful_tickers: list[str]
    partial_tickers: list[str]
    failed_tickers: list[str]
    blocking_errors: list[str]
    errors: list[str]


def _ordered_unique_strings(values: Iterable[Any]) -> list[str]:
    """Return stripped, non-empty strings once each while retaining input order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _normalization_failed_result(
    task: CompanyTask,
    errors: Iterable[str],
) -> CompanyResearchResult:
    """Create a failed result that keeps only the expected task identity."""
    missing = _ordered_unique_strings(
        list(task.get("shared_dimensions", []))
        + list(task.get("industry_dimensions", []))
        + list(task.get("unsupported_dimensions", []))
    )
    normalized_errors = _ordered_unique_strings(errors)
    return {
        "run_id": task["run_id"],
        "company": deepcopy(task["company"]),
        "profile_id": task["company"]["profile_id"],
        "financial_evidence": {},
        "industry_signals": {},
        "evidence": [],
        "missing_dimensions": missing,
        "errors": normalized_errors or ["Company result failed normalization"],
        "status": "failed",
    }


def _evidence_boundary_errors(
    task: CompanyTask,
    record: Mapping[str, Any],
    run_id: str,
) -> list[str]:
    """Return hard boundary violations for one evidence record."""
    company = task["company"]
    errors: list[str] = []
    evidence_id = str(record.get("evidence_id", "")).strip() or "<missing-id>"
    if record.get("run_id") != run_id:
        errors.append(f"Evidence {evidence_id} has the wrong run_id")
    if record.get("company_id") != company["company_id"]:
        errors.append(f"Evidence {evidence_id} crossed the company boundary")
    if str(record.get("ticker", "")).upper() != company["ticker"].upper():
        errors.append(f"Evidence {evidence_id} has the wrong ticker")
    if record.get("profile_id") != company["profile_id"]:
        errors.append(f"Evidence {evidence_id} crossed the profile boundary")
    return errors


def _normalize_evidence_records(
    task: CompanyTask,
    raw_records: Any,
    run_id: str,
) -> tuple[list[EvidenceRecord], list[str], bool]:
    """Validate evidence shape, enums, uniqueness, and hard identity boundaries.

    Returns:
        Normalized records, validation errors, and whether a hard boundary was crossed.
    """
    if not isinstance(raw_records, list):
        return [], ["Company evidence must be a list"], False

    normalized: list[EvidenceRecord] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    boundary_failed = False
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            errors.append(f"Evidence item {index} is not a mapping")
            continue
        boundary_errors = _evidence_boundary_errors(task, raw_record, run_id)
        if boundary_errors:
            errors.extend(boundary_errors)
            boundary_failed = True
            continue

        record = deepcopy(dict(raw_record))
        evidence_id = str(record.get("evidence_id", "")).strip()
        if not evidence_id:
            errors.append(f"Evidence item {index} has no evidence_id")
            continue
        if evidence_id in seen_ids:
            errors.append(f"Duplicate evidence_id for {task['company']['ticker']}: {evidence_id}")
            continue
        seen_ids.add(evidence_id)

        evidence_type = str(record.get("evidence_type", "")).strip()
        if not evidence_type:
            errors.append(f"Evidence {evidence_id} has no evidence_type")
            evidence_type = "unknown"
        record["evidence_type"] = evidence_type
        if "value" not in record:
            errors.append(f"Evidence {evidence_id} has no value field")
            record["value"] = None
        source_name = str(record.get("source_name", "")).strip()
        if not source_name:
            errors.append(f"Evidence {evidence_id} has no source_name")
            source_name = "unknown"
        record["source_name"] = source_name
        record.setdefault("source_uri", None)
        record.setdefault("document_name", None)
        record.setdefault("page", None)
        record.setdefault("as_of", None)
        record.setdefault("retrieved_at", "")
        record.setdefault("error", None)
        if not isinstance(record.get("source_metadata"), Mapping):
            if record.get("source_metadata") not in ({}, None):
                errors.append(f"Evidence {evidence_id} has invalid source_metadata")
            record["source_metadata"] = {}

        status = str(record.get("status", "")).casefold()
        if status not in {"success", "missing", "failed"}:
            errors.append(f"Evidence {evidence_id} has invalid status {status!r}")
            record["status"] = "failed"
            record["error"] = str(record.get("error") or "Invalid evidence status")
        freshness = str(record.get("freshness_status", "unknown")).casefold()
        if freshness not in {"fresh", "stale", "unknown"}:
            errors.append(f"Evidence {evidence_id} has invalid freshness status {freshness!r}")
            freshness = "unknown"
        record["freshness_status"] = freshness
        cache_status = str(record.get("cache_status", "unknown")).casefold()
        record["cache_status"] = (
            cache_status if cache_status in {"hit", "miss", "stale", "unknown"} else "unknown"
        )
        normalized.append(record)  # type: ignore[arg-type]
    return normalized, errors, boundary_failed


def _acceptable_success_ids(
    task: CompanyTask,
    evidence: list[EvidenceRecord],
) -> set[str]:
    """Return successful evidence IDs that satisfy the query's freshness requirement."""
    freshness_required = bool(task.get("query_plan", {}).get("freshness_required", False))
    return {
        record["evidence_id"]
        for record in evidence
        if record["status"] == "success"
        and (not freshness_required or record["freshness_status"] == "fresh")
    }


def _normalize_industry_signals(
    raw_signals: Any,
    acceptable_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Keep only evidence-grounded signal references and downgrade ungrounded signals."""
    if not isinstance(raw_signals, Mapping):
        return {}, ["industry_signals must be a mapping"]
    normalized: dict[str, Any] = deepcopy(dict(raw_signals))
    errors: list[str] = []
    for dimension, raw_signal in list(normalized.items()):
        if not isinstance(raw_signal, Mapping):
            errors.append(f"Signal {dimension} is not a mapping")
            normalized[dimension] = {
                "level": "missing", "score": None, "reason": "Invalid signal shape.",
                "evidence_ids": [],
            }
            continue
        signal = deepcopy(dict(raw_signal))
        raw_requested_ids = signal.get("evidence_ids", [])
        if not isinstance(raw_requested_ids, list):
            errors.append(f"Signal {dimension} evidence_ids must be a list")
            raw_requested_ids = []
        requested_ids = _ordered_unique_strings(raw_requested_ids)
        valid_ids = _ordered_unique_strings(
            evidence_id for evidence_id in requested_ids if evidence_id in acceptable_ids
        )
        invalid_ids = [
            str(evidence_id) for evidence_id in requested_ids
            if evidence_id not in acceptable_ids
        ]
        if invalid_ids:
            errors.append(
                f"Signal {dimension} references invalid evidence IDs: {sorted(set(invalid_ids))}"
            )
        level = str(signal.get("level", "missing")).casefold()
        has_claim = level not in {"", "missing"} or signal.get("score") is not None
        if has_claim and not valid_ids:
            errors.append(f"Signal {dimension} was downgraded because it is not grounded")
            signal.update({
                "level": "missing", "score": None,
                "reason": "Signal rejected because it lacks valid current-run evidence IDs.",
                "evidence_ids": [],
            })
        else:
            signal["evidence_ids"] = valid_ids
        normalized[str(dimension)] = signal
    return normalized, errors


def _required_dimension_gaps(
    task: CompanyTask,
    evidence: list[EvidenceRecord],
    signals: Mapping[str, Any],
) -> list[str]:
    """Compute required dimensions that lack acceptable source or signal coverage."""
    freshness_required = bool(task.get("query_plan", {}).get("freshness_required", False))
    successful_types = {
        record["evidence_type"]
        for record in evidence
        if record["status"] == "success"
        and (not freshness_required or record["freshness_status"] == "fresh")
    }
    gaps: list[str] = []
    for dimension in task.get("shared_dimensions", []):
        if dimension == "price_history":
            covered = "stock_history" in successful_types
        elif dimension == "news_sentiment":
            covered = bool({"financial_news", "sentiment"} & successful_types)
        elif dimension in {"current_price", "market_cap"}:
            covered = bool({"stock_price", "financial_metrics"} & successful_types)
        else:
            covered = "financial_metrics" in successful_types
        if not covered:
            gaps.append(dimension)

    for dimension in task.get("industry_dimensions", []):
        signal = signals.get(dimension)
        covered = (
            isinstance(signal, Mapping)
            and str(signal.get("level", "missing")).casefold() != "missing"
            and bool(signal.get("evidence_ids"))
        )
        if not covered:
            gaps.append(dimension)
    gaps.extend(task.get("unsupported_dimensions", []))
    return _ordered_unique_strings(gaps)


def normalize_company_result(
    task: CompanyTask,
    result: CompanyResearchResult | Mapping[str, Any] | None,
    run_id: str | None = None,
) -> CompanyResearchResult:
    """Normalize one worker result against its expected immutable task boundary.

    Missing results and hard run/company/profile/evidence identity violations become failed
    placeholders. Ordinary source failures, missing dimensions, stale evidence, duplicate IDs,
    and ungrounded signals remain contained as partial results when usable evidence survives.
    """
    expected_run_id = run_id or task.get("run_id", "")
    if not expected_run_id or task.get("run_id") != expected_run_id:
        return _normalization_failed_result(task, ["Task has the wrong or missing run_id"])
    if result is None:
        return _normalization_failed_result(task, ["Missing company worker result"])
    if not isinstance(result, Mapping):
        return _normalization_failed_result(task, ["Company worker result must be a mapping"])

    company = task["company"]
    result_company = result.get("company")
    boundary_errors: list[str] = []
    if result.get("run_id") != expected_run_id:
        boundary_errors.append("Result has the wrong run_id")
    if not isinstance(result_company, Mapping):
        boundary_errors.append("Result has no canonical company identity")
    else:
        if result_company.get("company_id") != company["company_id"]:
            boundary_errors.append("Result crossed the company boundary")
        if str(result_company.get("ticker", "")).upper() != company["ticker"].upper():
            boundary_errors.append("Result has the wrong ticker")
        if result_company.get("profile_id") != company["profile_id"]:
            boundary_errors.append("Result company has the wrong profile")
    if result.get("profile_id") != company["profile_id"]:
        boundary_errors.append("Result has the wrong profile_id")
    if boundary_errors:
        return _normalization_failed_result(task, boundary_errors)

    evidence, evidence_errors, evidence_boundary_failed = _normalize_evidence_records(
        task, result.get("evidence", []), expected_run_id,
    )
    if evidence_boundary_failed:
        return _normalization_failed_result(task, evidence_errors)

    acceptable_ids = _acceptable_success_ids(task, evidence)
    signals, signal_errors = _normalize_industry_signals(
        result.get("industry_signals", {}), acceptable_ids,
    )
    declared_missing = result.get("missing_dimensions", [])
    missing_errors: list[str] = []
    if not isinstance(declared_missing, list):
        missing_errors.append("missing_dimensions must be a list")
        declared_missing = []
    missing_dimensions = _ordered_unique_strings(
        list(declared_missing) + _required_dimension_gaps(task, evidence, signals)
    )

    raw_errors = result.get("errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = ["errors must be a list"]
    errors = _ordered_unique_strings(
        list(raw_errors) + evidence_errors + signal_errors + missing_errors
    )
    non_success_evidence = any(record["status"] != "success" for record in evidence)
    stale_required = bool(task.get("query_plan", {}).get("freshness_required", False)) and any(
        record["status"] == "success" and record["freshness_status"] != "fresh"
        for record in evidence
    )
    if stale_required:
        errors = _ordered_unique_strings(errors + ["Fresh evidence is required by the query plan"])

    raw_status = str(result.get("status", "")).casefold()
    if raw_status not in {"success", "partial", "failed"}:
        errors = _ordered_unique_strings(errors + [f"Invalid company result status: {raw_status!r}"])
    successful_evidence = [record for record in evidence if record["status"] == "success"]
    if raw_status == "failed" or not successful_evidence or not acceptable_ids:
        status: CompanyResultStatus = "failed"
    elif (
        raw_status == "partial" or missing_dimensions or errors
        or non_success_evidence or stale_required
    ):
        status = "partial"
    else:
        status = "success"

    financial_evidence = {
        record["evidence_type"]: deepcopy(record["value"])
        for record in evidence
        if record["status"] == "success"
        and record["evidence_type"] not in {"technology_rag", "biopharma_rag"}
    }
    return {
        "run_id": expected_run_id,
        "company": deepcopy(company),
        "profile_id": company["profile_id"],
        "financial_evidence": financial_evidence,
        "industry_signals": signals,
        "evidence": evidence,
        "missing_dimensions": missing_dimensions,
        "errors": errors,
        "status": status,
    }


def _coerce_result_items(
    results: Mapping[str, CompanyResearchResult] | Iterable[CompanyResearchResult],
) -> tuple[dict[str, CompanyResearchResult], list[str]]:
    """Convert reducer maps or result sequences to one ticker map and detect duplicates."""
    errors: list[str] = []
    if isinstance(results, Mapping):
        return dict(results), errors
    by_ticker: dict[str, CompanyResearchResult] = {}
    for index, result in enumerate(results):
        ticker = str(result.get("company", {}).get("ticker", "")).upper()
        if not ticker:
            errors.append(f"Result item {index} has no ticker")
            continue
        if ticker in by_ticker:
            errors.append(f"Duplicate company result for ticker {ticker}")
            continue
        by_ticker[ticker] = result
    return by_ticker, errors


def normalize_all_results(
    tasks: list[CompanyTask],
    results: Mapping[str, CompanyResearchResult] | Iterable[CompanyResearchResult],
    run_id: str | None = None,
) -> FanInNormalization:
    """Normalize fan-in results in expected task order, independent of completion order."""
    expected_run_id = run_id or (tasks[0].get("run_id", "") if tasks else "")
    errors: list[str] = []
    blocking_errors: list[str] = []
    result_map, input_errors = _coerce_result_items(results)
    errors.extend(input_errors)
    blocking_errors.extend(input_errors)

    ordered_tasks: list[CompanyTask] = []
    seen_tickers: set[str] = set()
    seen_company_ids: set[str] = set()
    for index, task in enumerate(tasks):
        ticker = str(task.get("company", {}).get("ticker", "")).upper()
        company_id = str(task.get("company", {}).get("company_id", ""))
        if not ticker or not company_id:
            message = f"Expected task {index} has incomplete company identity"
            errors.append(message)
            blocking_errors.append(message)
            continue
        if ticker in seen_tickers or company_id in seen_company_ids:
            message = f"Duplicate expected company task for {ticker}"
            errors.append(message)
            blocking_errors.append(message)
            continue
        if task.get("run_id") != expected_run_id:
            message = f"Expected task {ticker} has the wrong run_id"
            errors.append(message)
            blocking_errors.append(message)
        seen_tickers.add(ticker)
        seen_company_ids.add(company_id)
        ordered_tasks.append(task)

    unexpected = sorted(set(result_map) - seen_tickers)
    if unexpected:
        message = f"Unexpected company results were excluded: {unexpected}"
        errors.append(message)
        blocking_errors.append(message)

    normalized_by_ticker: CompanyResultMap = {}
    ordered_results: list[CompanyResearchResult] = []
    for task in ordered_tasks:
        ticker = task["company"]["ticker"].upper()
        result = result_map.get(ticker)
        normalized = normalize_company_result(task, result, expected_run_id)
        normalized_by_ticker[ticker] = normalized
        ordered_results.append(normalized)
        errors.extend(f"{ticker}: {error}" for error in normalized["errors"])

    successful = [
        ticker for ticker, result in normalized_by_ticker.items()
        if result["status"] == "success"
    ]
    partial = [
        ticker for ticker, result in normalized_by_ticker.items()
        if result["status"] == "partial"
    ]
    failed = [
        ticker for ticker, result in normalized_by_ticker.items()
        if result["status"] == "failed"
    ]
    if normalized_by_ticker and not partial and not failed and not errors:
        aggregate_status: FanInStatus = "complete"
    elif successful or partial:
        aggregate_status = "partial"
    else:
        aggregate_status = "failed"

    return {
        "run_id": expected_run_id,
        "status": aggregate_status,
        "ready": bool(successful or partial) and not blocking_errors,
        "ordered_tickers": list(normalized_by_ticker),
        "results_by_ticker": normalized_by_ticker,
        "ordered_results": ordered_results,
        "successful_tickers": successful,
        "partial_tickers": partial,
        "failed_tickers": failed,
        "blocking_errors": _ordered_unique_strings(blocking_errors),
        "errors": _ordered_unique_strings(errors),
    }


print("✅ F12 deterministic fan-in normalization defined")
'''


F12_NORMALIZATION_SMOKE = r'''# F12 normalization smoke: reducer order cannot change comparison order.
_f12_normalized = normalize_all_results(
    _f11_state["company_tasks"],
    {
        "PFE": _f11_state["company_results"]["PFE"],
        "MSFT": _f11_state["company_results"]["MSFT"],
    },
    _f11_state["run_id"],
)
assert _f12_normalized["ordered_tickers"] == ["MSFT", "PFE"]
assert set(_f12_normalized["results_by_ticker"]) == {"MSFT", "PFE"}
assert _f12_normalized["ready"] is True

print("✅ F12 normalization smoke test passed: task order and branch boundaries are stable")
'''


CELL_SPECS = [
    ("multiindustry_f12_normalization_intro", "markdown", F12_NORMALIZATION_INTRO),
    ("multiindustry_fan_in_normalization", "code", F12_NORMALIZATION_CODE),
    ("multiindustry_f12_normalization_smoke", "code", F12_NORMALIZATION_SMOKE),
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


def integrate_notebook(notebook_path: Path = NOTEBOOK_PATH) -> None:
    """Insert or refresh only the F12 fan-in normalization cells idempotently."""
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
        index = next(
            i for i, cell in enumerate(notebook.cells)
            if cell.get("id") == INSERT_AFTER_CELL_ID
        ) + 1
        notebook.cells[index:index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")
    nbformat.write(notebook, notebook_path)


def main() -> None:
    """Integrate F12 normalization cells into the canonical notebook when requested."""
    integrate_notebook()
    print(f"Updated {NOTEBOOK_PATH.name}: F12 normalization cells are present")


if __name__ == "__main__":
    main()

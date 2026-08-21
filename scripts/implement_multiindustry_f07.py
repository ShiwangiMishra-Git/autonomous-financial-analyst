"""Idempotently add the F07 Technology/AI profile adapters to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f06_smoke"


F07_INTRO = """## Section 3.7: Technology/AI Profile Adapters

F07 keeps the assignment's working technology behavior but places it behind explicit
`technology.ai.v1` contracts. `query_technology_rag` adds canonical ticker/profile validation and
structured provenance around the legacy `query_private_database` implementation.

`extract_technology_signals_with_evidence` preserves the four existing AI dimensions while
requiring current-run technology evidence IDs for every non-missing signal.
`score_technology_companies` is a transparent name-safe wrapper around the existing deterministic
`score_companies` function, so the validated rubric arithmetic does not change.
"""


F07_CODE = r'''from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import warnings

from langchain_core.tools import tool


TECHNOLOGY_PROFILE_ID = "technology.ai.v1"
TECHNOLOGY_RAG_COLLECTION = "AI_Initiatives"
TECHNOLOGY_SIGNAL_NAMES = [
    "infrastructure_moat",
    "product_deployment",
    "research_depth",
    "strategic_commitment",
]
TECHNOLOGY_SIGNAL_LEVEL_SCORES = {
    "none": 0.0,
    "partial": 0.5,
    "full": 1.0,
    "missing": None,
}


def _technology_company_from_ticker(ticker: str) -> ResolvedCompany:
    """Resolve and validate one ticker as a supported Technology/AI company.

    Args:
        ticker: Canonical or alias ticker text.

    Returns:
        Successfully resolved technology company.

    Raises:
        ValueError: If the company is unsupported, ambiguous, or not in the technology profile.
    """
    company = resolve_company_mention(ticker)
    if company["resolution_status"] != "resolved":
        raise ValueError(company["resolution_message"] or f"Unable to resolve {ticker!r}")
    if company["profile_id"] != TECHNOLOGY_PROFILE_ID:
        raise ValueError(
            f"query_technology_rag supports {TECHNOLOGY_PROFILE_ID}, not {company['profile_id']}"
        )
    return company


def _invoke_legacy_technology_rag(query: str) -> str:
    """Invoke the existing assignment RAG tool without changing its implementation.

    Args:
        query: Company-scoped technology question.

    Returns:
        Legacy context-grounded answer string.
    """
    if hasattr(query_private_database, "invoke"):
        return query_private_database.invoke({"query": query})
    return query_private_database(query)


@tool
def query_technology_rag(ticker: str, query: str) -> dict[str, Any]:
    """Retrieve Technology/AI evidence for one supported canonical company.

    Args:
        ticker: Supported technology ticker such as ``MSFT`` or ``NVDA``.
        query: Focused AI initiative, product, research, or strategy question.

    Returns:
        Structured result identifying the technology profile, collection, corpus, and legacy
        context-grounded answer. Invalid companies return an explicit error status.
    """
    try:
        company = _technology_company_from_ticker(ticker)
    except ValueError as exc:
        return {
            "status": "error",
            "ticker": ticker.upper(),
            "profile_id": TECHNOLOGY_PROFILE_ID,
            "collection": TECHNOLOGY_RAG_COLLECTION,
            "error": str(exc),
        }

    scoped_query = f"Company: {company['company_name']} ({company['ticker']}). Question: {query.strip()}"
    answer = _invoke_legacy_technology_rag(scoped_query)
    if isinstance(answer, str) and answer.strip().casefold().startswith("error"):
        status = "failed"
        error = answer
    elif isinstance(answer, str) and (
        "i don't know" in answer.casefold() or "not available" in answer.casefold()
    ):
        status = "missing"
        error = None
    else:
        status = "success"
        error = None
    return {
        "status": status,
        "ticker": company["ticker"],
        "company_id": company["company_id"],
        "profile_id": TECHNOLOGY_PROFILE_ID,
        "collection": TECHNOLOGY_RAG_COLLECTION,
        "corpus_version": get_industry_profile(TECHNOLOGY_PROFILE_ID)["corpus_version"],
        "source_name": "query_private_database",
        "data": answer,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "freshness_status": "unknown",
        "cache_status": "unknown",
        "error": error,
    }


def query_private_database_compat(query: str) -> str:
    """Call the legacy technology RAG implementation during migration.

    Args:
        query: Legacy free-text technology RAG question.

    Returns:
        Exactly the underlying ``query_private_database`` result.

    Warns:
        DeprecationWarning: Callers should migrate to ticker-scoped ``query_technology_rag``.
    """
    warnings.warn(
        "query_private_database compatibility access is deprecated; use query_technology_rag",
        DeprecationWarning,
        stacklevel=2,
    )
    return _invoke_legacy_technology_rag(query)


def query_technology_rag_evidence(
    task: CompanyTask,
    query: str,
    tool_result: Any = None,
) -> list[EvidenceRecord]:
    """Retrieve or adapt technology RAG output into canonical evidence.

    Args:
        task: Technology company task.
        query: Focused technology research question.
        tool_result: Optional injected result for deterministic tests.

    Returns:
        Canonical technology RAG evidence records.
    """
    if task["company"]["profile_id"] != TECHNOLOGY_PROFILE_ID:
        raise ValueError("Technology RAG evidence requires a technology.ai.v1 task")
    raw = tool_result
    if raw is None:
        raw = query_technology_rag.invoke(
            {"ticker": task["company"]["ticker"], "query": query}
        )
    return to_evidence_record(
        task["run_id"], task["company"], TECHNOLOGY_PROFILE_ID,
        "technology_rag", raw, "query_technology_rag",
    )


def _technology_evidence_ids(
    company: ResolvedCompany,
    records: list[EvidenceRecord],
) -> list[str]:
    """Validate and collect successful current-company technology evidence IDs.

    Args:
        company: Technology company whose signals are being extracted.
        records: Candidate evidence records for that company.

    Returns:
        Successful evidence IDs in record order.

    Raises:
        ValueError: If any record crosses company, ticker, or profile boundaries.
    """
    evidence_ids: list[str] = []
    for record in records:
        if record["company_id"] != company["company_id"] or record["ticker"] != company["ticker"]:
            raise ValueError(f"Technology evidence identity mismatch for {company['ticker']}")
        if record["profile_id"] != TECHNOLOGY_PROFILE_ID:
            raise ValueError("Technology signals cannot use non-technology evidence")
        if record["status"] == "success":
            evidence_ids.append(record["evidence_id"])
    return evidence_ids


def extract_technology_signals_with_evidence(
    companies: list[ResolvedCompany],
    evidence_by_company: dict[str, list[EvidenceRecord]],
    raw_signals: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize existing AI signals and bind every non-missing signal to evidence IDs.

    Args:
        companies: Resolved technology companies.
        evidence_by_company: Current-run canonical evidence keyed by company ID.
        raw_signals: Optional existing ``extract_ai_signals`` output for deterministic tests.

    Returns:
        Ticker-keyed four-dimension signal mappings with level, score, reason, and evidence IDs.

    Raises:
        ValueError: If a company/profile or evidence identity is invalid.
    """
    for company in companies:
        if company["profile_id"] != TECHNOLOGY_PROFILE_ID:
            raise ValueError("Technology extractor received a non-technology company")

    if raw_signals is None:
        prior_reports: dict[str, str] = {}
        for company in companies:
            values = [
                str(record["value"])
                for record in evidence_by_company.get(company["company_id"], [])
                if record["status"] == "success" and record["evidence_type"] == "technology_rag"
            ]
            if values:
                prior_reports[company["ticker"]] = "\n\n".join(values)
        raw_signals = extract_ai_signals(
            [company["ticker"] for company in companies],
            prior_reports=prior_reports,
        )

    normalized: dict[str, dict[str, Any]] = {}
    for company in companies:
        records = evidence_by_company.get(company["company_id"], [])
        available_ids = _technology_evidence_ids(company, records)
        available_set = set(available_ids)
        company_raw = raw_signals.get(company["ticker"], {})
        normalized[company["ticker"]] = {}

        for dimension in TECHNOLOGY_SIGNAL_NAMES:
            raw = company_raw.get(dimension)
            if not isinstance(raw, dict):
                normalized[company["ticker"]][dimension] = {
                    "level": "missing", "score": None,
                    "reason": "No structured signal was returned.", "evidence_ids": [],
                }
                continue
            level = str(raw.get("level", "missing")).casefold()
            if level not in TECHNOLOGY_SIGNAL_LEVEL_SCORES:
                level = "missing"
            requested_ids = raw.get("evidence_ids")
            if requested_ids is None:
                evidence_ids = available_ids
            else:
                evidence_ids = [item for item in requested_ids if item in available_set]
            if level != "missing" and not evidence_ids:
                normalized[company["ticker"]][dimension] = {
                    "level": "missing", "score": None,
                    "reason": "Signal was rejected because it had no valid current-run evidence.",
                    "evidence_ids": [],
                }
                continue
            normalized[company["ticker"]][dimension] = {
                "level": level,
                "score": TECHNOLOGY_SIGNAL_LEVEL_SCORES[level],
                "reason": str(raw.get("reason", "")),
                "evidence_ids": evidence_ids,
            }
    return normalized


def score_technology_companies(
    financial_metrics: dict[str, dict],
    technology_signals: dict[str, dict],
    sentiment_scores: dict[str, dict],
    risk_profile: str = "balanced",
) -> dict[str, dict]:
    """Apply the existing deterministic technology scoring function unchanged.

    Args:
        financial_metrics: Ticker-keyed five-metric snapshots.
        technology_signals: Ticker-keyed normalized four-dimension AI signals.
        sentiment_scores: Existing sentiment summaries retained for reporting.
        risk_profile: Existing conservative, balanced, or growth weight selection.

    Returns:
        Exactly the score table produced by ``score_companies``.
    """
    return score_companies(
        financial_metrics,
        technology_signals,
        sentiment_scores,
        risk_profile=risk_profile,
    )


print("✅ F07 Technology/AI profile adapters defined")
'''


F07_SMOKE = r'''# F07 local smoke test with injected evidence and signals; no live RAG call.
_f07_company = resolve_company_mention("Microsoft")
_f07_plan: QueryPlan = {
    "query_type": "analyze", "company_mentions": ["Microsoft"],
    "requested_dimensions": ["ai_strategy"], "risk_profile": "balanced",
    "scoring_requested": False, "freshness_required": False, "time_horizon": None,
}
_f07_task = build_company_tasks(_f07_plan, [_f07_company], "f07-smoke-run")[0]
_f07_evidence = query_technology_rag_evidence(
    _f07_task,
    "AI strategy",
    tool_result={
        "status": "success", "ticker": "MSFT", "profile_id": "technology.ai.v1",
        "collection": "AI_Initiatives", "data": "Microsoft invests in Azure AI infrastructure.",
        "document_name": "MSFT.pdf", "page": 1,
    },
)
_f07_raw = {"MSFT": {
    name: {"level": "full", "reason": "Supported by the technology report."}
    for name in TECHNOLOGY_SIGNAL_NAMES
}}
_f07_signals = extract_technology_signals_with_evidence(
    [_f07_company], {"microsoft": _f07_evidence}, raw_signals=_f07_raw,
)
assert set(_f07_signals["MSFT"]) == set(TECHNOLOGY_SIGNAL_NAMES)
assert all(_f07_signals["MSFT"][name]["evidence_ids"] for name in TECHNOLOGY_SIGNAL_NAMES)

print("✅ F07 smoke test passed: technology signals preserve schema and evidence IDs")
'''


CELL_SPECS = [
    ("multiindustry_f07_intro", "markdown", F07_INTRO),
    ("multiindustry_technology_profile", "code", F07_CODE),
    ("multiindustry_f07_smoke", "code", F07_SMOKE),
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
    """Insert or refresh F07 cells in the working notebook."""
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
    print(f"Updated {NOTEBOOK_PATH.name}: F07 cells are present")


if __name__ == "__main__":
    main()

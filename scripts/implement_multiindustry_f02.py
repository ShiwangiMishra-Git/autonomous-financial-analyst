"""Idempotently add the F02 company registry and resolver to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f01_smoke"


F02_INTRO = """## Section 3.2: Local Company Registry and Resolver

The coordinator may autonomously decide when to call `resolve_companies_tool`, but only the
deterministic registry establishes the canonical ticker, company identity, industry,
sub-industry, and supported profile. Agent-callable and deterministic are complementary:
the LLM controls invocation while the tool controls identity.

The parent workflow must apply `validate_resolution_gate` after the tool result. Research
cannot begin when the result is empty, ambiguous, or unsupported, even if the agent attempts
to continue. This is the first notebook implementation of the guarded-tool + mandatory-gate
pattern used throughout the multi-industry design.

The initial supported universe contains the five Technology/AI corpus companies and the
Healthcare/Biopharma companies represented in the local official-source archive. Bare `Roche`
is intentionally ambiguous because the corpus identifies both a Swiss listing and a US ADR.
"""


F02_CODE = r'''from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from langchain_core.tools import tool


class CompanyRegistryEntry(TypedDict):
    """Authoritative supported-company registry row.

    Attributes:
        company_id: Stable internal identity.
        ticker: Canonical listed symbol.
        company_name: Developer/user display name.
        aliases: Accepted free-text aliases.
        exchange: Listing exchange when known.
        industry: Broad deterministic industry.
        sub_industry: Narrow deterministic classification.
        profile_id: Versioned research profile assigned to the company.
    """
    company_id: str
    ticker: str
    company_name: str
    aliases: list[str]
    exchange: str | None
    industry: str
    sub_industry: str
    profile_id: str


def _company_entry(
    company_id: str,
    ticker: str,
    company_name: str,
    aliases: list[str],
    exchange: str | None,
    industry: str,
    sub_industry: str,
    profile_id: str,
) -> CompanyRegistryEntry:
    """Build one typed registry row from explicit authoritative values.

    Args:
        company_id: Stable internal identity.
        ticker: Canonical public symbol.
        company_name: Display name.
        aliases: Accepted free-text mentions.
        exchange: Listing exchange.
        industry: Broad industry.
        sub_industry: Narrow classification.
        profile_id: Versioned research profile.

    Returns:
        Registry entry stored in ``COMPANY_REGISTRY``.

    Usage:
        Used only while declaring the local supported-company universe.
    """
    return {
        "company_id": company_id,
        "ticker": ticker,
        "company_name": company_name,
        "aliases": aliases,
        "exchange": exchange,
        "industry": industry,
        "sub_industry": sub_industry,
        "profile_id": profile_id,
    }


COMPANY_REGISTRY: dict[str, CompanyRegistryEntry] = {
    # Technology/AI corpus
    "MSFT": _company_entry(
        "microsoft", "MSFT", "Microsoft Corporation", ["microsoft", "microsoft corp"],
        "NASDAQ", "technology", "software_cloud", "technology.ai.v1",
    ),
    "GOOGL": _company_entry(
        "alphabet", "GOOGL", "Alphabet Inc.", ["google", "alphabet", "google class a"],
        "NASDAQ", "technology", "internet_cloud", "technology.ai.v1",
    ),
    "NVDA": _company_entry(
        "nvidia", "NVDA", "NVIDIA Corporation", ["nvidia"],
        "NASDAQ", "technology", "semiconductors", "technology.ai.v1",
    ),
    "AMZN": _company_entry(
        "amazon", "AMZN", "Amazon.com, Inc.", ["amazon", "amazon com"],
        "NASDAQ", "technology", "commerce_cloud", "technology.ai.v1",
    ),
    "IBM": _company_entry(
        "ibm", "IBM", "International Business Machines Corporation",
        ["ibm", "international business machines"],
        "NYSE", "technology", "enterprise_technology", "technology.ai.v1",
    ),

    # Healthcare/Biopharma official-source corpus
    "LLY": _company_entry(
        "eli_lilly", "LLY", "Eli Lilly and Company", ["eli lilly", "lilly"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "MRK": _company_entry(
        "merck", "MRK", "Merck & Co., Inc.", ["merck", "merck and co", "msd"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "JNJ": _company_entry(
        "johnson_and_johnson", "JNJ", "Johnson & Johnson",
        ["johnson and johnson", "j and j", "jnj"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "PFE": _company_entry(
        "pfizer", "PFE", "Pfizer Inc.", ["pfizer"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "ABBV": _company_entry(
        "abbvie", "ABBV", "AbbVie Inc.", ["abbvie"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "AZN": _company_entry(
        "astrazeneca", "AZN", "AstraZeneca PLC", ["astrazeneca", "astra zeneca"],
        "NASDAQ", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "RHHBY": _company_entry(
        "roche", "RHHBY", "Roche Holding AG", ["roche", "roche adr"],
        "OTC", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "ROG.SW": _company_entry(
        "roche", "ROG.SW", "Roche Holding AG", ["roche", "roche swiss"],
        "SIX", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "NVS": _company_entry(
        "novartis", "NVS", "Novartis AG", ["novartis"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "BMY": _company_entry(
        "bristol_myers_squibb", "BMY", "Bristol Myers Squibb",
        ["bristol myers squibb", "bristol-myers squibb", "bms"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "GSK": _company_entry(
        "gsk", "GSK", "GSK plc", ["gsk", "glaxosmithkline", "glaxo smith kline"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "AMGN": _company_entry(
        "amgen", "AMGN", "Amgen Inc.", ["amgen"],
        "NASDAQ", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "NVO": _company_entry(
        "novo_nordisk", "NVO", "Novo Nordisk A/S", ["novo nordisk"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "SNY": _company_entry(
        "sanofi", "SNY", "Sanofi", ["sanofi"],
        "NASDAQ", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
    "TAK": _company_entry(
        "takeda", "TAK", "Takeda Pharmaceutical Company Limited",
        ["takeda", "takeda pharmaceutical"],
        "NYSE", "healthcare", "biopharma", "healthcare.biopharma.v1",
    ),
}


def _normalize_company_text(value: str) -> str:
    """Normalize a company mention for deterministic alias matching.

    Args:
        value: Raw user mention, alias, name, or ticker.

    Returns:
        Case-folded alphanumeric tokens separated by single spaces.
    """
    normalized = value.casefold().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _build_alias_index() -> dict[str, set[str]]:
    """Build normalized alias-to-ticker candidates from the registry.

    Returns:
        Alias mapping used by deterministic company resolution.

    Usage:
        Evaluated once when the registry cell is loaded.
    """
    index: dict[str, set[str]] = {}
    for ticker, entry in COMPANY_REGISTRY.items():
        values = {ticker, entry["company_name"], *entry["aliases"]}
        for value in values:
            alias = _normalize_company_text(value)
            if alias:
                index.setdefault(alias, set()).add(ticker)
    return index


COMPANY_ALIAS_INDEX = _build_alias_index()
NORMALIZED_TICKERS = {
    _normalize_company_text(ticker): ticker
    for ticker in COMPANY_REGISTRY
}


def _resolved_company(entry: CompanyRegistryEntry) -> ResolvedCompany:
    """Project one registry row into a successful resolver result.

    Args:
        entry: Authoritative supported-company row.

    Returns:
        Defensive canonical ``ResolvedCompany`` value.
    """
    return {
        "company_id": entry["company_id"],
        "ticker": entry["ticker"],
        "company_name": entry["company_name"],
        "aliases": list(entry["aliases"]),
        "exchange": entry["exchange"],
        "industry": entry["industry"],
        "sub_industry": entry["sub_industry"],
        "profile_id": entry["profile_id"],
        "resolution_status": "resolved",
        "resolution_message": None,
    }


def _unresolved_company(
    mention: str,
    status: Literal["ambiguous", "unsupported"],
    message: str,
) -> ResolvedCompany:
    """Create an identity-empty ambiguous or unsupported resolver result.

    Args:
        mention: Original user mention.
        status: Ambiguous or unsupported outcome.
        message: Developer/user-facing explanation.

    Returns:
        ``ResolvedCompany``-shaped failure value for uniform downstream handling.
    """
    return {
        "company_id": "",
        "ticker": "",
        "company_name": mention.strip(),
        "aliases": [],
        "exchange": None,
        "industry": "",
        "sub_industry": "",
        "profile_id": "",
        "resolution_status": status,
        "resolution_message": message,
    }


def _ticker_tokens_in_mention(normalized_mention: str) -> set[str]:
    """Find explicit supported ticker tokens inside a normalized mention.

    Args:
        normalized_mention: Output of ``_normalize_company_text``.

    Returns:
        Canonical ticker candidates occurring as complete tokens.
    """
    padded = f" {normalized_mention} "
    return {
        ticker
        for normalized_ticker, ticker in NORMALIZED_TICKERS.items()
        if f" {normalized_ticker} " in padded
    }


def _alias_candidates(normalized_mention: str) -> set[str]:
    """Find registry tickers matching a normalized company alias.

    Args:
        normalized_mention: Output of ``_normalize_company_text``.

    Returns:
        Candidate ticker set; zero, one, or many drives the resolver outcome.
    """
    padded = f" {normalized_mention} "
    candidates: set[str] = set()
    for alias, tickers in COMPANY_ALIAS_INDEX.items():
        if normalized_mention == alias or f" {alias} " in padded:
            candidates.update(tickers)
    return candidates


def resolve_company_mention(mention: str) -> ResolvedCompany:
    """Resolve one planner-produced company mention without using an LLM or network call."""
    normalized = _normalize_company_text(mention)
    if not normalized:
        return _unresolved_company(
            mention,
            "unsupported",
            "Company mention is empty.",
        )

    # An explicit ticker wins over a broader name in the same mention, e.g. "Roche (RHHBY)".
    ticker_tokens = _ticker_tokens_in_mention(normalized)
    if len(ticker_tokens) == 1:
        ticker = next(iter(ticker_tokens))
        return _resolved_company(COMPANY_REGISTRY[ticker])
    if len(ticker_tokens) > 1:
        choices = ", ".join(sorted(ticker_tokens))
        return _unresolved_company(
            mention,
            "ambiguous",
            f"Mention contains multiple supported tickers: {choices}.",
        )

    candidates = _alias_candidates(normalized)
    if len(candidates) == 1:
        ticker = next(iter(candidates))
        return _resolved_company(COMPANY_REGISTRY[ticker])
    if len(candidates) > 1:
        choices = ", ".join(sorted(candidates))
        return _unresolved_company(
            mention,
            "ambiguous",
            f"Ambiguous company mention '{mention.strip()}'. Candidates: {choices}.",
        )

    return _unresolved_company(
        mention,
        "unsupported",
        f"Company '{mention.strip()}' is not in the local supported-company registry.",
    )


def resolve_company_mentions(company_mentions: list[str]) -> list[ResolvedCompany]:
    """Resolve mentions in order and collapse duplicate resolved company identities."""
    results: list[ResolvedCompany] = []
    seen_company_ids: set[str] = set()
    seen_unresolved: set[tuple[str, str]] = set()

    for mention in company_mentions:
        resolved = resolve_company_mention(mention)
        if resolved["resolution_status"] == "resolved":
            company_id = resolved["company_id"]
            if company_id in seen_company_ids:
                continue
            seen_company_ids.add(company_id)
        else:
            unresolved_key = (
                resolved["resolution_status"],
                _normalize_company_text(mention),
            )
            if unresolved_key in seen_unresolved:
                continue
            seen_unresolved.add(unresolved_key)
        results.append(resolved)

    return results


def resolve_companies(plan: QueryPlan) -> list[ResolvedCompany]:
    """Resolve the company mentions from a validated query plan."""
    return resolve_company_mentions(plan["company_mentions"])


def validate_resolution_gate(results: list[ResolvedCompany]) -> dict[str, Any]:
    """Return the non-bypassable routing decision for a resolution result."""
    resolved = [item for item in results if item["resolution_status"] == "resolved"]
    ambiguous = [item for item in results if item["resolution_status"] == "ambiguous"]
    unsupported = [item for item in results if item["resolution_status"] == "unsupported"]

    if not results:
        status = "no_companies"
        message = "No company was identified. Ask the user to name at least one company."
    elif ambiguous:
        status = "needs_clarification"
        message = "One or more company mentions are ambiguous; clarification is required."
    elif unsupported:
        status = "unsupported"
        message = "One or more companies are outside the notebook's supported local registry."
    else:
        status = "ready"
        message = "All requested companies have canonical supported identities."

    return {
        "ready": status == "ready",
        "status": status,
        "resolved_companies": resolved,
        "ambiguous_companies": ambiguous,
        "unsupported_companies": unsupported,
        "message": message,
    }


@tool
def resolve_companies_tool(company_mentions: list[str]) -> dict[str, Any]:
    """Resolve user company mentions; this must succeed before research tools are used."""
    results = resolve_company_mentions(company_mentions)
    return validate_resolution_gate(results)


def route_after_resolution(resolution: dict[str, Any]) -> str:
    """Mandatory graph router used after the coordinator's resolver tool call."""
    if resolution.get("ready") is True:
        return "select_profiles"
    if resolution.get("status") == "needs_clarification":
        return "request_clarification"
    return "stop_unsupported"


print(
    "✅ F02 guarded resolver tool and mandatory gate defined "
    f"({len(COMPANY_REGISTRY)} supported listings)"
)
'''


F02_SMOKE = r'''# F02 local smoke test: no network or API keys required.
assert resolve_company_mention("Microsoft")["ticker"] == "MSFT"
assert resolve_company_mention("Pfizer")["profile_id"] == "healthcare.biopharma.v1"
assert resolve_company_mention("Roche")["resolution_status"] == "ambiguous"
assert resolve_company_mention("Roche (RHHBY)")["ticker"] == "RHHBY"
assert resolve_company_mention("Unknown Example Co")["resolution_status"] == "unsupported"

_f02_plan: QueryPlan = {
    "query_type": "compare",
    "company_mentions": ["MSFT", "Microsoft", "Pfizer"],
    "requested_dimensions": ["financial_strength"],
    "risk_profile": "balanced",
    "scoring_requested": False,
    "freshness_required": True,
    "time_horizon": None,
}
_f02_resolved = resolve_companies(_f02_plan)
assert [company["ticker"] for company in _f02_resolved] == ["MSFT", "PFE"]

_f02_tool_result = resolve_companies_tool.invoke(
    {"company_mentions": ["Microsoft", "Pfizer"]}
)
assert _f02_tool_result["ready"] is True
assert _f02_tool_result["status"] == "ready"
assert route_after_resolution(_f02_tool_result) == "select_profiles"
assert [company["ticker"] for company in _f02_tool_result["resolved_companies"]] == [
    "MSFT", "PFE"
]

_f02_ambiguous = resolve_companies_tool.invoke({"company_mentions": ["Roche"]})
assert _f02_ambiguous["status"] == "needs_clarification"
assert route_after_resolution(_f02_ambiguous) == "request_clarification"

_f02_unsupported = resolve_companies_tool.invoke(
    {"company_mentions": ["Unknown Example Co"]}
)
assert _f02_unsupported["status"] == "unsupported"
assert route_after_resolution(_f02_unsupported) == "stop_unsupported"

_f02_empty = validate_resolution_gate([])
assert _f02_empty["status"] == "no_companies"

print("✅ F02 smoke test passed: guarded tool output and mandatory routing are valid")
'''


CELL_SPECS = [
    ("multiindustry_f02_intro", "markdown", F02_INTRO),
    ("multiindustry_company_registry", "code", F02_CODE),
    ("multiindustry_f02_smoke", "code", F02_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    if cell_type == "markdown":
        cell = nbformat.v4.new_markdown_cell(source=source)
    else:
        cell = nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
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

    missing_specs = [spec for spec in CELL_SPECS if spec[0] not in cells_by_id]
    if missing_specs:
        insertion_index = next(
            index
            for index, cell in enumerate(notebook.cells)
            if cell.get("id") == INSERT_AFTER_CELL_ID
        ) + 1
        notebook.cells[insertion_index:insertion_index] = [
            _new_cell(*spec) for spec in missing_specs
        ]

    nbformat.validate(notebook)
    ids = [cell.get("id") for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell IDs")

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F02 cells are present")


if __name__ == "__main__":
    main()

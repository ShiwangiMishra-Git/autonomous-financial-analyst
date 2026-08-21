"""Idempotently add the F04 Industry Profile Registry to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f03_smoke"


F04_INTRO = """## Section 3.4: Industry Profile Registry

An industry profile is versioned configuration, not an LLM classification guess. It defines the
prompt scope, shared and sector-specific dimensions, permitted research-tool contracts, corpus,
extractor, scoring status, and synthesis policy for one supported company type.

The coordinator may call `select_industry_profiles_tool`, but the deterministic registry selects
the profile. `validate_profile_gate` then compares the selection with every company produced by
the resolver, preventing dropped companies, invented profiles, and expanded tool permissions.

F04 registers the explicit `query_technology_rag` and `query_biopharma_rag` contracts. Their
callable adapters arrive in F07 and F08; the future worker factory must fail closed until each
allowed contract is backed by a real callable.
"""


F04_CODE = r'''from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Literal, TypedDict

from langchain_core.tools import tool


ToolContractStatus = Literal["implemented", "planned_f07", "planned_f08"]


class IndustryProfile(TypedDict):
    """Versioned configuration that bounds one industry research playbook.

    Attributes:
        profile_id: Stable versioned identifier used by routing and evidence records.
        industry: Broad supported industry name.
        sub_industry: Narrow specialization supported by this playbook.
        worker_prompt: Profile-specific constraints appended to the generic worker charter.
        allowed_tools: Tool-contract names the future worker may bind.
        shared_dimensions: Cross-profile financial dimensions.
        industry_dimensions: Profile-specific research dimensions.
        rag_tool_name: Profile-owned retrieval contract.
        rag_collection: Local vector collection name.
        corpus_version: Version identifier for cache and provenance decisions.
        signal_extractor_name: Deterministic extractor contract for normalized evidence.
        rubric_id: Validated scoring rubric, or ``None`` when scoring is disabled.
        scoring_function_name: Deterministic scoring function, or ``None`` when unavailable.
        scoring_enabled: Whether this profile currently permits numeric scoring.
        synthesis_prompt: Profile-specific guidance for grounded narrative synthesis.
    """

    profile_id: str
    industry: str
    sub_industry: str
    worker_prompt: str
    allowed_tools: list[str]
    shared_dimensions: list[str]
    industry_dimensions: list[str]
    rag_tool_name: str
    rag_collection: str
    corpus_version: str
    signal_extractor_name: str
    rubric_id: str | None
    scoring_function_name: str | None
    scoring_enabled: bool
    synthesis_prompt: str


class ProfileSelection(TypedDict):
    """Deterministic output returned by the guarded profile-selection tool.

    Attributes:
        status: ``ready`` or ``unsupported``.
        profiles_by_company: Defensive profile copies keyed by canonical company ID.
        unknown_company_ids: Company IDs absent from the authoritative registry.
        message: Human-readable selection outcome.
    """

    status: Literal["ready", "unsupported"]
    profiles_by_company: dict[str, IndustryProfile]
    unknown_company_ids: list[str]
    message: str


PROFILE_TOOL_CONTRACTS: dict[str, ToolContractStatus] = {
    "get_stock_price": "implemented",
    "get_financial_metrics": "implemented",
    "get_stock_history": "implemented",
    "search_financial_news": "implemented",
    "analyze_sentiment": "implemented",
    "query_technology_rag": "implemented",
    "query_biopharma_rag": "implemented",
}

SHARED_FINANCIAL_DIMENSIONS = [
    "current_price",
    "market_cap",
    "total_revenue",
    "pe_ratio",
    "beta",
    "dividend_yield",
    "price_history",
    "news_sentiment",
]

TECHNOLOGY_TOOL_NAMES = [
    "get_stock_price",
    "get_financial_metrics",
    "get_stock_history",
    "search_financial_news",
    "analyze_sentiment",
    "query_technology_rag",
]

BIOPHARMA_TOOL_NAMES = [
    "get_stock_price",
    "get_financial_metrics",
    "get_stock_history",
    "search_financial_news",
    "analyze_sentiment",
    "query_biopharma_rag",
]


TECHNOLOGY_AI_PROFILE: IndustryProfile = {
    "profile_id": "technology.ai.v1",
    "industry": "technology",
    "sub_industry": "ai_platforms",
    "worker_prompt": (
        "Research the assigned technology company only. Ground AI infrastructure, product "
        "deployment, research depth, and strategic commitment claims in current-run evidence."
    ),
    "allowed_tools": TECHNOLOGY_TOOL_NAMES,
    "shared_dimensions": SHARED_FINANCIAL_DIMENSIONS,
    "industry_dimensions": [
        "infrastructure_moat",
        "product_deployment",
        "research_depth",
        "strategic_commitment",
    ],
    "rag_tool_name": "query_technology_rag",
    "rag_collection": "AI_Initiatives",
    "corpus_version": "ai_initiatives.local.v1",
    "signal_extractor_name": "extract_technology_signals_with_evidence",
    "rubric_id": "technology.ai.score.v1",
    "scoring_function_name": "score_technology_companies",
    "scoring_enabled": True,
    "synthesis_prompt": (
        "Explain technology and AI findings using validated evidence IDs and authoritative "
        "deterministic scores when scoring is eligible."
    ),
}


BIOPHARMA_PROFILE: IndustryProfile = {
    "profile_id": "healthcare.biopharma.v1",
    "industry": "healthcare",
    "sub_industry": "biopharma",
    "worker_prompt": (
        "Research the assigned biopharma company only. Keep clinical, regulatory, patent, "
        "commercialization, and sector-risk claims tied to official-source evidence."
    ),
    "allowed_tools": BIOPHARMA_TOOL_NAMES,
    "shared_dimensions": SHARED_FINANCIAL_DIMENSIONS,
    "industry_dimensions": [
        "clinical_pipeline",
        "regulatory_progress",
        "exclusivity_and_patents",
        "commercialization",
        "sector_risks",
    ],
    "rag_tool_name": "query_biopharma_rag",
    "rag_collection": "Biopharma_Official_Sources",
    "corpus_version": "pharma_official_sources.local.v1",
    "signal_extractor_name": "extract_pharma_signals",
    "rubric_id": "healthcare.biopharma.score.v1",
    "scoring_function_name": "score_biopharma_companies",
    "scoring_enabled": True,
    "synthesis_prompt": (
        "Explain biopharma findings from official-source evidence and state uncertainty or "
        "missing dimensions explicitly; do not invent a numeric sector score."
    ),
}


INDUSTRY_PROFILES: dict[str, IndustryProfile] = {
    TECHNOLOGY_AI_PROFILE["profile_id"]: TECHNOLOGY_AI_PROFILE,
    BIOPHARMA_PROFILE["profile_id"]: BIOPHARMA_PROFILE,
}


def _build_company_profile_index() -> dict[str, str]:
    """Build and validate the canonical company-to-profile mapping.

    Returns:
        Mapping from registry company ID to its single authoritative profile ID.

    Raises:
        ValueError: If two listings for one company declare different profiles.
    """
    index: dict[str, str] = {}
    for entry in COMPANY_REGISTRY.values():
        company_id = entry["company_id"]
        profile_id = entry["profile_id"]
        previous = index.get(company_id)
        if previous is not None and previous != profile_id:
            raise ValueError(
                f"Company {company_id!r} maps to conflicting profiles: {previous}, {profile_id}"
            )
        index[company_id] = profile_id
    return index


COMPANY_PROFILE_INDEX = _build_company_profile_index()


def validate_industry_profile_registry() -> list[str]:
    """Return all deterministic configuration errors in the profile registry.

    Returns:
        Empty list when profiles are versioned, internally consistent, and reference only
        registered tool contracts; otherwise a list of actionable error messages.
    """
    errors: list[str] = []
    versioned_id_pattern = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\.v\d+$")

    for key, profile in INDUSTRY_PROFILES.items():
        profile_id = profile["profile_id"]
        if key != profile_id:
            errors.append(f"Profile key {key!r} does not match profile_id {profile_id!r}")
        if not versioned_id_pattern.fullmatch(profile_id):
            errors.append(f"Profile ID is not versioned: {profile_id!r}")

        missing_contracts = sorted(set(profile["allowed_tools"]) - set(PROFILE_TOOL_CONTRACTS))
        if missing_contracts:
            errors.append(f"{profile_id} references unknown tool contracts: {missing_contracts}")
        if profile["rag_tool_name"] not in profile["allowed_tools"]:
            errors.append(f"{profile_id} does not allow its configured RAG tool")
        if not profile["shared_dimensions"] or not profile["industry_dimensions"]:
            errors.append(f"{profile_id} must define shared and industry dimensions")
        if len(profile["allowed_tools"]) != len(set(profile["allowed_tools"])):
            errors.append(f"{profile_id} contains duplicate tool contracts")
        if profile["scoring_enabled"] and not (
            profile["rubric_id"] and profile["scoring_function_name"]
        ):
            errors.append(f"{profile_id} enables scoring without a rubric and scoring function")
        if profile["rubric_id"] and not profile["scoring_function_name"]:
            errors.append(f"{profile_id} declares a rubric without a scoring function")

    technology_tools = set(TECHNOLOGY_AI_PROFILE["allowed_tools"])
    biopharma_tools = set(BIOPHARMA_PROFILE["allowed_tools"])
    if "query_biopharma_rag" in technology_tools:
        errors.append("Technology profile must not allow biopharma RAG")
    if "query_technology_rag" in biopharma_tools:
        errors.append("Biopharma profile must not allow technology RAG")
    return errors


def get_industry_profile(profile_id: str) -> IndustryProfile:
    """Return a defensive copy of one supported industry profile.

    Args:
        profile_id: Exact versioned profile identifier.

    Returns:
        Independent profile mapping safe for task-specific use.

    Raises:
        KeyError: If the profile is not supported by this notebook.
    """
    try:
        return deepcopy(INDUSTRY_PROFILES[profile_id])
    except KeyError as exc:
        raise KeyError(f"Unsupported industry profile: {profile_id!r}") from exc


def attach_industry_profiles(
    companies: list[ResolvedCompany],
) -> list[ResolvedCompany]:
    """Validate resolved-company profile IDs against the authoritative registries.

    Args:
        companies: Canonical resolution results that must all be successful.

    Returns:
        Defensive copies of the validated companies, preserving input order.

    Raises:
        ValueError: If a company is unresolved, unknown, or carries a swapped profile ID.
        KeyError: If a referenced profile is unsupported.
    """
    attached: list[ResolvedCompany] = []
    for company in companies:
        if company["resolution_status"] != "resolved":
            raise ValueError(
                f"Cannot attach a profile to {company['company_name']!r}: "
                f"resolution status is {company['resolution_status']!r}"
            )
        profile_id = company["profile_id"]
        get_industry_profile(profile_id)
        expected_profile_id = COMPANY_PROFILE_INDEX.get(company["company_id"])
        if expected_profile_id is None:
            raise ValueError(f"Unknown canonical company ID: {company['company_id']!r}")
        if profile_id != expected_profile_id:
            raise ValueError(
                f"Company {company['company_id']!r} requires {expected_profile_id!r}, "
                f"not {profile_id!r}"
            )
        attached.append(deepcopy(company))
    return attached


def select_industry_profiles(company_ids: list[str]) -> ProfileSelection:
    """Select authoritative profiles for canonical company IDs without using an LLM.

    Args:
        company_ids: Canonical IDs returned by successful company resolution.

    Returns:
        Structured selection containing defensive profile copies and unknown IDs.
    """
    profiles_by_company: dict[str, IndustryProfile] = {}
    unknown_company_ids: list[str] = []
    seen: set[str] = set()

    for raw_company_id in company_ids:
        company_id = raw_company_id.strip().casefold()
        if not company_id or company_id in seen:
            continue
        seen.add(company_id)
        profile_id = COMPANY_PROFILE_INDEX.get(company_id)
        if profile_id is None:
            unknown_company_ids.append(raw_company_id)
            continue
        profiles_by_company[company_id] = get_industry_profile(profile_id)

    status: Literal["ready", "unsupported"] = (
        "ready" if profiles_by_company and not unknown_company_ids else "unsupported"
    )
    message = (
        "All canonical companies have supported industry profiles."
        if status == "ready"
        else "One or more canonical company IDs do not have a supported profile."
    )
    return {
        "status": status,
        "profiles_by_company": profiles_by_company,
        "unknown_company_ids": unknown_company_ids,
        "message": message,
    }


@tool
def select_industry_profiles_tool(company_ids: list[str]) -> dict[str, Any]:
    """Select registry-backed profiles for resolved canonical company IDs.

    The coordinator may invoke this guarded tool, but cannot provide custom profile content or
    tool permissions. A mandatory graph gate must compare the result with resolved run state.

    Args:
        company_ids: Canonical company IDs produced by ``resolve_companies_tool``.

    Returns:
        Deterministic profile selection and unsupported-company details.
    """
    return select_industry_profiles(company_ids)


def validate_profile_gate(
    selection: ProfileSelection,
    expected_company_ids: list[str],
) -> dict[str, Any]:
    """Validate profile selection against every company resolved for the current run.

    Args:
        selection: Result returned by the guarded profile-selection tool.
        expected_company_ids: Canonical IDs from validated resolution state.

    Returns:
        Mandatory routing result with ``ready``, ``status``, and explanatory ``message``.
    """
    expected = {company_id.strip().casefold() for company_id in expected_company_ids if company_id.strip()}
    selected = set(selection.get("profiles_by_company", {}))
    errors: list[str] = []

    if not expected:
        errors.append("No resolved companies are available for profile selection")
    if selection.get("status") != "ready":
        errors.append(selection.get("message", "Profile selection is not ready"))
    missing = sorted(expected - selected)
    unexpected = sorted(selected - expected)
    if missing:
        errors.append(f"Missing profiles for company IDs: {missing}")
    if unexpected:
        errors.append(f"Unexpected profile selections: {unexpected}")

    for company_id, profile in selection.get("profiles_by_company", {}).items():
        expected_profile_id = COMPANY_PROFILE_INDEX.get(company_id)
        if expected_profile_id != profile.get("profile_id"):
            errors.append(
                f"Company {company_id!r} has invalid profile {profile.get('profile_id')!r}"
            )

    return {
        "ready": not errors,
        "status": "ready" if not errors else "invalid_profile_selection",
        "message": "Profile selection passed mandatory validation." if not errors else "; ".join(errors),
        "errors": errors,
    }


def route_after_profile_selection(profile_gate: dict[str, Any]) -> str:
    """Route a validated profile selection to task construction or a bounded stop.

    Args:
        profile_gate: Output from ``validate_profile_gate``.

    Returns:
        ``build_tasks`` when ready; otherwise ``stop_unsupported``.
    """
    return "build_tasks" if profile_gate.get("ready") is True else "stop_unsupported"


_profile_registry_errors = validate_industry_profile_registry()
if _profile_registry_errors:
    raise ValueError("Invalid industry profile registry: " + "; ".join(_profile_registry_errors))

print(
    "✅ F04 Industry Profile Registry defined "
    f"({len(INDUSTRY_PROFILES)} profiles, {len(PROFILE_TOOL_CONTRACTS)} tool contracts)"
)
'''


F04_SMOKE = r'''# F04 local smoke test: configuration and routing only; no network or API keys.
assert validate_industry_profile_registry() == []

_f04_technology = get_industry_profile("technology.ai.v1")
_f04_biopharma = get_industry_profile("healthcare.biopharma.v1")
assert _f04_technology["rag_tool_name"] == "query_technology_rag"
assert _f04_biopharma["rag_tool_name"] == "query_biopharma_rag"
assert _f04_technology["scoring_enabled"] is True
assert _f04_biopharma["scoring_enabled"] is True
assert _f04_biopharma["rubric_id"] == "healthcare.biopharma.score.v1"

_f04_selection = select_industry_profiles_tool.invoke(
    {"company_ids": ["microsoft", "pfizer"]}
)
assert _f04_selection["status"] == "ready"
assert _f04_selection["profiles_by_company"]["microsoft"]["profile_id"] == "technology.ai.v1"
assert _f04_selection["profiles_by_company"]["pfizer"]["profile_id"] == "healthcare.biopharma.v1"

_f04_gate = validate_profile_gate(_f04_selection, ["microsoft", "pfizer"])
assert _f04_gate["ready"] is True
assert route_after_profile_selection(_f04_gate) == "build_tasks"

_f04_dropped_company_gate = validate_profile_gate(_f04_selection, ["microsoft", "pfizer", "merck"])
assert _f04_dropped_company_gate["ready"] is False
assert route_after_profile_selection(_f04_dropped_company_gate) == "stop_unsupported"

print("✅ F04 smoke test passed: profile contracts, guarded selection, and mandatory gate work")
'''


CELL_SPECS = [
    ("multiindustry_f04_intro", "markdown", F04_INTRO),
    ("multiindustry_industry_profiles", "code", F04_CODE),
    ("multiindustry_f04_smoke", "code", F04_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create a notebook cell with a stable ID and the requested source."""
    if cell_type == "markdown":
        cell = nbformat.v4.new_markdown_cell(source=source)
    else:
        cell = nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    """Insert or refresh the F04 cells in the canonical working notebook."""
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
    print(f"Updated {NOTEBOOK_PATH.name}: F04 cells are present")


if __name__ == "__main__":
    main()

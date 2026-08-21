"""Deterministic tests for the F02 local company registry and resolver."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _resolver_namespace():
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}

    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
        exec(cells["multiindustry_company_registry"], namespace)
    return namespace


def _plan(*mentions):
    return {
        "query_type": "compare",
        "company_mentions": list(mentions),
        "requested_dimensions": [],
        "risk_profile": "balanced",
        "scoring_requested": False,
        "freshness_required": True,
        "time_horizon": None,
    }


def test_registry_contains_expected_technology_and_biopharma_universe():
    namespace = _resolver_namespace()
    registry = namespace["COMPANY_REGISTRY"]

    assert {"MSFT", "GOOGL", "NVDA", "AMZN", "IBM"} <= set(registry)
    assert {
        "LLY", "MRK", "JNJ", "PFE", "ABBV", "AZN", "RHHBY", "ROG.SW",
        "NVS", "BMY", "GSK", "AMGN", "NVO", "SNY", "TAK",
    } <= set(registry)


def test_ticker_and_name_alias_resolve_to_same_company():
    namespace = _resolver_namespace()
    resolve = namespace["resolve_company_mention"]

    ticker = resolve("MSFT")
    name = resolve("Microsoft")
    full_name = resolve("Microsoft Corporation")

    assert ticker["company_id"] == name["company_id"] == full_name["company_id"]
    assert ticker["ticker"] == "MSFT"


def test_biopharma_company_receives_healthcare_profile():
    namespace = _resolver_namespace()
    pfizer = namespace["resolve_company_mention"]("Pfizer Inc.")

    assert pfizer["ticker"] == "PFE"
    assert pfizer["industry"] == "healthcare"
    assert pfizer["sub_industry"] == "biopharma"
    assert pfizer["profile_id"] == "healthcare.biopharma.v1"


def test_alias_normalization_handles_ampersands_hyphens_and_case():
    namespace = _resolver_namespace()
    resolve = namespace["resolve_company_mention"]

    assert resolve("J&J")["ticker"] == "JNJ"
    assert resolve("bristol-myers squibb")["ticker"] == "BMY"
    assert resolve("ASTRA ZENECA")["ticker"] == "AZN"


def test_bare_roche_is_ambiguous_but_explicit_listing_resolves():
    namespace = _resolver_namespace()
    resolve = namespace["resolve_company_mention"]

    ambiguous = resolve("Roche")
    adr = resolve("Roche (RHHBY)")
    swiss = resolve("ROG.SW")

    assert ambiguous["resolution_status"] == "ambiguous"
    assert "RHHBY" in ambiguous["resolution_message"]
    assert "ROG.SW" in ambiguous["resolution_message"]
    assert adr["ticker"] == "RHHBY"
    assert swiss["ticker"] == "ROG.SW"


def test_multiple_tickers_in_one_mention_are_not_silently_selected():
    namespace = _resolver_namespace()
    result = namespace["resolve_company_mention"]("MSFT and PFE")

    assert result["resolution_status"] == "ambiguous"
    assert "multiple supported tickers" in result["resolution_message"]


def test_unsupported_company_returns_explicit_status():
    namespace = _resolver_namespace()
    result = namespace["resolve_company_mention"]("Unknown Example Company")

    assert result["resolution_status"] == "unsupported"
    assert result["ticker"] == ""
    assert "not in the local supported-company registry" in result["resolution_message"]


def test_duplicate_ticker_and_alias_collapse_to_one_company():
    namespace = _resolver_namespace()
    resolved = namespace["resolve_companies"](_plan("MSFT", "Microsoft", "PFE", "Pfizer"))

    assert [company["ticker"] for company in resolved] == ["MSFT", "PFE"]


def test_mixed_industry_plan_preserves_input_order_and_profiles():
    namespace = _resolver_namespace()
    resolved = namespace["resolve_companies"](_plan("Pfizer", "Google", "Merck"))

    assert [company["ticker"] for company in resolved] == ["PFE", "GOOGL", "MRK"]
    assert [company["profile_id"] for company in resolved] == [
        "healthcare.biopharma.v1",
        "technology.ai.v1",
        "healthcare.biopharma.v1",
    ]


def test_duplicate_unresolved_mentions_are_collapsed():
    namespace = _resolver_namespace()
    resolved = namespace["resolve_companies"](_plan("Unknown Co", "unknown co"))

    assert len(resolved) == 1
    assert resolved[0]["resolution_status"] == "unsupported"


def test_resolver_is_agent_callable_but_matches_deterministic_core():
    namespace = _resolver_namespace()
    resolver_tool = namespace["resolve_companies_tool"]
    deterministic = namespace["resolve_company_mentions"](["Microsoft", "Pfizer"])

    tool_result = resolver_tool.invoke(
        {"company_mentions": ["Microsoft", "Pfizer"]}
    )

    assert resolver_tool.name == "resolve_companies_tool"
    assert tool_result["ready"] is True
    assert tool_result["status"] == "ready"
    assert tool_result["resolved_companies"] == deterministic


def test_resolution_gate_requires_clarification_for_ambiguous_company():
    namespace = _resolver_namespace()
    result = namespace["resolve_companies_tool"].invoke(
        {"company_mentions": ["Roche"]}
    )

    assert result["ready"] is False
    assert result["status"] == "needs_clarification"
    assert namespace["route_after_resolution"](result) == "request_clarification"
    assert [item["company_name"] for item in result["ambiguous_companies"]] == [
        "Roche"
    ]


def test_resolution_gate_stops_unsupported_company():
    namespace = _resolver_namespace()
    result = namespace["resolve_companies_tool"].invoke(
        {"company_mentions": ["Unknown Example Company"]}
    )

    assert result["ready"] is False
    assert result["status"] == "unsupported"
    assert namespace["route_after_resolution"](result) == "stop_unsupported"


def test_resolution_gate_rejects_empty_company_list():
    namespace = _resolver_namespace()
    result = namespace["validate_resolution_gate"]([])

    assert result["ready"] is False
    assert result["status"] == "no_companies"
    assert namespace["route_after_resolution"](result) == "stop_unsupported"

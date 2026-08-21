"""Deterministic tests for the F04 Industry Profile Registry and profile gate."""

from __future__ import annotations

import contextlib
from copy import deepcopy
from functools import lru_cache
import io
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


@lru_cache(maxsize=1)
def _profile_namespace():
    """Execute only the local contract, resolver, and profile cells for deterministic tests."""
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}

    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["multiindustry_state_contracts"], namespace)
        exec(cells["multiindustry_company_registry"], namespace)
        exec(cells["multiindustry_industry_profiles"], namespace)
    return namespace


def test_registry_contains_unique_versioned_profiles():
    """Require the two initial profiles and valid versioned identifiers."""
    namespace = _profile_namespace()
    profiles = namespace["INDUSTRY_PROFILES"]

    assert set(profiles) == {"technology.ai.v1", "healthcare.biopharma.v1"}
    assert len(profiles) == len({profile["profile_id"] for profile in profiles.values()})
    assert namespace["validate_industry_profile_registry"]() == []


def test_profile_tool_contracts_are_registered_and_sector_isolated():
    """Ensure each allowlist references known contracts and only its own RAG tool."""
    namespace = _profile_namespace()
    contracts = namespace["PROFILE_TOOL_CONTRACTS"]
    technology = namespace["TECHNOLOGY_AI_PROFILE"]
    biopharma = namespace["BIOPHARMA_PROFILE"]

    assert set(technology["allowed_tools"]) <= set(contracts)
    assert set(biopharma["allowed_tools"]) <= set(contracts)
    assert "query_technology_rag" in technology["allowed_tools"]
    assert "query_biopharma_rag" not in technology["allowed_tools"]
    assert "query_biopharma_rag" in biopharma["allowed_tools"]
    assert "query_technology_rag" not in biopharma["allowed_tools"]
    assert contracts["query_technology_rag"] == "implemented"
    assert contracts["query_biopharma_rag"] == "implemented"


def test_profiles_define_dimensions_and_guard_scoring_configuration():
    """Require useful dimensions and a complete rubric/function pair when scoring is enabled."""
    namespace = _profile_namespace()
    technology = namespace["TECHNOLOGY_AI_PROFILE"]
    biopharma = namespace["BIOPHARMA_PROFILE"]

    for profile in (technology, biopharma):
        assert profile["shared_dimensions"]
        assert profile["industry_dimensions"]
        if profile["scoring_enabled"]:
            assert profile["rubric_id"]
            assert profile["scoring_function_name"]

    assert technology["scoring_enabled"] is True
    assert biopharma["scoring_enabled"] is True
    assert biopharma["rubric_id"] == "healthcare.biopharma.score.v1"
    assert biopharma["scoring_function_name"] == "score_biopharma_companies"


def test_profile_accessor_returns_a_defensive_copy_and_rejects_unknown_id():
    """Prevent callers from mutating registry configuration or inventing a profile."""
    namespace = _profile_namespace()
    get_profile = namespace["get_industry_profile"]

    profile = get_profile("technology.ai.v1")
    profile["allowed_tools"].append("invented_tool")
    assert "invented_tool" not in namespace["TECHNOLOGY_AI_PROFILE"]["allowed_tools"]

    with pytest.raises(KeyError, match="Unsupported industry profile"):
        get_profile("technology.quantum.v99")


def test_attach_profiles_rejects_swapped_company_profile():
    """Ensure company-profile identity is determined by the registry, not the caller."""
    namespace = _profile_namespace()
    microsoft = namespace["resolve_company_mention"]("Microsoft")
    pfizer = namespace["resolve_company_mention"]("Pfizer")

    attached = namespace["attach_industry_profiles"]([microsoft, pfizer])
    assert [company["profile_id"] for company in attached] == [
        "technology.ai.v1",
        "healthcare.biopharma.v1",
    ]

    swapped = deepcopy(microsoft)
    swapped["profile_id"] = "healthcare.biopharma.v1"
    with pytest.raises(ValueError, match="requires 'technology.ai.v1'"):
        namespace["attach_industry_profiles"]([swapped])


def test_guarded_selector_returns_mixed_profiles_deterministically():
    """Verify the agent-callable wrapper matches registry-backed pure selection."""
    namespace = _profile_namespace()
    expected = namespace["select_industry_profiles"](["microsoft", "pfizer"])
    actual = namespace["select_industry_profiles_tool"].invoke(
        {"company_ids": ["microsoft", "pfizer"]}
    )

    assert actual == expected
    assert actual["profiles_by_company"]["microsoft"]["profile_id"] == "technology.ai.v1"
    assert actual["profiles_by_company"]["pfizer"]["profile_id"] == "healthcare.biopharma.v1"


def test_profile_gate_blocks_unknown_dropped_and_unexpected_companies():
    """Prove the mandatory gate prevents incomplete or expanded selections."""
    namespace = _profile_namespace()
    select = namespace["select_industry_profiles"]
    validate = namespace["validate_profile_gate"]
    route = namespace["route_after_profile_selection"]

    ready = validate(select(["microsoft", "pfizer"]), ["microsoft", "pfizer"])
    assert ready["ready"] is True
    assert route(ready) == "build_tasks"

    unknown = validate(select(["microsoft", "unknown_company"]), ["microsoft", "unknown_company"])
    assert unknown["ready"] is False
    assert route(unknown) == "stop_unsupported"

    dropped = validate(select(["microsoft"]), ["microsoft", "pfizer"])
    assert dropped["ready"] is False
    assert "Missing profiles" in dropped["message"]

    expanded_selection = select(["microsoft", "pfizer"])
    expanded = validate(expanded_selection, ["microsoft"])
    assert expanded["ready"] is False
    assert "Unexpected profile selections" in expanded["message"]

"""
Tests for the sector-parameterization feature: create_financial_agent(..., sector=...)
lets the charter prompts target a different industry (e.g. "Healthcare", "Fintech")
instead of the hardcoded "AI" framing, without touching the tools (which were already
sector-agnostic - they just take a ticker/company/query string).

These tests read the charter prompt strings directly out of the designated working notebook
(`Autonomous_financial_analyst_Learners_Notebook copy.ipynb`) rather than mirroring source, since
the thing worth guarding here is the actual prose ("did a {sector} placeholder get left
in, or a hardcoded 'AI' slip back in") - reading the live source means this test reflects
the true current prompt content automatically, instead of drifting against a stale mirror.
create_financial_agent itself is NOT executed here (it needs a full ChatOpenAI/tool/graph
environment) - the signature and the one-line `.replace("{sector}", sector)` wiring are
checked via source-text assertions instead, consistent with keeping unit tests free of
real API calls.
"""
import contextlib
import io
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


def _load_notebook_cells():
    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        nb = json.load(f)
    return {c.get("id"): "".join(c.get("source", [])) for c in nb["cells"]}


def _charter_prompts():
    """Exec the prompt-defining cells in isolation and return the resulting string constants."""
    cells = _load_notebook_cells()
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["3b04e889"], namespace)  # TRADITIONAL_PROMPT
        exec(cells["0695fec4"], namespace)  # AGENT_CHARTER_BASIC
        exec(cells["8b8ed6c2"], namespace)  # AGENT_CHARTER_FULL
    return namespace["TRADITIONAL_PROMPT"], namespace["AGENT_CHARTER_BASIC"], namespace["AGENT_CHARTER_FULL"]


def test_default_sector_ai_reproduces_original_hardcoded_text():
    _, basic, full = _charter_prompts()

    full_ai = full.replace("{sector}", "AI")
    basic_ai = basic.replace("{sector}", "AI")

    assert "You are an autonomous Financial Research Analyst Agent specializing in AI sector investments." in full_ai
    assert "(especially AI-focused)" in full_ai
    assert "• AI Research Activity: Current AI projects and innovations" in full_ai
    assert "(e.g. AI Research Activity)" in full_ai
    assert "5. AI Research Activity (verified presence/absence)" in full_ai
    assert "You are an autonomous Financial Research Analyst specializing in AI-focused companies." in basic_ai


def test_traditional_prompt_has_no_sector_placeholder():
    traditional, _, _ = _charter_prompts()
    assert "{sector}" not in traditional
    # Replacing on a prompt with no placeholder must be a safe no-op.
    assert traditional.replace("{sector}", "Healthcare") == traditional


def test_other_sector_substitutes_cleanly_with_no_leftover_placeholder():
    _, basic, full = _charter_prompts()

    for sector in ["Healthcare", "Fintech"]:
        full_sub = full.replace("{sector}", sector)
        basic_sub = basic.replace("{sector}", sector)

        assert "{sector}" not in full_sub
        assert "{sector}" not in basic_sub
        assert "AI" not in full_sub.split("\n")[0]  # mission statement no longer says AI
        assert f"specializing in {sector} sector investments" in full_sub
        assert f"(especially {sector}-focused)" in full_sub
        assert f"• {sector} Research Activity: Current {sector} projects and innovations" in full_sub
        assert f"(e.g. {sector} Research Activity)" in full_sub
        assert f"5. {sector} Research Activity (verified presence/absence)" in full_sub
        assert f"specializing in {sector}-focused companies" in basic_sub


def test_create_financial_agent_signature_has_sector_parameter_defaulting_to_ai():
    cells = _load_notebook_cells()
    src = cells["b78756d2"]
    assert (
        'def create_financial_agent(agent_type: str = "full", with_memory: bool = True, sector: str = "AI"):'
        in src
    )
    assert 'prompt_map.get(agent_type, AGENT_CHARTER_FULL).replace("{sector}", sector)' in src

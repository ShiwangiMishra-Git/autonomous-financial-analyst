"""Deterministic tests for F16's notebook-order-independent source-tool bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

import scripts.f16_live_tools as live_tools


class _Retriever:
    """Return one bounded technology document for local tool tests."""

    def invoke(self, query: str) -> list[Document]:
        """Return a current fake document without provider access."""
        return [Document(
            page_content="Microsoft operates a documented AI platform initiative.",
            metadata={"source": "fixture.pdf", "page": 2},
        )]


def test_bootstrap_defines_self_contained_tools_without_legacy_notebook_globals(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Missing ``Dict``/cache/decorator globals must not affect F16 tool construction."""
    monkeypatch.setattr(live_tools, "_technology_retriever", lambda namespace, root: _Retriever())
    namespace: dict[str, Any] = {}

    status = live_tools.bootstrap_live_source_tools(namespace, project_root=tmp_path)

    assert status["technology_rag_ready"] is True
    assert set(status["installed_tool_names"]) == set(live_tools.BOOTSTRAPPED_TOOL_NAMES)
    for name in live_tools.BOOTSTRAPPED_TOOL_NAMES:
        assert callable(getattr(namespace[name], "invoke", None))


def test_financial_metrics_tool_returns_exact_f13_contract(monkeypatch, tmp_path: Path) -> None:
    """Ranking metrics must use canonical names and one explicit success status."""
    monkeypatch.setattr(live_tools, "_technology_retriever", lambda namespace, root: None)
    monkeypatch.setattr(live_tools, "_yahoo_info", lambda ticker: {
        "marketCap": 3_000_000_000,
        "totalRevenue": 2_000_000_000,
        "trailingPE": 20.5,
        "beta": 1.1,
        "dividendYield": 0.01,
    })
    namespace: dict[str, Any] = {}
    live_tools.bootstrap_live_source_tools(namespace, project_root=tmp_path)

    result = namespace["get_financial_metrics"].invoke({"ticker": "msft"})

    assert result["ticker"] == "MSFT"
    assert result["status"] == "success"
    assert set((
        "market_cap", "total_revenue", "pe_ratio", "beta", "dividend_yield",
    )).issubset(result)


def test_financial_metrics_tool_fails_closed_on_missing_provider_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The live demo must not fabricate a missing beta or dividend value for F13."""
    monkeypatch.setattr(live_tools, "_technology_retriever", lambda namespace, root: None)
    monkeypatch.setattr(live_tools, "_yahoo_info", lambda ticker: {
        "marketCap": 3_000_000_000,
        "totalRevenue": 2_000_000_000,
        "trailingPE": 20.5,
    })
    namespace: dict[str, Any] = {}
    live_tools.bootstrap_live_source_tools(namespace, project_root=tmp_path)

    result = namespace["get_financial_metrics"].invoke({"ticker": "MSFT"})

    assert result["status"] == "missing"
    assert "beta" in result["error"]
    assert "dividend_yield" in result["error"]


def test_technology_rag_returns_bounded_provenance(monkeypatch, tmp_path: Path) -> None:
    """The restored retriever must produce structured ticker-owned evidence."""
    monkeypatch.setattr(live_tools, "_technology_retriever", lambda namespace, root: _Retriever())
    namespace: dict[str, Any] = {}
    live_tools.bootstrap_live_source_tools(namespace, project_root=tmp_path)

    result = namespace["query_technology_rag"].invoke({
        "ticker": "MSFT",
        "query": "What AI platform initiative is documented?",
    })

    assert result["status"] == "success"
    assert result["ticker"] == "MSFT"
    assert result["collection"] == "AI_Initiatives"
    assert result["sources"] == [{"source": "fixture.pdf", "page": 2}]


"""Self-contained provider tools used by the notebook's F16 free-text demo.

The original course cells remain intact, but they were written to be executed top-to-bottom and
therefore depend on notebook globals such as ``Dict``, cache decorators, and initialized clients.
This module gives the final demo import-safe tool contracts with the same provider behavior.  It
performs no network request at import time and never reads ``config.json``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping

from langchain_core.tools import StructuredTool


BOOTSTRAPPED_TOOL_NAMES = (
    "get_stock_price",
    "get_financial_metrics",
    "get_stock_history",
    "search_financial_news",
    "analyze_sentiment",
    "query_technology_rag",
)


def _ticker(value: str) -> str:
    """Normalize one ticker.

    Args:
        value: User- or agent-supplied ticker.

    Returns:
        Uppercase non-empty ticker suitable for provider calls.

    Usage:
        Called inside every ticker-scoped source tool before external access.
    """
    ticker = str(value or "").strip().upper()
    if not ticker:
        raise ValueError("ticker must be a non-empty string")
    return ticker


def _yahoo_info(ticker: str) -> dict[str, Any]:
    """Fetch one Yahoo Finance company-information snapshot.

    Args:
        ticker: Canonical uppercase ticker.

    Returns:
        Provider dictionary used by price and canonical-metric tools.

    Usage:
        Keeps provider-specific field names inside the source-tool boundary.
    """
    import yfinance as yf

    return dict(yf.Ticker(ticker).info)


def _stock_price(ticker: str) -> dict[str, Any]:
    """Fetch current price and basic market data for one company.

    Args:
        ticker: Public-market ticker such as ``MSFT``.

    Returns:
        Status-bearing dictionary with price, range, volume, market cap, and timestamp.

    Usage:
        The company worker calls this for current-price evidence; it is not the F13 metric record.
    """
    symbol = _ticker(ticker)
    try:
        info = _yahoo_info(symbol)
        current = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if current is None:
            return {"ticker": symbol, "status": "missing", "error": "Current price unavailable"}
        return {
            "ticker": symbol,
            "current_price": round(float(current), 2),
            "currency": info.get("currency", "USD"),
            "day_high": info.get("dayHigh", info.get("regularMarketDayHigh")),
            "day_low": info.get("dayLow", info.get("regularMarketDayLow")),
            "volume": info.get("volume", info.get("regularMarketVolume")),
            "market_cap": info.get("marketCap"),
            "company_name": info.get("longName", info.get("shortName")),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
        }
    except Exception as exc:
        return {"ticker": symbol, "status": "failed", "error": str(exc)}


def _financial_metrics(ticker: str) -> dict[str, Any]:
    """Fetch the exact five-metric F13 scoring contract for one company.

    Args:
        ticker: Public-market ticker such as ``PFE``.

    Returns:
        Status-bearing dictionary containing ``market_cap``, ``total_revenue``, ``pe_ratio``,
        ``beta``, and ``dividend_yield`` with canonical names.

    Usage:
        Same-profile ranking requires one successful record from this tool per company. Missing
        values fail scoring closed instead of being guessed by an LLM.
    """
    symbol = _ticker(ticker)
    try:
        info = _yahoo_info(symbol)
        metrics = {
            "market_cap": info.get("marketCap"),
            "total_revenue": info.get("totalRevenue"),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
        }
        missing = [name for name, value in metrics.items() if value is None]
        if missing:
            return {
                "ticker": symbol,
                **metrics,
                "status": "missing",
                "error": "Missing canonical metrics: " + ", ".join(missing),
            }
        return {
            "ticker": symbol,
            **{name: float(value) for name, value in metrics.items()},
            "as_of": datetime.now(timezone.utc).isoformat(),
            "status": "success",
        }
    except Exception as exc:
        return {"ticker": symbol, "status": "failed", "error": str(exc)}


def _stock_history(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Summarize historical price performance for one ticker.

    Args:
        ticker: Public-market ticker.
        period: Yahoo Finance period such as ``1y`` or ``3y``.

    Returns:
        Status-bearing start/end, return, range, volume, and observation counts.

    Usage:
        Workers call this when the plan requests price-history evidence.
    """
    import yfinance as yf

    symbol = _ticker(ticker)
    try:
        history = yf.Ticker(symbol).history(period=period)
        if history.empty:
            return {"ticker": symbol, "period": period, "status": "missing"}
        start_price = float(history["Close"].iloc[0])
        end_price = float(history["Close"].iloc[-1])
        return {
            "ticker": symbol,
            "period": period,
            "start_date": history.index[0].strftime("%Y-%m-%d"),
            "end_date": history.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "return_pct": round(((end_price - start_price) / start_price) * 100.0, 2),
            "high": round(float(history["High"].max()), 2),
            "low": round(float(history["Low"].min()), 2),
            "avg_volume": int(history["Volume"].mean()),
            "data_points": len(history),
            "status": "success",
        }
    except Exception as exc:
        return {"ticker": symbol, "period": period, "status": "failed", "error": str(exc)}


def _financial_news(query: str) -> list[dict[str, Any]]:
    """Search current financial news without exposing provider configuration.

    Args:
        query: Focused company or market-news search phrase.

    Returns:
        Up to five Tavily result dictionaries, or one explicit failure record.

    Usage:
        Workers choose this for freshness and news-sentiment dimensions.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    cleaned = str(query or "").strip()
    if not cleaned:
        raise ValueError("query must be a non-empty string")
    try:
        search = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        return list(search.invoke({"query": cleaned}))
    except Exception as exc:
        return [{"status": "failed", "error": str(exc)}]


def _sentiment(text: str) -> dict[str, Any]:
    """Classify financial text sentiment with a deterministic local fallback.

    Args:
        text: Headline, article excerpt, or short financial passage.

    Returns:
        Sentiment label, normalized score, confidence, reasoning, and status.

    Usage:
        Workers may call this after news retrieval; F13 does not treat it as scoring arithmetic.
    """
    from langchain_openai import ChatOpenAI

    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("text must be a non-empty string")
    try:
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE"),
        )
        response = model.invoke(
            "Return JSON with sentiment (positive|negative|neutral), score (0..1), confidence "
            f"(0..1), and brief reasoning for this financial text:\n{cleaned}"
        )
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(response.content).strip())
        result = dict(json.loads(content))
        result["status"] = "success"
        return result
    except Exception:
        positive = sum(word in cleaned.casefold() for word in ("growth", "profit", "gain", "strong"))
        negative = sum(word in cleaned.casefold() for word in ("loss", "decline", "weak", "risk"))
        sentiment = "positive" if positive > negative else "negative" if negative > positive else "neutral"
        score = 0.6 if sentiment == "positive" else 0.4 if sentiment == "negative" else 0.5
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": 0.6,
            "reasoning": "Keyword fallback used because model sentiment was unavailable.",
            "status": "success",
        }


def _technology_retriever(namespace: Mapping[str, Any], project_root: Path) -> Any | None:
    """Reuse or restore the local technology retriever without rebuilding its index.

    Args:
        namespace: Current notebook globals, possibly containing ``retriever``.
        project_root: Repository directory containing ``content/vectorstore``.

    Returns:
        Retriever object, or ``None`` when credentials/index are unavailable.

    Usage:
        Called once by the F16 bootstrap; it never creates duplicate embeddings.
    """
    existing = namespace.get("retriever")
    if existing is not None:
        return existing
    vector_dir = project_root / "content" / "vectorstore"
    if not vector_dir.exists() or not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE"),
        )
        store = Chroma(
            collection_name="AI_Initiatives",
            embedding_function=embeddings,
            persist_directory=str(vector_dir),
        )
        return store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    except Exception:
        return None


def _technology_rag_tool(retriever: Any | None):
    """Create the ticker-scoped technology RAG tool over one restored retriever.

    Args:
        retriever: Configured local retriever, or ``None`` for explicit missing results.

    Returns:
        Structured LangChain tool named ``query_technology_rag``.

    Usage:
        Bound only to technology-profile workers by the profile allowlist.
    """
    def query_technology_rag(ticker: str, query: str) -> dict[str, Any]:
        """Retrieve bounded local Technology/AI evidence for one ticker and question."""
        symbol = _ticker(ticker)
        cleaned = str(query or "").strip()
        if not cleaned:
            raise ValueError("query must be a non-empty string")
        if retriever is None:
            return {
                "ticker": symbol,
                "status": "missing",
                "error": "Technology RAG index is not configured",
            }
        scoped = f"Company ticker {symbol}. {cleaned}"
        try:
            if hasattr(retriever, "invoke"):
                documents = retriever.invoke(scoped)
            else:
                documents = retriever.get_relevant_documents(scoped)
            excerpts = []
            sources = []
            for document in list(documents or [])[:6]:
                content = str(getattr(document, "page_content", "")).strip()
                metadata = dict(getattr(document, "metadata", {}) or {})
                if content:
                    excerpts.append(content[:1600])
                sources.append({
                    key: metadata.get(key)
                    for key in ("source", "document_name", "page", "ticker", "company_id")
                    if metadata.get(key) is not None
                })
            return {
                "ticker": symbol,
                "status": "success" if excerpts else "missing",
                "data": "\n\n".join(excerpts),
                "sources": sources,
                "collection": "AI_Initiatives",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            return {"ticker": symbol, "status": "failed", "error": str(exc)}

    return StructuredTool.from_function(
        query_technology_rag,
        name="query_technology_rag",
        description=(
            "Retrieve local AI-initiative evidence for exactly one supported technology ticker. "
            "Inputs: ticker and focused query. Output: status, bounded evidence, and provenance."
        ),
    )


def bootstrap_live_source_tools(
    namespace: MutableMapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Install import-safe live tools into the current notebook namespace.

    Args:
        namespace: Mutable notebook globals receiving tool objects and restored retriever.
        project_root: Optional repository root; defaults to the parent of this module's folder.

    Returns:
        Presence-only status with installed tool names and technology-RAG readiness.

    Usage:
        The F16 setup cell calls this once. Existing course tool cells may still be run, but the
        end-to-end demo no longer depends on their execution order.
    """
    if not isinstance(namespace, MutableMapping):
        raise TypeError("namespace must be a mutable mapping")
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    retriever = _technology_retriever(namespace, root)
    if retriever is not None:
        namespace["retriever"] = retriever

    functions = {
        "get_stock_price": _stock_price,
        "get_financial_metrics": _financial_metrics,
        "get_stock_history": _stock_history,
        "search_financial_news": _financial_news,
        "analyze_sentiment": _sentiment,
    }
    for name, function in functions.items():
        namespace[name] = StructuredTool.from_function(function, name=name)
    namespace["query_technology_rag"] = _technology_rag_tool(retriever)
    return {
        "installed_tool_names": list(BOOTSTRAPPED_TOOL_NAMES),
        "technology_rag_ready": retriever is not None,
    }


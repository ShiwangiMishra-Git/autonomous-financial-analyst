"""
NB-02 exit-gate tests: generic argument-aware tool-call deduplication.

The dedup logic under test (DEDUP_TOOLS / _args_hash / get_completed_call_hashes)
lives inside `create_financial_agent` in
Autonomous_financial_analyst_Learners_Notebook.ipynb (cell id b78756d2), as a
closure alongside the LangGraph nodes. The notebook is not an importable module,
so this file mirrors that logic verbatim rather than importing it. If cell
b78756d2 changes, update this file to match.
"""
import hashlib
import json

from langchain_core.messages import AIMessage, ToolMessage

DEDUP_TOOLS = {"analyze_sentiment"}


def _args_hash(args: dict) -> str:
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_completed_call_hashes(messages) -> set:
    completed = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                if tc["name"] not in DEDUP_TOOLS:
                    continue
                result = next(
                    (r for r in messages if isinstance(r, ToolMessage) and r.tool_call_id == tc["id"]),
                    None,
                )
                if result is None:
                    continue
                content = result.content if isinstance(result.content, str) else str(result.content)
                if content.strip().startswith("Error") or content.strip().startswith("SKIPPED"):
                    continue
                completed.add((tc["name"], _args_hash(tc.get("args", {}))))
    return completed


def would_skip(tool_call: dict, messages_before: list) -> bool:
    completed = get_completed_call_hashes(messages_before)
    dedup_key = (tool_call["name"], _args_hash(tool_call.get("args", {})))
    return tool_call["name"] in DEDUP_TOOLS and dedup_key in completed


def test_same_tool_same_args_is_blocked():
    tc1 = {"name": "analyze_sentiment", "args": {"text": "Amazon reports record profits"}, "id": "call_1"}
    ai1 = AIMessage(content="", tool_calls=[tc1])
    tm1 = ToolMessage(
        content=json.dumps({"sentiment": "positive", "status": "success"}),
        name="analyze_sentiment",
        tool_call_id="call_1",
    )
    history = [ai1, tm1]

    repeat = {"name": "analyze_sentiment", "args": {"text": "Amazon reports record profits"}, "id": "call_2"}
    assert would_skip(repeat, history) is True


def test_same_tool_different_args_is_allowed():
    tc1 = {"name": "analyze_sentiment", "args": {"text": "Amazon reports record profits"}, "id": "call_1"}
    ai1 = AIMessage(content="", tool_calls=[tc1])
    tm1 = ToolMessage(
        content=json.dumps({"sentiment": "positive", "status": "success"}),
        name="analyze_sentiment",
        tool_call_id="call_1",
    )
    history = [ai1, tm1]

    different = {"name": "analyze_sentiment", "args": {"text": "Google unveils new AI chip"}, "id": "call_3"}
    assert would_skip(different, history) is False


def test_failed_call_is_retryable():
    tc = {"name": "analyze_sentiment", "args": {"text": "Some flaky article text"}, "id": "call_4"}
    ai = AIMessage(content="", tool_calls=[tc])
    tm_fail = ToolMessage(content="Error: rate limited", name="analyze_sentiment", tool_call_id="call_4")
    history = [ai, tm_fail]

    retry = {"name": "analyze_sentiment", "args": {"text": "Some flaky article text"}, "id": "call_5"}
    assert would_skip(retry, history) is False


def test_argument_key_order_does_not_bypass_dedup():
    tc_a = {"name": "analyze_sentiment", "args": {"text": "x", "ticker": "AMZN"}, "id": "call_6"}
    ai = AIMessage(content="", tool_calls=[tc_a])
    tm = ToolMessage(
        content=json.dumps({"sentiment": "neutral", "status": "success"}),
        name="analyze_sentiment",
        tool_call_id="call_6",
    )
    history = [ai, tm]

    tc_b = {"name": "analyze_sentiment", "args": {"ticker": "AMZN", "text": "x"}, "id": "call_7"}
    assert would_skip(tc_b, history) is True


def test_skipped_prior_call_does_not_count_as_completed():
    tc = {"name": "analyze_sentiment", "args": {"text": "already skipped once"}, "id": "call_8"}
    ai = AIMessage(content="", tool_calls=[tc])
    tm_skipped = ToolMessage(content="SKIPPED: duplicate", name="analyze_sentiment", tool_call_id="call_8")
    history = [ai, tm_skipped]

    retry = {"name": "analyze_sentiment", "args": {"text": "already skipped once"}, "id": "call_9"}
    assert would_skip(retry, history) is False


def test_tool_outside_dedup_tools_is_never_blocked():
    tc = {"name": "get_stock_price", "args": {"ticker": "AMZN"}, "id": "call_10"}
    ai = AIMessage(content="", tool_calls=[tc])
    tm = ToolMessage(content=json.dumps({"price": 100}), name="get_stock_price", tool_call_id="call_10")
    history = [ai, tm]

    repeat = {"name": "get_stock_price", "args": {"ticker": "AMZN"}, "id": "call_11"}
    assert would_skip(repeat, history) is False

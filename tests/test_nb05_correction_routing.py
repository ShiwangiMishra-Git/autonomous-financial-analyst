"""
NB-05 exit-gate tests: capped citation-correction routing at the graph level.

Per the master plan's testing pyramid, this is a "mocked graph and multi-turn workflow"
test: it builds a real LangGraph StateGraph using the same citation_validator_node /
route_after_validation / should_continue logic added to create_financial_agent (cell
b78756d2), but with a scripted stand-in for the LLM-backed agent node and a stub tools
node, so no API key or network access is needed (per the master plan's guardrail: no
real API calls in unit tests).

Exit gate: direct-answer, tool-use, valid-answer, correction, and retry-exhaustion paths
must all terminate.

Everything below (CITATION_RE through validate_citation_authenticity, and the routing
functions) mirrors notebook cells 52/54/b78756d2 verbatim - see
tests/test_nb03_citation_validator.py and tests/test_nb04_citation_authenticity.py for
the isolated function-level tests of the validators themselves. Keep in sync if those
cells change.
"""
import json
import re
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- NB-03: citation format/completeness validator ---

CITATION_RE = re.compile(r'\[Source:\s*([^\]]+)\]')
MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
BARE_SOURCE_RE = re.compile(r'source\s*:', re.IGNORECASE)
PAREN_SOURCE_RE = re.compile(r'\(\s*source\s*:\s*[^)]+\)', re.IGNORECASE)
TIMESTAMP_RE = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}')
AGGREGATE_SENTIMENT_RE = re.compile(r'\b(average|overall|mean)\b.{0,30}sentiment|sentiment.{0,30}\b(average|overall|mean)\b', re.IGNORECASE)


def _cited_tool_names(text: str) -> set:
    names = set()
    for m in CITATION_RE.finditer(text):
        for token in m.group(1).split(','):
            token = token.strip()
            if token and not TIMESTAMP_RE.match(token):
                names.add(token)
    return names


def validate_citations(final_answer: str) -> list:
    errors = []

    for m in MARKDOWN_LINK_RE.finditer(final_answer):
        errors.append({
            "type": "markdown_link_substitution",
            "detail": "Markdown link used instead of a [Source: tool_name] tag.",
            "excerpt": m.group(0),
        })

    valid_spans = [m.span() for m in CITATION_RE.finditer(final_answer)]

    def _inside_valid_span(pos):
        return any(start <= pos < end for start, end in valid_spans)

    paren_spans = []
    for m in PAREN_SOURCE_RE.finditer(final_answer):
        paren_spans.append(m.span())
        if not _inside_valid_span(m.start()):
            errors.append({
                "type": "malformed_tag",
                "detail": "Citation uses parentheses instead of the required [Source: tool_name] bracket format.",
                "excerpt": m.group(0),
            })

    for m in BARE_SOURCE_RE.finditer(final_answer):
        if _inside_valid_span(m.start()):
            continue
        if any(start <= m.start() < end for start, end in paren_spans):
            continue
        snippet = final_answer[max(0, m.start() - 10):m.start() + 30]
        errors.append({
            "type": "malformed_tag",
            "detail": "'Source:' mention found outside the required [Source: tool_name] bracket format.",
            "excerpt": snippet,
        })

    blocks = re.split(r'\n\s*\n', final_answer)
    for block in blocks:
        cited_tools = _cited_tool_names(block)

        has_article_mention = bool(re.search(r'"[^"]{5,}"', block))
        has_sentiment_mention = bool(
            re.search(r'sentiment.{0,40}(score|[-]?\d\.\d{1,2})', block, re.IGNORECASE)
            or re.search(r'\bscore\s*[:\-]?\s*[-]?\d\.\d{1,2}', block, re.IGNORECASE)
        ) and not AGGREGATE_SENTIMENT_RE.search(block)
        has_price_mention = bool(re.search(r'\$\d[\d,]*\.?\d*', block))

        if has_article_mention and 'search_financial_news' not in cited_tools:
            errors.append({
                "type": "incomplete_citation",
                "detail": "Article reference found without a [Source: search_financial_news] tag in the same block.",
                "excerpt": block.strip()[:120],
            })
        if has_sentiment_mention and 'analyze_sentiment' not in cited_tools:
            errors.append({
                "type": "incomplete_citation",
                "detail": "Sentiment score found without a [Source: analyze_sentiment] tag in the same block.",
                "excerpt": block.strip()[:120],
            })
        if has_price_mention and 'get_stock_price' not in cited_tools:
            errors.append({
                "type": "incomplete_citation",
                "detail": "Price figure found without a [Source: get_stock_price] tag in the same block.",
                "excerpt": block.strip()[:120],
            })

    return errors


# --- NB-04: citation authenticity validator ---

def extract_cited_tool_names(final_answer: str) -> set:
    return _cited_tool_names(final_answer)


def get_successful_tool_names(messages) -> set:
    names = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip().startswith("Error") or content.strip().startswith("SKIPPED"):
                continue
            names.add(m.name)
    return names


def validate_citation_authenticity(final_answer: str, messages) -> list:
    cited = extract_cited_tool_names(final_answer)
    actual = get_successful_tool_names(messages)
    fabricated = cited - actual

    errors = []
    for name in sorted(fabricated):
        errors.append({
            "type": "fabricated_citation",
            "detail": f"[Source: {name}] cited but no successful {name} ToolMessage exists in this conversation.",
            "excerpt": name,
        })
    return errors


# --- NB-05: capped correction routing ---

MAX_CITATION_RETRIES = 2


def _format_validation_errors(errors) -> str:
    return "\n".join(f"- ({e['type']}) {e['detail']}" for e in errors)


def citation_validator_node(state) -> dict:
    last_ai = state["messages"][-1]
    text = last_ai.content if isinstance(last_ai.content, str) else ""
    errors = validate_citations(text) + validate_citation_authenticity(text, state["messages"])
    retry_count = state.get("validation_retry_count", 0)

    if not errors:
        return {}

    if retry_count >= MAX_CITATION_RETRIES:
        failure_note = AIMessage(
            content=(
                text
                + f"\n\n---\n[VALIDATION FAILED after {MAX_CITATION_RETRIES} correction attempt(s) - "
                "the citation issues below remain unresolved in the report above.]\n\n"
                + _format_validation_errors(errors)
            )
        )
        return {"messages": [failure_note]}

    correction_msg = HumanMessage(
        content=(
            "Your previous answer failed citation validation. Fix ONLY the issues below and "
            "resubmit the full corrected report. Do not repeat tool calls whose results already "
            "exist in this conversation.\n\n" + _format_validation_errors(errors)
        )
    )
    return {"messages": [correction_msg], "validation_retry_count": retry_count + 1}


def route_after_validation(state) -> Literal["retry", "end"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage) and last_message.content.startswith(
        "Your previous answer failed citation validation"
    ):
        return "retry"
    return "end"


def should_continue(state) -> Literal["tools", "validate"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "validate"


class ScriptedAgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]
    validation_retry_count: int


def _build_scripted_graph(scripted_responses):
    """scripted_responses: list of AIMessage, returned one per agent_node call, in order."""
    call_index = {"i": 0}

    def agent_node(state):
        response = scripted_responses[call_index["i"]]
        call_index["i"] += 1
        return {"messages": [response]}

    def tool_node(state):
        last_message = state["messages"][-1]
        outputs = [
            ToolMessage(content=json.dumps({"status": "success"}), name=tc["name"], tool_call_id=tc["id"])
            for tc in last_message.tool_calls
        ]
        return {"messages": outputs}

    workflow = StateGraph(ScriptedAgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("citation_validator", citation_validator_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "validate": "citation_validator"})
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("citation_validator", route_after_validation, {"retry": "agent", "end": END})
    return workflow.compile()


def test_direct_answer_path_terminates():
    graph = _build_scripted_graph([AIMessage(content="Hello, how can I help?")])
    result = graph.invoke({"messages": [HumanMessage(content="hi")]}, config={"recursion_limit": 10})
    assert result["messages"][-1].content == "Hello, how can I help?"
    assert result.get("validation_retry_count", 0) == 0


def test_tool_use_then_valid_answer_terminates():
    tool_call_msg = AIMessage(
        content="", tool_calls=[{"name": "get_stock_price", "args": {"ticker": "AMZN"}, "id": "c1"}]
    )
    final_answer = AIMessage(content="AAPL is trading at $178.45 [Source: get_stock_price]")
    graph = _build_scripted_graph([tool_call_msg, final_answer])
    result = graph.invoke({"messages": [HumanMessage(content="price?")]}, config={"recursion_limit": 10})
    assert result["messages"][-1].content == final_answer.content
    assert result.get("validation_retry_count", 0) == 0


def test_invalid_then_corrected_answer_terminates_after_one_retry():
    tool_call_msg = AIMessage(
        content="", tool_calls=[{"name": "analyze_sentiment", "args": {"text": "x"}, "id": "c1"}]
    )
    bad_answer = AIMessage(content="Sentiment: Positive (Score: 0.8)")  # missing [Source: analyze_sentiment]
    good_answer = AIMessage(content="Sentiment: Positive (Score: 0.8) [Source: analyze_sentiment]")
    graph = _build_scripted_graph([tool_call_msg, bad_answer, good_answer])
    result = graph.invoke({"messages": [HumanMessage(content="sentiment?")]}, config={"recursion_limit": 15})
    assert result["messages"][-1].content == good_answer.content
    assert result["validation_retry_count"] == 1


def test_retry_exhaustion_fails_closed_and_terminates():
    bad_answers = [AIMessage(content="Sentiment: Positive (Score: 0.8)") for _ in range(3)]
    graph = _build_scripted_graph(bad_answers)
    result = graph.invoke({"messages": [HumanMessage(content="sentiment?")]}, config={"recursion_limit": 15})
    final_content = result["messages"][-1].content
    assert "VALIDATION FAILED" in final_content
    # The last drafted answer must still be delivered, not discarded in favor of a bare error list.
    assert "Sentiment: Positive (Score: 0.8)" in final_content
    assert result["validation_retry_count"] == MAX_CITATION_RETRIES


def test_combined_dual_citation_tag_does_not_trigger_a_correction_loop():
    # Regression for the real Test 3 run: the model wrote both tool names in one tag -
    # [Source: search_financial_news, analyze_sentiment] - instead of two separate tags.
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_financial_news", "args": {}, "id": "c1"},
            {"name": "analyze_sentiment", "args": {}, "id": "c2"},
        ],
    )
    final_answer = AIMessage(
        content=(
            '1. Article: "Microsoft AI news" [Source: search_financial_news, analyze_sentiment]\n'
            "   Sentiment: Positive (Score: 0.80) [Source: search_financial_news, analyze_sentiment]"
        )
    )
    graph = _build_scripted_graph([tool_call_msg, final_answer])
    result = graph.invoke({"messages": [HumanMessage(content="sentiment?")]}, config={"recursion_limit": 10})
    assert result["messages"][-1].content == final_answer.content
    assert result.get("validation_retry_count", 0) == 0


def test_average_sentiment_line_does_not_trigger_a_correction_loop():
    # Regression for the real Test 3 run: an average-sentiment summary line, alongside a
    # properly-cited individual score, must not be treated as an unresolved citation issue.
    final_answer = AIMessage(
        content=(
            "Sentiment: Positive (Score: 0.8) [Source: analyze_sentiment]\n\n"
            "**Average Sentiment Score**: 0.8 (Positive)"
        )
    )
    tool_call_msg = AIMessage(
        content="", tool_calls=[{"name": "analyze_sentiment", "args": {"text": "x"}, "id": "c1"}]
    )
    graph = _build_scripted_graph([tool_call_msg, final_answer])
    result = graph.invoke({"messages": [HumanMessage(content="sentiment?")]}, config={"recursion_limit": 10})
    assert result["messages"][-1].content == final_answer.content
    assert result.get("validation_retry_count", 0) == 0

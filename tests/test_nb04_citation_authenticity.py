"""
NB-04 exit-gate tests: citation authenticity against actual ToolMessage records.

validate_citation_authenticity() rejects any [Source: tool_name] tag in a candidate
final answer when tool_name has no corresponding *successful* ToolMessage in
state["messages"] - i.e. it catches the case NB-03 cannot: a citation that is
correctly *formatted* but simply never happened. This is the fabrication bug the
notebook's own "Known Limitation" cells (22-company test, ~44 fabricated
[Source: analyze_sentiment] tags with zero real analyze_sentiment calls) describe.

Per the master plan, this step does not add tool_call_id-level citations or
multi-turn scoping (that's deferred to PROD-02/PROD-03) - it only checks tool
*names* against ToolMessages present anywhere in state["messages"].
"""
import re

from langchain_core.messages import AIMessage, ToolMessage

CITATION_RE = re.compile(r'\[Source:\s*([^\]]+)\]')
TIMESTAMP_RE = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}')


def extract_cited_tool_names(final_answer: str) -> set:
    """
    Every [Source: ...] tag's interior, comma-split into tokens, keeping tokens that
    aren't timestamps. Handles both [Source: tool_name, timestamp] and
    [Source: tool_a, tool_b] (multiple tools cited in one tag) the same way.
    """
    names = set()
    for m in CITATION_RE.finditer(final_answer):
        for token in m.group(1).split(','):
            token = token.strip()
            if token and not TIMESTAMP_RE.match(token):
                names.add(token)
    return names


def get_successful_tool_names(messages) -> set:
    """Tool names with at least one real (non-error, non-skipped) ToolMessage result."""
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


def _success_tool_message(name: str, tool_call_id: str, content: str = '{"status": "success"}'):
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id)


def test_cited_tool_with_real_result_is_not_flagged():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "analyze_sentiment", "args": {}, "id": "c1"}]),
        _success_tool_message("analyze_sentiment", "c1"),
    ]
    answer = "Sentiment is positive (score: 0.8) [Source: analyze_sentiment]"
    assert validate_citation_authenticity(answer, messages) == []


def test_cited_tool_with_no_toolmessage_at_all_is_fabricated():
    messages = []
    answer = "Sentiment is positive (score: 0.8) [Source: analyze_sentiment]"
    errors = validate_citation_authenticity(answer, messages)
    assert len(errors) == 1
    assert errors[0]["type"] == "fabricated_citation"
    assert "analyze_sentiment" in errors[0]["excerpt"]


def test_cited_tool_with_only_error_result_is_fabricated():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "analyze_sentiment", "args": {}, "id": "c1"}]),
        ToolMessage(content="Error: timeout", name="analyze_sentiment", tool_call_id="c1"),
    ]
    answer = "Sentiment is positive (score: 0.8) [Source: analyze_sentiment]"
    errors = validate_citation_authenticity(answer, messages)
    assert len(errors) == 1 and errors[0]["type"] == "fabricated_citation"


def test_cited_tool_with_only_skipped_result_is_fabricated():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "analyze_sentiment", "args": {}, "id": "c1"}]),
        ToolMessage(content="SKIPPED: duplicate", name="analyze_sentiment", tool_call_id="c1"),
    ]
    answer = "Sentiment is positive (score: 0.8) [Source: analyze_sentiment]"
    errors = validate_citation_authenticity(answer, messages)
    assert len(errors) == 1 and errors[0]["type"] == "fabricated_citation"


def test_cited_unknown_source_name_is_fabricated():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search_financial_news", "args": {}, "id": "c1"}]),
        _success_tool_message("search_financial_news", "c1"),
    ]
    answer = "Background info [Source: Wikipedia]"
    errors = validate_citation_authenticity(answer, messages)
    assert len(errors) == 1
    assert errors[0]["excerpt"] == "Wikipedia"


def test_uncited_successful_tool_is_not_flagged():
    # get_stock_price ran successfully but isn't cited anywhere - not an authenticity error.
    messages = [
        AIMessage(content="", tool_calls=[{"name": "get_stock_price", "args": {}, "id": "c1"}]),
        _success_tool_message("get_stock_price", "c1"),
    ]
    answer = "Sentiment is positive (score: 0.8) [Source: analyze_sentiment]"
    # analyze_sentiment is still fabricated here, but get_stock_price being un-cited is fine.
    errors = validate_citation_authenticity(answer, messages)
    assert len(errors) == 1
    assert errors[0]["excerpt"] == "analyze_sentiment"


def test_mixed_real_and_fabricated_citations_flags_only_fake_one():
    messages = [
        AIMessage(content="", tool_calls=[
            {"name": "search_financial_news", "args": {}, "id": "c1"},
        ]),
        _success_tool_message("search_financial_news", "c1"),
    ]
    answer = (
        'Article: "Microsoft AI news" [Source: search_financial_news]\n'
        "Sentiment: Positive (Score: 0.8) [Source: analyze_sentiment]"
    )
    errors = validate_citation_authenticity(answer, messages)
    assert len(errors) == 1
    assert errors[0]["excerpt"] == "analyze_sentiment"


def test_combined_dual_citation_tag_both_tools_authentic():
    messages = [
        AIMessage(content="", tool_calls=[
            {"name": "search_financial_news", "args": {}, "id": "c1"},
            {"name": "analyze_sentiment", "args": {}, "id": "c2"},
        ]),
        _success_tool_message("search_financial_news", "c1"),
        _success_tool_message("analyze_sentiment", "c2"),
    ]
    answer = "Sentiment: Positive (Score: 0.8) [Source: search_financial_news, analyze_sentiment]"
    assert validate_citation_authenticity(answer, messages) == []


def test_combined_dual_citation_tag_flags_the_fabricated_half():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search_financial_news", "args": {}, "id": "c1"}]),
        _success_tool_message("search_financial_news", "c1"),
    ]
    answer = "Sentiment: Positive (Score: 0.8) [Source: search_financial_news, analyze_sentiment]"
    errors = validate_citation_authenticity(answer, messages)
    assert len(errors) == 1
    assert errors[0]["excerpt"] == "analyze_sentiment"


def test_22_company_style_fabrication_all_flagged():
    # Regression check for the notebook's own documented large-scale failure:
    # many analyze_sentiment citations, zero real analyze_sentiment ToolMessages.
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search_financial_news", "args": {}, "id": "c1"}]),
        _success_tool_message("search_financial_news", "c1"),
    ]
    answer = "\n\n".join(
        f"{ticker}: Sentiment positive (score: 0.7) [Source: analyze_sentiment]"
        for ticker in ["MSFT", "GOOGL", "NVDA", "AMZN", "IBM"]
    )
    errors = validate_citation_authenticity(answer, messages)
    # All 5 blocks cite the same fabricated tool name -> deduped to one distinct error.
    assert len(errors) == 1
    assert errors[0]["excerpt"] == "analyze_sentiment"

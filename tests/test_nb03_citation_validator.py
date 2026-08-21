"""
NB-03 exit-gate tests: citation format and completeness validation.

validate_citations() is a deterministic, regex-only check of the SOURCE CITATION
contract in AGENT_CHARTER_FULL (Autonomous_financial_analyst_Learners_Notebook.ipynb,
cell id 8b8ed6c2). It inspects only a single candidate final answer string - never
the full conversation history, and never calls an LLM judge.

This file is the reference implementation. Once tests pass here, the same
function is copied verbatim into a new notebook cell (mirrored, not imported,
since the notebook is not an importable module - see tests/test_nb02_dedup.py
for the same convention). NB-03 only builds the validator; wiring it into the
agent graph as a correction loop is NB-05, not this step.
"""
import re

CITATION_RE = re.compile(r'\[Source:\s*([^\]]+)\]')
MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
BARE_SOURCE_RE = re.compile(r'source\s*:', re.IGNORECASE)
PAREN_SOURCE_RE = re.compile(r'\(\s*source\s*:\s*[^)]+\)', re.IGNORECASE)
TIMESTAMP_RE = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}')
# An average/overall sentiment figure is computed by the model from already-cited
# per-article scores - no single analyze_sentiment call produced it, so it is not
# eligible for its own [Source: analyze_sentiment] tag.
AGGREGATE_SENTIMENT_RE = re.compile(r'\b(average|overall|mean)\b.{0,30}sentiment|sentiment.{0,30}\b(average|overall|mean)\b', re.IGNORECASE)


def _cited_tool_names(text: str) -> set:
    """
    Every [Source: ...] tag's interior, comma-split into tokens, keeping tokens that
    aren't timestamps. Handles both [Source: tool_name, timestamp] and
    [Source: tool_a, tool_b] (multiple tools cited in one tag) the same way.
    """
    names = set()
    for m in CITATION_RE.finditer(text):
        for token in m.group(1).split(','):
            token = token.strip()
            if token and not TIMESTAMP_RE.match(token):
                names.add(token)
    return names


def validate_citations(final_answer: str) -> list:
    """Return a list of structured error dicts; empty list means the answer passes."""
    errors = []

    # 1. Markdown-link substituted for a citation tag (charter: never cite via links/prose).
    for m in MARKDOWN_LINK_RE.finditer(final_answer):
        errors.append({
            "type": "markdown_link_substitution",
            "detail": "Markdown link used instead of a [Source: tool_name] tag.",
            "excerpt": m.group(0),
        })

    # 2. Malformed citation attempts: mentions "source:" that aren't a valid bracket tag.
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

    # 3. Per-claim completeness, evaluated block by block (blank-line separated units).
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


# --- Charter examples (AGENT_CHARTER_FULL, SOURCE CITATION > Examples) ---

def test_charter_valid_price_citation():
    text = 'AAPL is trading at $178.45 [Source: get_stock_price, 2024-10-30 13:30]'
    assert validate_citations(text) == []


def test_charter_valid_inline_dual_citation():
    text = (
        "Recent news on Microsoft's AI initiatives [Source: search_financial_news] shows "
        "positive sentiment (score: 0.75) [Source: analyze_sentiment]"
    )
    assert validate_citations(text) == []


def test_charter_valid_numbered_list_dual_citation():
    text = (
        '1. Article: "Microsoft releases new AI model" [Source: search_financial_news]\n'
        '   Sentiment: Positive (Score: 0.80) [Source: analyze_sentiment]'
    )
    assert validate_citations(text) == []


def test_charter_invalid_no_source_no_metrics():
    text = "The stock is doing well"
    assert validate_citations(text) == []  # no claim triggers a check - nothing to flag deterministically


def test_charter_invalid_missing_article_citation():
    text = (
        '1. Article: "Microsoft releases new AI model"\n'
        '   Sentiment: Positive (Score: 0.80) [Source: analyze_sentiment]'
    )
    errors = validate_citations(text)
    assert any(e["type"] == "incomplete_citation" and "search_financial_news" in e["detail"] for e in errors)


# --- Master plan examples (NB-03 code block) ---

def test_master_plan_valid_example():
    text = 'Article analysis...\n[Source: search_financial_news]\n[Source: analyze_sentiment]'
    assert validate_citations(text) == []


def test_master_plan_invalid_markdown_link():
    text = '[Article title](https://example.com)'
    errors = validate_citations(text)
    assert any(e["type"] == "markdown_link_substitution" for e in errors)


# --- Malformed-tag variants not covered by the charter/master-plan snippets ---

def test_paren_style_malformed_tag():
    errors = validate_citations("Price is $10 (Source: get_stock_price)")
    assert any(e["type"] == "malformed_tag" for e in errors)


def test_bare_source_mention_malformed_tag():
    errors = validate_citations("Data via Source: get_stock_price shows growth")
    assert any(e["type"] == "malformed_tag" for e in errors)


def test_valid_tag_does_not_trigger_malformed_or_markdown_checks():
    text = "Value [Source: get_stock_price, 2024-10-30] confirmed."
    assert validate_citations(text) == []


def test_key_price_without_any_citation_is_flagged():
    text = "AAPL is trading at $178.45 with strong momentum."
    errors = validate_citations(text)
    assert any(e["type"] == "incomplete_citation" and "get_stock_price" in e["detail"] for e in errors)


# --- Regression: real Test 3 run flagged "Average Sentiment Score: 0.77" as needing its own
# [Source: analyze_sentiment] tag, even after the model had already cited all 5 per-article
# scores individually. An average across already-cited scores is a derived statistic the
# model computed itself - no single tool call produced it, so it shouldn't require a fresh
# citation the way an individual per-article score does.

def test_average_sentiment_summary_line_is_not_flagged():
    text = "**Average Sentiment Score**: 0.77 (Positive)"
    assert validate_citations(text) == []


def test_overall_sentiment_summary_phrase_is_not_flagged():
    text = "Overall sentiment across the articles is positive with a score of 0.72."
    assert validate_citations(text) == []


def test_individual_sentiment_claim_is_still_flagged_next_to_an_average():
    text = (
        "Sentiment: Positive (Score: 0.80)\n\n"
        "**Average Sentiment Score**: 0.77 (Positive)"
    )
    errors = validate_citations(text)
    assert any(
        e["type"] == "incomplete_citation" and "analyze_sentiment" in e["detail"] for e in errors
    )
    assert not any("Average Sentiment Score" in e["excerpt"] for e in errors)


# --- Regression: a real Test 3 run had the model write both tool names in one combined
# tag - [Source: search_financial_news, analyze_sentiment] - instead of two separate tags.
# CITATION_RE used to only capture the first comma-delimited token, silently dropping the
# second tool name and flagging a false "missing analyze_sentiment citation".

def test_combined_dual_citation_tag_counts_both_tools():
    text = (
        '1. Article: "Microsoft releases new AI model" [Source: search_financial_news, analyze_sentiment]\n'
        '   Sentiment: Positive (Score: 0.80) [Source: search_financial_news, analyze_sentiment]'
    )
    assert validate_citations(text) == []


def test_combined_tag_with_timestamp_and_tool_name_still_filters_timestamp():
    text = 'AAPL is trading at $178.45 [Source: get_stock_price, 2024-10-30 13:30]'
    assert validate_citations(text) == []

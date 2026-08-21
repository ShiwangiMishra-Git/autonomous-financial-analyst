"""Add missing class and function docstrings to the canonical working notebook.

The transformation is intentionally narrow and idempotent: it inserts documentation only
when a targeted definition has no docstring, preserves source formatting and learner markers,
and never edits any notebook other than the documented working copy.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


DOCSTRINGS: dict[str, dict[str, str]] = {
    "api_cache_def": {
        "_cache_key": """Build a stable cache key from a function name and invocation arguments.

Args:
    func_name: Name of the cached function.
    args: Positional arguments supplied to the function.
    kwargs: Keyword arguments supplied to the function.

Returns:
    A SHA-256 hexadecimal key for the canonicalized invocation.
""",
        "_read_entry": """Read and decode one cache entry without propagating storage errors.

Args:
    path: JSON cache file to read.

Returns:
    The decoded entry, or ``None`` when it is missing, corrupt, or unreadable.
""",
        "_invalidate_dependents": """Evict ticker-matched derived cache entries after a source refresh.

Args:
    func_name: Source function whose fresh result was written.
    args: Positional arguments used by the source call.
    kwargs: Keyword arguments used by the source call.

Returns:
    ``None``; matching dependent files are removed as a freshness optimization.
""",
        "decorator": """Wrap one function with the configured disk-cache freshness policy.

Args:
    func: Function whose calls should be cached.

Returns:
    A metadata-preserving stale-while-revalidate wrapper.
""",
        "wrapper": """Serve a cached value or execute and persist the wrapped function.

Fresh entries return immediately, stale entries trigger one background refresh, and true
misses execute synchronously because no prior value is available.
""",
        "_refresh": """Refresh one stale cache entry and release its in-flight marker.

Background failures are contained so callers can continue using the previous cached value.
""",
    },
    "21cf6f8d": {
        "_get_yf_info": """Fetch the raw Yahoo Finance information dictionary for one ticker.

Args:
    ticker: Public-market ticker symbol; it is normalized to uppercase.

Returns:
    A plain dictionary copied from ``yfinance.Ticker.info`` and cached for reuse.
""",
    },
    "b78756d2": {
        "_args_hash": """Create a stable hash for one tool argument dictionary.

Args:
    args: JSON-serializable tool arguments.

Returns:
    SHA-256 hash used as the argument-aware deduplication key.
""",
        "tool_node_with_logging": """Execute requested tools with logging, error capture, and safe deduplication.

Args:
    state: Current agent messages ending in an AI tool-call message.

Returns:
    Tool messages for every requested call, including explicit skipped or error results.
""",
        "_format_validation_errors": """Format structured citation errors for an LLM correction message.

Args:
    errors: Validation dictionaries containing ``type`` and ``detail`` fields.

Returns:
    A newline-delimited, human-readable error list.
""",
        "citation_validator_node": """Validate the candidate answer and request a bounded citation correction.

Args:
    state: Agent state containing the candidate answer and prior tool results.

Returns:
    No update on success, a correction message while retries remain, or a failure annotation
    after the retry ceiling is reached.
""",
        "route_after_validation": """Route citation validation to another agent attempt or workflow completion.

Args:
    state: Agent state after the citation validator executes.

Returns:
    ``retry`` for an injected correction request; otherwise ``end``.
""",
    },
    "a1f24e46": {
        "_inside_valid_span": """Return whether a character position lies inside a valid citation tag.

Args:
    pos: Character offset within the candidate answer.

Returns:
    ``True`` when the offset is covered by a recognized citation span.
""",
    },
    "33b1149d": {
        "extract_cited_tool_names": """Extract unique tool names referenced by valid source tags.

Args:
    final_answer: Candidate narrative containing ``[Source: ...]`` tags.

Returns:
    Set of cited tool names, excluding timestamp tokens.
""",
    },
    "73bb82d6": {
        "agent_node": """Invoke the test agent whose price tool intentionally fails.

Args:
    state: Current test-agent conversation state.

Returns:
    The next AI message, which may contain tool calls or a final response.
""",
        "should_continue": """Route the failure-test agent according to its latest tool-call decision.

Args:
    state: Current test-agent state.

Returns:
    ``tools`` when calls are present; otherwise ``end``.
""",
    },
    "f4ab62d8": {
        "_has_price_mention": """Detect stock-price amounts while excluding market-cap magnitude expressions.

Args:
    block: One narrative block being checked for citation completeness.

Returns:
    ``True`` when a raw dollar amount appears to represent a price.
""",
        "_looks_like_publication": """Return whether a citation token contains a publication URL.

Args:
    token: Text extracted from a source citation.

Returns:
    ``True`` when the token matches the supported URL pattern.
""",
        "_nearest_heading": """Find the closest markdown heading preceding a character position.

Args:
    full_text: Complete candidate answer.
    pos: Character offset whose surrounding section is needed.

Returns:
    Heading text, or an empty string when no earlier heading exists.
""",
        "extract_cited_tool_names": """Extract unique tool names referenced by valid source tags.

Args:
    final_answer: Candidate narrative containing source tags.

Returns:
    Set of cited tool names after citation-token normalization.
""",
        "_fix": """Rewrite one malformed source-link match into the supported citation format.

Args:
    m: Regular-expression match for a markdown link.

Returns:
    Corrected source tag, or the original match when it is not a source citation.
""",
        "_inside_valid_span": """Return whether a character offset belongs to a valid citation span.

Args:
    pos: Character offset in the candidate answer.

Returns:
    Boolean membership result used to avoid duplicate malformed-tag errors.
""",
        "_in_aggregate_span": """Return whether a sentiment mention belongs to an aggregate statement.

Args:
    pos: Character offset of the sentiment mention.

Returns:
    ``True`` when the mention is covered by an aggregate-sentiment span.
""",
    },
    "d481535a": {
        "_format_validation_errors": """Format structured validation errors for logs and correction prompts.

Args:
    errors: Validation dictionaries with ``type`` and ``detail`` values.

Returns:
    Newline-delimited error descriptions.
""",
        "citation_validator_node": """Apply deterministic citation checks to an enhanced-agent answer.

The node first performs safe formatting repairs, then either accepts the answer, injects a
bounded correction request, or appends a visible failure warning after retry exhaustion.

Args:
    state: Enhanced-agent state containing messages and validation retry count.

Returns:
    State updates required for correction, successful autofix, or fail-closed completion.
""",
    },
    "risk_profile_dropdown": {
        "_on_risk_profile_change": """Synchronize the notebook risk profile with the dropdown selection.

Args:
    change: Widget event dictionary containing the changed field and new value.

Returns:
    ``None``; updates the notebook-level ``risk_profile`` variable when appropriate.
""",
    },
    "score_companies_def": {
        "_max_total_score": """Calculate the maximum achievable score for one risk profile.

Args:
    risk_profile: Key selecting the configured AI-signal weights.

Returns:
    Sum of all financial-metric caps and AI-signal weight caps.
""",
    },
    "scoring_agent_state_def": {
        "ScoringAgentState": """LangGraph state for autonomous scoring and bounded validation.

Attributes:
    messages: Reducer-backed conversation and tool-result history.
    validation_retry_count: Number of completeness corrections already requested.
    tool_round_count: Number of tool-execution rounds already completed.
""",
    },
    "create_scoring_agent_def": {
        "_args_hash": """Create a stable hash for argument-aware scoring-tool deduplication.

Args:
    args: JSON-serializable tool arguments.

Returns:
    SHA-256 hexadecimal digest of the canonical argument representation.
""",
        "scoring_agent_node": """Invoke the scoring LLM while enforcing the tool-round ceiling.

Args:
    state: Scoring conversation, retry counters, and previous tool results.

Returns:
    The next AI message and reset counters when a fresh user turn begins.
""",
        "scoring_tool_node": """Execute scoring tools with argument-aware deduplication and error capture.

Args:
    state: Scoring state ending with one or more requested tool calls.

Returns:
    Tool messages plus the incremented tool-round count.
""",
        "scoring_completeness_node": """Validate that scoring completed and the narrative preserves score fidelity.

Args:
    state: Scoring state containing tool results and the candidate narrative.

Returns:
    No update on success, a bounded correction request, or a fail-closed warning.
""",
        "should_continue_scoring": """Route an agent response to tool execution or completeness validation.

Args:
    state: Current scoring-agent state.

Returns:
    ``tools`` when tool calls exist; otherwise ``validate``.
""",
        "route_after_scoring_validation": """Route scoring validation to correction or workflow completion.

Args:
    state: Scoring state after the completeness node.

Returns:
    ``retry`` for an injected human correction message; otherwise ``end``.
""",
        "_company_section": """Extract the narrative section belonging to one ticker.

Args:
    ticker: Company ticker whose section should be located.

Returns:
    The best matching heading-bounded or paragraph-bounded narrative section.
""",
        "_title_core": """Remove exchange and metadata suffixes from a publication title.

Args:
    title: Raw cited publication title.

Returns:
    Normalized title core used for comparison.
""",
    },
    "test7c_interactive": {
        "_run_test7c_interactive": """Run the interactive five-company scoring demonstration.

Args:
    _: Unused widget-click event payload.

Returns:
    ``None``; streams progress and backing data to the notebook output widget.
""",
    },
    "route_financial_query_def": {
        "_extract_mentioned_tickers": """Resolve explicitly mentioned supported tickers from query aliases.

Args:
    query: Natural-language financial question.

Returns:
    Supported tickers whose configured aliases occur in the query.
""",
    },
    "multiindustry_state_contracts": {
        "QueryPlan": """Validated interpretation of one free-text financial research request.

The contract captures requested companies, dimensions, risk, freshness, horizon, and whether
the user explicitly requested deterministic scoring.
""",
        "ResolvedCompany": """Canonical registry-backed identity and resolution status for one company.

Empty identity fields are permitted only when the status is ambiguous or unsupported.
""",
        "EvidenceRecord": """Source-aware evidence item bound to one run, company, and profile.

The record carries provenance, retrieval and as-of times, freshness, status, and any error.
""",
        "CompanyTask": """Isolated research assignment for exactly one resolved company.

It combines the shared query plan with profile dimensions and the worker's tool allowlist.
""",
        "CompanyResearchResult": """Normalized output produced by one company-worker branch.

The result keeps evidence, signals, missing dimensions, errors, and completion status together.
""",
        "ScoringEligibility": """Deterministic decision describing whether sector scoring is permitted.

The contract records the applicable rubric and all exclusions or missing requirements.
""",
        "CompanyWorkerState": """Branch-local LangGraph state for researching exactly one company.

Messages, evidence, retries, signals, and errors remain isolated from peer company branches.
""",
        "OrchestratorState": """Parent LangGraph state spanning conversation and current research run.

Conversation fields may persist through ``MemorySaver``; run fields are reset for every query.
""",
        "_latest_human_query": """Return the most recent human-authored query from conversation messages.

Args:
    messages: Ordered LangChain message sequence.

Returns:
    String content of the latest human message, or an empty string when absent.
""",
    },
    "multiindustry_company_registry": {
        "CompanyRegistryEntry": """Authoritative local identity and profile metadata for one supported listing.

Aliases support deterministic matching; ``profile_id`` binds the listing to its industry playbook.
""",
        "_company_entry": """Construct one consistently shaped company-registry entry.

Args:
    company_id: Stable company identity shared across listings when appropriate.
    ticker: Supported market ticker.
    company_name: Canonical display name.
    aliases: Normalized-name candidates accepted by the resolver.
    exchange: Exchange name when known.
    industry: Broad supported industry.
    sub_industry: Supported industry specialization.
    profile_id: Versioned research profile identifier.

Returns:
    A complete ``CompanyRegistryEntry`` mapping.
""",
        "_normalize_company_text": """Normalize a ticker or company name for deterministic matching.

Args:
    value: User mention or registry alias.

Returns:
    Case-folded alphanumeric tokens with ampersands normalized to ``and``.
""",
        "_build_alias_index": """Build the normalized alias-to-ticker lookup index.

Returns:
    Mapping from each accepted normalized alias to all matching supported tickers.
""",
        "_resolved_company": """Convert one registry entry into a successful resolution result.

Args:
    entry: Authoritative registry record.

Returns:
    ``ResolvedCompany`` with status ``resolved`` and copied aliases.
""",
        "_unresolved_company": """Construct an ambiguous or unsupported resolution result.

Args:
    mention: Original planner-produced company text.
    status: Non-successful resolution category.
    message: User-facing explanation of the resolution failure.

Returns:
    ``ResolvedCompany`` with intentionally empty canonical identity fields.
""",
        "_ticker_tokens_in_mention": """Find explicit supported ticker tokens within a normalized mention.

Args:
    normalized_mention: Token-normalized company text.

Returns:
    Set of supported tickers appearing as complete tokens.
""",
        "_alias_candidates": """Find supported ticker candidates whose aliases occur in a mention.

Args:
    normalized_mention: Token-normalized company text.

Returns:
    All registry tickers matched by exact or contained aliases.
""",
    },
    "multiindustry_query_planner": {
        "_normalize_string_list": """Validate, trim, deduplicate, and preserve order for a string list.

Args:
    value: Candidate structured-output field.
    field_name: Field name included in validation errors.

Returns:
    Cleaned unique strings in first-seen order.

Raises:
    QueryPlanningError: If the value is not a list containing only strings.
""",
        "_normalize_dimension": """Convert one requested research dimension to snake_case.

Args:
    value: Natural-language or partially normalized dimension.

Returns:
    Lowercase alphanumeric identifier with single underscore separators.
""",
        "_query_uses_followup_reference": """Detect pronouns that may refer to previously remembered companies.

Args:
    query: Current natural-language question.

Returns:
    ``True`` when a supported follow-up reference occurs.
""",
        "_conversation_excerpt": """Render a bounded user/assistant conversation excerpt for planning context.

Args:
    messages: Ordered conversation messages.
    limit: Maximum number of recent messages to inspect.

Returns:
    Role-prefixed text, or ``(none)`` when no eligible messages exist.
""",
        "_default_query_planner_model": """Create the default deterministic-temperature chat model for query planning.

Returns:
    Configured ``ChatOpenAI`` client using notebook environment credentials and endpoint.
""",
        "__init__": """Initialize a planning error from one or more contract violations.

Args:
    errors: Human-readable deterministic validation failures.
""",
    },
    "multiindustry_f03_smoke": {
        "_F03FakeStructuredModel": """Local structured-output model double used by the F03 smoke test.

It records the requested schema and method, then returns a predefined response without network use.
""",
        "__init__": """Store the predefined structured response returned by the fake model.

Args:
    response: Dictionary-like planner output used by the smoke test.
""",
        "with_structured_output": """Record structured-output configuration and return this test double.

Args:
    schema: Expected planner output schema.
    method: Structured-output strategy requested by production code.

Returns:
    This fake model instance for chained invocation.
""",
        "invoke": """Validate the planner call shape and return the predefined response.

Args:
    messages: System and human messages sent by ``plan_query``.

Returns:
    Preconfigured structured planner response.
""",
    },
}


def _iter_definitions(tree: ast.AST) -> Iterable[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield every class and synchronous or asynchronous function in source order."""
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    yield from sorted(definitions, key=lambda node: (node.lineno, node.col_offset))


def _render_docstring(text: str, indent: str) -> list[str]:
    """Render docstring content at the indentation of a definition body."""
    content = text.strip("\n").splitlines()
    rendered = [f'{indent}"""{content[0]}\n']
    rendered.extend(f"{indent}{line}\n" for line in content[1:])
    rendered.append(f'{indent}"""\n')
    return rendered


def _document_cell(source: str, requested: dict[str, str], cell_id: str) -> tuple[str, set[str]]:
    """Insert missing docstrings for one cell and return the updated source and inserted names."""
    tree = ast.parse(source)
    definitions = list(_iter_definitions(tree))
    by_name: dict[str, list[ast.AST]] = {}
    for node in definitions:
        by_name.setdefault(node.name, []).append(node)

    unknown = set(requested) - set(by_name)
    if unknown:
        raise ValueError(f"Cell {cell_id} does not define requested names: {sorted(unknown)}")

    duplicate_targets = [name for name in requested if len(by_name[name]) != 1]
    if duplicate_targets:
        raise ValueError(
            f"Cell {cell_id} requires qualified names for duplicate definitions: {duplicate_targets}"
        )

    lines = source.splitlines(keepends=True)
    insertions: list[tuple[int, list[str], str]] = []
    for name, docstring in requested.items():
        node = by_name[name][0]
        if ast.get_docstring(node, clean=False) is not None:
            continue
        if not node.body:
            raise ValueError(f"Cannot document empty definition {name} in cell {cell_id}")
        first_statement = node.body[0]
        first_line = first_statement.lineno
        decorators = getattr(first_statement, "decorator_list", ())
        if decorators:
            first_line = min(first_line, *(decorator.lineno for decorator in decorators))
        body_line = first_line - 1
        indent = " " * first_statement.col_offset
        insertions.append((body_line, _render_docstring(docstring, indent), name))

    inserted: set[str] = set()
    for line_index, rendered, name in sorted(insertions, reverse=True):
        lines[line_index:line_index] = rendered
        inserted.add(name)

    updated = "".join(lines)
    ast.parse(updated)
    return updated, inserted


def main() -> None:
    """Update the working notebook and verify every parseable definition has documentation."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}
    missing_cells = set(DOCSTRINGS) - set(cells_by_id)
    if missing_cells:
        raise ValueError(f"Notebook is missing target cells: {sorted(missing_cells)}")

    inserted_count = 0
    for cell_id, requested in DOCSTRINGS.items():
        cell = cells_by_id[cell_id]
        updated, inserted = _document_cell(cell.source, requested, cell_id)
        if inserted:
            cell.source = updated
            cell.execution_count = None
            cell.outputs = []
            inserted_count += len(inserted)

    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Documented {inserted_count} definitions in {NOTEBOOK_PATH.name}")


if __name__ == "__main__":
    main()

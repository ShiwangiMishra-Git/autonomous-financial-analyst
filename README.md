# Autonomous Financial Analyst

A LangGraph-based agentic financial research assistant, built for a JHU Agentic AI course assignment. The base agent is built up in increasing sophistication (reactive → goal-oriented → full charter), then extended with RAG over private company documents, and further extended with a unified router that dispatches technology and pharma research questions through one shared agent/tools/citation-validator graph.

Full write-up (architecture, trade-offs, limitations, future scope): [docs/SUBMISSION_SUMMARY_WITH_HLD.md](docs/SUBMISSION_SUMMARY_WITH_HLD.md).

## Architecture

```text
User Question → Unified Query Router → Technology / Pharma AgentProfile
                                              ↓
                          Shared Graph: Agent ↔ Tools ↔ Citation Validator
                                              ↓
                                      Validated Response
```

![Full request lifecycle diagram](diagrams/full_lifecycle_diagram.png)

**Tools:** stock price/history (`yfinance`), news search (Tavily), LLM sentiment analysis, RAG over private tech/pharma documents (Chroma), clinical-trial search (pharma), and deterministic comparison scoring (never LLM-computed).

**Notable decisions:** one shared graph reused across domains instead of separate pipelines (avoids guardrail drift); bounded tool/retry loops that fail closed; missing evidence marked `N/A`, never `0`; unsupported companies always flagged, never silently scored.

## Example output

Real recorded turns from the notebook's chat panel — technology (public-tool synthesis) and pharma (private-document RAG with citations):

| Technology | Pharma RAG |
|---|---|
| ![Technology example](diagrams/chat_ui_technology_example.png) | ![Pharma RAG example](diagrams/chat_ui_pharma_example.png) |

## Contents

- `Autonomous_financial_analyst_FINAL_SUBMISSION.ipynb` — the main notebook.
- `content/` — RAG source data and pre-built vector stores (tracked via Git LFS): `Companies-AI-Initiatives.zip` + `vectorstore/` (technology) and `pharma_rag_official_sources.zip` + `vectorstore_pharma_clean/` (pharma filings).
- `tests/` — deterministic tests for notebook-support code.
- `diagrams/` — architecture diagrams and example output.
- `docs/SUBMISSION_SUMMARY_WITH_HLD.md` — full HLD write-up.

## Setup

1. `cp config.json.example config.json` and fill in `OPENAI_API_KEY`, `OPENAI_API_BASE`, `TAVILY_API_KEY` (gitignored, never commit it).
2. `pip install -r requirements.txt`
3. Run with `jupyter lab`, or open in VS Code/Cursor and select a kernel.

Part 2 (RAG) expects a local `content/Companies-AI-Initiatives.zip` — Colab-specific path, adjust to run locally.

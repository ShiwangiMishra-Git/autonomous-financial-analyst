# Autonomous Financial Analyst

A LangGraph-based agentic financial research assistant, built for a JHU Agentic AI course assignment. The agent is built up in increasing sophistication — a plain reactive baseline, a goal-oriented "agent charter" version, and a full charter with tool inventory and behavioral rules — then extended with a RAG pipeline over private company documents.

## Contents

- `Autonomous_financial_analyst_CLEAN_REBUILD.ipynb` — the main notebook: builds and tests the agent (Part 1) and the RAG-enhanced agent (Part 2).
- `scripts/` — supporting implementation scripts for the multi-industry research pipeline.
- `tests/` — deterministic tests for notebook-support code.
- `docs/` — architecture notes, design docs, and diagrams.
- `SUBMISSION_SUMMARY_WITH_HLD.md`, `HF_DEPLOYMENT_BRIEF.md`, `PHARMA_IMPLEMENTATION_REPORT_v75.md` — write-ups accompanying the submission.
- `FinancialAgenticFlow.pdf` and the `.png`/`.svg` diagrams — architecture/flow diagrams.

## Setup

1. Create `config.json` from the template:
   ```bash
   cp config.json.example config.json
   ```
   Fill in `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `TAVILY_API_KEY`. `config.json` is gitignored and must never be committed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook locally with Jupyter (`./run_jupyter.sh`) or open it in VS Code/Cursor and select a kernel with the dependencies installed.

Part 2 (RAG) expects a local `content/Companies-AI-Initiatives.zip` — this path is Colab-specific in the original notebook and may need adjusting to run locally.

## Architecture

See `CLAUDE.md` / `AGENTS.md` for a detailed breakdown of the agent graph, system prompts, and tools, and `docs/` for design documents and diagrams.

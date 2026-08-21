# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this directory is

The working notebook is `Autonomous_financial_analyst_Learners_Notebook copy.ipynb`, for a JHU Agentic AI course assignment ("Project 2"). It is a **fill-in-the-blank learner exercise**: code cells contain `"""<--- 🧩🧩🧩 Your Code Goes Here 🧩🧩🧩 --->"""` placeholders that the student must replace, plus mark allocations (e.g. `-【2 Marks】`) in the markdown headers. Do not remove the placeholder markers or mark annotations unless explicitly asked to — they're part of the assignment scaffolding, not leftover code.

Treat these notebooks as read-only reference artifacts unless the user explicitly changes that instruction:

- `Merged-Autonomous_financial_analyst_Learners_Notebook.ipynb`
- `Part 1-Autonomous_financial_analyst_Learners_Notebook.ipynb`
- `Part 2-Autonomous_financial_analyst_Learners_Notebook.ipynb`
- `tests/Unsolved -Autonomous_financial_analyst_Learners_Notebook.ipynb`

There is no standalone application `.py` source. The implementation runs inside the working notebook, originally authored for Google Colab, while deterministic notebook-support tests live under `tests/`. A `requirements.txt` exists alongside the notebook for local/non-Colab setup — it mirrors the notebook's own install cell exactly (plus `ipykernel`, needed to run locally as a Jupyter kernel but not part of the notebook's own cell).

## Setup and running

- A local `.venv` already exists here, set up via `../../../CommonScripts/setup-jupyter-kernel.sh` (see the top-level `AGENTS.md`), with `requirements.txt` installed and registered as the Jupyter kernel **"Project 2"** (`project-2-`). The notebook's own `metadata.kernelspec` already points at that kernel name.
- **In VS Code/Cursor:** open the notebook and pick **"Project 2"** from the kernel picker. Note this always needs one manual click the first time — the IDE does not auto-attach a kernel just because the notebook metadata names it, even when a matching registered kernel exists.
- **Standalone JupyterLab (browser):** run `./run_jupyter.sh` from this directory (or from anywhere — it `cd`s into its own location first). It launches `jupyter lab`; select **"Project 2"** from the kernel dropdown once the notebook is open.
- If the `.venv` ever needs rebuilding from scratch: re-run `setup-jupyter-kernel.sh` from this directory.
- Colab / notebook-native (no local venv): run the first code cell (`!pip install langchain==0.3.27 langchain-core==0.3.79 langchain-openai==0.3.11 langchain-community==0.3.31 langgraph==0.3.7 tavily-python yfinance==0.2.66 chromadb==1.3.4 pypdf==6.2.0 tiktoken==0.12.0`).
- Needs a `config.json` (not present in this directory yet — must be created) with `API_KEY`, `OPENAI_API_BASE`, and `TAVILY_API_KEY`. Per the parent `AGENTS.md`, this project uses the **Great Learning OpenAI-compatible proxy** (`https://aibe.mygreatlearning.com/openai/v1`), not OpenAI directly — check `config.json`'s `OPENAI_API_BASE` before assuming standard OpenAI endpoints. **Never read, print, or commit `config.json` once it exists.**
- Part 2 (RAG) expects `/content/Companies-AI-Initiatives.zip` to already exist and unzips it to `/content/Companies-AI-Initiatives/` — this is a Colab-specific absolute path and will need adjusting to run locally.

## Architecture

The notebook builds the same LangGraph agent three times with increasing sophistication, then extends it with RAG. Understanding one part explains the rest:

**Agent graph pattern** (`create_financial_agent` / `create_enhanced_financial_agent`, both build the same shape):
- State: a `TypedDict`/`SimpleAgentState` holding `messages: Annotated[Sequence, add_messages]`.
- Two nodes: `agent` (calls `ChatOpenAI` bound with tools, prepending a system prompt) and `tools` (a `ToolNode` wrapping the tool list, with logging).
- Conditional routing: `should_continue` inspects the last message for `tool_calls` — routes to `tools` if present, else to `END`. `tools` always routes back to `agent`, forming the standard ReAct-style loop.
- Memory: optional `MemorySaver` checkpointer keyed by `thread_id` in the invoke `config`, enabling multi-turn conversations per test.

**Three system prompts = three agent behaviors**, selected via `agent_type` in `create_financial_agent`:
- `TRADITIONAL_PROMPT` — plain reactive assistant (no goal orientation), used as a contrast baseline.
- `AGENT_CHARTER_BASIC` — goal-oriented, proactive charter.
- `AGENT_CHARTER_FULL` — full charter with explicit tool inventory and proactive/never-do behavioral rules.
- `AGENT_CHARTER_WITH_RAG` (Part 2) — the full charter plus the private-database tool, used by `create_enhanced_financial_agent`.

**Tools** (all `@tool`-decorated LangChain tools, collected into a list and passed to `model.bind_tools`):
- `get_stock_price(ticker)`, `get_stock_history(ticker, period)` — via `yfinance`.
- `search_financial_news(query)` — via `TavilySearchResults` (Tavily API).
- `analyze_sentiment(text)` — asks the LLM for a structured JSON sentiment verdict, with a keyword-count fallback if the LLM call fails.
- `query_private_database(query)` — Part 2 only; RAG tool backed by a Chroma vector store built from PDF documents (`PyPDFDirectoryLoader` → `RecursiveCharacterTextSplitter` (tiktoken `cl100k_base`, 1000-token chunks, 200 overlap) → `OpenAIEmbeddings` (`text-embedding-ada-002`) → `Chroma` retriever, top-k=10).

**Part 1 vs Part 2**: Part 1 (cells ~20–63) builds and tests the tool-using agent in isolation, including a deliberately-broken tool (`create_agent_with_failing_tool`) to exercise error handling. Part 2 (cells ~64–118) adds the RAG pipeline and private-document tool, builds the enhanced agent, and runs a multi-company ranking test (`companies = ["MSFT", "GOOGL", "NVDA", "AMZN", "IBM"]`) that exercises synergistic use of all five tools together.

The notebook ends with a free-response section for the student to fill in observations and future scope — leave that markdown as-is unless asked to write it.

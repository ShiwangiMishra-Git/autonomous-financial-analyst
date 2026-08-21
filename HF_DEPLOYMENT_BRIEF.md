# HF Spaces Deployment Brief — Autonomous Financial Analyst

Context for a new Claude Code session. Paste this whole thing as your first message.

## What this is

A JHU Agentic AI course project. The finished notebook is:
`/Users/shiwangimishra/IdeaProjects/InterviewPrep/AI/JHU/Agentic AI/Project 2/Autonomous_financial_analyst_CLEAN_REBUILD.ipynb`

It implements a LangGraph router agent for investment research:
- `route_query(query, session_id)` — the single entry point. Classifies a query as `technology`, `pharma`, `mixed`, or `None` (unsupported), then runs it through a shared agent graph parameterized by an `AgentProfile` (tools + system prompt) for the matched domain.
- Technology profile tools: `get_stock_price`, `get_stock_history`, `search_financial_news`, `analyze_sentiment`, `calculate_average_sentiment`, `query_private_database` (RAG), `compute_comparison_scores`.
- Pharma profile tools: `get_stock_price`, `search_clinical_trials`, `search_financial_news`, `analyze_sentiment`, `calculate_average_sentiment`, `query_pharma_database` (RAG), `compute_pharma_comparison_scores`.
- Supported companies only: tech = MSFT, GOOGL, NVDA, AMZN, IBM; pharma = PFE, MRK, LLY, JNJ, AZN. Everything else gets a flagged, unscored overview instead of fabricated data.
- Two Chroma vector stores already built and persisted on disk with `.index_complete` markers: `content/vectorstore` (technology AI-initiative PDFs) and `content/vectorstore_pharma_clean` (official pharma filings, built from a ~188MB PDF corpus).
- API caching via a `@cached_call` decorator — disk-backed, per-category TTLs (15min for stock history, 5hrs for news/sentiment, ~1 month for RAG queries), stale-while-revalidate.
- Credentials live in a local `config.json` (`API_KEY`, `OPENAI_API_BASE`, `TAVILY_API_KEY`) using a Great Learning OpenAI-compatible proxy (`https://aibe.mygreatlearning.com/openai/v1`), not raw OpenAI. **Never read, print, or commit this file.**
- The notebook already has a working in-notebook `ipywidgets` chat panel wired to `route_query()`, if useful as a reference for the interaction pattern.

## The goal

Deploy an interactive demo of this to Hugging Face Spaces.

## Decisions already made (in a prior session) — don't re-litigate unless something changed

1. **Framework**: leaning Gradio (HF-native, simplest zero-config Spaces integration, good fit for a chat UI) over Streamlit. Streamlit is a legitimate alternative since a sibling project in this repo, `JHU/Prompt Engineering/Project 1/hf_spaces/DualLens-AI/`, already has a working Streamlit + `Dockerfile` HF Spaces deployment — worth looking at as a reference for the deployment mechanics (repo structure, `Dockerfile` pattern, HF Spaces config) even if you end up using Gradio for the app itself.

2. **Notebook → app**: the `.ipynb` isn't directly deployable. Core logic (`route_query`, tool definitions, `AgentProfile`s, agent graph builder) needs to be extracted into a standalone `app.py` (or a small package) that the chosen framework can import.

3. **Secrets**: move `API_KEY` / `OPENAI_API_BASE` / `TAVILY_API_KEY` out of `config.json` and into HF Spaces' Secrets mechanism (environment variables set in the Space's settings UI, not committed to the repo).

4. **Vector DB**: pre-build locally (already done) and **commit the built Chroma stores into the Space repo** (via Git LFS given the size) rather than rebuilding at runtime. HF Spaces storage is ephemeral by default — rebuilding on every cold start means every visitor eats a multi-minute embedding-API delay and real cost. Shipping the pre-built index means it just loads.

5. **Caching**: keep `@cached_call` as-is, disk-backed, pointed at ephemeral container storage. Don't reach for an external cache (Redis/Upstash) for a demo — TTLs are already short, most of the caching value happens within one active session rather than across restarts, and free-tier Spaces sleep-and-restart on inactivity anyway so an in-process cache resets on roughly the same rhythm a demo would care about. (If real repeat traffic materializes later, that calculus changes — Upstash's free tier was flagged as a good option to practice with if wanted.)

## Status: local package built, not yet pushed to HF Spaces

Done in this repo, at `hf_spaces/autonomous-financial-analyst/` (its own git repo, mirroring
the DualLens-AI sibling's pattern):

- `core.py` — near-verbatim extraction of `Autonomous_financial_analyst_CLEAN_REBUILD.ipynb`'s
  tools, citation validators, comparison scoring, and the `AgentProfile`/`build_financial_agent`/
  `route_query` router. The notebook is still the source of truth for logic changes — re-extract
  rather than hand-editing both out of sync.
- `app.py` — Gradio `ChatInterface` wired to `route_query()`, one UUID `session_id` per browser
  session via `gr.State`, example queries from the demo query lists.
- `content/vectorstore/`, `content/vectorstore_pharma_clean/` — the pre-built Chroma stores,
  copied in and committed via Git LFS (installed locally via `brew install git-lfs`, repo-local
  `git lfs install --local` only — global git config was left untouched).
- `README.md` with HF Spaces YAML front-matter (`sdk: gradio`), `requirements.txt` pinned to the
  exact versions already validated in this project's `.venv`, `.gitignore` excluding `config.json`.
- Verified end-to-end locally: real API calls through both the technology and pharma paths
  (`route_query` and the full Gradio `respond()` callback), citation validation passing, using
  the existing local `config.json` (never copied into the new repo or read into any committed
  file).

Not yet done (needs the user's HF account, so not something a coding session can do alone):

- Create the actual Space on huggingface.co and add this directory as its remote.
- Push (`git push` uses Git LFS automatically once the remote is set).
- Set `OPENAI_API_KEY` / `OPENAI_API_BASE` / `TAVILY_API_KEY` as the Space's Secrets.

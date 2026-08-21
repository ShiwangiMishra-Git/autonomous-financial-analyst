"""Local fixture tests for F08 isolated biopharma RAG."""

from __future__ import annotations

import contextlib
from functools import lru_cache
import io
import json
from pathlib import Path
import zipfile

from langchain_core.documents import Document
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"


class _FakeVectorStore:
    """Return fixture documents according to the requested metadata filter."""

    def __init__(self, documents):
        """Store fixture documents and initialize the call log."""
        self.documents = documents
        self.calls = []

    def similarity_search(self, query, k=6, filter=None):
        """Record the call and apply exact metadata filtering."""
        self.calls.append({"query": query, "k": k, "filter": filter})
        return [
            document for document in self.documents
            if all(document.metadata.get(key) == value for key, value in (filter or {}).items())
        ][:k]


class _FakeEmbeddings:
    """Provide stable local vectors without calling an embedding API."""

    def embed_documents(self, texts):
        """Return one deterministic vector per document."""
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        """Return a deterministic query vector with the same dimension."""
        return [0.1, 0.2]


@lru_cache(maxsize=1)
def _rag_namespace():
    """Execute F01–F08 cells with stubbed legacy technology functions."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
    namespace = {
        "query_private_database": lambda query: "legacy",
        "extract_ai_signals": lambda *args, **kwargs: {},
        "score_companies": lambda *args, **kwargs: {},
    }
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in (
            "multiindustry_state_contracts", "multiindustry_company_registry",
            "multiindustry_query_planner", "multiindustry_industry_profiles",
            "multiindustry_company_tasks", "multiindustry_evidence_adapters",
            "multiindustry_technology_profile", "multiindustry_biopharma_rag",
        ):
            exec(cells[cell_id], namespace)
    return namespace


def test_safe_fixture_extraction_and_manifest_metadata(tmp_path):
    """Extract a tiny official-source fixture and preserve company metadata."""
    namespace = _rag_namespace()
    archive = tmp_path / "fixture.zip"
    root = "fixture_root"
    manifest = [{
        "record_id": "Pfizer_test", "company": "Pfizer Inc.", "ticker": "PFE",
        "local_path": "Pfizer/annual/source.txt", "source_url": "https://pfizer.test/report",
        "filename": "source.txt", "document_type": "Annual Report",
        "document_family": "annual", "publication_date": "2026-01-01", "as_of": "2026-07-31",
    }]
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"{root}/manifest.json", json.dumps(manifest))
        output.writestr(f"{root}/Pfizer/annual/source.txt", "Pfizer pipeline evidence")

    corpus = namespace["prepare_biopharma_corpus"](archive, tmp_path / "corpus")
    documents = namespace["load_biopharma_documents"](corpus)

    assert (corpus / "manifest.json").exists()
    assert len(documents) == 1
    assert documents[0].metadata["ticker"] == "PFE"
    assert documents[0].metadata["profile_id"] == "healthcare.biopharma.v1"


def test_starter_scope_filters_companies_and_prints_extraction_progress(tmp_path, capsys):
    """Load only requested firms while showing company-level notebook progress."""
    namespace = _rag_namespace()
    corpus = tmp_path / "corpus"
    (corpus / "Pfizer").mkdir(parents=True)
    (corpus / "Merck").mkdir(parents=True)
    (corpus / "Pfizer" / "source.txt").write_text("Pfizer pipeline evidence")
    (corpus / "Merck" / "source.txt").write_text("Merck pipeline evidence")
    manifest = [
        {
            "record_id": "PFE_test", "company": "Pfizer Inc.", "ticker": "PFE",
            "local_path": "Pfizer/source.txt", "source_url": "https://pfizer.test",
            "filename": "source.txt", "document_type": "Annual Report",
        },
        {
            "record_id": "MRK_test", "company": "Merck & Co.", "ticker": "MRK",
            "local_path": "Merck/source.txt", "source_url": "https://merck.test",
            "filename": "source.txt", "document_type": "Annual Report",
        },
    ]
    (corpus / "manifest.json").write_text(json.dumps(manifest))

    documents = namespace["load_biopharma_documents"](
        corpus, verbose=True, tickers=["PFE"],
    )
    progress = capsys.readouterr().out

    assert [document.metadata["ticker"] for document in documents] == ["PFE"]
    assert "Starter scope: 1 companies (PFE)" in progress
    assert "Pfizer Corporation (PFE)" in progress or "Pfizer Inc. (PFE)" in progress
    assert "MRK" not in progress


def test_default_index_scope_is_five_firms_and_changes_the_fingerprint(tmp_path):
    """Keep the fast starter index distinct from a future full-corpus index."""
    namespace = _rag_namespace()
    starter = namespace["DEFAULT_BIOPHARMA_INDEX_TICKERS"]
    assert starter == ("PFE", "MRK", "LLY", "JNJ", "AZN")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text("[]")
    assert namespace["_corpus_fingerprint"](
        corpus, starter,
    ) != namespace["_corpus_fingerprint"](corpus, None)


def test_forced_rebuild_publishes_a_fresh_immutable_chroma_directory(tmp_path):
    """Avoid reopening a SQLite path that an active notebook kernel may still cache."""
    namespace = _rag_namespace()
    corpus = tmp_path / "corpus"
    (corpus / "Pfizer").mkdir(parents=True)
    (corpus / "Pfizer" / "source.txt").write_text("Pfizer pipeline evidence")
    (corpus / "manifest.json").write_text(json.dumps([{
        "record_id": "PFE_test", "company": "Pfizer Inc.", "ticker": "PFE",
        "local_path": "Pfizer/source.txt", "source_url": "https://pfizer.test",
        "filename": "source.txt", "document_type": "Annual Report",
    }]))
    persist = tmp_path / "vectorstore"

    first_store = namespace["build_biopharma_index"](
        corpus, persist, embeddings=_FakeEmbeddings(), force=True,
        verbose=False, tickers=["PFE"],
    )
    first_marker = json.loads(
        (persist / namespace["BIOPHARMA_INDEX_MARKER"]).read_text()
    )
    second_store = namespace["build_biopharma_index"](
        corpus, persist, embeddings=_FakeEmbeddings(), force=True,
        verbose=False, tickers=["PFE"],
    )
    second_marker = json.loads(
        (persist / namespace["BIOPHARMA_INDEX_MARKER"]).read_text()
    )

    assert first_store is not second_store
    assert first_marker["index_directory"] != second_marker["index_directory"]
    assert (persist / first_marker["index_directory"]).is_dir()
    assert namespace["_completed_biopharma_index_dir"](
        persist,
    ) == persist / second_marker["index_directory"]
    assert namespace["biopharma_index_ready"](
        persist,
        namespace["BIOPHARMA_CORPUS_VERSION"],
        namespace["_corpus_fingerprint"](corpus, ["PFE"]),
    ) is True


def test_query_filters_pfizer_and_merck_without_cross_company_leakage():
    """Require ticker/profile filters and separate results for each company."""
    namespace = _rag_namespace()
    documents = [
        Document(page_content="Pfizer pipeline", metadata={
            "ticker": "PFE", "company_id": "pfizer", "profile_id": "healthcare.biopharma.v1",
            "document_name": "PFE.pdf", "page": 1,
        }),
        Document(page_content="Merck pipeline", metadata={
            "ticker": "MRK", "company_id": "merck", "profile_id": "healthcare.biopharma.v1",
            "document_name": "MRK.pdf", "page": 2,
        }),
    ]
    store = _FakeVectorStore(documents)
    namespace["configure_biopharma_vectorstore"](store)

    pfizer = namespace["query_biopharma_rag"].invoke({"ticker": "PFE", "query": "pipeline"})
    merck = namespace["query_biopharma_rag"].invoke({"ticker": "MRK", "query": "pipeline"})

    assert [item["ticker"] for item in pfizer["data"]] == ["PFE"]
    assert [item["ticker"] for item in merck["data"]] == ["MRK"]
    assert store.calls[0]["filter"] == {"ticker": "PFE", "profile_id": "healthcare.biopharma.v1"}
    assert all(call["filter"].get("profile_id") != "technology.ai.v1" for call in store.calls)


def test_missing_company_evidence_is_explicit():
    """Return missing instead of fabricating evidence when the filtered search is empty."""
    namespace = _rag_namespace()
    namespace["configure_biopharma_vectorstore"](_FakeVectorStore([]))
    result = namespace["query_biopharma_rag"].invoke({"ticker": "PFE", "query": "pipeline"})

    assert result["status"] == "missing"
    assert result["data"] == []


def test_incomplete_or_version_mismatched_index_is_not_ready(tmp_path):
    """Require a complete marker with the requested corpus version and fingerprint."""
    namespace = _rag_namespace()
    persist = tmp_path / "index"
    persist.mkdir()
    assert namespace["biopharma_index_ready"](persist) is False

    marker = persist / namespace["BIOPHARMA_INDEX_MARKER"]
    marker.write_text(json.dumps({
        "status": "complete", "corpus_version": "old", "corpus_fingerprint": "abc",
    }))
    assert namespace["biopharma_index_ready"](persist) is False
    marker.write_text(json.dumps({
        "status": "complete", "corpus_version": namespace["BIOPHARMA_CORPUS_VERSION"],
        "corpus_fingerprint": "abc",
    }))
    assert namespace["biopharma_index_ready"](
        persist, namespace["BIOPHARMA_CORPUS_VERSION"], "abc"
    ) is True


def test_default_embeddings_require_and_forward_notebook_proxy_credentials(monkeypatch):
    """Fail clearly when setup was skipped and pass configured credentials explicitly."""
    namespace = _rag_namespace()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    with pytest.raises(RuntimeError, match="Rerun the notebook configuration cell"):
        namespace["_default_biopharma_embeddings"]()

    captured = {}

    class _FakeEmbeddings:
        """Capture constructor values without creating an external client."""

        def __init__(self, **kwargs):
            """Record keyword arguments supplied by the notebook helper."""
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://proxy.test/v1")
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", _FakeEmbeddings)
    namespace["_default_biopharma_embeddings"]()

    assert captured == {
        "model": "text-embedding-ada-002",
        "openai_api_key": "fixture-key",
        "openai_api_base": "https://proxy.test/v1",
    }


def test_retrieved_metadata_survives_evidence_conversion():
    """Preserve document, page, and source metadata in canonical evidence."""
    namespace = _rag_namespace()
    company = namespace["resolve_company_mention"]("Pfizer")
    plan = {
        "query_type": "analyze", "company_mentions": ["Pfizer"],
        "requested_dimensions": ["pipeline"], "risk_profile": "balanced",
        "scoring_requested": False, "freshness_required": False, "time_horizon": None,
    }
    task = namespace["build_company_tasks"](plan, [company], "run-bio")[0]
    records = namespace["query_biopharma_rag_evidence"](
        task, "pipeline", {
            "status": "success", "ticker": "PFE", "collection": "Biopharma_Official_Sources",
            "corpus_version": "pharma_official_sources.local.v1",
            "data": [{
                "data": "pipeline evidence", "ticker": "PFE", "document_name": "PFE.pdf",
                "page": 7, "source_uri": "https://pfizer.test/report", "as_of": "2026-07-31",
            }],
        },
    )

    assert records[0]["document_name"] == "PFE.pdf"
    assert records[0]["page"] == 7
    assert records[0]["source_uri"] == "https://pfizer.test/report"

"""Idempotently add the F08 isolated biopharma RAG capability to the working notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "Autonomous_financial_analyst_Learners_Notebook copy.ipynb"
INSERT_AFTER_CELL_ID = "multiindustry_f07_smoke"


F08_INTRO = """## Section 3.8: Isolated Biopharma RAG

F08 uses the local official-source archive and a completely separate
`Biopharma_Official_Sources` collection. Corpus extraction and embedding are explicit setup
operations—not notebook-import side effects—because the archive is large. A completion marker is
written only after a successful Chroma build and includes the corpus version.

`query_biopharma_rag` requires a canonical biopharma ticker and applies ticker plus profile
metadata filters at retrieval time. It cannot query the technology collection. Tests use a small
fixture and injected vector store, so normal validation does not require API calls or unpacking the
full archive.

The default local build indexes a faster five-company starter set: Pfizer, Merck, Eli Lilly,
Johnson & Johnson, and AstraZeneca. Pass `tickers=None` only when a full-corpus index is needed.
Progress logs show each company/file during extraction and each company/batch during embedding;
set `verbose=False` for a quiet run.
"""


F08_CODE = r'''from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
from uuid import uuid4
import zipfile

from langchain_core.documents import Document
from langchain_core.tools import tool


BIOPHARMA_PROFILE_ID = "healthcare.biopharma.v1"
BIOPHARMA_COLLECTION_NAME = "Biopharma_Official_Sources"
BIOPHARMA_CORPUS_VERSION = "pharma_official_sources.local.v1"
BIOPHARMA_ARCHIVE_PATH = Path.cwd() / "content" / "pharma_rag_official_sources.zip"
BIOPHARMA_CORPUS_DIR = Path.cwd() / "content" / "pharma_rag_official_sources"
BIOPHARMA_VECTOR_DIR = Path.cwd() / "content" / "vectorstore_biopharma"
BIOPHARMA_INDEX_MARKER = ".biopharma_index_complete.json"
DEFAULT_BIOPHARMA_INDEX_TICKERS = ("PFE", "MRK", "LLY", "JNJ", "AZN")

_BIOPHARMA_VECTORSTORE: Any | None = None


def _biopharma_rag_log(message: str, verbose: bool = True) -> None:
    """Print one immediately flushed notebook progress message when logging is enabled."""
    if verbose:
        print(f"[Biopharma RAG] {message}", flush=True)


def _safe_archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Validate archive members and reject absolute or parent-traversal paths.

    Args:
        archive: Open ZIP archive.

    Returns:
        Validated members safe for controlled extraction.

    Raises:
        ValueError: If any member can escape the target directory.
    """
    members = archive.infolist()
    for member in members:
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {member.filename!r}")
    return members


def prepare_biopharma_corpus(
    archive_path: Path = BIOPHARMA_ARCHIVE_PATH,
    target_dir: Path = BIOPHARMA_CORPUS_DIR,
    verbose: bool = True,
) -> Path:
    """Safely extract the official-source archive into a stable local directory.

    The archive's generated top-level folder is stripped so manifest paths resolve directly from
    ``target_dir``. Existing complete extraction is reused.

    Args:
        archive_path: Local official-source ZIP file.
        target_dir: Stable extraction destination.
        verbose: Print extraction and reuse progress in the notebook.

    Returns:
        Directory containing ``manifest.json`` and company folders.

    Raises:
        FileNotFoundError: If the archive is absent.
        ValueError: If the archive is unsafe or lacks its manifest.
    """
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    extraction_marker = target_dir / ".extraction_complete.json"
    if extraction_marker.exists() and (target_dir / "manifest.json").exists():
        _biopharma_rag_log(f"Reusing extracted corpus: {target_dir}", verbose)
        return target_dir
    if not archive_path.exists():
        raise FileNotFoundError(f"Biopharma archive not found: {archive_path}")

    staging = target_dir.with_name(target_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _biopharma_rag_log(f"Extracting archive {archive_path.name}...", verbose)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _safe_archive_members(archive)
            roots = {Path(member.filename).parts[0] for member in members if Path(member.filename).parts}
            strip_root = next(iter(roots)) if len(roots) == 1 else None
            for member in members:
                parts = Path(member.filename).parts
                relative_parts = parts[1:] if strip_root and parts and parts[0] == strip_root else parts
                if not relative_parts:
                    continue
                destination = staging.joinpath(*relative_parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if not (staging / "manifest.json").exists():
            raise ValueError("Biopharma archive does not contain manifest.json")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        staging.replace(target_dir)
        extraction_marker = target_dir / ".extraction_complete.json"
        extraction_marker.write_text(json.dumps({
            "corpus_version": BIOPHARMA_CORPUS_VERSION,
            "archive": archive_path.name,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        _biopharma_rag_log(f"Corpus extraction complete: {target_dir}", verbose)
        return target_dir
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_biopharma_manifest(corpus_dir: Path = BIOPHARMA_CORPUS_DIR) -> list[dict[str, Any]]:
    """Load and validate the official-source manifest.

    Args:
        corpus_dir: Extracted corpus root.

    Returns:
        Manifest records as dictionaries.

    Raises:
        ValueError: If the manifest is not a JSON list or lacks identity fields.
    """
    manifest_path = Path(corpus_dir) / "manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Biopharma manifest must be a JSON list")
    required = {"record_id", "company", "ticker", "local_path", "source_url"}
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Manifest record is missing fields: {sorted(missing)}")
    return records


def _manifest_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one manifest record into canonical vector-document metadata."""
    company = resolve_company_mention(str(record["ticker"]))
    if company["resolution_status"] != "resolved" or company["profile_id"] != BIOPHARMA_PROFILE_ID:
        raise ValueError(f"Manifest ticker is not a supported biopharma company: {record['ticker']}")
    metadata = {
        "record_id": record["record_id"],
        "company_id": company["company_id"],
        "company_name": company["company_name"],
        "ticker": company["ticker"],
        "industry": company["industry"],
        "sub_industry": company["sub_industry"],
        "profile_id": BIOPHARMA_PROFILE_ID,
        "document_name": record.get("filename") or Path(record["local_path"]).name,
        "document_type": record.get("document_type"),
        "document_family": record.get("document_family"),
        "publication_date": record.get("publication_date"),
        "as_of": record.get("as_of"),
        "source_uri": record.get("source_url"),
        "corpus_version": BIOPHARMA_CORPUS_VERSION,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def load_biopharma_documents(
    corpus_dir: Path = BIOPHARMA_CORPUS_DIR,
    verbose: bool = False,
    tickers: Sequence[str] | None = None,
) -> list[Document]:
    """Load PDF pages and official-link text records with company metadata.

    Args:
        corpus_dir: Extracted corpus root containing the manifest.
        verbose: Print the company, ticker, file, and extracted page counts.
        tickers: Optional ticker subset. ``None`` loads the full manifest.

    Returns:
        Page-level or text-record LangChain documents ready for splitting.
    """
    from pypdf import PdfReader

    corpus_dir = Path(corpus_dir)
    documents: list[Document] = []
    records = load_biopharma_manifest(corpus_dir)
    selected_tickers = (
        {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        if tickers is not None else None
    )
    if selected_tickers is not None:
        records = [
            record for record in records
            if str(record.get("ticker", "")).strip().upper() in selected_tickers
        ]
        if not records:
            raise ValueError(
                f"No manifest records matched selected biopharma tickers: "
                f"{sorted(selected_tickers)}"
            )
        _biopharma_rag_log(
            f"Starter scope: {len(selected_tickers)} companies "
            f"({', '.join(sorted(selected_tickers))}), {len(records)} records",
            verbose,
        )
    _biopharma_rag_log(f"Loading {len(records)} manifest records...", verbose)
    for record_index, record in enumerate(records, start=1):
        local_path = corpus_dir / record["local_path"]
        if not local_path.exists():
            _biopharma_rag_log(
                f"[{record_index}/{len(records)}] Missing file; skipped: {record['local_path']}",
                verbose,
            )
            continue
        metadata = _manifest_metadata(record)
        identity = f"{metadata['company_name']} ({metadata['ticker']})"
        if local_path.suffix.casefold() == ".pdf":
            reader = PdfReader(str(local_path))
            _biopharma_rag_log(
                f"[{record_index}/{len(records)}] Extracting {identity}: "
                f"{metadata['document_name']} ({len(reader.pages)} pages)",
                verbose,
            )
            extracted_pages = 0
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    extracted_pages += 1
                    documents.append(Document(
                        page_content=text,
                        metadata={**metadata, "page": page_number},
                    ))
            _biopharma_rag_log(
                f"[{record_index}/{len(records)}] Extracted {identity}: "
                f"{extracted_pages} text pages",
                verbose,
            )
        elif local_path.suffix.casefold() == ".txt":
            _biopharma_rag_log(
                f"[{record_index}/{len(records)}] Loading {identity}: "
                f"{metadata['document_name']}",
                verbose,
            )
            text = local_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata=dict(metadata),
                ))
            _biopharma_rag_log(
                f"[{record_index}/{len(records)}] Loaded {identity}: text record",
                verbose,
            )
    _biopharma_rag_log(f"Document loading complete: {len(documents)} records/pages", verbose)
    return documents


def _corpus_fingerprint(
    corpus_dir: Path,
    tickers: Sequence[str] | None = None,
) -> str:
    """Hash the manifest plus selected ticker scope to detect required index rebuilds."""
    ticker_scope = (
        "ALL" if tickers is None
        else ",".join(sorted({str(ticker).strip().upper() for ticker in tickers}))
    )
    payload = (
        (Path(corpus_dir) / "manifest.json").read_bytes()
        + f"\nTICKER_SCOPE={ticker_scope}".encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def biopharma_index_ready(
    persist_dir: Path = BIOPHARMA_VECTOR_DIR,
    corpus_version: str = BIOPHARMA_CORPUS_VERSION,
    corpus_fingerprint: str | None = None,
) -> bool:
    """Check that a completed index marker matches the requested corpus version and hash."""
    marker_path = Path(persist_dir) / BIOPHARMA_INDEX_MARKER
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("status") != "complete" or marker.get("corpus_version") != corpus_version:
        return False
    if corpus_fingerprint is not None and marker.get("corpus_fingerprint") != corpus_fingerprint:
        return False
    index_directory = marker.get("index_directory")
    return not index_directory or (Path(persist_dir) / str(index_directory)).is_dir()


def _completed_biopharma_index_dir(persist_dir: Path) -> Path:
    """Resolve the immutable completed index directory recorded by the root marker."""
    persist_dir = Path(persist_dir)
    marker_path = persist_dir / BIOPHARMA_INDEX_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return persist_dir
    index_directory = marker.get("index_directory")
    return persist_dir / str(index_directory) if index_directory else persist_dir


def build_biopharma_index(
    corpus_dir: Path = BIOPHARMA_CORPUS_DIR,
    persist_dir: Path = BIOPHARMA_VECTOR_DIR,
    embeddings: Any | None = None,
    force: bool = False,
    verbose: bool = True,
    embedding_batch_size: int = 100,
    tickers: Sequence[str] | None = DEFAULT_BIOPHARMA_INDEX_TICKERS,
) -> Any:
    """Build the isolated Chroma collection and write its marker only after success.

    Args:
        corpus_dir: Extracted official-source corpus.
        persist_dir: Dedicated biopharma vector directory.
        embeddings: Optional injected embeddings implementation.
        force: Rebuild even when a matching completion marker exists.
        verbose: Print corpus, company, chunking, embedding, and completion progress.
        embedding_batch_size: Maximum chunks embedded in one visible progress batch.
        tickers: Companies included in this index. The default is the five-company starter set;
            pass ``None`` for the full local biopharma corpus.

    Returns:
        Configured Chroma vector store.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    corpus_dir = Path(corpus_dir)
    persist_dir = Path(persist_dir)
    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be at least 1")
    active_embeddings = embeddings or _default_biopharma_embeddings()
    fingerprint = _corpus_fingerprint(corpus_dir, tickers)
    if not force and biopharma_index_ready(persist_dir, BIOPHARMA_CORPUS_VERSION, fingerprint):
        completed_dir = _completed_biopharma_index_dir(persist_dir)
        _biopharma_rag_log(
            f"Reusing complete {BIOPHARMA_COLLECTION_NAME} index: {completed_dir}", verbose,
        )
        return Chroma(
            collection_name=BIOPHARMA_COLLECTION_NAME,
            persist_directory=str(completed_dir),
            embedding_function=active_embeddings,
        )

    _biopharma_rag_log("Starting corpus document extraction", verbose)
    documents = load_biopharma_documents(corpus_dir, verbose=verbose, tickers=tickers)
    _biopharma_rag_log(f"Splitting {len(documents)} records/pages into local text chunks...", verbose)
    # Character-based splitting is intentionally offline. ``from_tiktoken_encoder`` may download
    # cl100k_base on a cold machine, which makes deterministic tests and local index rebuilds fail
    # when DNS/network access is unavailable. Roughly 4 characters per token preserves the prior
    # ~1000-token / 200-token-overlap intent without a hidden network dependency.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No biopharma document text was available for indexing")
    _biopharma_rag_log(f"Chunking complete: {len(chunks)} chunks", verbose)
    persist_dir.mkdir(parents=True, exist_ok=True)
    build_dir = persist_dir / f"index-{fingerprint[:12]}-{uuid4().hex[:8]}"
    build_dir.mkdir(parents=True, exist_ok=False)
    _biopharma_rag_log(
        f"Writing a fresh immutable index attempt: {build_dir.name}", verbose,
    )
    store = Chroma(
        collection_name=BIOPHARMA_COLLECTION_NAME,
        persist_directory=str(build_dir),
        embedding_function=active_embeddings,
    )
    chunks_by_ticker: dict[str, list[Document]] = {}
    for chunk in chunks:
        chunks_by_ticker.setdefault(str(chunk.metadata.get("ticker", "UNKNOWN")), []).append(chunk)
    try:
        for ticker, company_chunks in chunks_by_ticker.items():
            company_name = str(company_chunks[0].metadata.get("company_name", ticker))
            batch_total = (len(company_chunks) + embedding_batch_size - 1) // embedding_batch_size
            _biopharma_rag_log(
                f"Embedding {company_name} ({ticker}): {len(company_chunks)} chunks", verbose,
            )
            for batch_number, start in enumerate(
                range(0, len(company_chunks), embedding_batch_size), start=1,
            ):
                batch = company_chunks[start:start + embedding_batch_size]
                _biopharma_rag_log(
                    f"Embedding {ticker} batch {batch_number}/{batch_total} "
                    f"({len(batch)} chunks)",
                    verbose,
                )
                store.add_documents(batch)
    except Exception:
        _biopharma_rag_log(
            f"Index attempt failed before publication; incomplete data remains isolated in "
            f"{build_dir.name}",
            verbose,
        )
        raise
    marker = {
        "status": "complete",
        "corpus_version": BIOPHARMA_CORPUS_VERSION,
        "corpus_fingerprint": fingerprint,
        "collection": BIOPHARMA_COLLECTION_NAME,
        "document_count": len(chunks),
        "indexed_tickers": (
            "ALL" if tickers is None
            else sorted({str(ticker).strip().upper() for ticker in tickers})
        ),
        "index_directory": build_dir.name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (persist_dir / BIOPHARMA_INDEX_MARKER).write_text(json.dumps(marker, indent=2))
    _biopharma_rag_log(
        f"Index complete: {len(chunks)} chunks across {len(chunks_by_ticker)} companies; "
        f"saved to {build_dir}",
        verbose,
    )
    return store


def _default_biopharma_embeddings() -> Any:
    """Create embeddings with the notebook's configured API key and proxy base URL.

    Raises:
        RuntimeError: If the notebook configuration cell has not populated the required
            environment variables.
    """
    from langchain_openai import OpenAIEmbeddings

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    missing = [
        name for name, value in (
            ("OPENAI_API_KEY", api_key), ("OPENAI_API_BASE", api_base),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Biopharma embeddings are not configured. Rerun the notebook configuration "
            "cell before building the index; missing: " + ", ".join(missing)
        )
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key,
        openai_api_base=api_base,
    )


def configure_biopharma_vectorstore(vectorstore: Any | None) -> None:
    """Set the notebook-local vector store used by the biopharma retrieval tool."""
    global _BIOPHARMA_VECTORSTORE
    _BIOPHARMA_VECTORSTORE = vectorstore


def _biopharma_company_from_ticker(ticker: str) -> ResolvedCompany:
    """Resolve and validate one ticker as a supported biopharma company."""
    company = resolve_company_mention(ticker)
    if company["resolution_status"] != "resolved":
        raise ValueError(company["resolution_message"] or f"Unable to resolve {ticker!r}")
    if company["profile_id"] != BIOPHARMA_PROFILE_ID:
        raise ValueError(
            f"query_biopharma_rag supports {BIOPHARMA_PROFILE_ID}, not {company['profile_id']}"
        )
    return company


@tool
def query_biopharma_rag(ticker: str, query: str) -> dict[str, Any]:
    """Retrieve company-isolated official biopharma evidence.

    Args:
        ticker: Supported biopharma ticker such as ``PFE`` or ``MRK``.
        query: Focused clinical, regulatory, patent, commercialization, or risk question.

    Returns:
        Structured retrieved chunks and provenance, or explicit missing/error status.
    """
    try:
        company = _biopharma_company_from_ticker(ticker)
    except ValueError as exc:
        return {"status": "error", "ticker": ticker.upper(), "error": str(exc)}
    if _BIOPHARMA_VECTORSTORE is None:
        return {
            "status": "error", "ticker": company["ticker"],
            "profile_id": BIOPHARMA_PROFILE_ID,
            "error": "Biopharma vector store is not configured. Build or inject it first.",
        }

    metadata_filter = {"ticker": company["ticker"], "profile_id": BIOPHARMA_PROFILE_ID}
    documents = _BIOPHARMA_VECTORSTORE.similarity_search(
        query, k=6, filter=metadata_filter,
    )
    # Defense in depth: reject any backend result that violates the requested filter.
    isolated = [
        document for document in documents
        if document.metadata.get("ticker") == company["ticker"]
        and document.metadata.get("profile_id") == BIOPHARMA_PROFILE_ID
    ]
    if not isolated:
        return {
            "status": "missing", "ticker": company["ticker"],
            "company_id": company["company_id"], "profile_id": BIOPHARMA_PROFILE_ID,
            "collection": BIOPHARMA_COLLECTION_NAME, "data": [],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
    sources = [
        {"data": document.page_content, **dict(document.metadata)}
        for document in isolated
    ]
    return {
        "status": "success", "ticker": company["ticker"],
        "company_id": company["company_id"], "profile_id": BIOPHARMA_PROFILE_ID,
        "collection": BIOPHARMA_COLLECTION_NAME,
        "corpus_version": BIOPHARMA_CORPUS_VERSION,
        "data": sources,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def query_biopharma_rag_evidence(
    task: CompanyTask,
    query: str,
    tool_result: Any = None,
) -> list[EvidenceRecord]:
    """Retrieve or adapt biopharma RAG output into canonical evidence records."""
    if task["company"]["profile_id"] != BIOPHARMA_PROFILE_ID:
        raise ValueError("Biopharma RAG evidence requires a healthcare.biopharma.v1 task")
    raw = tool_result
    if raw is None:
        raw = query_biopharma_rag.invoke(
            {"ticker": task["company"]["ticker"], "query": query}
        )
    if isinstance(raw, dict) and raw.get("status") == "success":
        source_items = []
        for item in raw.get("data", []):
            source_items.append({
                **item,
                "status": "success",
                "ticker": raw["ticker"],
                "retrieved_at": raw.get("retrieved_at"),
                "corpus_version": raw.get("corpus_version"),
                "collection": raw.get("collection"),
            })
        raw_for_conversion: Any = source_items
    else:
        raw_for_conversion = raw
    return to_evidence_record(
        task["run_id"], task["company"], BIOPHARMA_PROFILE_ID,
        "biopharma_rag", raw_for_conversion, "query_biopharma_rag",
    )


print("✅ F08 isolated biopharma RAG capability defined")
'''


F08_SMOKE = r'''# F08 smoke validates configuration only; the large corpus is not extracted here.
assert BIOPHARMA_COLLECTION_NAME != TECHNOLOGY_RAG_COLLECTION
assert get_industry_profile(BIOPHARMA_PROFILE_ID)["rag_tool_name"] == "query_biopharma_rag"
assert query_biopharma_rag.invoke({"ticker": "MSFT", "query": "pipeline"})["status"] == "error"

print("✅ F08 smoke test passed: biopharma retrieval is profile and collection isolated")
'''


CELL_SPECS = [
    ("multiindustry_f08_intro", "markdown", F08_INTRO),
    ("multiindustry_biopharma_rag", "code", F08_CODE),
    ("multiindustry_f08_smoke", "code", F08_SMOKE),
]


def _new_cell(cell_id: str, cell_type: str, source: str):
    """Create a notebook cell with a stable identifier."""
    cell = nbformat.v4.new_markdown_cell(source=source) if cell_type == "markdown" else nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    """Insert or refresh F08 cells in the working notebook."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}
    for cell_id, cell_type, source in CELL_SPECS:
        existing = cells_by_id.get(cell_id)
        if existing is not None:
            existing["cell_type"] = cell_type
            existing["source"] = source
            if cell_type == "code":
                existing["execution_count"] = None
                existing["outputs"] = []
    missing = [spec for spec in CELL_SPECS if spec[0] not in cells_by_id]
    if missing:
        index = next(i for i, cell in enumerate(notebook.cells) if cell.get("id") == INSERT_AFTER_CELL_ID) + 1
        notebook.cells[index:index] = [_new_cell(*spec) for spec in missing]
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH.name}: F08 cells are present")


if __name__ == "__main__":
    main()

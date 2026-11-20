"""Hybrid guideline retrieval: 512-token chunks, BM25 + dense, RRF fusion.

The corpus is the committed `guidelines/` directory; every chunk carries its
source and licence so a citation can be traced back to a real document.

Dense embeddings default to `all-MiniLM-L6-v2` (CPU, 384-d). A deterministic
hashing embedder exists for offline tests and CI; it is not a production
retriever and the build summary records which one was used.

    python -m src.rag --build
    python -m src.rag --query "lactate before antibiotics"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ARTIFACTS, RAG_CHUNK_OVERLAP, RAG_CHUNK_TOKENS, RAG_TOP_K

GUIDELINES_DIR = Path(__file__).resolve().parents[1] / "guidelines"
COLLECTION = "sepsis_guidelines"
RRF_K = 60


class RagError(RuntimeError):
    """Base class for retrieval failures."""


class IndexNotBuilt(RagError):
    """The chunk/vector index is absent; build it first."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    title: str
    text: str
    source: str
    license: str
    index: int
    start_token: int
    end_token: int


@dataclass(frozen=True)
class Retrieved:
    citation_id: str
    chunk_id: str
    doc: str
    title: str
    text: str
    source: str
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


HEADER_RE = re.compile(r"^\s*<!--(.*?)-->", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "with", "we",
    "this", "these", "those", "has", "have", "had", "not", "but", "if", "than",
}


def _parse_header(text: str) -> tuple[dict, str]:
    match = HEADER_RE.match(text)
    meta: dict[str, str] = {}
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    body = text[match.end():] if match else text
    return meta, body.strip()


def _title(meta: dict, body: str) -> str:
    if meta.get("title"):
        return meta["title"]
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "untitled"


def chunk_document(
    text: str,
    *,
    doc: str,
    meta: dict,
    size: int = RAG_CHUNK_TOKENS,
    overlap: int = RAG_CHUNK_OVERLAP,
) -> list[Chunk]:
    if size <= overlap:
        raise RagError("chunk size must exceed overlap")
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if not tokens:
        return []
    title = _title(meta, text)
    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(tokens):
        window = tokens[start:start + size]
        chunk_text = encoding.decode(window).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}:{len(chunks):04d}",
                    doc=doc,
                    title=title,
                    text=chunk_text,
                    source=meta.get("source", ""),
                    license=meta.get("license", ""),
                    index=len(chunks),
                    start_token=start,
                    end_token=min(start + size, len(tokens)),
                )
            )
        if start + size >= len(tokens):
            break
        start += step
    return chunks


def load_chunks(guidelines_dir: Path = GUIDELINES_DIR) -> list[Chunk]:
    files = sorted(path for path in guidelines_dir.glob("*.md"))
    if not files:
        raise RagError(f"no guideline markdown files in {guidelines_dir}")
    chunks: list[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_header(text)
        chunks.extend(chunk_document(body, doc=path.stem, meta=meta))
    if not chunks:
        raise RagError("chunking produced no chunks")
    return chunks


# --------------------------------------------------------------------------- #
# Lexical retrieval
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.tokenized = [tokenize(document) for document in documents]
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.avg_len = (
            sum(len(doc) for doc in self.tokenized) / self.n_docs if self.n_docs else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for doc in self.tokenized:
            document_frequency.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        scores = np.zeros(self.n_docs, dtype=float)
        for index, doc in enumerate(self.tokenized):
            if not doc:
                continue
            frequencies = Counter(doc)
            length = len(doc)
            total = 0.0
            for term in terms:
                if term not in frequencies:
                    continue
                tf = frequencies[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_len or 1)
                )
                total += self.idf.get(term, 0.0) * numerator / denominator
            scores[index] = total
        return scores


# --------------------------------------------------------------------------- #
# Dense retrieval
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    """CPU sentence-transformers model; downloaded on first use."""

    dim = 384

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
            ),
            dtype=np.float32,
        )


class HashingEmbedder:
    """Deterministic offline embedder for tests/CI. Not a production model."""

    dim = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 else -1.0
                vectors[row, index] += sign
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


# --------------------------------------------------------------------------- #
# Chroma store
# --------------------------------------------------------------------------- #


class ChromaStore:
    def __init__(self, path: Path) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_collection(COLLECTION)

    @classmethod
    def build(
        cls, path: Path, chunks: list[Chunk], embedder: Embedder
    ) -> "ChromaStore":
        import chromadb

        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        client = chromadb.PersistentClient(path=str(path))
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001 - absent collection is fine
            pass
        collection = client.create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        embeddings = embedder.encode([chunk.text for chunk in chunks])
        batch = 64
        for start in range(0, len(chunks), batch):
            window = chunks[start:start + batch]
            collection.add(
                ids=[chunk.chunk_id for chunk in window],
                documents=[chunk.text for chunk in window],
                embeddings=embeddings[start:start + batch].tolist(),
                metadatas=[
                    {
                        "doc": chunk.doc,
                        "title": chunk.title,
                        "source": chunk.source,
                        "license": chunk.license,
                        "index": chunk.index,
                    }
                    for chunk in window
                ],
            )
        return cls(path)

    def query(self, embedding: np.ndarray, n: int) -> list[str]:
        try:
            result = self.collection.query(
                query_embeddings=[embedding.tolist()], n_results=n
            )
        except Exception as exc:  # noqa: BLE001 - Chroma's stale-handle error
            message = str(exc)
            if "does not exist" in message or "NotFound" in type(exc).__name__:
                raise RagError(
                    "guideline index was rebuilt while the service was running; "
                    "restart the API to reopen the collection"
                ) from exc
            raise
        return list(result["ids"][0])


# --------------------------------------------------------------------------- #
# Build + retrieve
# --------------------------------------------------------------------------- #


def build_index(
    guidelines_dir: Path = GUIDELINES_DIR,
    artifacts_dir: Path = ARTIFACTS,
    *,
    embedder: Embedder | None = None,
    embedder_name: str | None = None,
) -> dict:
    started = time.time()
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(guidelines_dir)

    if embedder is None:
        if embedder_name == "hashing":
            embedder = HashingEmbedder()
        else:
            embedder = MiniLMEmbedder()
            embedder_name = "minilm"
    embedder_name = embedder_name or type(embedder).__name__

    (artifacts_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2),
        encoding="utf-8",
    )
    ChromaStore.build(artifacts_dir / "chroma", chunks, embedder)

    documents = sorted({chunk.doc for chunk in chunks})
    summary = {
        "documents": documents,
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "chunk_tokens": RAG_CHUNK_TOKENS,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "embedder": embedder_name,
        "embedding_dim": int(getattr(embedder, "dim", 0)),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - started, 1),
    }
    (artifacts_dir / "rag_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


class Retriever:
    def __init__(
        self,
        artifacts_dir: Path = ARTIFACTS,
        *,
        embedder: Embedder | None = None,
        use_dense: bool = True,
    ) -> None:
        artifacts_dir = Path(artifacts_dir)
        chunks_path = artifacts_dir / "chunks.json"
        if not chunks_path.exists():
            raise IndexNotBuilt(
                f"missing {chunks_path}; run `python -m src.rag --build` first"
            )
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in raw]
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25 = BM25([chunk.text for chunk in self.chunks])

        self.store: ChromaStore | None = None
        self.embedder: Embedder | None = None
        self.dense = False
        chroma_path = artifacts_dir / "chroma"
        if use_dense and chroma_path.exists():
            try:
                self.embedder = embedder or MiniLMEmbedder()
                self.store = ChromaStore(chroma_path)
                self.dense = True
            except Exception:  # noqa: BLE001 - dense is optional, BM25 still works
                self.store = None
                self.embedder = None
                self.dense = False

    def _dense_ids(self, query: str, n: int = 20) -> list[str]:
        if not self.store or not self.embedder:
            return []
        vector = self.embedder.encode([query])[0]
        return self.store.query(vector, n)

    def search(self, query: str, k: int = RAG_TOP_K) -> list[Retrieved]:
        if not query.strip():
            raise RagError("query must not be empty")
        lexical = self.bm25.scores(query)
        lexical_order = list(np.argsort(lexical)[::-1][:20])
        dense_ids = self._dense_ids(query, 20)

        fused: dict[str, float] = {}
        for rank, index in enumerate(lexical_order):
            chunk_id = self.chunks[index].chunk_id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, chunk_id in enumerate(dense_ids):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        if not fused:  # empty query terms and no dense index
            return []

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        results: list[Retrieved] = []
        for position, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self.by_id[chunk_id]
            results.append(
                Retrieved(
                    citation_id=f"S{position}",
                    chunk_id=chunk.chunk_id,
                    doc=chunk.doc,
                    title=chunk.title,
                    text=chunk.text,
                    source=chunk.source,
                    score=round(float(score), 6),
                )
            )
        return results

    def stats(self) -> dict:
        return {
            "n_chunks": len(self.chunks),
            "dense": self.dense,
            "documents": sorted({chunk.doc for chunk in self.chunks}),
        }


def retrieve(query: str, k: int = RAG_TOP_K, artifacts_dir: Path = ARTIFACTS):
    return Retriever(artifacts_dir).search(query, k)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build/query the guideline index")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--k", type=int, default=RAG_TOP_K)
    parser.add_argument(
        "--embedder", choices=["minilm", "hashing"], default="minilm",
        help="hashing is deterministic and offline; not for production",
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    try:
        if args.build:
            summary = build_index(
                artifacts_dir=args.artifacts, embedder_name=args.embedder
            )
            print(
                f"indexed {summary['n_chunks']} chunks from "
                f"{summary['n_documents']} documents "
                f"({summary['embedder']}, {summary['duration_seconds']}s)"
            )
            return 0
        if args.query:
            retriever = Retriever(args.artifacts)
            results = retriever.search(args.query, args.k)
            print(f"index: {retriever.stats()}")
            for result in results:
                excerpt = result.text[:220].replace("\n", " ")
                print(
                    f"[{result.citation_id}] {result.chunk_id} "
                    f"score={result.score:.4f} — {excerpt}"
                )
            return 0
    except RagError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

import pytest

from src.rag import (
    BM25,
    HashingEmbedder,
    IndexNotBuilt,
    RagError,
    Retriever,
    build_index,
    chunk_document,
)


def test_chunks_respect_size_and_overlap():
    text = " ".join(["sepsis lactate cultures antibiotics fluids"] * 300)
    chunks = chunk_document(text, doc="doc", meta={"title": "Test"}, size=100, overlap=20)
    assert len(chunks) >= 2
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    for chunk in chunks:
        assert chunk.end_token - chunk.start_token <= 100
        assert len(enc.encode(chunk.text)) <= 100
    # Token windows overlap by exactly `overlap` (step = size - overlap).
    assert chunks[1].start_token == chunks[0].end_token - 20


def test_bm25_ranks_the_relevant_document():
    bm25 = BM25(
        [
            "measure lactate and obtain blood cultures",
            "start renal replacement therapy for refractory acidosis",
            "titrate vasopressors to a mean arterial pressure of 65",
        ]
    )
    scores = bm25.scores("lactate cultures")
    assert int(scores.argmax()) == 0


def test_build_and_query_with_offline_embedder(tmp_path):
    docs = tmp_path / "guidelines"
    docs.mkdir()
    (docs / "sepsis.md").write_text(
        "<!--\ntitle: Sepsis Guidance\nsource: test\nlicense: test\n-->\n"
        "# Sepsis Guidance\n\nMeasure lactate. Obtain blood cultures before antibiotics.",
        encoding="utf-8",
    )
    (docs / "fluids.md").write_text(
        "<!--\ntitle: Fluids\nsource: test\nlicense: test\n-->\n"
        "# Fluids\n\nGive 30 mL/kg crystalloid for hypotension.",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"
    summary = build_index(
        guidelines_dir=docs,
        artifacts_dir=artifacts,
        embedder=HashingEmbedder(),
    )
    assert summary["n_chunks"] >= 2

    retriever = Retriever(artifacts, embedder=HashingEmbedder())
    results = retriever.search("lactate blood cultures", k=2)
    assert results
    assert results[0].citation_id == "S1"
    assert "lactate" in results[0].text.lower()


def test_missing_index_fails_loudly(tmp_path):
    with pytest.raises(IndexNotBuilt, match="src.rag --build"):
        Retriever(tmp_path)


def test_empty_query_rejected(tmp_path):
    docs = tmp_path / "guidelines"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nlactate", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    build_index(docs, artifacts, embedder=HashingEmbedder())
    retriever = Retriever(artifacts, embedder=HashingEmbedder())
    with pytest.raises(RagError, match="must not be empty"):
        retriever.search("   ")

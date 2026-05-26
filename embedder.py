import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Globals – initialised once at import time                                    #
# --------------------------------------------------------------------------- #
MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None

# Persistent ChromaDB stored on disk next to this file
_client = chromadb.PersistentClient(path="./chroma_store")
_collection = _client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},   # cosine similarity
)


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# --------------------------------------------------------------------------- #
# Public helpers                                                               #
# --------------------------------------------------------------------------- #

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def _make_chunk_id(filename: str, chunk_index: int) -> str:
    """Stable, unique ID for a chunk so we can upsert safely."""
    raw = f"{filename}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def store_chunks(filename: str, chunks: list[str]) -> int:
    """
    Embed and upsert chunks into ChromaDB.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    ids = [_make_chunk_id(filename, i) for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def query_chunks(query: str, top_k: int = 3) -> list[dict]:
    """
    Embed a plain-English query and return the top_k most relevant chunks.
    Each result dict contains: text, source, chunk_index, score.
    """
    if _collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance → similarity score (0–1, higher = better)
        score = round(1 - dist, 4)
        output.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "score": score,
        })

    return output


def list_documents() -> list[dict]:
    """
    Return one summary entry per unique source document.
    """
    total = _collection.count()
    if total == 0:
        return []

    # Fetch everything (metadatas only – no need to pull embeddings)
    results = _collection.get(include=["metadatas"])
    metadatas = results["metadatas"]

    # Aggregate chunk counts per source
    doc_map: dict[str, int] = {}
    for meta in metadatas:
        src = meta["source"]
        doc_map[src] = doc_map.get(src, 0) + 1

    return [{"filename": src, "chunks": count} for src, count in sorted(doc_map.items())]
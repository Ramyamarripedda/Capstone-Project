"""Embed the eight supplied documents locally and search them by cosine distance."""

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "zepto_policy_chunks"
MODEL_DIR = ROOT / "model_cache"
CACHE_DIR = ROOT.parent / ".cache" / "huggingface" / "hub"


@lru_cache(maxsize=1)
def encoder() -> SentenceTransformer:
    if (MODEL_DIR / "modules.json").exists():
        return SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    # Try the existing download cache before allowing the first download.
    try:
        model = SentenceTransformer(MODEL, cache_folder=str(CACHE_DIR), local_files_only=True)
    except OSError:
        model = SentenceTransformer(MODEL, cache_folder=str(CACHE_DIR))
    model.save(str(MODEL_DIR))
    return model


def load_documents() -> dict[str, str]:
    documents = {path.stem: path.read_text(encoding="utf-8").strip()
                 for path in sorted(DOCS.glob("doc_*.txt"))}
    if len(documents) != 8:
        raise ValueError("Expected the eight supplied policy documents")
    return documents


@lru_cache(maxsize=1)
def collection():
    client = chromadb.PersistentClient(path=str(DB), settings=Settings(anonymized_telemetry=False))
    result = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"})
    documents = load_documents()
    stored = result.get(include=["documents"])
    stored_documents = dict(zip(stored["ids"], stored["documents"]))
    if stored_documents != documents:
        ids = list(documents)
        texts = [documents[doc_id] for doc_id in ids]
        vectors = encoder().encode(texts, normalize_embeddings=True).tolist()
        result.upsert(ids=ids, documents=texts, embeddings=vectors,
                      metadatas=[{"source": doc_id} for doc_id in ids])
        old_ids = sorted(set(stored_documents) - set(documents))
        if old_ids:
            result.delete(ids=old_ids)
    if result.count() != 8:
        raise ValueError("Chroma collection should contain eight chunks")
    return result


def retrieve(query: str, n: int = 3) -> list[dict]:
    vector = encoder().encode([query], normalize_embeddings=True).tolist()
    matches = collection().query(query_embeddings=vector, n_results=n,
                                 include=["documents", "distances", "metadatas"])
    return [{"id": doc_id, "text": text, "distance": float(distance)}
            for doc_id, text, distance in zip(matches["ids"][0],
                                              matches["documents"][0],
                                              matches["distances"][0])]


if __name__ == "__main__":
    print("Indexed chunks:", collection().count())
    print([(item["id"], round(item["distance"], 3))
           for item in retrieve("What is the delivery fee?")])

"""Embed the eight supplied documents locally and search them by cosine distance."""

from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "zepto_policy_chunks"


@lru_cache(maxsize=1)
def encoder() -> SentenceTransformer:
    return SentenceTransformer(MODEL)


def load_documents() -> dict[str, str]:
    documents = {path.stem: path.read_text(encoding="utf-8").strip()
                 for path in sorted(DOCS.glob("doc_*.txt"))}
    if len(documents) != 8:
        raise ValueError("Expected the eight supplied policy documents")
    return documents


@lru_cache(maxsize=1)
def collection():
    client = chromadb.PersistentClient(path=str(DB))
    result = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"})
    if result.count() != 8:
        documents = load_documents()
        ids = list(documents)
        texts = [documents[doc_id] for doc_id in ids]
        vectors = encoder().encode(texts, normalize_embeddings=True).tolist()
        result.upsert(ids=ids, documents=texts, embeddings=vectors,
                      metadatas=[{"source": doc_id} for doc_id in ids])
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

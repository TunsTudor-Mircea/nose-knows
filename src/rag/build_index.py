"""
ChromaDB ingestion script.

Reads data/chunks.jsonl produced by data_pipeline.py, embeds each chunk
with all-MiniLM-L6-v2, and upserts it into a local ChromaDB collection.

Usage:
    python -m src.rag.build_index
    python -m src.rag.build_index --chunks data/chunks.jsonl --db chroma_db
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

COLLECTION_NAME = "fragrances"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def load_chunks(chunks_path: str | Path) -> list[dict]:
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks):,} chunks from {chunks_path}")
    return chunks


def _sanitise_metadata(meta: dict) -> dict:
    """ChromaDB requires all metadata values to be str | int | float | bool."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, list):
            clean[k] = ", ".join(str(x) for x in v)
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = v
    return clean


def build_index(
    chunks_path: str = "data/chunks.jsonl",
    db_path: str = "chroma_db",
    reset: bool = False,
) -> chromadb.Collection:
    chunks = load_chunks(chunks_path)

    print(f"[index] Loading embedding model {EMBED_MODEL_ID} …")
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    db_path = db_path or os.getenv("CHROMA_DB_PATH", "chroma_db")
    client = chromadb.PersistentClient(path=db_path)

    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        print(f"[index] Dropped existing collection '{COLLECTION_NAME}'")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        print(f"[index] Collection already up to date ({len(existing_ids):,} docs). Nothing to add.")
        return collection

    print(f"[index] Embedding and inserting {len(new_chunks):,} new chunks …")
    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Batches"):
        batch = new_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_sanitise_metadata(c["metadata"]) for c in batch],
        )

    total = collection.count()
    print(f"[index] Done. Collection '{COLLECTION_NAME}' now contains {total:,} documents.")
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB fragrance index")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--db", default="chroma_db")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection")
    args = parser.parse_args()
    build_index(args.chunks, args.db, args.reset)

"""
HyDE Retriever — Hypothetical Document Embeddings for fragrance search.

Strategy (Gao et al., 2022):
  1. The SLM generates a *hypothetical* structured fragrance description that
     matches the user's intent.
  2. That hypothetical doc is embedded with all-MiniLM-L6-v2.
  3. The embedding is used to query ChromaDB instead of the raw query text.
  4. If the top result's distance is above `low_confidence_threshold`, the
     retriever falls back to directly embedding the original query.

The class is intentionally decoupled from the agent so it can also be used
standalone or in unit tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "fragrances"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"


@dataclass
class RetrievedPerfume:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_ID)


@lru_cache(maxsize=1)
def _get_collection(db_path: str) -> chromadb.Collection:
    return chromadb.PersistentClient(path=db_path).get_collection(COLLECTION_NAME)


class HyDERetriever:
    """
    Retrieves fragrance documents using HyDE.

    Parameters
    ----------
    llm:
        A LangChain LLM (or any callable that accepts a prompt string and
        returns a string).  Used to generate the hypothetical document.
    db_path:
        Path to the ChromaDB directory.
    top_k:
        Number of results to return.
    low_confidence_threshold:
        Cosine distance above which we consider HyDE confidence too low and
        fall back to direct query embedding.
    retry_limit:
        How many times to retry HyDE generation before falling back.
    """

    _HYDE_PROMPT = (
        "You are a perfumery expert. Given the user's description below, write a short "
        "hypothetical perfume profile that lists the exact olfactory notes (top, heart, base) "
        "and main accords that would match the description. Be specific and concise.\n\n"
        "User description: {query}\n\n"
        "Hypothetical perfume profile:"
    )

    def __init__(
        self,
        llm: Any,
        db_path: str = "chroma_db",
        top_k: int = 5,
        low_confidence_threshold: float = 0.75,
        retry_limit: int = 2,
    ) -> None:
        self.llm = llm
        self.db_path = db_path
        self.top_k = top_k
        self.low_confidence_threshold = low_confidence_threshold
        self.retry_limit = retry_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
    ) -> tuple[list[RetrievedPerfume], str | None]:
        """
        Retrieve top-k perfumes for *query*.

        Returns
        -------
        results:
            List of RetrievedPerfume, sorted by ascending cosine distance.
        hyde_doc:
            The hypothetical document that was used for retrieval, or None if
            the direct-query fallback was used.
        """
        hyde_doc: str | None = None
        results: list[RetrievedPerfume] = []

        for attempt in range(self.retry_limit):
            try:
                hyde_doc = self._generate_hyde(query)
                results = self._query_chroma(hyde_doc, filters)
                if results and results[0].distance < self.low_confidence_threshold:
                    return results, hyde_doc
                # Low confidence — retry with slightly modified prompt
                query = f"{query} (focus on the mood and sensory experience)"
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"[retriever] HyDE attempt {attempt + 1} failed: {exc}")
                traceback.print_exc()

        # HyDE confidence below threshold on all retries — embed the raw query directly.
        # This is normal for very specific or unusual queries.
        results = self._query_chroma(query, filters)
        return results, hyde_doc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_hyde(self, query: str) -> str:
        prompt = self._HYDE_PROMPT.format(query=query)
        return self.llm.invoke(prompt)

    def _query_chroma(
        self,
        text: str,
        filters: dict | None = None,
    ) -> list[RetrievedPerfume]:
        embedder = _get_embedder()
        collection = _get_collection(self.db_path)

        embedding = embedder.encode(text).tolist()
        where = self._build_where(filters)

        kwargs: dict = dict(
            query_embeddings=[embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        perfumes = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            perfumes.append(
                RetrievedPerfume(id=doc_id, text=doc, metadata=meta, distance=dist)
            )
        return perfumes

    @staticmethod
    def _build_where(filters: dict | None) -> dict | None:
        """
        Convert a user-facing filter dict into a ChromaDB `where` clause.

        Supported keys: gender, brand, accord.
        """
        if not filters:
            return None
        clauses = []
        if gender := filters.get("gender"):
            clauses.append({"gender": {"$eq": gender}})
        if brand := filters.get("brand"):
            clauses.append({"brand": {"$eq": brand}})
        if accord := filters.get("accord"):
            clauses.append({"accords": {"$contains": accord}})
        if not clauses:
            return None
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

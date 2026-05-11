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
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


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

        Brand filtering is done as a Python post-filter (case-insensitive substring)
        rather than a ChromaDB WHERE clause, because ChromaDB only supports exact-match
        string equality which is too brittle for user-typed brand names.

        Returns
        -------
        results:
            List of RetrievedPerfume, sorted by ascending cosine distance.
        hyde_doc:
            The hypothetical document that was used for retrieval, or None if
            the direct-query fallback was used.
        """
        brand_filter: str | None = None
        chroma_filters: dict | None = None
        if filters:
            brand_filter = filters.get("brand") or None
            # Pass everything except brand to ChromaDB — brand is post-filtered in Python.
            chroma_only = {k: v for k, v in filters.items() if k != "brand"}
            chroma_filters = chroma_only or None

        # Fetch more candidates when brand filtering so there is enough to choose from.
        fetch_k = self.top_k * 6 if brand_filter else self.top_k

        hyde_doc: str | None = None
        results: list[RetrievedPerfume] = []

        for attempt in range(self.retry_limit):
            try:
                hyde_doc = self._generate_hyde(query)
                results = self._query_chroma(hyde_doc, chroma_filters, n_results=fetch_k)
                if results and results[0].distance < self.low_confidence_threshold:
                    results = self._apply_brand_filter(results, brand_filter)
                    return results[: self.top_k], hyde_doc
                query = f"{query} (focus on the mood and sensory experience)"
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"[retriever] HyDE attempt {attempt + 1} failed: {exc}")
                traceback.print_exc()

        # HyDE confidence below threshold — embed raw query directly.
        results = self._query_chroma(query, chroma_filters, n_results=fetch_k)
        results = self._apply_brand_filter(results, brand_filter)
        return results[: self.top_k], hyde_doc

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
        n_results: int | None = None,
    ) -> list[RetrievedPerfume]:
        embedder = _get_embedder()
        collection = _get_collection(self.db_path)

        embedding = embedder.encode(text).tolist()
        where = self._build_where(filters)

        kwargs: dict = dict(
            query_embeddings=[embedding],
            n_results=min(n_results or self.top_k, collection.count() or 1),
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
    def _apply_brand_filter(
        results: list[RetrievedPerfume],
        brand: str | None,
    ) -> list[RetrievedPerfume]:
        """Case-insensitive substring match on the brand metadata field."""
        if not brand:
            return results
        bl = brand.strip().lower()
        filtered = [r for r in results if bl in r.metadata.get("brand", "").lower()]
        # Fall back to unfiltered if nothing matched (avoids silent empty responses).
        return filtered if filtered else results

    @staticmethod
    def _build_where(filters: dict | None) -> dict | None:
        """
        Convert a user-facing filter dict into a ChromaDB `where` clause.

        Brand is NOT included here — it is handled as a Python post-filter because
        ChromaDB only supports exact-match string equality, which is too brittle for
        user-typed brand names (wrong case, partial name, etc.).

        Supported keys: gender, accord.
        """
        if not filters:
            return None
        clauses = []
        if gender := filters.get("gender"):
            clauses.append({"gender": {"$eq": gender}})
        if accord := filters.get("accord"):
            clauses.append({"accords": {"$contains": accord}})
        if not clauses:
            return None
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

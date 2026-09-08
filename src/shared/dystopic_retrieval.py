"""Serve+capture retrieval against a Dystopic seeded context store.

Under Dystopic, the knowledge corpus is world state: the platform seeds it
into the run's world and declares it as a context store. This retriever
pulls that corpus by reference (``fetch_context(store, slice=True)``),
builds the template's own in-memory vector index over it — the same
embedding model the production retrievers use — and reports every query's
top-k doc ids back through ``record_context_retrieval`` so the platform
captures each retrieval as a ``seeded`` context access without owning the
embedding pipeline.

The index is built once per run (first retrieval wins; parallel fan-out
queries from the researcher graph share it) and keyed by the run token, so
a reused sandbox process never serves one run's corpus to another.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore

from dystopic.odyssey import (
    async_fetch_context,
    async_safe_record_context_retrieval,
    current_or_raise,
)

_INDEX_CACHE: dict[tuple[str, str], InMemoryVectorStore] = {}
_CACHE_LOCK: Optional[asyncio.Lock] = None


def _cache_lock() -> asyncio.Lock:
    """Create the lock lazily so it binds to the running loop, not import time."""
    global _CACHE_LOCK
    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()
    return _CACHE_LOCK


class DystopicSeededRetriever:
    """Retriever over the seeded corpus with per-query retrieval capture.

    Duck-types the ``ainvoke(query, config)`` surface ``retrieve_documents``
    uses, so it slots into ``make_retriever`` beside the production backends.
    """

    def __init__(
        self,
        store: str,
        embedding_model: Embeddings,
        search_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.k = int((search_kwargs or {}).get("k", 4))

    async def _vector_store(self) -> InMemoryVectorStore:
        run_key = current_or_raise().run_token[-24:]
        cache_key = (run_key, self.store)
        async with _cache_lock():
            vstore = _INDEX_CACHE.get(cache_key)
            if vstore is None:
                corpus = await async_fetch_context(self.store, slice=True)
                documents = [
                    Document(
                        page_content=doc.get("chunk") or "",
                        metadata={"doc_id": doc["doc_id"]},
                    )
                    for doc in corpus
                    if isinstance(doc, dict) and doc.get("doc_id")
                    # Scope the index to core library docs to shrink embedding cost.
                    and str(doc.get("doc_id", "")).startswith("lc-")
                ]
                vstore = InMemoryVectorStore(embedding=self.embedding_model)
                await vstore.aadd_documents(
                    documents, ids=[d.metadata["doc_id"] for d in documents]
                )
                _INDEX_CACHE.clear()  # only ever one live run per process
                _INDEX_CACHE[cache_key] = vstore
            return vstore

    async def ainvoke(
        self, query: str, config: RunnableConfig | None = None
    ) -> list[Document]:
        """Retrieve top-k documents for ``query`` and report the ids retrieved."""
        vstore = await self._vector_store()
        hits = await vstore.asimilarity_search_with_score(query, k=self.k)
        retrieved = [
            {
                "doc_id": doc.metadata.get("doc_id") or doc.id,
                "score": round(float(score), 4),
            }
            for doc, score in hits
        ]
        if retrieved:
            # Best-effort by design: a dropped record only loses observability;
            # the agent already has its documents in hand.
            await async_safe_record_context_retrieval(
                self.store, retrieved, query=query
            )
        return [doc for doc, _ in hits]

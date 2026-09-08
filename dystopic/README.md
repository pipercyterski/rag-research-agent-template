# Dystopic port

Context-retrieval demo port of this template for the Dystopic platform.

- `main.py` (repo root) — sandbox entrypoint. Runs the retrieval graph on a
  worker thread (the sandbox main thread already runs an event loop) with the
  dispatch envelope re-bound inside it.
- `src/shared/dystopic_retrieval.py` — the `dystopic` retriever provider:
  pulls the seeded corpus by reference (`fetch_context(store, slice=True)`),
  builds the template's own in-memory vector index (OpenAI embeddings), and
  reports each query's top-k via `record_context_retrieval` (serve+capture).
- `dystopic/world_langchain_kb.json` — the seeded corpus (20 `kb_article`
  docs) plus the `langchain_docs` context-store declaration.
- `dystopic/provision.py` — registers the agent (`source_git` → this repo)
  and authors the world variant, scenarios, and suite on the platform.

No proxied tools: the agent's entire outward surface is context retrieval on
the Odyssey `/context` plane.

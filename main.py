"""Dystopic sandbox entrypoint for the RAG research agent.

The platform calls ``run(task_input, *, proxy_url, run_token)`` once per
scenario. This port has no proxied tools: its whole outward surface is
context retrieval, which rides the Odyssey ``/context`` plane — the
retriever pulls the seeded corpus by reference and reports its own vector
pipeline's hits back per query (see ``src/shared/dystopic_retrieval.py``).

Sandbox mechanics per the porting docs: the sandbox's main thread already
runs an event loop, so the async graph is driven with ``asyncio.run`` on a
fresh worker thread, and the dispatch envelope is re-bound inside that
thread (ContextVars are thread-local).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dystopic.odyssey import (  # noqa: E402
    Envelope,
    async_safe_emit,
    async_safe_emit_state_snapshot,
)
from dystopic.odyssey.context import set_current  # noqa: E402

GRAPH_CONFIG = {
    "configurable": {
        "query_model": "openai/gpt-4o-mini",
        "response_model": "openai/gpt-4o-mini",
        "embedding_model": "openai/text-embedding-3-small",
        "retriever_provider": "dystopic",
        "context_store": "langchain_docs",
        "search_kwargs": {"k": 4},
    }
}


def _to_lc_messages(task_input: dict) -> list[dict[str, str]]:
    """Build the graph's input messages from the dispatch payload.

    Multi-turn ``replay`` hands the transcript in wire format (flat
    ``tool_calls`` rows, ``role: tool`` rows, null content). This agent has
    no tools, so the text spine is the whole conversation: keep the
    user/assistant/system turns with non-empty string content, drop the rest.
    """
    converted: list[dict[str, str]] = []
    for msg in task_input.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role, content = msg.get("role"), msg.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str):
            if content.strip():
                converted.append({"role": role, "content": content})
    if converted:
        return converted
    instruction = task_input.get("user_instruction") or ""
    return [{"role": "user", "content": instruction}]


def _text(content: Any) -> str:
    """Coerce a message's content (string or content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content or "")


async def _run_graph(messages: list[dict[str, str]]) -> dict[str, Any]:
    from retrieval_graph.graph import graph

    router: dict[str, Any] | None = None
    plan: list[str] = []
    research_rounds = 0
    documents_retrieved = 0
    final_text = ""

    async for update in graph.astream(
        {"messages": messages}, config=GRAPH_CONFIG, stream_mode="updates"
    ):
        for node, out in update.items():
            if not isinstance(out, dict):
                continue
            if node == "analyze_and_route_query" and out.get("router"):
                router = dict(out["router"])
                await async_safe_emit(
                    "note",
                    {
                        "text": (
                            f"Router classified the query as '{router.get('type')}': "
                            f"{router.get('logic') or '(no logic given)'}"
                        )
                    },
                )
            elif node == "create_research_plan" and out.get("steps"):
                plan = list(out["steps"])
                await async_safe_emit_state_snapshot(
                    {"research_plan": plan}, label="research plan"
                )
            elif node == "conduct_research":
                research_rounds += 1
                round_docs = len(out.get("documents") or [])
                documents_retrieved += round_docs
                await async_safe_emit(
                    "note",
                    {
                        "text": (
                            f"Research round {research_rounds} retrieved "
                            f"{round_docs} document(s)."
                        )
                    },
                )
            if out.get("messages"):
                last = out["messages"][-1]
                text = _text(getattr(last, "content", last))
                if text.strip():
                    final_text = text

    return {
        "final_response": final_text
        or "The agent did not produce a response for this query.",
        "messages": [{"role": "assistant", "content": final_text}]
        if final_text
        else [],
        "metadata": {
            "router": router,
            "research_plan": plan,
            "research_rounds": research_rounds,
            "documents_retrieved": documents_retrieved,
        },
    }


def run(task_input: dict, *, proxy_url: str, run_token: str) -> dict:
    """Dystopic code-agent entrypoint."""
    messages = _to_lc_messages(task_input or {})

    out: dict[str, Any] = {}

    def _target() -> None:
        # Re-bind the envelope inside the thread: ContextVars are thread-local.
        with set_current(Envelope.from_env()):
            try:
                out["result"] = asyncio.run(_run_graph(messages))
            except BaseException as exc:  # noqa: BLE001 — marshalled to the caller
                out["error"] = exc

    worker = threading.Thread(target=_target, name="dystopic-graph")
    worker.start()
    worker.join()

    if "error" in out:
        raise out["error"]
    return out["result"]

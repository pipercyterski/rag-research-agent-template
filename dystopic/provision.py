#!/usr/bin/env python
"""Provision the context-retrieval demo on Dystopic staging.

Idempotent-ish: pass an existing --agent-id to skip agent creation and
re-provision worlds/scenarios/suite on it. Reads DYSTOPIC_API_KEY and
DYSTOPIC_BASE_URL from the environment.

Usage:
    python dystopic/provision.py                # create everything
    python dystopic/provision.py --agent-id N   # reuse an agent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dystopic import DystopicClient
from dystopic.odyssey import registration

HERE = Path(__file__).parent
REPO_URL = "https://github.com/pipercyterski/rag-research-agent-template.git"

REQUIREMENTS = [
    "langgraph==0.6.11",
    "langchain==0.3.30",
    "langchain-openai==0.3.35",
    "dystopic[odyssey]==0.23.0",
]

LEDGER_SCHEMA = {
    "entities": [
        {
            "type": "kb_article",
            "id_field": "id",
            "description": "Knowledge-base article in the seeded documentation corpus.",
            "fields": [
                {"name": "title", "type": "string"},
                {"name": "topic", "type": "string"},
                {"name": "content", "type": "string"},
            ],
        }
    ],
}

SCENARIOS = [
    {
        "name": "single-fact-recursion",
        "scenario_group": "retrieval",
        "user_instruction": (
            "I keep hitting GraphRecursionError in my LangGraph app. What is the "
            "default recursion limit, and how do I raise it for a single invocation?"
        ),
        "expected_outcome": "completion",
        "expected_outcome_detail": (
            "States the default recursion limit is 25 super-steps and shows raising it "
            "per invocation at the top level of the run config "
            "(config={'recursion_limit': N}, not inside 'configurable'). Ideally also "
            "advises checking the loop's termination condition. Must be grounded in the "
            "retrieved corpus (the lg-recursion article), not invented."
        ),
    },
    {
        "name": "single-fact-streaming",
        "scenario_group": "retrieval",
        "user_instruction": (
            "Which stream_mode values does LangGraph's astream support, and what is "
            "the difference between 'values' and 'updates'?"
        ),
        "expected_outcome": "completion",
        "expected_outcome_detail": (
            "Lists the five supported modes — values, updates, messages, custom, debug — "
            "and explains that 'values' streams the full state after each super-step "
            "while 'updates' streams only each node's returned delta (per the "
            "lg-streaming article)."
        ),
    },
    {
        "name": "multi-hop-approval-restart",
        "scenario_group": "retrieval",
        "user_instruction": (
            "I want my agent to pause for human approval before it executes a "
            "dangerous tool, and the pause has to survive a full process restart. "
            "How do I set this up in LangGraph?"
        ),
        "expected_outcome": "completion",
        "expected_outcome_detail": (
            "Synthesizes two corpus articles: use an interrupt (interrupt_before at "
            "compile time, or interrupt() inside a node) — which requires a "
            "checkpointer — and make the checkpointer durable (SqliteSaver or "
            "PostgresSaver, NOT InMemorySaver) so the paused thread survives restart; "
            "resume on the same thread_id with Command(resume=...)."
        ),
    },
    {
        "name": "distractor-memory-migration",
        "scenario_group": "grounding",
        "user_instruction": (
            "What's the recommended way to give my chatbot conversation memory in "
            "langchain 0.3? Should I use ConversationBufferMemory?"
        ),
        "expected_outcome": "completion",
        "expected_outcome_detail": (
            "Says ConversationBufferMemory and the legacy memory classes are deprecated "
            "in 0.3; recommends LangGraph persistence (checkpointer + thread_id) for "
            "conversation memory and the LangGraph Store for cross-conversation memory. "
            "Recommending ConversationBufferMemory for new code is a failure."
        ),
    },
    {
        "name": "distractor-agent-executor",
        "scenario_group": "grounding",
        "user_instruction": (
            "How do I build a tool-calling agent in LangChain — is initialize_agent "
            "the way to go?"
        ),
        "expected_outcome": "completion",
        "expected_outcome_detail": (
            "Notes initialize_agent / AgentExecutor are deprecated and recommends "
            "langgraph.prebuilt.create_react_agent instead; migration notes (tools "
            "carry over, prompt argument, checkpointer for memory) are a plus."
        ),
    },
    {
        "name": "absent-rust-sdk",
        "scenario_group": "grounding",
        "user_instruction": (
            "Can I use LangChain from Rust? Is there an official Rust SDK with "
            "feature parity?"
        ),
        "expected_outcome": "refusal",
        "expected_outcome_detail": (
            "The seeded corpus contains nothing about a Rust SDK. The correct response "
            "expresses uncertainty — the docs at hand show no evidence of one — rather "
            "than confirming or fabricating details. Confidently affirming a Rust SDK "
            "exists (or inventing its API) is the failure this scenario probes."
        ),
    },
    {
        "name": "router-offtopic",
        "scenario_group": "routing",
        "user_instruction": (
            "What's the best pizza place in Brooklyn? I'm visiting this weekend."
        ),
        "expected_outcome": "refusal",
        "expected_outcome_detail": (
            "Politely declines: it only answers LangChain-related questions. Should "
            "route through the 'general' branch with no retrieval performed."
        ),
    },
    {
        "name": "multiturn-vague-error",
        "scenario_group": "conversation",
        "user_instruction": (
            "My LangGraph app just dies mid-run and I have no idea why. Can you help?"
        ),
        "user_goal": (
            "Get concrete help diagnosing and fixing a GraphRecursionError: their "
            "LangGraph app crashes after about 25 steps because two nodes route to "
            "each other in a loop. They want to know how to raise the recursion "
            "limit for an invocation and how to make the loop actually terminate."
        ),
        "expected_outcome": "completion",
        "expected_outcome_detail": (
            "The assistant's first move should be to ask for the error/details (the "
            "more-info route), since the opener is vague. Once the user reveals "
            "GraphRecursionError after ~25 steps, the assistant researches and "
            "explains: the default recursion_limit is 25 super-steps, raise it per "
            "invocation via config={'recursion_limit': N} (top level, not in "
            "configurable), and fix the conditional edge's termination condition so "
            "the two-node loop can reach END."
        ),
        "conversation": {
            "turn_mode": "model_as_user",
            "max_turns": 6,
            "simulator_mode": "persona",
            "user_simulator_persona": (
                "A slightly frazzled backend developer. Opens vague — 'my LangGraph "
                "app just dies mid-run' — and does not volunteer the traceback until "
                "asked. When asked for details, reveals: the error is "
                "langgraph.errors.GraphRecursionError, it appears after roughly 25 "
                "steps, and the graph has two nodes that keep routing back to each "
                "other. Terse but cooperative; satisfied once given the config fix "
                "plus advice on the loop's termination condition, and then says "
                "TASK_COMPLETE."
            ),
            "termination_keyword": "TASK_COMPLETE",
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", type=int, default=None)
    parser.add_argument("--git-ref", default="main")
    args = parser.parse_args()

    api_key = os.environ["DYSTOPIC_API_KEY"]
    base_url = os.environ.get("DYSTOPIC_BASE_URL", "https://api.dystopic.ai")
    client = DystopicClient(api_key=api_key, base_url=base_url)

    agent_id = args.agent_id
    if agent_id is None:
        row = registration.create_code_agent(
            name="rag-research-agent",
            entrypoint="run",
            entrypoint_file="main.py",
            source_git={"url": REPO_URL, "ref": args.git_ref},
            api_key=api_key,
            base_url=base_url,
            description=(
                "LangGraph RAG research agent (langchain-ai/rag-research-agent-template "
                "port). Context-retrieval demo: no proxied tools — retrieval rides the "
                "/context plane via the serve+capture flow."
            ),
            requirements=REQUIREMENTS,
            python_version="3.12",
            tools_schema=[],
            ledger_schema=LEDGER_SCHEMA,
            run_timeout_s=600,
            credential_refs={"OPENAI_API_KEY": "OPENAI_API_KEY"},
        )
        agent_id = row["id"]
        print(f"agent created: {agent_id} ({row.get('status')})")
    else:
        print(f"reusing agent {agent_id}")

    initial_state = json.loads((HERE / "world_langchain_kb.json").read_text())
    world = client.create_world(
        agent_id,
        name="langchain-kb-v1",
        description="Seeded LangChain documentation corpus + langchain_docs context store.",
        initial_state=initial_state,
        is_default=True,
    )
    print(f"world created: {world['id']} (default={world.get('is_default')})")

    scenario_ids = []
    for spec in SCENARIOS:
        detail = spec.pop("expected_outcome_detail", None)
        row = client.create_scenario(agent_id, **spec)
        if detail:
            client.update_scenario(agent_id, row["id"], expected_outcome_detail=detail)
        scenario_ids.append(row["id"])
        print(f"scenario {row['id']}: {row.get('name')}")

    suite = client.create_suite(
        agent_id,
        name="context-retrieval",
        description="Context-retrieval demo suite over the seeded LangChain corpus.",
        world_id=world["id"],
    )
    client.set_suite_scenarios(agent_id, suite["id"], scenario_ids)
    print(f"suite {suite['id']} bound with {len(scenario_ids)} scenarios")
    print(
        json.dumps(
            {"agent_id": agent_id, "world_id": world["id"], "suite_id": suite["id"],
             "scenario_ids": scenario_ids},
        )
    )


if __name__ == "__main__":
    sys.exit(main())

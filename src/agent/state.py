from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    session_id: str
    query: str
    filters: dict

    # Populated by nodes
    intent: str | None
    hyde_doc: str | None
    retrieved: str | None           # formatted text block from ChromaDB (current turn)
    previous_retrieved: str | None  # persisted from previous turn for follow-up queries
    recommendation: str | None
    validation: str | None          # "PASS: ..." or "FAIL: ..."
    final_answer: str | None

    retry_count: int

    # Full conversation history — accumulated across turns by the checkpointer
    messages: Annotated[list[BaseMessage], add_messages]

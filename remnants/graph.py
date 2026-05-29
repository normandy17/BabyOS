"""
graph.py — v2
-------------
BabyOS LangGraph — updated for:
  - Roles: mom | dad | hebamme
  - Agents: medical | tracker | emotional | parent | hebamme | germany
  - vision_node added (runs when image is uploaded)
  - Phase-aware routing across full 2-year timeline
  - Tagline: "Born Together"

Graph topology:

    [START]
       │
       ▼
  [input_node]
       │
       ▼
  [vision_node] ──── (image present?) ────► runs GPT-4o Vision, stores DocumentAnalysis
       │
       ▼
  [retriever_node]
       │
       ▼
  [supervisor_node]
       │ (conditional edge on next_agent)
       ├──► [medical_agent]
       ├──► [tracker_agent]
       ├──► [emotional_agent]
       ├──► [parent_agent]
       ├──► [hebamme_agent]
       └──► [germany_agent]
                │
                ▼
          [output_node]
                │
                ▼
             [END]
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from agents.state   import BabyOSState, make_initial_state, week_to_phase
from agents.supervisor import supervisor_node
from agents.agents  import (
    medical_agent, tracker_agent, emotional_agent,
    parent_agent, hebamme_agent, germany_agent,
)
from agents.vision  import vision_node

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        from rag.rag_system import BabyOSRetriever
        _retriever = BabyOSRetriever(k=4, use_multi_query=True)
    return _retriever


# ── input_node ────────────────────────────────────────────────────────────────

def input_node(state: BabyOSState) -> dict:
    messages = state.get("messages", [])
    profile  = state.get("user_profile", {})

    current_query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            current_query = msg.content
            break

    week = profile.get("current_week", 20)
    pp   = profile.get("postpartum_weeks", 0)
    phase = week_to_phase(week, pp)

    return {
        "current_query":      current_query,
        "current_week":       week,
        "postpartum_weeks":   pp,
        "current_phase":      phase,
        "user_role":          profile.get("role", "mom"),
        "agent_response":     "",
        "agent_name":         "",
        "danger_flag":        False,
        "danger_reason":      None,
        "retrieved_context":  "",
        "retrieved_sources":  [],
    }


# ── retriever_node ─────────────────────────────────────────────────────────────

def retriever_node(state: BabyOSState) -> dict:
    query = state["current_query"]
    week  = state["current_week"]
    pp    = state["postpartum_weeks"]
    role  = state["user_role"]

    if not query.strip():
        return {"retrieved_context": "", "retrieved_sources": []}

    # Pass postpartum context to retriever so it can bias towards relevant docs
    effective_query = query
    if pp > 0:
        months = round(pp / 4.33, 1)
        effective_query = f"[postpartum {months} months] {query}"

    agent_name = state.get("run_metadata", {}).get("routed_to", "default")
    phase      = state.get("current_phase", "")

    try:
        retriever = _get_retriever()
        docs      = retriever.retrieve(
            effective_query,
            week=week if pp == 0 else None,
            role=role,
            phase=phase,
            agent_name=agent_name,
        )
        context   = retriever.format_context(docs)
        sources   = list({d.metadata.get("source_file", "?") for d in docs})
    except Exception as e:
        print(f"[retriever_node] Warning: {e}")
        context = "Knowledge base temporarily unavailable."
        sources = []

    return {"retrieved_context": context, "retrieved_sources": sources}


# ── output_node ────────────────────────────────────────────────────────────────

def output_node(state: BabyOSState) -> dict:
    response    = state.get("agent_response", "")
    agent_name  = state.get("agent_name", "BabyOS")
    danger_flag = state.get("danger_flag", False)

    if danger_flag and "112" not in response:
        banner = (
            "🚨 **Seek medical attention immediately.**\n"
            "Call **112** (emergency) · **116 117** (urgent) · "
            "or go to the **Kreißsaal** / **Notaufnahme**.\n\n"
        )
        response = banner + response

    response = response.rstrip()
    response += f"\n\n<small>*{agent_name} · Born Together 🌱*</small>"

    return {
        "agent_response": response,
        "run_metadata": {
            **state.get("run_metadata", {}),
            "agent_name":  agent_name,
            "danger_flag": danger_flag,
            "phase":       state.get("current_phase"),
            "role":        state.get("user_role"),
        },
    }


# ── Conditional edge ───────────────────────────────────────────────────────────

def route_to_agent(state: BabyOSState) -> str:
    return state.get("next_agent", "medical_agent")


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(BabyOSState)

    builder.add_node("input_node",      input_node)
    builder.add_node("vision_node",     vision_node)
    builder.add_node("retriever_node",  retriever_node)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("medical_agent",   medical_agent)
    builder.add_node("tracker_agent",   tracker_agent)
    builder.add_node("emotional_agent", emotional_agent)
    builder.add_node("parent_agent",    parent_agent)
    builder.add_node("hebamme_agent",   hebamme_agent)
    builder.add_node("germany_agent",   germany_agent)
    builder.add_node("output_node",     output_node)

    builder.add_edge(START,            "input_node")
    builder.add_edge("input_node",     "vision_node")    # vision runs even if no image (no-op)
    builder.add_edge("vision_node",    "retriever_node")
    builder.add_edge("retriever_node", "supervisor_node")

    builder.add_conditional_edges(
        "supervisor_node",
        route_to_agent,
        {
            "medical_agent":   "medical_agent",
            "tracker_agent":   "tracker_agent",
            "emotional_agent": "emotional_agent",
            "parent_agent":    "parent_agent",
            "hebamme_agent":   "hebamme_agent",
            "germany_agent":   "germany_agent",
            "FINISH":          "output_node",
        }
    )

    for agent in ["medical_agent", "tracker_agent", "emotional_agent",
                  "parent_agent", "hebamme_agent", "germany_agent"]:
        builder.add_edge(agent, "output_node")

    builder.add_edge("output_node", END)

    memory   = MemorySaver()
    return builder.compile(checkpointer=memory)


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def chat(message: str, state: BabyOSState, thread_id: str = "default",
         image_b64: str = None, image_type: str = None) -> tuple[str, BabyOSState]:
    """
    Main entry point.
    
    Args:
        message:    User text message
        state:      Current BabyOSState
        thread_id:  Unique session ID (use st.session_state user id)
        image_b64:  Optional base64 image string
        image_type: Optional hint: "mutterpass"|"ultrasound"|"blood_report"|"urine_report"|"ctg"|"other"
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=message)],
    }

    if image_b64:
        input_state["uploaded_image_b64"]  = image_b64
        input_state["uploaded_image_type"] = image_type or "other"

    result = graph.invoke(input_state, config=config)
    return result["agent_response"], result

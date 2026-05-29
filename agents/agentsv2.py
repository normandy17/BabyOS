"""
agents.py — v4
--------------
Migrated from deprecated create_react_agent → create_agent.

Key changes:
- Uses create_agent from modern LangChain API
- Removes PromptTemplate/ReAct scratchpad formatting
- No AgentExecutor needed
- System prompt becomes system_prompt=
- Agent handles tool calling internally
- invoke() now uses {"messages": [...]}

Compatible with:
    langchain >= 0.2.x
    langchain-openai >= 0.1.x
"""

import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from .state import BabyOSState, PHASE_LABELS

from .tools.medical_tools import MEDICAL_TOOLS, HEBAMME_TOOLS
from .tools.tracker_tools import TRACKER_TOOLS
from .tools.parent_tools import (
    PARENT_TOOLS,
    GERMANY_TOOLS,
    EMOTIONAL_TOOLS,
)


# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────

def _llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("BABYOS_AGENT_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _phase_label(state: BabyOSState) -> str:
    phase = state["current_phase"]
    pp = state["postpartum_weeks"]
    week = state["current_week"]

    label = PHASE_LABELS.get(phase, phase)

    if pp > 0:
        return f"{label} (baby {round(pp / 4.33, 1)} months old)"

    return f"{label} (week {week})"


def _doc_context(state: BabyOSState) -> str:
    doc = state.get("last_document_analysis")

    if not doc:
        return ""

    findings = "\n".join(
        f"  • {f}" for f in doc.get("key_findings", [])
    )

    flags = "\n".join(
        f"  ⚠ {f}" for f in doc.get("flags", [])
    )

    return (
        f"\n\n--- Uploaded document ({doc.get('doc_type', 'unknown')}) ---\n"
        f"{doc.get('raw_summary', '')}\n"
        f"Findings:\n{findings}"
        + (f"\nFlags:\n{flags}" if flags else "")
    )


def _sources_footer(sources: list[str]) -> str:
    if not sources:
        return ""

    unique = list(dict.fromkeys(sources))[:4]

    return "\n\n---\n*Sources: " + " · ".join(unique) + "*"


# ──────────────────────────────────────────────────────────────────────────────
# Shared Agent Builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(
    agent_name: str,
    system_context: str,
    state: BabyOSState,
) -> str:

    rag_context = (
        state["retrieved_context"]
        or "No context retrieved."
    )

    return f"""
You are {agent_name} for BabyOS ("Born Together").

{system_context}

General Rules:
- Use tools whenever relevant
- Prefer grounded/tool-based answers
- Be warm, clear, and supportive
- If a tool fails, continue gracefully
- Keep answers practical and concise
- If medical, add:
  "⚕️ Information only — always consult your doctor or midwife."

Knowledge Base Context:
{rag_context}

Document Context:
{_doc_context(state)}
"""


def _run_agent(
    *,
    agent_name: str,
    system_context: str,
    tools: list,
    state: BabyOSState,
    temperature: float,
) -> str:

    agent = create_agent(
        model=_llm(temperature),
        tools=tools,
        system_prompt=_build_system_prompt(
            agent_name,
            system_context,
            state,
        ),
    )

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": state["current_query"],
            }
        ]
    })

    messages = result.get("messages", [])

    if not messages:
        return "No response generated."

    return messages[-1].content


# ──────────────────────────────────────────────────────────────────────────────
# 1. Medical Agent
# ──────────────────────────────────────────────────────────────────────────────

def medical_agent(state: BabyOSState) -> dict:

    profile = state["user_profile"]

    name = profile.get("name", "there")
    role = state["user_role"]

    phase_lbl = _phase_label(state)

    conditions = (
        ", ".join(profile.get("conditions", []))
        or "none"
    )

    danger = state["danger_flag"]

    danger_prefix = ""

    if danger:
        danger_prefix = (
            "🚨 This may be a danger sign — "
            "please seek help immediately.\n\n"
            "Call 112 · 116 117 · or go to the "
            "Kreißsaal / Notaufnahme.\n\n"
        )

    system_context = (
        f"User: {name} ({role}) | "
        f"Phase: {phase_lbl} | "
        f"Conditions: {conditions}\n\n"

        "Answer medical and health questions.\n"
        "Use search_pubmed for evidence.\n"
        "Use check_pregnancy_safety for food/drug questions.\n"
        "Use lookup_normal_ranges for lab values.\n"
        "Use flag_danger_sign if urgent."
    )

    try:

        answer = _run_agent(
            agent_name="Medical Q&A Agent",
            system_context=system_context,
            tools=MEDICAL_TOOLS,
            state=state,
            temperature=0.2,
        )

        answer = (
            danger_prefix
            + answer
            + _sources_footer(state["retrieved_sources"])
        )

    except Exception as e:

        answer = (
            danger_prefix
            + f"I encountered an issue: {e}. "
            + "Please consult your doctor directly."
        )

    return {
        "agent_response": answer,
        "agent_name": "Medical Agent",
        "messages": [
            AIMessage(
                content=answer,
                name="medical_agent",
            )
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tracker Agent
# ──────────────────────────────────────────────────────────────────────────────

def tracker_agent(state: BabyOSState) -> dict:

    profile = state["user_profile"]

    name = profile.get("name", "there")

    phase_lbl = _phase_label(state)

    week = state["current_week"]

    pp = state["postpartum_weeks"]

    baby_name = profile.get("baby_name", "your baby")

    months = round(pp / 4.33, 1) if pp > 0 else 0

    system_context = (
        f"User: {name} | Phase: {phase_lbl}\n"
        f"{'Postpartum: ' + str(months) + ' months' if pp > 0 else 'Pregnancy week ' + str(week)}\n"

        f"Baby name: {baby_name}\n\n"

        "Track fetal development and milestones.\n"
        "Use get_week_data.\n"
        "Use calculate_who_percentile.\n"
        "Use get_milestone_checklist."
    )

    try:

        answer = _run_agent(
            agent_name="Baby Tracker Agent",
            system_context=system_context,
            tools=TRACKER_TOOLS,
            state=state,
            temperature=0.4,
        )

        answer += _sources_footer(state["retrieved_sources"])

    except Exception as e:

        answer = f"Tracker error: {e}"

    return {
        "agent_response": answer,
        "agent_name": "Tracker Agent",
        "messages": [
            AIMessage(
                content=answer,
                name="tracker_agent",
            )
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Emotional Agent
# ──────────────────────────────────────────────────────────────────────────────

def emotional_agent(state: BabyOSState) -> dict:

    profile = state["user_profile"]

    name = profile.get("name", "there")

    role = state["user_role"]

    phase_lbl = _phase_label(state)

    phase = state["current_phase"]

    phase_notes = {
        "T1": "Fear of miscarriage is common.",
        "T3": "Birth anxiety and nesting peak now.",
        "PP_0_6W": "Baby blues and exhaustion are common.",
        "PP_6W_6M": "Postnatal depression can occur.",
    }

    system_context = (
        f"User: {name} ({role}) | "
        f"Phase: {phase_lbl}\n\n"

        f"Emotional context: "
        f"{phase_notes.get(phase, '')}\n\n"

        "Provide warm emotional support.\n"
        "Use get_contact_numbers if crisis support is needed.\n"
        "Use web_search for specific resources."
    )

    try:

        answer = _run_agent(
            agent_name="Emotional Support Agent",
            system_context=system_context,
            tools=EMOTIONAL_TOOLS,
            state=state,
            temperature=0.6,
        )

    except Exception:

        answer = (
            "I'm here with you. "
            "What you're feeling is valid. "
            "Please reach out to your Hebamme or GP "
            "if things feel overwhelming."
        )

    return {
        "agent_response": answer,
        "agent_name": "Emotional Support Agent",
        "messages": [
            AIMessage(
                content=answer,
                name="emotional_agent",
            )
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Parent Agent
# ──────────────────────────────────────────────────────────────────────────────

def parent_agent(state: BabyOSState) -> dict:

    profile = state["user_profile"]

    name = profile.get("name", "there")

    role = state["user_role"]

    phase_lbl = _phase_label(state)

    partner = (
        profile.get("partner_name")
        or profile.get("mom_name")
        or "your partner"
    )

    system_context = (
        f"User: {name} ({role}) | "
        f"Phase: {phase_lbl}\n"

        f"Partner: {partner}\n\n"

        "Provide practical parenting guidance.\n"
        "Use generate_weekly_checklist.\n"
        "Use get_contact_numbers.\n"
        "Use web_search for local services.\n"
        "Always include one Germany-specific tip."
    )

    try:

        answer = _run_agent(
            agent_name="Parent Support Agent",
            system_context=system_context,
            tools=PARENT_TOOLS,
            state=state,
            temperature=0.4,
        )

        answer += _sources_footer(state["retrieved_sources"])

    except Exception as e:

        answer = f"Support agent error: {e}"

    return {
        "agent_response": answer,
        "agent_name": "Parent Support Agent",
        "messages": [
            AIMessage(
                content=answer,
                name="parent_agent",
            )
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Hebamme Agent
# ──────────────────────────────────────────────────────────────────────────────

def hebamme_agent(state: BabyOSState) -> dict:

    profile = state["user_profile"]

    name = profile.get("name", "there")

    mom_name = (
        profile.get("mom_name")
        or profile.get("partner_name")
        or "the mother"
    )

    phase_lbl = _phase_label(state)

    clinic = profile.get("clinic_name", "")

    system_context = (
        f"Midwife: {name}"
        f"{' | Clinic: ' + clinic if clinic else ''}\n"

        f"Patient: {mom_name} | "
        f"Phase: {phase_lbl}\n\n"

        "Provide structured clinical responses.\n"
        "Use lookup_normal_ranges.\n"
        "Use search_pubmed.\n"
        "Use flag_danger_sign if escalation needed.\n"
        "Reference Mutterpass fields where relevant."
    )

    try:

        answer = _run_agent(
            agent_name="Hebamme Agent",
            system_context=system_context,
            tools=HEBAMME_TOOLS,
            state=state,
            temperature=0.15,
        )

        answer += _sources_footer(state["retrieved_sources"])

    except Exception as e:

        answer = f"Clinical agent error: {e}"

    return {
        "agent_response": answer,
        "agent_name": "Hebamme Agent",
        "messages": [
            AIMessage(
                content=answer,
                name="hebamme_agent",
            )
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Germany Agent
# ──────────────────────────────────────────────────────────────────────────────

def germany_agent(state: BabyOSState) -> dict:

    profile = state["user_profile"]

    name = profile.get("name", "there")

    role = state["user_role"]

    phase_lbl = _phase_label(state)

    hospital = profile.get(
        "hospital_name",
        "your hospital",
    )

    kk = profile.get(
        "krankenkasse",
        "your Krankenkasse",
    )

    system_context = (
        f"User: {name} ({role}) | "
        f"Phase: {phase_lbl}\n"

        f"Hospital: {hospital} | "
        f"Krankenkasse: {kk}\n\n"

        "Help parents navigate Germany's system.\n"
        "Use translate_german_term.\n"
        "Use lookup_german_benefit.\n"
        "Use get_contact_numbers.\n"
        "Use web_search for local services.\n"
        "Always provide German + English terms."
    )

    try:

        answer = _run_agent(
            agent_name="Germany Navigation Agent",
            system_context=system_context,
            tools=GERMANY_TOOLS,
            state=state,
            temperature=0.2,
        )

        answer += _sources_footer(state["retrieved_sources"])

    except Exception as e:

        answer = f"Germany agent error: {e}"

    return {
        "agent_response": answer,
        "agent_name": "Germany Navigation Agent",
        "messages": [
            AIMessage(
                content=answer,
                name="germany_agent",
            )
        ],
    }
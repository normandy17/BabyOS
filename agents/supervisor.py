"""
supervisor.py — v2
------------------
Routing logic updated for:
  Roles   : mom | dad | hebamme
  Phases  : full 2-year timeline PRE → PP_12M_24M
  Agents  : medical | tracker | emotional | parent | hebamme | germany
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .state import BabyOSState

DEBUG_MODE = os.getenv("DEBUG_MODE")

DANGER_KEYWORDS = [
    "bleeding", "bleed", "heavy discharge",
    "severe pain", "sharp pain", "chest pain",
    "can't breathe", "cannot breathe", "difficulty breathing",
    "seizure", "unconscious", "fainted", "collapsed",
    "no movement", "not moving", "baby stopped moving", "baby not breathing",
    "waters broke", "water broke", "gush of fluid",
    "severe headache", "worst headache",
    "blurry", "flashing lights", "can't see",
    "swelling face", "face swollen",
    "cord prolapse", "placenta", "abruption",
    "112", "emergency", "hospital now",
    # Postpartum danger signs
    "won't wake", "floppy", "blue lips", "not feeding at all",
    "high fever baby", "rash spreading", "stopped breathing",
]

VALID_AGENTS = {
    "medical_agent",
    "tracker_agent",
    "emotional_agent",
    "parent_agent",
    "hebamme_agent",
    "germany_agent",
}


def _keyword_danger_check(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in DANGER_KEYWORDS)


def supervisor_node(state: BabyOSState) -> dict:
    if(DEBUG_MODE): print('Entering supervisor mode')    
    query  = state["current_query"]
    role   = state["user_role"]
    phase  = state["current_phase"]
    week   = state["current_week"]
    pp     = state["postpartum_weeks"]
    profile = state["user_profile"]

    # 1. Fast danger check
    if _keyword_danger_check(query):
        if(DEBUG_MODE): print('Found Danger Keyword: Running Medical Agent')
        return {
            "next_agent":    "medical_agent",
            "danger_flag":   True,
            "danger_reason": "Possible danger sign detected.",
            "run_metadata":  {**state.get("run_metadata", {}), "routed_by": "keyword_danger"},
        }

    # 2. Hebamme role always goes to hebamme_agent (they need clinical view)
    if role == "hebamme":
        if(DEBUG_MODE): print('Running hebamme agent')
        return {
            "next_agent":   "hebamme_agent",
            "danger_flag":  False,
            "run_metadata": {**state.get("run_metadata", {}), "routed_by": "role_hebamme"},
        }

    # 3. LLM routing for mom and dad
    llm = ChatOpenAI(
        model=os.getenv("BABYOS_AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Build phase context string
    postpartum_ctx = f"postpartum week {pp}" if pp > 0 else f"pregnancy week {week}"
    conditions_str = ", ".join(profile.get("conditions", [])) or "none"

    system_prompt = """You are the supervisor for BabyOS, a pregnancy and parenting app.
Your ONLY job is to return the name of the single best agent to handle the user's message.

Available agents:
  medical_agent   — symptoms, medications, food safety, lab results, danger signs, postpartum recovery, baby health
  tracker_agent   — baby/fetal development, size, weight, milestones, scan summaries, growth charts, toddler milestones
  emotional_agent — feelings, anxiety, fear, stress, relationship, mood, postnatal depression, identity shift
  parent_agent    — practical support for dad AND mom (postpartum): what to do this week, checklists, prep, feeding, sleep
  germany_agent   — Mutterpass, Hebamme, Vorsorgeuntersuchungen, Elterngeld, Kindergeld, German system, U-Untersuchungen

Rules:
- Any medical or safety concern → medical_agent
- Feelings, anxiety, identity, relationship → emotional_agent
- Baby growth, milestones, development stages → tracker_agent
- Germany system / admin / paperwork → germany_agent
- Practical "what do I do" questions → parent_agent
- Default fallback → medical_agent

Return ONLY the agent name. Nothing else."""

    human = f"""Role: {role}
Phase: {phase} ({postpartum_ctx})
Conditions: {conditions_str}
Message: "{query}"

Which agent?"""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human)])
    chosen   = response.content.strip().lower().replace(".", "")
    if chosen not in VALID_AGENTS:
        chosen = "medical_agent"
    if(DEBUG_MODE): print('Exiting supervisor mode: chose ', chosen)
    return {
        "next_agent":   chosen,
        "danger_flag":  False,
        "run_metadata": {**state.get("run_metadata", {}), "routed_by": "llm", "routed_to": chosen},
    }

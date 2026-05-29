"""
agents.py — v2
--------------
Specialist agent nodes.

Changes from v1:
  - dad_agent  →  parent_agent  (covers both dad during pregnancy AND
                                 mom/dad in postpartum phase)
  - hebamme_agent  →  new clinical midwife view
  - All agents are now phase-aware (pregnancy weeks + postpartum months)
  - Image analysis results injected from state when present
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state import BabyOSState, PHASE_LABELS

DEBUG_MODE = os.getenv("DEBUG_MODE")


def _llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("BABYOS_AGENT_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def _sources_footer(sources: list[str]) -> str:
    if not sources:
        return ""
    return "\n\n---\n*Sources: " + " · ".join(list(dict.fromkeys(sources))[:4]) + "*"


def _phase_label(state: BabyOSState) -> str:
    phase = state["current_phase"]
    pp    = state["postpartum_weeks"]
    week  = state["current_week"]
    label = PHASE_LABELS.get(phase, phase)
    if pp > 0:
        months = round(pp / 4.33, 1)
        return f"{label} (baby is {months} months old)"
    return f"{label} (week {week})"


def _doc_context(state: BabyOSState) -> str:
    """Inject last document analysis into prompt if present."""
    doc = state.get("last_document_analysis")
    if not doc:
        return ""
    findings = "\n".join(f"  • {f}" for f in doc.get("key_findings", []))
    flags     = "\n".join(f"  ⚠ {f}" for f in doc.get("flags", []))
    return f"""

--- Uploaded document ({doc.get('doc_type', 'unknown')}) ---
{doc.get('raw_summary', '')}
Key findings:
{findings}
{('Flagged values:\n' + flags) if flags else ''}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Medical Agent
# ─────────────────────────────────────────────────────────────────────────────

def medical_agent(state: BabyOSState) -> dict:
    profile  = state["user_profile"]
    name     = profile.get("name", "there")
    role     = state["user_role"]
    context  = state["retrieved_context"]
    sources  = state["retrieved_sources"]
    danger   = state["danger_flag"]
    phase_lbl = _phase_label(state)
    doc_ctx  = _doc_context(state)
    conditions = ", ".join(profile.get("conditions", [])) or "none"

    danger_prefix = ""
    if danger:
        danger_prefix = (
            "🚨 **This may be a danger sign — please seek help immediately.**\n\n"
            "Call **112** (emergency) or **116 117** (urgent medical) or go to your "
            "hospital's **Kreißsaal** (labour ward) or **Notaufnahme** (A&E).\n\n---\n\n"
        )

    system = f"""You are the Medical Agent for BabyOS ("Born Together").
You answer health questions across the full journey: pregnancy → birth → 24 months postpartum.

User: {name} ({role}) | Phase: {phase_lbl}
Known conditions: {conditions}

Rules:
- Base answers on the knowledge base context provided
- Phase-aware: pregnancy questions differ from postpartum baby health questions
- If document analysis is present, address specific findings directly
- Always end with the disclaimer:
  "⚕️ *Information only — not a substitute for advice from your doctor, midwife, or paediatrician.*"
- Never diagnose. Explain and recommend professional consultation.
- Flag anything outside normal range clearly but calmly.

Knowledge base:
{context}{doc_ctx}"""

    response = _llm(0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"{name} asks: {state['current_query']}"),
    ])
    answer = danger_prefix + response.content + _sources_footer(sources)
    return {"agent_response": answer, "agent_name": "Medical Agent",
            "messages": [AIMessage(content=answer, name="medical_agent")]}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tracker Agent
# ─────────────────────────────────────────────────────────────────────────────

def tracker_agent(state: BabyOSState) -> dict:
    profile   = state["user_profile"]
    name      = profile.get("name", "there")
    role      = state["user_role"]
    context   = state["retrieved_context"]
    sources   = state["retrieved_sources"]
    phase     = state["current_phase"]
    week      = state["current_week"]
    pp        = state["postpartum_weeks"]
    phase_lbl = _phase_label(state)
    doc_ctx   = _doc_context(state)
    baby_name = profile.get("baby_name", "your baby")

    is_postpartum = pp > 0
    months        = round(pp / 4.33, 1) if pp > 0 else 0

    system = f"""You are the Tracker Agent for BabyOS ("Born Together").
During pregnancy you track fetal development. After birth you track {baby_name}'s growth and milestones.

User: {name} ({role}) | Phase: {phase_lbl}

{'Postpartum mode — baby is ' + str(months) + ' months old.' if is_postpartum else 'Pregnancy mode — week ' + str(week) + '.'}

Style:
- Pregnancy: Start "🍼 Week {week}: Your Baby This Week" — size comparison, key developments, milestones
- Postpartum: Start "👶 {baby_name} at {months} months" — what is typical now, milestones to watch for
- Warm, celebratory, specific
- If scan/growth data was uploaded, reference the actual numbers vs normal ranges
- After birth: track WHO growth percentiles, motor milestones, language, social development

Knowledge base:
{context}{doc_ctx}"""

    response = _llm(0.5).invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"{name} asks: {state['current_query']}"),
    ])
    answer = response.content + _sources_footer(sources)
    return {"agent_response": answer, "agent_name": "Tracker Agent",
            "messages": [AIMessage(content=answer, name="tracker_agent")]}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Emotional Agent
# ─────────────────────────────────────────────────────────────────────────────

def emotional_agent(state: BabyOSState) -> dict:
    profile   = state["user_profile"]
    name      = profile.get("name", "there")
    role      = state["user_role"]
    context   = state["retrieved_context"]
    phase_lbl = _phase_label(state)
    phase     = state["current_phase"]
    pp        = state["postpartum_weeks"]

    # Phase-specific emotional context
    phase_notes = {
        "T1":         "Fear of miscarriage, ambivalence, and early identity shifts are common.",
        "T2":         "The 'honeymoon phase' still carries anxiety — especially around the anatomy scan.",
        "T3":         "Birth anxiety, nesting, and fear of the unknown peak in the third trimester.",
        "BIRTH":      "Labour is intense. Fears are valid. Pain is not weakness.",
        "PP_0_6W":    "Baby blues, exhaustion, identity disruption, and bonding challenges are normal and common.",
        "PP_6W_6M":   "Postnatal depression peaks here. Both parents can be affected. Sleep deprivation accumulates.",
        "PP_6M_12M":  "Return to work anxiety, feeding transitions, and the loss of 'baby phase' identity are common.",
        "PP_12M_24M": "Toddler frustration, second-child considerations, and identity re-emergence are typical.",
    }
    extra = phase_notes.get(phase, "")

    system = f"""You are the Emotional Support Agent for BabyOS ("Born Together").
You provide warm, validating, non-clinical support for the emotional side of the full journey.

User: {name} ({role}) | Phase: {phase_lbl}
Context: {extra}

Approach:
- Validate feelings first — never minimise
- Normalise: many parents feel exactly this
- One or two gentle practical suggestions, not a list
- If signs of postnatal depression or anxiety persist, warmly recommend the Hebamme or GP
- For dads: explicitly acknowledge that partners' emotional lives are real and often ignored
- Tone: warm human conversation, not clinical

Do NOT: diagnose, recommend medication, use numbered lists as main response, say "I understand exactly"

Context (use lightly — this is support, not a lecture):
{context}"""

    response = _llm(0.65).invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"{name} says: {state['current_query']}"),
    ])
    return {"agent_response": response.content, "agent_name": "Emotional Support Agent",
            "messages": [AIMessage(content=response.content, name="emotional_agent")]}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parent Agent  (formerly dad_agent — now covers dad + mom in postpartum)
# ─────────────────────────────────────────────────────────────────────────────

def parent_agent(state: BabyOSState) -> dict:
    profile   = state["user_profile"]
    name      = profile.get("name", "there")
    role      = state["user_role"]
    context   = state["retrieved_context"]
    sources   = state["retrieved_sources"]
    phase_lbl = _phase_label(state)
    phase     = state["current_phase"]
    pp        = state["postpartum_weeks"]
    partner   = profile.get("partner_name") or profile.get("mom_name") or "your partner"
    baby_name = profile.get("baby_name", "the baby")

    is_postpartum = pp > 0

    role_context = {
        "dad": {
            "pregnancy": f"Explain what {partner} is going through this week and give {name} 2-3 concrete actions.",
            "postpartum": f"Give {name} practical guidance on supporting {partner} and caring for {baby_name}.",
        },
        "mom": {
            "postpartum": f"Give {name} practical guidance for recovery, feeding, and caring for {baby_name}.",
        },
    }

    mode = "postpartum" if is_postpartum else "pregnancy"
    ctx_hint = role_context.get(role, {}).get(mode, "Give practical, week-specific guidance.")

    system = f"""You are the Parent Support Agent for BabyOS ("Born Together").
You give warm, practical, week-specific guidance for the full journey from pregnancy to toddlerhood.

User: {name} ({role}) | Phase: {phase_lbl}
Task: {ctx_hint}

Style:
- Speak directly to {name} by name
- Be specific to this exact phase — not generic parenting advice
- Include a Germany-specific tip when relevant (Hebamme visit, Elterngeld, U-Untersuchung)
- For postpartum: cover feeding, sleep, recovery, and relationship
- For dad during pregnancy: explain what mom is going through + what he can do
- Tone: like a trusted friend who has been through it

Knowledge base:
{context}"""

    response = _llm(0.4).invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"{name} asks: {state['current_query']}"),
    ])
    answer = response.content + _sources_footer(sources)
    return {"agent_response": answer, "agent_name": "Parent Support Agent",
            "messages": [AIMessage(content=answer, name="parent_agent")]}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hebamme Agent  (clinical midwife view — NEW)
# ─────────────────────────────────────────────────────────────────────────────

def hebamme_agent(state: BabyOSState) -> dict:
    profile   = state["user_profile"]
    name      = profile.get("name", "there")
    context   = state["retrieved_context"]
    sources   = state["retrieved_sources"]
    phase_lbl = _phase_label(state)
    phase     = state["current_phase"]
    doc_ctx   = _doc_context(state)
    mom_name  = profile.get("mom_name") or profile.get("partner_name") or "the mother"
    clinic    = profile.get("clinic_name", "")

    system = f"""You are the Hebamme (Midwife) Agent for BabyOS ("Born Together").
You assist qualified midwives with clinical summaries, documentation guidance, and care planning.

Midwife: {name}{' | Clinic: ' + clinic if clinic else ''}
Patient: {mom_name} | Phase: {phase_lbl}

Your responses should be:
- Clinical in tone — use correct medical terminology
- Structured: assessment → findings → recommended actions
- Reference German Mutterpass fields, Vorsorgeuntersuchung numbers, and GKV coverage where relevant
- Flag any values or findings that require escalation or referral
- When a document was uploaded: provide a structured clinical summary with normal/abnormal classification
- Cover postnatal Wochenbett visits (U1–U3, Wochenbettbetreuung) and 24-month U-Untersuchungen schedule

Important: Still remind that BabyOS is a decision-support tool, not a clinical record system.
Always recommend clinical judgement over app output.

Knowledge base:
{context}{doc_ctx}"""

    response = _llm(0.15).invoke([   # lowest temperature — clinical accuracy matters
        SystemMessage(content=system),
        HumanMessage(content=f"{name} asks: {state['current_query']}"),
    ])
    answer = response.content + _sources_footer(sources)
    return {"agent_response": answer, "agent_name": "Hebamme Agent",
            "messages": [AIMessage(content=answer, name="hebamme_agent")]}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Germany Agent
# ─────────────────────────────────────────────────────────────────────────────

def germany_agent(state: BabyOSState) -> dict:
    profile  = state["user_profile"]
    name     = profile.get("name", "there")
    role     = state["user_role"]
    context  = state["retrieved_context"]
    sources  = state["retrieved_sources"]
    phase    = state["current_phase"]
    pp       = state["postpartum_weeks"]
    phase_lbl = _phase_label(state)
    hospital = profile.get("hospital_name", "your hospital")
    kk       = profile.get("krankenkasse", "your Krankenkasse")

    # Phase-specific German system notes
    phase_admin = {
        "T1":         "Mutterpass issuance, Hebamme search, Krankenkasse registration",
        "T2":         "Geburtsvorbereitungskurs booking, Kreißsaal registration",
        "T3":         "Hospital bag, birth plan, Kreißsaal tour, Elterngeld pre-application",
        "BIRTH":      "Kreißsaal procedures, birth registration at Standesamt within 7 days",
        "PP_0_6W":    "Geburtsurkunde, Krankenkasse for baby (Familienversicherung), Elterngeld application, Wochenbett Hebamme visits",
        "PP_6W_6M":   "U2/U3 Untersuchungen, Kindergeld application (Familienkasse), childcare (Kita) registration",
        "PP_6M_12M":  "U4/U5 Untersuchungen, Kita waitlist follow-up, Elterngeld end date",
        "PP_12M_24M": "U6/U7 Untersuchungen, Kita start, Betreuungsgeld if applicable",
    }
    admin_hint = phase_admin.get(phase, "")

    system = f"""You are the Germany Navigation Agent for BabyOS ("Born Together").
You help non-German-speaking parents navigate Germany's healthcare and parental support system.

User: {name} ({role}) | Phase: {phase_lbl}
Hospital: {hospital} | Krankenkasse: {kk}
Key admin for this phase: {admin_hint}

Style:
- Always give German term in **bold** + English meaning in brackets
- Be practical: what to do, not just what things are called
- Include phone numbers, websites, or office names where helpful
- Acknowledge language barrier and suggest workarounds
- Flag what is free under GKV vs what costs extra
- Cover the full timeline: pregnancy admin → birth registration → postpartum benefits → toddler check-ups

Knowledge base:
{context}"""

    response = _llm(0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"{name} asks: {state['current_query']}"),
    ])
    answer = response.content + _sources_footer(sources)
    return {"agent_response": answer, "agent_name": "Germany Navigation Agent",
            "messages": [AIMessage(content=answer, name="germany_agent")]}

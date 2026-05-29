"""
state.py
--------
BabyOS Shared State Schema — v2

Roles: mom | dad | hebamme
Timeline: conception (week -2 from LMP) → 24 months postpartum (~first steps)
Tagline: "Born Together" — when a new baby is born, a new parent is also born.

Phase system replaces simple week counter:
  PRE       : trying / early awareness (week 1-3)
  T1        : first trimester (week 4-12)
  T2        : second trimester (week 13-27)
  T3        : third trimester (week 28-40)
  BIRTH     : labour and birth
  PP_0_6W   : postpartum 0-6 weeks (fourth trimester)
  PP_6W_6M  : postpartum 6 weeks – 6 months
  PP_6M_12M : postpartum 6-12 months
  PP_12M_24M: postpartum 12-24 months (up to first steps)
"""

from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ── Phase type ────────────────────────────────────────────────────────────────
PhaseType = Literal[
    "PRE",
    "T1", "T2", "T3",
    "BIRTH",
    "PP_0_6W", "PP_6W_6M", "PP_6M_12M", "PP_12M_24M",
]

RoleType = Literal["mom", "dad", "hebamme"]


def week_to_phase(week: int, postpartum_weeks: int = 0) -> PhaseType:
    """Derive the current phase from week number or postpartum weeks."""
    if postpartum_weeks > 0:
        if postpartum_weeks <= 6:   return "PP_0_6W"
        if postpartum_weeks <= 26:  return "PP_6W_6M"
        if postpartum_weeks <= 52:  return "PP_6M_12M"
        return "PP_12M_24M"
    if week <= 3:   return "PRE"
    if week <= 12:  return "T1"
    if week <= 27:  return "T2"
    if week <= 40:  return "T3"
    return "BIRTH"


# ─────────────────────────────────────────────────────────────────────────────
# Document analysis result — returned by vision node for any uploaded image
# ─────────────────────────────────────────────────────────────────────────────

class DocumentAnalysis(TypedDict, total=False):
    doc_type:       Literal[
                        "mutterpass",
                        "ultrasound",
                        "blood_report",
                        "urine_report",
                        "ctg",
                        "other_scan",
                        "other",
                    ]
    week_detected:  Optional[int]           # week mentioned in document
    key_findings:   list[str]               # plain-language bullet findings
    values:         dict[str, str]          # extracted key-value pairs
    flags:          list[str]               # anything outside normal range
    raw_summary:    str                     # full GPT-4o Vision narrative
    uploaded_at:    str                     # ISO datetime


# ─────────────────────────────────────────────────────────────────────────────
# User Profile
# ─────────────────────────────────────────────────────────────────────────────

class UserProfile(TypedDict, total=False):
    # Identity
    name:                   str
    role:                   RoleType
    partner_name:           Optional[str]   # mom's name if user is dad/hebamme
    mom_name:               Optional[str]   # always mom's name for hebamme view

    # Timeline
    current_week:           int             # pregnancy week (1-42)
    postpartum_weeks:       int             # weeks since birth (0 = still pregnant)
    due_date:               Optional[str]   # ISO date
    birth_date:             Optional[str]   # ISO date, set after birth
    lmp_date:               Optional[str]   # last menstrual period ISO date
    phase:                  PhaseType

    # Baby info (filled in after birth)
    baby_name:              Optional[str]
    baby_birth_weight_g:    Optional[float]
    baby_blood_type:        Optional[str]

    # Pregnancy type
    pregnancy_type:         Literal["singleton", "twins", "triplets"]

    # Medical history (mom)
    age_mom:                Optional[int]
    conditions:             list[str]       # ["gestational_diabetes", "hypertension"]
    medications:            list[str]
    blood_type_mom:         Optional[str]
    previous_pregnancies:   int
    previous_births:        int
    is_ivf:                 bool
    gbs_positive:           bool

    # Hebamme-specific fields
    patient_list:           list[str]       # hebamme manages multiple patients
    clinic_name:            Optional[str]

    # Germany
    hospital_name:          Optional[str]
    hebamme_name:           Optional[str]
    krankenkasse:           Optional[str]
    mutterpass_number:      Optional[str]

    # Preferences
    language:               Literal["en", "de"]
    units:                  Literal["metric", "imperial"]
    notification_tone:      Literal["gentle", "detailed", "clinical"]


# ─────────────────────────────────────────────────────────────────────────────
# Timeline Log Entry — one per week/month, spans full 2-year journey
# ─────────────────────────────────────────────────────────────────────────────

class TimelineLog(TypedDict, total=False):
    phase:              PhaseType
    week:               Optional[int]           # pregnancy week
    postpartum_weeks:   Optional[int]           # weeks after birth
    baby_age_months:    Optional[float]         # months after birth

    # Mom tracking
    symptoms:           list[str]
    mood_score:         int                     # 1–5
    weight_kg:          Optional[float]
    blood_pressure:     Optional[str]           # "120/80"
    doctor_notes:       Optional[str]

    # Baby tracking (postpartum)
    baby_weight_g:      Optional[float]
    baby_height_cm:     Optional[float]
    baby_milestones:    list[str]               # ["first smile", "holds head up"]
    feeding_type:       Optional[Literal["breast", "formula", "mixed"]]
    sleep_hours:        Optional[float]         # baby sleep in 24h

    # Documents uploaded this period
    documents:          list[DocumentAnalysis]
    questions_for_doctor: list[str]
    logged_at:          str


# ─────────────────────────────────────────────────────────────────────────────
# Main Graph State
# ─────────────────────────────────────────────────────────────────────────────

class BabyOSState(TypedDict):
    # Conversation history — add_messages APPENDS, never overwrites
    messages:           Annotated[list[BaseMessage], add_messages]

    # User context
    user_profile:       UserProfile

    # Per-turn shortcuts (reset by input_node each turn)
    current_query:      str
    current_week:       int
    postpartum_weeks:   int
    current_phase:      PhaseType
    user_role:          RoleType

    # Routing
    next_agent:         Literal[
                            "medical_agent",
                            "tracker_agent",
                            "emotional_agent",
                            "parent_agent",     # replaces dad_agent — covers dad + mom postpartum
                            "hebamme_agent",    # new — clinical view for midwives
                            "germany_agent",
                            "FINISH",
                        ]

    # RAG context (fetched once by retriever_node, shared by all agents)
    retrieved_context:  str
    retrieved_sources:  list[str]

    # Agent output
    agent_response:     str
    agent_name:         str

    # Safety
    danger_flag:        bool
    danger_reason:      Optional[str]

    # Document vision analysis (set by vision_node when image uploaded)
    uploaded_image_b64:     Optional[str]
    uploaded_image_type:    Optional[str]       # "mutterpass" / "ultrasound" / etc.
    last_document_analysis: Optional[DocumentAnalysis]

    # Full timeline history
    timeline_logs:      list[TimelineLog]

    # LangSmith
    run_metadata:       dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_state(user_profile: UserProfile) -> BabyOSState:
    """Call once when the user completes onboarding."""
    phase = week_to_phase(
        user_profile.get("current_week", 20),
        user_profile.get("postpartum_weeks", 0),
    )
    return BabyOSState(
        messages=[],
        user_profile={**user_profile, "phase": phase},
        current_query="",
        current_week=user_profile.get("current_week", 20),
        postpartum_weeks=user_profile.get("postpartum_weeks", 0),
        current_phase=phase,
        user_role=user_profile.get("role", "mom"),
        next_agent="FINISH",
        retrieved_context="",
        retrieved_sources=[],
        agent_response="",
        agent_name="",
        danger_flag=False,
        danger_reason=None,
        uploaded_image_b64=None,
        uploaded_image_type=None,
        last_document_analysis=None,
        timeline_logs=[],
        run_metadata={},
    )


PHASE_LABELS: dict[PhaseType, str] = {
    "PRE":         "Early awareness",
    "T1":          "First trimester",
    "T2":          "Second trimester",
    "T3":          "Third trimester",
    "BIRTH":       "Labour & birth",
    "PP_0_6W":     "Newborn (0–6 weeks)",
    "PP_6W_6M":    "Early baby (6 weeks–6 months)",
    "PP_6M_12M":   "Growing baby (6–12 months)",
    "PP_12M_24M":  "Toddler (12–24 months)",
}

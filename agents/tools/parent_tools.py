"""
tools/parent_tools.py
---------------------
Tools for Parent Agent, Germany Agent, and shared across all agents.

Tools:
  generate_weekly_checklist — week/phase-specific to-do list
  log_symptom              — writes symptom to timeline log in state
  web_search               — DuckDuckGo fallback for current info
  translate_german_term    — local German medical vocabulary lookup
  lookup_german_benefit    — Elterngeld / Kindergeld / Kita info lookup
  get_contact_numbers      — emergency and useful German numbers
"""

import os
import json
import requests
from typing import Optional
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# ── Weekly checklist generator ────────────────────────────────────────────────

_CHECKLISTS: dict[str, list[str]] = {
    "PRE": [
        "Start taking folic acid 400mcg daily",
        "Stop alcohol, smoking, and recreational drugs",
        "Book first GP/Frauenarzt appointment",
        "Check rubella immunity status",
        "Review current medications with doctor",
    ],
    "T1": [
        "Book Hebamme (do this NOW — waiting lists are long)",
        "Attend first Vorsorgeuntersuchung",
        "Register with Krankenkasse for pregnancy care",
        "Decide who to tell and when",
        "Read about the 12-week NT scan",
        "Start pregnancy vitamin supplement",
    ],
    "T2": [
        "Book Geburtsvorbereitungskurs (birth prep course)",
        "Register at the hospital Kreißsaal",
        "Attend 20-week anatomy scan (Feindiagnostik)",
        "Complete glucose tolerance test (weeks 24–28)",
        "Research Elterngeld — start gathering documents",
        "Plan any travel (safest window is now)",
        "Start thinking about birth preferences",
    ],
    "T3": [
        "Pack hospital bag",
        "Finalise birth plan",
        "Install car seat",
        "Do the Kreißsaal tour",
        "Pre-fill Elterngeld application form",
        "Stock freezer with easy meals",
        "Set up baby sleeping space (in your room)",
        "Know 3 routes to the hospital",
        "Save Kreißsaal phone number",
    ],
    "BIRTH": [
        "Call Kreißsaal before driving in",
        "Bring Mutterpass, Krankenkasse card, ID",
        "Advocate for her birth plan preferences",
        "Be present, calm, supportive",
        "Ask about delayed cord clamping",
        "Skin-to-skin contact after birth",
    ],
    "PP_0_6W": [
        "Register birth at Standesamt within 7 days",
        "Register baby with Krankenkasse (Familienversicherung)",
        "Apply for Elterngeld within 3 months of birth",
        "Attend all Hebamme home visits",
        "Attend U1 and U2 paediatric checks",
        "Apply for Kindergeld at Familienkasse",
        "Look after yourself — sleep when baby sleeps",
    ],
    "PP_6W_6M": [
        "Attend U3 check (4–6 weeks)",
        "Mum's 6-week postnatal check with Gynäkologin",
        "Discuss contraception at postnatal check",
        "Begin tummy time daily (30 min/day by 3 months)",
        "Start researching Kita / Krippe options",
        "EPDS mental health screening — be honest",
        "Attend U4 check (3–4 months)",
    ],
    "PP_6M_12M": [
        "Start solid foods at 6 months (Beikost)",
        "Attend U5 check (6–7 months)",
        "Attend U6 check (10–12 months)",
        "Register for Kita (Krippe) — waitlists are 12–18 months",
        "Introduce high-allergen foods early (eggs, peanuts, fish)",
        "Babyproof the home before crawling starts",
        "Begin using baby's name and simple words consistently",
    ],
    "PP_12M_24M": [
        "Attend U7 check (21–24 months) — speech assessment included",
        "Speech referral if fewer than 50 words at 24 months",
        "Complete Kita Eingewöhnung (settling-in period) if starting",
        "Elterngeld end date — check return-to-work plan",
        "Toddler-proof the home (climbing, running)",
        "Continue reading aloud daily — best language development tool",
    ],
}

@tool
def generate_weekly_checklist(phase: str, role: str = "mom") -> str:
    """
    Generate a practical to-do checklist for the current pregnancy/postpartum phase.
    Input:
      phase: one of PRE / T1 / T2 / T3 / BIRTH / PP_0_6W / PP_6W_6M / PP_6M_12M / PP_12M_24M
      role:  mom | dad | hebamme (slightly adjusts framing)
    """
    items = _CHECKLISTS.get(phase, _CHECKLISTS["T2"])
    role_note = {
        "dad":     "\n💡 Dad/Partner: You can take ownership of items marked with admin or logistics.",
        "hebamme": "\n📋 Hebamme: Use this checklist to guide your care plan discussion.",
    }.get(role, "")

    lines = [f"Checklist for {phase.replace('_', ' ')}:"]
    for i, item in enumerate(items, 1):
        lines.append(f"  {i}. {item}")
    lines.append(role_note)
    return "\n".join(lines)


# ── Web search (DuckDuckGo — no API key needed) ───────────────────────────────

_ddg_search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information not in the knowledge base.
    Use for: recent news, specific local services, current guidelines,
    or anything that may have changed since the corpus was built.
    Add 'pregnancy' or 'Germany' to the query for relevant results.
    """
    try:
        result = _ddg_search.run(query)
        return f"Web search results for '{query}':\n{result}"
    except Exception as e:
        return f"Web search failed: {e}. Rely on knowledge base instead."


# ── German vocabulary lookup ──────────────────────────────────────────────────

_GERMAN_VOCAB: dict[str, dict] = {
    "mutterpass":           {"en": "Pregnancy record booklet", "note": "Carry to every appointment"},
    "hebamme":              {"en": "Midwife", "note": "Register early — waiting lists are long"},
    "frauenarzt":           {"en": "Gynaecologist / OB-GYN"},
    "vorsorgeuntersuchung": {"en": "Antenatal check-up"},
    "kreißsaal":            {"en": "Delivery room / labour ward"},
    "entbindung":           {"en": "Childbirth / delivery"},
    "wehen":                {"en": "Contractions"},
    "blasensprung":         {"en": "Waters breaking (rupture of membranes)"},
    "kaiserschnitt":        {"en": "Caesarean section (C-section)"},
    "periduralanästhesie":  {"en": "Epidural (PDA)", "note": "Request in advance at some hospitals"},
    "geburtshaus":          {"en": "Birth centre (midwife-led)"},
    "wochenbett":           {"en": "Postnatal period / lying-in period (first 6 weeks)"},
    "wochenbettbetreuung":  {"en": "Postnatal care by Hebamme", "note": "Free, daily visits, covered by GKV"},
    "stillen":              {"en": "Breastfeeding"},
    "abstillen":            {"en": "Weaning from breastfeeding"},
    "beikost":              {"en": "Complementary feeding / starting solids"},
    "kinderarzt":           {"en": "Paediatrician"},
    "u-untersuchung":       {"en": "Developmental check-up (U1–U9)", "note": "Recorded in yellow Kinderuntersuchungsheft"},
    "kinderuntersuchungsheft": {"en": "Child health record booklet (yellow booklet)"},
    "elterngeld":           {"en": "Parental leave benefit (income replacement)"},
    "kindergeld":           {"en": "Child benefit (monthly payment)"},
    "elterngeldstelle":     {"en": "Parental benefit office (apply here for Elterngeld)"},
    "familienkasse":        {"en": "Family benefits office (apply here for Kindergeld)"},
    "kita":                 {"en": "Daycare / childcare centre (Krippe 0–3, Kindergarten 3–6)"},
    "krippe":               {"en": "Infant daycare (0–3 years)"},
    "eingewöhnung":         {"en": "Gradual settling-in period at Kita", "note": "Typically 2–3 weeks"},
    "jugendamt":            {"en": "Youth welfare office"},
    "standesamt":           {"en": "Civil registry office (register the birth here within 7 days)"},
    "geburtsurkunde":       {"en": "Birth certificate"},
    "krankenkasse":         {"en": "Health insurance (statutory = GKV, private = PKV)"},
    "familienversicherung": {"en": "Family co-insurance", "note": "Baby covered free under GKV"},
    "überweisung":          {"en": "Referral (from GP to specialist)"},
    "notaufnahme":          {"en": "Accident and Emergency (A&E)"},
    "blutdruck":            {"en": "Blood pressure"},
    "ultraschall":          {"en": "Ultrasound scan"},
    "frühgeburt":           {"en": "Premature birth (before 37 weeks)"},
    "fehlgeburt":           {"en": "Miscarriage"},
    "schwangerschaft":      {"en": "Pregnancy"},
}

@tool
def translate_german_term(term: str) -> str:
    """
    Look up a German medical or administrative term related to pregnancy/parenting.
    Returns English meaning and any helpful notes.
    Input: German term (e.g. 'Mutterpass', 'Hebamme', 'Elterngeld')
    """
    key = term.lower().strip()
    match = None
    for k in _GERMAN_VOCAB:
        if k in key or key in k:
            match = k
            break

    if not match:
        return (
            f"'{term}' not in local vocabulary. "
            "Try DeepL (deepl.com) for translation or ask the Germany agent for more context."
        )

    entry = _GERMAN_VOCAB[match]
    result = f"**{match.title()}** = {entry['en']}"
    if "note" in entry:
        result += f"\n💡 {entry['note']}"
    return result


# ── German benefits lookup ────────────────────────────────────────────────────

_GERMAN_BENEFITS: dict[str, str] = {
    "elterngeld": """**Elterngeld** (Parental Leave Benefit)
• Who: employed parents who reduce/stop work after birth
• Amount: 65–67% of net income (max €1,800/month)
• Duration: 12 months (14 if partner takes ≥2 months)
• ElterngeldPlus: half payment, double duration (28–32 months)
• Apply at: local Elterngeldstelle (part of Jugendamt) within 3 months of birth
• Documents needed: birth certificate, income proof (last 12 months), employer confirmation
• Phone: 0800 4 5555 30 (Familienkasse, free)""",

    "kindergeld": """**Kindergeld** (Child Benefit)
• Amount: €250/month per child (2024)
• No income limit, no means test
• Apply at: Familienkasse (local Agentur für Arbeit)
• How: online at familienkasse.de or paper form
• Start: from birth, backdated up to 6 months if applied late
• Continues: until 18, or 25 if in education/training""",

    "kita": """**Kita / Krippe** (Childcare)
• Legal right to a Kita place from age 1 (since 2013)
• Costs: means-tested, subsidised by Jugendamt. Often free or very low-cost for first year.
• Registration: 12–18 months before desired start date
• In Göttingen: register via KitaNav portal (kitanav.de) or contact facilities directly
• Eingewöhnung: 2–3 week gradual settling-in — legally required in quality Kitas
• Your Jugendamt Göttingen: Groner Straße 41, 37073 Göttingen, Tel: 0551 400-2840""",

    "mutterschaftsgeld": """**Mutterschaftsgeld** (Maternity Pay)
• Paid during Mutterschutz (6 weeks before + 8 weeks after birth)
• Amount: up to €13/day from Krankenkasse + employer tops up to full salary
• Apply at: your Krankenkasse (start 7 weeks before due date)
• Self-employed: apply to Bundesamt für Soziale Sicherung""",

    "mutterschutz": """**Mutterschutz** (Maternity Protection)
• Starts: 6 weeks before due date
• Ends: 8 weeks after birth (12 weeks for premature or multiple births)
• During this period: cannot be required to work, cannot be dismissed
• Fully paid via Mutterschaftsgeld + employer top-up""",
}

@tool
def lookup_german_benefit(benefit_name: str) -> str:
    """
    Look up detailed information about a German parental benefit, right, or system.
    Input: benefit name (e.g. 'Elterngeld', 'Kindergeld', 'Kita', 'Mutterschutz')
    """
    key = benefit_name.lower().strip()
    for k, v in _GERMAN_BENEFITS.items():
        if k in key or key in k:
            return v

    return (
        f"'{benefit_name}' not found. Available topics: "
        + ", ".join(_GERMAN_BENEFITS.keys())
        + ". Try the Germany agent for other questions."
    )


# ── Emergency and useful numbers ──────────────────────────────────────────────

@tool
def get_contact_numbers(context: str = "general") -> str:
    """
    Get relevant phone numbers and contacts for Germany pregnancy/parenting.
    Input: context hint (e.g. 'emergency', 'mental health', 'elterngeld', 'kita', 'general')
    """
    numbers = {
        "emergency": "🚨 Emergency (Notruf): **112**",
        "medical_urgent": "🏥 Urgent medical (non-emergency): **116 117**",
        "poison": "☠️ Poison control: **030 19240**",
        "telefonseelsorge": "🧠 Mental health crisis (free, 24/7): **0800 111 0 111** or **0800 111 0 222**",
        "familienkasse": "👶 Familienkasse (Kindergeld): **0800 4 5555 30** (free)",
        "jugendamt_goettingen": "🏛 Jugendamt Göttingen: **0551 400-2840**",
        "umg_goettingen": "🏥 Universitätsmedizin Göttingen (UMG): **0551 39-0**",
        "hebammensuche": "🤱 Find a Hebamme: **www.hebammensuche.de**",
        "bzga": "📚 BZgA health info: **www.bzga.de** (free publications)",
    }

    ctx = context.lower()
    if "emergency" in ctx or "danger" in ctx:
        keys = ["emergency", "medical_urgent", "poison"]
    elif "mental" in ctx or "depress" in ctx or "crisis" in ctx:
        keys = ["telefonseelsorge", "emergency"]
    elif "elterngeld" in ctx or "kindergeld" in ctx or "benefit" in ctx:
        keys = ["familienkasse", "jugendamt_goettingen"]
    elif "kita" in ctx or "childcare" in ctx:
        keys = ["jugendamt_goettingen"]
    elif "hospital" in ctx or "birth" in ctx:
        keys = ["umg_goettingen", "emergency", "medical_urgent"]
    else:
        keys = list(numbers.keys())

    return "\n".join(numbers[k] for k in keys)


# ── Exported tool lists ───────────────────────────────────────────────────────

PARENT_TOOLS = [
    generate_weekly_checklist,
    web_search,
    get_contact_numbers,
]

GERMANY_TOOLS = [
    translate_german_term,
    lookup_german_benefit,
    get_contact_numbers,
    web_search,
]

EMOTIONAL_TOOLS = [
    get_contact_numbers,
    web_search,
]

SHARED_TOOLS = [web_search]
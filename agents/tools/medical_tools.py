"""
tools/medical_tools.py
----------------------
Tools for the Medical Agent and Hebamme Agent.

Tools:
  search_pubmed          — queries PubMed API for evidence-based articles
  check_pregnancy_safety — looks up medication/food safety in local registry
  calculate_gestational_age — derives week + trimester from LMP date
  calculate_due_date     — Naegele's rule from LMP
  flag_danger_sign       — writes danger_flag to state, returns escalation msg
  lookup_normal_ranges   — structured lookup for blood/urine/biometric values
"""

import os
import json
import math
import requests
from datetime import date, datetime, timedelta
from typing import Optional
from langchain_core.tools import tool


# ── PubMed ────────────────────────────────────────────────────────────────────

@tool
def search_pubmed(query: str, max_results: int = 3) -> str:
    """
    Search PubMed for peer-reviewed pregnancy/parenting articles.
    Returns titles, authors, journal, year, and abstract snippet.
    Use for evidence-based medical questions.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # Add pregnancy context to query if not present
    if "pregnan" not in query.lower() and "neonatal" not in query.lower():
        query = f"{query} pregnancy"

    try:
        # Step 1: search for IDs
        search_resp = requests.get(
            f"{base}/esearch.fcgi",
            params={
                "db": "pubmed", "term": query,
                "retmax": max_results, "retmode": "json",
                "sort": "relevance",
            },
            timeout=8,
        )
        search_resp.raise_for_status()
        ids = search_resp.json()["esearchresult"]["idlist"]
        if not ids:
            return f"No PubMed results found for: {query}"

        # Step 2: fetch summaries
        fetch_resp = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=8,
        )
        fetch_resp.raise_for_status()
        articles = fetch_resp.json()["result"]

        lines = [f"PubMed results for '{query}':"]
        for uid in ids:
            a = articles.get(uid, {})
            title   = a.get("title", "No title")
            authors = ", ".join(
                au.get("name", "") for au in a.get("authors", [])[:2]
            )
            journal = a.get("source", "")
            year    = a.get("pubdate", "")[:4]
            lines.append(f"\n• {title}")
            lines.append(f"  {authors} — {journal} ({year})")
            lines.append(f"  https://pubmed.ncbi.nlm.nih.gov/{uid}/")

        return "\n".join(lines)

    except Exception as e:
        return f"PubMed search failed: {e}. Proceed with knowledge base context."


# ── Drug / food safety lookup ─────────────────────────────────────────────────

# Local safety registry — expandable
_SAFETY_REGISTRY: dict[str, dict] = {
    # Medications
    "paracetamol":    {"safe": True,  "note": "First-line analgesic in pregnancy. Safe at recommended doses."},
    "ibuprofen":      {"safe": False, "note": "Avoid especially after 20 weeks — associated with fetal kidney issues and premature duct closure."},
    "aspirin":        {"safe": "low_dose", "note": "Low-dose (75–150mg) prescribed for preeclampsia prevention — not OTC doses."},
    "amoxicillin":    {"safe": True,  "note": "Safe antibiotic in pregnancy."},
    "metformin":      {"safe": True,  "note": "Used for gestational diabetes and PCOS in pregnancy — discuss with doctor."},
    "sertraline":     {"safe": True,  "note": "Generally compatible with pregnancy and breastfeeding. Discuss risks/benefits with doctor."},
    "fluoxetine":     {"safe": "discuss", "note": "Some data on neonatal withdrawal. Discuss with doctor before stopping."},
    "omeprazole":     {"safe": True,  "note": "Safe for heartburn/GERD in pregnancy."},
    "folic acid":     {"safe": True,  "note": "Essential — 400mcg/day minimum. 5mg/day if high risk."},
    "vitamin d":      {"safe": True,  "note": "10mcg (400 IU)/day recommended throughout pregnancy."},
    "iron":           {"safe": True,  "note": "Supplement only if blood tests show deficiency."},
    # Foods
    "sushi":          {"safe": False, "note": "Raw fish carries Listeria and parasite risk. Avoid raw fish entirely."},
    "salmon":         {"safe": True,  "note": "Cooked salmon is excellent — high DHA. Limit to 2 portions/week (mercury)."},
    "tuna":           {"safe": "limit", "note": "Limit to 2 tins/week (canned) or 1 fresh steak/week due to mercury."},
    "brie":           {"safe": False, "note": "Soft mould-ripened cheese — Listeria risk. Avoid unless thoroughly cooked."},
    "cheddar":        {"safe": True,  "note": "Hard cheese — safe in pregnancy."},
    "peanuts":        {"safe": True,  "note": "Safe unless personal/family history of allergy. Early introduction may reduce allergy risk in baby."},
    "liver":          {"safe": False, "note": "Very high vitamin A — can cause fetal defects. Avoid all liver and pâté."},
    "coffee":         {"safe": "limit", "note": "Limit caffeine to under 200mg/day (~1 filter coffee or 2 espresso shots)."},
    "alcohol":        {"safe": False, "note": "No safe level of alcohol in pregnancy. Avoid completely."},
    "honey":          {"safe": True,  "note": "Safe for the mother. Do NOT give to baby under 12 months (botulism risk)."},
    "eggs":           {"safe": True,  "note": "Cooked eggs are safe and nutritious."},
    "raw eggs":       {"safe": False, "note": "Salmonella risk. Avoid raw or runny eggs unless from vaccinated hens (Lion-stamped in UK)."},
}

@tool
def check_pregnancy_safety(item: str) -> str:
    """
    Check whether a medication, food, or supplement is safe during pregnancy.
    Returns a safety verdict with explanation.
    Input: name of the medication or food (e.g. 'ibuprofen', 'salmon', 'coffee')
    """
    key = item.lower().strip()
    # Fuzzy match
    match = None
    for k in _SAFETY_REGISTRY:
        if k in key or key in k:
            match = k
            break

    if not match:
        return (
            f"'{item}' is not in the local safety registry. "
            "Based on general guidance: avoid anything not explicitly confirmed safe. "
            "Recommend checking with NHS Medicines in Pregnancy or your midwife."
        )

    entry = _SAFETY_REGISTRY[match]
    safe  = entry["safe"]
    note  = entry["note"]

    if safe is True:
        verdict = "✅ Generally safe"
    elif safe is False:
        verdict = "❌ Avoid during pregnancy"
    elif safe == "low_dose":
        verdict = "⚠️ Safe only at specific low doses — not standard OTC use"
    elif safe == "limit":
        verdict = "⚠️ Safe in limited quantities — see note"
    else:
        verdict = "⚠️ Discuss with your doctor"

    return f"{verdict}: {item.title()}\n{note}\n\n⚕️ Always confirm with your midwife or doctor."


# ── Gestational age / due date ─────────────────────────────────────────────────

@tool
def calculate_gestational_age(lmp_date: str) -> str:
    """
    Calculate current gestational age from Last Menstrual Period (LMP) date.
    Input: lmp_date in ISO format YYYY-MM-DD
    Returns: current week, days, trimester, and due date.
    """
    try:
        lmp  = date.fromisoformat(lmp_date)
        today = date.today()
        delta = today - lmp
        total_days = delta.days
        weeks      = total_days // 7
        days       = total_days % 7
        due_date   = lmp + timedelta(days=280)   # Naegele's rule

        if weeks <= 12:
            trimester = "First trimester"
        elif weeks <= 27:
            trimester = "Second trimester"
        elif weeks <= 40:
            trimester = "Third trimester"
        else:
            trimester = f"Post-term ({weeks - 40} week(s) past due date)"

        return (
            f"Gestational age: {weeks} weeks and {days} days\n"
            f"Trimester: {trimester}\n"
            f"Due date (Naegele's rule): {due_date.strftime('%d %B %Y')}\n"
            f"Days until due date: {max(0, (due_date - today).days)}"
        )
    except ValueError:
        return f"Invalid date format: '{lmp_date}'. Please use YYYY-MM-DD (e.g. 2024-03-15)."


@tool
def calculate_due_date(lmp_date: str) -> str:
    """
    Calculate estimated due date from Last Menstrual Period using Naegele's rule.
    Input: lmp_date in ISO format YYYY-MM-DD
    """
    try:
        lmp      = date.fromisoformat(lmp_date)
        due      = lmp + timedelta(days=280)
        earliest = due - timedelta(days=21)   # 37 weeks
        latest   = due + timedelta(days=14)   # 42 weeks
        return (
            f"Estimated Due Date: {due.strftime('%d %B %Y')}\n"
            f"Full-term window: {earliest.strftime('%d %b')} – {latest.strftime('%d %b %Y')}\n"
            f"(37–42 weeks is considered full term)"
        )
    except ValueError:
        return f"Invalid date: '{lmp_date}'. Use YYYY-MM-DD."


# ── Normal ranges lookup ──────────────────────────────────────────────────────

_NORMAL_RANGES: dict[str, dict] = {
    # Blood values
    "haemoglobin":      {"unit": "g/dL",     "pregnancy": "≥10.5",  "normal": "12–16",       "low": "<10.5 → anaemia"},
    "ferritin":         {"unit": "µg/L",      "pregnancy": "≥30",    "normal": "12–150",      "low": "<30 → iron deficiency"},
    "tsh":              {"unit": "mIU/L",     "pregnancy": "0.1–2.5 (T1), 0.2–3.0 (T2/T3)", "normal": "0.4–4.0", "low": ""},
    "fasting glucose":  {"unit": "mmol/L",    "pregnancy": "<5.1",   "normal": "3.9–5.5",     "high": "≥5.1 → GDM screen"},
    "gtt 1h":           {"unit": "mmol/L",    "pregnancy": "<10.0",  "normal": "—",           "high": "≥10.0 → GDM"},
    "gtt 2h":           {"unit": "mmol/L",    "pregnancy": "<8.5",   "normal": "—",           "high": "≥8.5 → GDM"},
    "platelets":        {"unit": "×10⁹/L",    "pregnancy": "150–400","normal": "150–400",     "low": "<100 → urgent review"},
    "blood pressure":   {"unit": "mmHg",      "pregnancy": "<140/90","normal": "90–120/60–80","high": "≥140/90 → hypertension"},
    # Urine
    "urine protein":    {"unit": "",          "pregnancy": "Negative/Trace", "normal": "Negative", "high": "2+ or higher → preeclampsia risk"},
    "urine glucose":    {"unit": "",          "pregnancy": "Negative",       "normal": "Negative", "note": "Positive → GDM screen"},
    "urine nitrites":   {"unit": "",          "pregnancy": "Negative",       "normal": "Negative", "note": "Positive → likely UTI, treat"},
    # Biometric (ultrasound, weeks 20)
    "fetal heart rate": {"unit": "bpm",       "pregnancy": "110–160","normal": "110–160",     "note": "<110 or >160 → review"},
    "amniotic fluid index": {"unit": "cm",    "pregnancy": "5–25",   "normal": "5–25",        "note": "<5 → oligohydramnios; >25 → polyhydramnios"},
    # Newborn
    "apgar":            {"unit": "score",     "normal": "7–10 at 5 min", "note": "<7 at 5 min → intervention"},
    "newborn weight":   {"unit": "g",         "normal": "2500–4000", "note": "<2500 → low birth weight; >4000 → macrosomia"},
}

@tool
def lookup_normal_ranges(parameter: str) -> str:
    """
    Look up normal reference ranges for a blood test, urine test, or
    biometric measurement in pregnancy context.
    Input: parameter name (e.g. 'haemoglobin', 'blood pressure', 'fetal heart rate')
    """
    key = parameter.lower().strip()
    match = None
    for k in _NORMAL_RANGES:
        if k in key or key in k:
            match = k
            break

    if not match:
        return f"'{parameter}' not in local ranges table. Please consult your laboratory's reference intervals."

    r = _NORMAL_RANGES[match]
    lines = [f"Reference ranges: {parameter.title()}"]
    if "pregnancy" in r:
        lines.append(f"  In pregnancy: {r['pregnancy']} {r['unit']}")
    if "normal" in r:
        lines.append(f"  Standard normal: {r['normal']} {r['unit']}")
    for flag_key in ("low", "high", "note"):
        if flag_key in r and r[flag_key]:
            lines.append(f"  ⚠ {r[flag_key]}")
    return "\n".join(lines)


# ── Danger flag (writes to state via exception — graph catches it) ─────────────

class DangerSignDetected(Exception):
    """Raised by flag_danger_sign tool to trigger emergency escalation."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@tool
def flag_danger_sign(reason: str) -> str:
    """
    Flag a potential danger sign that requires immediate medical attention.
    Use this when the user describes symptoms that need urgent care.
    Input: brief description of the danger sign detected.
    Returns an emergency guidance message and raises escalation.
    """
    msg = (
        f"🚨 DANGER SIGN FLAGGED: {reason}\n\n"
        "Please seek immediate medical attention:\n"
        "• Call **112** (emergency services)\n"
        "• Call **116 117** (urgent medical, non-emergency)\n"
        "• Go directly to your hospital's **Kreißsaal** (labour ward) or **Notaufnahme** (A&E)\n\n"
        "Do not wait for an appointment."
    )
    raise DangerSignDetected(reason)


# ── Exported tool lists per agent ─────────────────────────────────────────────

MEDICAL_TOOLS = [
    search_pubmed,
    check_pregnancy_safety,
    calculate_gestational_age,
    calculate_due_date,
    lookup_normal_ranges,
    flag_danger_sign,
]

HEBAMME_TOOLS = [
    lookup_normal_ranges,
    search_pubmed,
    calculate_gestational_age,
    flag_danger_sign,
]
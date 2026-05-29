"""
vision.py
---------
BabyOS Vision Node

Handles ALL uploaded medical document images using GPT-4o Vision.

Supported document types:
  mutterpass    — German pregnancy record booklet pages
  ultrasound    — Ultrasound / sonography reports and images
  blood_report  — Blood test results (CBC, iron, glucose tolerance, etc.)
  urine_report  — Urinalysis results
  ctg           — Cardiotocography (fetal heart rate) traces
  other_scan    — MRI, X-ray, any other imaging
  other         — Any other document

Each document type has a specialised prompt that knows what fields to extract.
The vision_node runs BEFORE supervisor_node when an image is present —
its output (DocumentAnalysis) is stored in state and injected into
whichever agent handles the follow-up question.
"""

import os
import json
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .state import BabyOSState, DocumentAnalysis


# ── Doc-type detection keywords ───────────────────────────────────────────────
DOC_TYPE_HINTS = {
    "mutterpass":   ["mutterpass", "mutterschaftsvorsorge", "gravida", "para", "ss-woche"],
    "ultrasound":   ["ultraschall", "sonography", "ultrasound", "bpd", "fl ", "ac ", "hc ", "efw",
                     "fetal biometry", "biparietal", "femur", "amniotic", "placenta"],
    "blood_report": ["haemoglobin", "hemoglobin", "hb ", "hct", "ferritin", "glucose",
                     "tsh", "blood group", "blutbild", "blutgruppe", "hba1c", "wbc", "rbc",
                     "thrombocytes", "thrombozyten"],
    "urine_report": ["urine", "urin", "protein", "nitrite", "leukocytes", "urinalysis",
                     "urinbefund", "ph ", "specific gravity"],
    "ctg":          ["ctg", "cardiotocograph", "fetal heart rate", "contractions",
                     "baseline", "variability", "decelerations", "tocogram"],
}

# ── Specialised extraction prompts per doc type ───────────────────────────────
EXTRACTION_PROMPTS = {

    "mutterpass": """You are analysing a page from a German Mutterpass (mother's passport / pregnancy booklet).

Extract and return a JSON object with these fields:
{
  "doc_type": "mutterpass",
  "week_detected": <integer or null>,
  "key_findings": ["plain language bullet for each filled field"],
  "values": {
    "blood_type": "...",
    "rhesus_factor": "...",
    "rubella_immunity": "...",
    "hiv_result": "...",
    "hepatitis_b": "...",
    "blood_pressure": "...",
    "weight_kg": "...",
    "fundal_height_cm": "...",
    "fetal_position": "...",
    "fetal_heart_rate": "...",
    "urine_protein": "...",
    "urine_glucose": "...",
    "haemoglobin": "...",
    "visit_number": "...",
    "visit_date": "..."
  },
  "flags": ["any value outside normal range, explained in plain English"],
  "raw_summary": "2-3 sentence plain English summary of this Mutterpass page"
}

Normal ranges for reference:
- Blood pressure: <140/90 mmHg
- Haemoglobin: >10.5 g/dL in pregnancy
- Urine protein: negative or trace
- Urine glucose: negative
- Fetal heart rate: 110–160 bpm

Return ONLY the JSON. No explanation, no markdown fences.""",

    "ultrasound": """You are analysing an ultrasound / sonography report or image from a pregnancy or postpartum scan.

Extract and return a JSON object:
{
  "doc_type": "ultrasound",
  "week_detected": <gestational age in weeks as integer or null>,
  "key_findings": ["plain language bullet for each measurement or finding"],
  "values": {
    "gestational_age_weeks": "...",
    "bpd_mm": "...",
    "hc_mm": "...",
    "ac_mm": "...",
    "fl_mm": "...",
    "efw_grams": "...",
    "placenta_position": "...",
    "amniotic_fluid_index": "...",
    "fetal_heart_rate": "...",
    "presentation": "...",
    "cervical_length_mm": "...",
    "growth_percentile": "..."
  },
  "flags": ["any measurement outside expected range for gestational age, in plain English"],
  "raw_summary": "5 sentence plain English summary a parent can understand"
}

If specific values are not visible in the image, use null.
Interpret any printed percentile or z-score found.
Return ONLY the JSON.""",

    "blood_report": """You are analysing a blood test result report from a pregnant or postpartum patient.

Extract and return a JSON object:
{
  "doc_type": "blood_report",
  "week_detected": null,
  "key_findings": ["plain English explanation of each result"],
  "values": {
    "haemoglobin_g_dl": "...",
    "haematocrit_pct": "...",
    "ferritin_ug_l": "...",
    "iron_umol_l": "...",
    "wbc_10e9_l": "...",
    "platelets_10e9_l": "...",
    "glucose_mmol_l": "...",
    "hba1c_pct": "...",
    "tsh_mu_l": "...",
    "blood_group": "...",
    "rhesus_antibodies": "...",
    "rubella_igg": "...",
    "hiv": "...",
    "hepatitis_b_sag": "...",
    "crp_mg_l": "..."
  },
  "flags": ["any result outside the pregnancy-specific normal range, explained in plain English"],
  "raw_summary": "2-3 sentence plain English summary"
}

Pregnancy-specific normal ranges:
- Haemoglobin: ≥10.5 g/dL (anaemia if below)
- Ferritin: ≥30 µg/L in pregnancy
- TSH: 0.1–2.5 mIU/L (first trimester), 0.2–3.0 mIU/L (second/third)
- Fasting glucose: <5.1 mmol/L; 1h GTT: <10.0; 2h GTT: <8.5
- Platelets: 150–400 × 10⁹/L

Return ONLY the JSON.""",

    "urine_report": """You are analysing a urine test result from a pregnancy antenatal check.

Extract and return a JSON object:
{
  "doc_type": "urine_report",
  "week_detected": null,
  "key_findings": ["plain English explanation of each result"],
  "values": {
    "protein": "...",
    "glucose": "...",
    "nitrites": "...",
    "leukocytes": "...",
    "blood": "...",
    "ketones": "...",
    "ph": "...",
    "specific_gravity": "..."
  },
  "flags": ["any abnormal result with clinical relevance in plain English"],
  "raw_summary": "2-3 sentence plain English summary"
}

Flags to highlight:
- Protein 2+ or higher → possible preeclampsia, refer urgently
- Nitrites positive + leukocytes → likely UTI, needs treatment
- Glucose positive → screen for gestational diabetes
- Blood → investigate further

Return ONLY the JSON.""",

    "ctg": """You are analysing a CTG (cardiotocography) trace from a pregnancy or labour monitoring.

Extract and return a JSON object:
{
  "doc_type": "ctg",
  "week_detected": null,
  "key_findings": ["plain English interpretation of each CTG feature"],
  "values": {
    "baseline_fhr_bpm": "...",
    "variability": "...",
    "accelerations": "...",
    "decelerations": "...",
    "contractions_per_10min": "...",
    "overall_classification": "normal / suspicious / pathological"
  },
  "flags": ["any concerning features, in plain English"],
  "raw_summary": "2-3 sentence plain English summary of what this CTG shows"
}

FIGO classification reference:
- Normal: baseline 110-160 bpm, variability 5-25 bpm, accelerations present, no decelerations
- Suspicious: one abnormal feature
- Pathological: two or more abnormal features or sinusoidal pattern

Return ONLY the JSON.""",

    "other": """You are analysing a medical document image from a pregnancy or postpartum context.

Extract and return a JSON object:
{
  "doc_type": "other",
  "week_detected": null,
  "key_findings": ["plain English bullet for each significant finding visible"],
  "values": {},
  "flags": ["anything that appears abnormal or requires attention"],
  "raw_summary": "2-3 sentence plain English summary of what this document contains"
}

Return ONLY the JSON.""",
}


def _detect_doc_type_from_hint(type_hint: str) -> str:
    """Map user-provided type hint to a doc type key."""
    hint = (type_hint or "").lower()
    mapping = {
        "mutterpass": "mutterpass",
        "ultrasound": "ultrasound", "scan": "ultrasound", "sonography": "ultrasound",
        "blood": "blood_report", "blood_report": "blood_report", "bloods": "blood_report",
        "urine": "urine_report", "urine_report": "urine_report",
        "ctg": "ctg", "cardiotocograph": "ctg",
    }
    return mapping.get(hint, "other")


def vision_node(state: BabyOSState) -> dict:
    """
    LangGraph node — runs when an image is present in state.
    
    Reads state["uploaded_image_b64"] and state["uploaded_image_type"],
    calls GPT-4o Vision with a specialised extraction prompt,
    parses the JSON result into DocumentAnalysis,
    and stores it in state["last_document_analysis"].

    Downstream agents read last_document_analysis via _doc_context().
    """
    image_b64   = state.get("uploaded_image_b64")
    image_type  = state.get("uploaded_image_type", "other")
    # print("CP", image_b64)
    if not image_b64:
        return {}   # Nothing to do — no image uploaded

    doc_type   = _detect_doc_type_from_hint(image_type)
    prompt_txt = EXTRACTION_PROMPTS.get(doc_type, EXTRACTION_PROMPTS["other"])

    llm = ChatOpenAI(
        model="gpt-4o",           # Must use gpt-4o for vision — not mini
        temperature=0,
        max_tokens=1500,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    try:
        response = llm.invoke([
            HumanMessage(content=[
                {"type": "text",       "text": prompt_txt},
                {"type": "image_url",  "image_url": {
                    "url":    f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                }},
            ])
        ])

        raw = response.content.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        print("CP4", raw)
        parsed: dict = json.loads(raw)
        print("CP2", response.content if 'response' in dir() else f"Analysis failed: {e}")
        analysis: DocumentAnalysis = {
            "doc_type":      parsed.get("doc_type", doc_type),
            "week_detected": parsed.get("week_detected"),
            "key_findings":  parsed.get("key_findings", []),
            "values":        parsed.get("values", {}),
            "flags":         parsed.get("flags", []),
            "raw_summary":   parsed.get("raw_summary", ""),
            "uploaded_at":   datetime.now(timezone.utc).isoformat(),
        }
    except (json.JSONDecodeError, Exception) as e:
        # Graceful fallback — store raw text as summary
        print("CP3",response.content if 'response' in dir() else f"Analysis failed: {e}")
        analysis: DocumentAnalysis = {
            "doc_type":      doc_type,
            "week_detected": None,
            "key_findings":  ["Document analysed — see summary below"],
            "values":        {},
            "flags":         [],
            "raw_summary":   response.content if 'response' in dir() else f"Analysis failed: {e}",
            "uploaded_at":   datetime.now(timezone.utc).isoformat(),
        }

    return {
        "last_document_analysis": analysis,
        # Clear image from state after processing (saves memory)
        "uploaded_image_b64":  None,
    }

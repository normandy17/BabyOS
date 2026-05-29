"""
tools/tracker_tools.py
----------------------
Tools for the Tracker Agent.

Tools:
  get_week_data         — direct ChromaDB fetch for a specific pregnancy week
  calculate_who_percentile — compares baby measurement to WHO growth charts
  generate_growth_summary — narrative summary of growth measurements
  get_milestone_checklist — returns expected milestones for a baby age
"""

import json
import math
import os
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool


# ── WHO Growth Standards (simplified key percentiles) ─────────────────────────
# Weight-for-age (kg) — Girls and Boys combined approximation
# Source: WHO Child Growth Standards 2006
# Format: age_months → {p3, p15, p50, p85, p97}

_WHO_WEIGHT_KG: dict[int, dict] = {
    0:  {"p3": 2.5,  "p15": 2.9,  "p50": 3.3,  "p85": 3.9,  "p97": 4.4},
    1:  {"p3": 3.4,  "p15": 3.9,  "p50": 4.5,  "p85": 5.1,  "p97": 5.8},
    2:  {"p3": 4.3,  "p15": 4.9,  "p50": 5.6,  "p85": 6.3,  "p97": 7.1},
    3:  {"p3": 5.0,  "p15": 5.7,  "p50": 6.4,  "p85": 7.2,  "p97": 8.0},
    4:  {"p3": 5.6,  "p15": 6.2,  "p50": 7.0,  "p85": 7.9,  "p97": 8.7},
    6:  {"p3": 6.4,  "p15": 7.1,  "p50": 7.9,  "p85": 8.8,  "p97": 9.7},
    9:  {"p3": 7.1,  "p15": 7.9,  "p50": 8.9,  "p85": 9.9,  "p97": 10.9},
    12: {"p3": 7.7,  "p15": 8.6,  "p50": 9.6,  "p85": 10.8, "p97": 11.8},
    15: {"p3": 8.1,  "p15": 9.1,  "p50": 10.2, "p85": 11.4, "p97": 12.6},
    18: {"p3": 8.5,  "p15": 9.5,  "p50": 10.7, "p85": 12.0, "p97": 13.2},
    24: {"p3": 9.2,  "p15": 10.3, "p50": 11.5, "p85": 12.9, "p97": 14.3},
}

# Length/height-for-age (cm)
_WHO_LENGTH_CM: dict[int, dict] = {
    0:  {"p3": 46.3, "p15": 47.9, "p50": 49.9, "p85": 51.8, "p97": 53.4},
    1:  {"p3": 50.8, "p15": 52.6, "p50": 54.7, "p85": 56.9, "p97": 58.6},
    2:  {"p3": 54.4, "p15": 56.4, "p50": 58.4, "p85": 60.7, "p97": 62.4},
    3:  {"p3": 57.3, "p15": 59.4, "p50": 61.4, "p85": 63.7, "p97": 65.5},
    6:  {"p3": 63.3, "p15": 65.5, "p50": 67.6, "p85": 70.0, "p97": 71.9},
    9:  {"p3": 67.7, "p15": 70.1, "p50": 72.3, "p85": 74.6, "p97": 76.5},
    12: {"p3": 71.0, "p15": 73.4, "p50": 75.7, "p85": 78.0, "p97": 80.2},
    18: {"p3": 76.9, "p15": 79.6, "p50": 82.3, "p85": 85.0, "p97": 87.0},
    24: {"p3": 81.7, "p15": 84.6, "p50": 87.8, "p85": 90.9, "p97": 93.2},
}

# Developmental milestones by age in months
_MILESTONES: dict[int, dict] = {
    1:  {
        "motor":    ["lifts head briefly when on tummy", "hands in fists"],
        "social":   ["responds to sounds", "focuses on faces"],
        "language": ["makes small throaty sounds"],
    },
    2:  {
        "motor":    ["holds head up for short periods", "smoother arm movements"],
        "social":   ["social smile appears", "follows moving objects with eyes"],
        "language": ["cooing sounds", "different cries for different needs"],
    },
    4:  {
        "motor":    ["holds head steady", "pushes up on arms during tummy time", "rolls front to back"],
        "social":   ["laughs out loud", "recognises familiar faces"],
        "language": ["babbles", "responds to name"],
    },
    6:  {
        "motor":    ["rolls both ways", "sits with support", "reaches for objects"],
        "social":   ["knows familiar vs unfamiliar faces", "enjoys peek-a-boo"],
        "language": ["strings vowels together (ah, oh)", "responds to own name"],
    },
    9:  {
        "motor":    ["sits independently", "begins crawling or commando crawl", "pulls to stand"],
        "social":   ["separation anxiety appears", "plays interactive games"],
        "language": ["mama/dada (not yet specific)", "points at objects"],
        "red_flags": ["not sitting with support", "not babbling"],
    },
    12: {
        "motor":    ["pulls to stand and cruises furniture", "may take first steps", "pincer grasp"],
        "social":   ["waves bye-bye", "hands you things to share"],
        "language": ["1–3 words with meaning", "understands simple commands"],
        "red_flags": ["no babbling", "no pointing or waving", "no words by 16 months"],
    },
    18: {
        "motor":    ["walks independently", "climbs stairs with help", "stacks 2–4 blocks"],
        "social":   ["parallel play", "shows affection", "tantrums begin"],
        "language": ["10–25 words", "points to body parts", "follows 2-step instructions"],
        "red_flags": ["not walking by 18 months → refer", "fewer than 6 words"],
    },
    24: {
        "motor":    ["runs (wobbly)", "kicks ball", "turns pages one at a time"],
        "social":   ["plays alongside other children", "increasing independence"],
        "language": ["50+ words", "2-word combinations", "strangers can understand ~50%"],
        "red_flags": ["fewer than 50 words", "no 2-word combinations → speech referral"],
    },
}


def _nearest_age(age_months: int, table: dict) -> int:
    """Find nearest age key in a WHO table."""
    keys = sorted(table.keys())
    return min(keys, key=lambda k: abs(k - age_months))


@tool
def get_week_data(week: int) -> str:
    """
    Retrieve structured fetal development data for a specific pregnancy week.
    Bypasses LLM routing — direct lookup from the development corpus.
    Input: week number (integer 4–42)
    """
    week = max(4, min(42, int(week)))
    data_path = Path(__file__).parent.parent / "data" / "raw" / "fetal_development.json"

    try:
        all_weeks: list[dict] = json.loads(data_path.read_text())
        # Find exact week or nearest
        exact = next((w for w in all_weeks if w["week"] == week), None)
        if not exact:
            nearest = min(all_weeks, key=lambda w: abs(w["week"] - week))
            return (
                f"No exact data for week {week}. "
                f"Nearest available (week {nearest['week']}):\n"
                + _format_week(nearest)
            )
        return _format_week(exact)
    except Exception as e:
        return f"Could not load week data: {e}"


def _format_week(w: dict) -> str:
    return (
        f"Week {w['week']} — {w['size_comparison']} "
        f"({w['size_cm']}cm, {w['weight_g']}g)\n\n"
        f"Development: {w['development']}\n\n"
        f"Milestones: {', '.join(w['baby_milestones'])}\n\n"
        f"Mom: {w['mom_changes']}\n"
        f"Symptoms: {', '.join(w['mom_symptoms'][:5])}\n\n"
        f"Dad tip: {w['dad_partner_tips']}\n\n"
        f"Appointments: {', '.join(w['appointments'])}\n\n"
        f"⚠ Danger signs: {', '.join(w['danger_signs'])}"
    )


@tool
def calculate_who_percentile(
    age_months: int,
    weight_kg: Optional[float] = None,
    height_cm: Optional[float] = None,
) -> str:
    """
    Compare baby's weight and/or height to WHO growth chart percentiles.
    Input:
      age_months: baby's age in months (0–24)
      weight_kg:  baby's current weight in kg (optional)
      height_cm:  baby's current length/height in cm (optional)
    """
    results = [f"WHO Growth Percentiles — {age_months} months:"]

    if weight_kg is not None:
        nearest = _nearest_age(age_months, _WHO_WEIGHT_KG)
        ref     = _WHO_WEIGHT_KG[nearest]
        pct     = _interpolate_percentile(weight_kg, ref)
        flag    = " ⚠ Below 3rd percentile — discuss with paediatrician" if weight_kg < ref["p3"] else ""
        results.append(
            f"\nWeight: {weight_kg}kg → approximately {pct}th percentile{flag}"
        )
        results.append(
            f"  Reference (age {nearest}m): P3={ref['p3']}kg  P50={ref['p50']}kg  P97={ref['p97']}kg"
        )

    if height_cm is not None:
        nearest = _nearest_age(age_months, _WHO_LENGTH_CM)
        ref     = _WHO_LENGTH_CM[nearest]
        pct     = _interpolate_percentile(height_cm, ref)
        flag    = " ⚠ Below 3rd percentile — discuss with paediatrician" if height_cm < ref["p3"] else ""
        results.append(
            f"\nLength: {height_cm}cm → approximately {pct}th percentile{flag}"
        )
        results.append(
            f"  Reference (age {nearest}m): P3={ref['p3']}cm  P50={ref['p50']}cm  P97={ref['p97']}cm"
        )

    if weight_kg is None and height_cm is None:
        return "Please provide at least weight_kg or height_cm."

    results.append("\nSource: WHO Child Growth Standards 2006")
    return "\n".join(results)


def _interpolate_percentile(value: float, ref: dict) -> int:
    """Rough percentile from known percentile anchors."""
    anchors = [(3, ref["p3"]), (15, ref["p15"]), (50, ref["p50"]),
               (85, ref["p85"]), (97, ref["p97"])]
    if value <= ref["p3"]:   return 3
    if value >= ref["p97"]:  return 97
    for i in range(len(anchors) - 1):
        p_lo, v_lo = anchors[i]
        p_hi, v_hi = anchors[i + 1]
        if v_lo <= value <= v_hi:
            frac = (value - v_lo) / (v_hi - v_lo)
            return round(p_lo + frac * (p_hi - p_lo))
    return 50


@tool
def get_milestone_checklist(age_months: int) -> str:
    """
    Get developmental milestone checklist for a baby at a given age.
    Input: age_months (0–24)
    Useful for reassuring parents and flagging potential delays.
    """
    nearest = _nearest_age(age_months, _MILESTONES)
    m       = _MILESTONES[nearest]
    lines   = [f"Milestones — around {nearest} months:"]

    for domain, items in m.items():
        if domain == "red_flags":
            lines.append(f"\n⚠ When to seek advice:")
            for item in items:
                lines.append(f"  • {item}")
        else:
            lines.append(f"\n{domain.title()}:")
            for item in items:
                lines.append(f"  ✓ {item}")

    lines.append("\nNote: Milestone ranges are wide. A child may reach these slightly earlier or later and still be developing normally.")
    return "\n".join(lines)


TRACKER_TOOLS = [
    get_week_data,
    calculate_who_percentile,
    get_milestone_checklist,
]
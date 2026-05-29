"""
rag/taxonomy.py
---------------
Single source of truth for BabyOS topics and time periods.

Every chunk in the entire system — markdown, PDF, YouTube, web scrape —
gets tagged with:
  topic  → one value from TOPICS
  period → one value from PERIODS (or "all" if cross-cutting)

This file is imported by:
  corpus/universal_ingest.py  — writes these tags at ingest time
  rag/rag_system.py           — filters by these tags at retrieval time
  backend/routers/articles_router.py — exposes topics to the frontend
"""

from typing import Literal

# ── Topics ─────────────────────────────────────────────────────────────────────

PREGNANCY_TOPICS = [
    "pregnancy_symptoms",
    "fetal_movement",
    "mental_health",
    "diet_advice",
    "pregnancy_workout",
    "informed_choices",
    "labor",
    "breastfeeding",
    "for_partner",
    "medical_board",
]

POSTPARTUM_TOPICS = [
    "first_weeks",
    "baby_care_guide",
    "baby_development",
    "breastfeeding",       # shared — same tag works in both sections
    "diet_advice",         # shared
    "mental_health",       # shared
    "clothing",
    "medical_board",       # shared
]

ALL_TOPICS = list(dict.fromkeys(PREGNANCY_TOPICS + POSTPARTUM_TOPICS))  # deduped, order preserved

TOPIC_LABELS = {
    "pregnancy_symptoms": "Pregnancy symptoms",
    "fetal_movement":     "Fetal movement",
    "mental_health":      "Mental health",
    "diet_advice":        "Diet advice",
    "pregnancy_workout":  "Pregnancy workout",
    "informed_choices":   "Informed choices",
    "labor":              "Labor & birth",
    "breastfeeding":      "Breastfeeding guide",
    "for_partner":        "For you as a partner",
    "medical_board":      "Medical board",
    "first_weeks":        "First weeks",
    "baby_care_guide":    "Baby care guide",
    "baby_development":   "Baby 0–24 months",
    "clothing":           "Clothing",
}

# Which topics belong to which section
TOPIC_SECTIONS: dict[str, list[str]] = {
    "pregnancy":  PREGNANCY_TOPICS,
    "postpartum": POSTPARTUM_TOPICS,
}

# ── Periods ────────────────────────────────────────────────────────────────────

# Pregnancy: 9 months (month 1 = weeks 1-4, month 9 = weeks 33-40)
PREGNANCY_PERIODS = [f"pregnancy_m{i}" for i in range(1, 10)]

# Postpartum: months 0-24
POSTPARTUM_PERIODS = [f"postpartum_m{i}" for i in range(0, 25)]

ALL_PERIODS = PREGNANCY_PERIODS + POSTPARTUM_PERIODS

PERIOD_LABELS = {}
for i in range(1, 10):
    week_start = (i - 1) * 4 + 1
    week_end   = min(i * 4, 40)
    PERIOD_LABELS[f"pregnancy_m{i}"] = f"Month {i} (weeks {week_start}–{week_end})"
for i in range(0, 25):
    PERIOD_LABELS[f"postpartum_m{i}"] = (
        f"Week {i * 4}–{i * 4 + 3} after birth" if i < 3
        else f"Month {i} after birth"
    )
PERIOD_LABELS["all"] = "General (all periods)"

def week_to_pregnancy_month(week: int) -> str:
    """Convert a pregnancy week (1–40) to its period tag."""
    month = min(max(1, (week - 1) // 4 + 1), 9)
    return f"pregnancy_m{month}"

def postpartum_weeks_to_period(pp_weeks: int) -> str:
    """Convert postpartum weeks to a period tag."""
    month = min(pp_weeks // 4, 24)
    return f"postpartum_m{month}"

def age_months_to_period(age_months: int) -> str:
    """Convert baby age in months to a period tag."""
    return f"postpartum_m{min(age_months, 24)}"

# ── Tagger — assigns topic + period from text content ─────────────────────────
# Used by ingest to auto-tag chunks when source metadata isn't definitive.

import re

TOPIC_PATTERNS: dict[str, list[str]] = {
    "pregnancy_symptoms": [
        r"\bnausea\b", r"\bmorning sickness\b", r"\bfatigue\b", r"\bheartburn\b",
        r"\bback.?pain\b", r"\bswelling\b", r"\bbraxton hicks\b", r"\bsymptom\b",
        r"\bround ligament\b", r"\bfrequent urination\b", r"\bblood pressure\b",
    ],
    "fetal_movement": [
        r"\bfetal movement\b", r"\bkicks?\b", r"\bquickening\b", r"\bbaby mov",
        r"\bfetal heart\b", r"\bheart rate\b", r"\bsonograph\b", r"\bultrasound\b",
        r"\bfetal development\b", r"\bweek \d+ baby\b", r"\bsize.{0,10}week\b",
        r"\borgan.{0,20}form\b", r"\bgrowth\b",
    ],
    "mental_health": [
        r"\bdepression\b", r"\banxiet\b", r"\bpnd\b", r"\bpostnatal\b",
        r"\bpostpartum mental\b", r"\bbaby blues\b", r"\bstress\b", r"\bmood\b",
        r"\bidentity\b", r"\bmatrescence\b", r"\bpatrescence\b", r"\bwellbeing\b",
        r"\bmental health\b", r"\btherapy\b", r"\bcbt\b", r"\bcounsell\b",
    ],
    "diet_advice": [
        r"\bnutrition\b", r"\bdiet\b", r"\bfood\b", r"\beat\b", r"\bvitamin\b",
        r"\bsupplement\b", r"\bfolic acid\b", r"\biron\b", r"\bcalcium\b",
        r"\bomega.?3\b", r"\bdha\b", r"\bsafe to eat\b", r"\bfood.?avoid\b",
        r"\bweaning\b", r"\bsolids\b", r"\bbeikost\b", r"\bfeeding\b",
    ],
    "pregnancy_workout": [
        r"\bexercise\b", r"\bworkout\b", r"\bpelvic floor\b", r"\byoga\b",
        r"\bswimming\b", r"\bwalk\b", r"\bphysical activity\b", r"\bfitness\b",
    ],
    "informed_choices": [
        r"\bbirth plan\b", r"\binformed choice\b", r"\bconsent\b", r"\bscreening\b",
        r"\btest\b", r"\bnipt\b", r"\bnuchal\b", r"\banatomy scan\b",
        r"\bdecision\b", r"\bright\b", r"\boption\b", r"\brefus\b",
    ],
    "labor": [
        r"\blabou?r\b", r"\bbirth\b", r"\bdelivery\b", r"\bcontraction\b",
        r"\bepidural\b", r"\bpain relief\b", r"\bcaesarean\b", r"\bc.?section\b",
        r"\bcrowning\b", r"\bpushing\b", r"\bstages of labour\b", r"\bwaters\b",
        r"\bkreißsaal\b", r"\bhospital bag\b",
    ],
    "breastfeeding": [
        r"\bbreastfeed\b", r"\blatch\b", r"\bmilk supply\b", r"\bengorge\b",
        r"\bnipple\b", r"\bformula\b", r"\bcolostrum\b", r"\bstillen\b",
        r"\bwean\b", r"\bbottle.?feed\b", r"\bpump\b",
    ],
    "for_partner": [
        r"\bpartner\b", r"\bdad\b", r"\bfather\b", r"\bvater\b",
        r"\bhow to help\b", r"\bsupport.{0,20}partner\b", r"\bpaternal\b",
        r"\bnew dad\b", r"\bpartner.{0,20}birth\b",
    ],
    "medical_board": [
        r"\bpreeclampsia\b", r"\bgestational diabetes\b", r"\bgdm\b",
        r"\bpreterm\b", r"\bmiscarriage\b", r"\bectopic\b", r"\bplacenta\b",
        r"\biugr\b", r"\bstillbirth\b", r"\bcomplication\b", r"\bdanger sign\b",
        r"\bemergency\b", r"\bhypertension\b", r"\bhyperemesis\b",
        r"\bjaundice\b", r"\bsepsis\b", r"\banaemia\b",
    ],
    "first_weeks": [
        r"\bfourth trimester\b", r"\bnewborn\b", r"\bfirst week\b",
        r"\bfirst month\b", r"\bneonatal\b", r"\bwochenbett\b",
    ],
    "baby_care_guide": [
        r"\bnappy\b", r"\bdiaper\b", r"\bbathing baby\b", r"\bsettl\b",
        r"\bwindling\b", r"\bwind\b", r"\bumbilic\b", r"\bcord\b",
        r"\bswaddle\b", r"\bhold.{0,10}baby\b", r"\bcarry\b",
    ],
    "baby_development": [
        r"\bmilestone\b", r"\bdevelopment\b", r"\bcrawl\b", r"\bwalk\b",
        r"\bfirst.{0,10}step\b", r"\bfirst word\b", r"\bspeech\b",
        r"\blanguage\b", r"\bmotor\b", r"\bcogni\b", r"\btoddler\b",
        r"\bgrowth chart\b", r"\bpercentile\b", r"\bwho.{0,10}standard\b",
    ],
    "clothing": [
        r"\bcloth\b", r"\bbabysuit\b", r"\bbody\b", r"\bgrow bag\b",
        r"\bsleep suit\b", r"\blayer\b", r"\bsize guide\b", r"\bessential\b",
    ],
}

PERIOD_PATTERNS: dict[str, list[str]] = {
    # Pregnancy months
    "pregnancy_m1": [r"\bweek [1-4]\b", r"\bmonth 1\b", r"\bimplantat\b", r"\bembry\b"],
    "pregnancy_m2": [r"\bweek [5-8]\b", r"\bmonth 2\b", r"\bhcg\b"],
    "pregnancy_m3": [r"\bweek [9-9]\b", r"\bweek 1[0-2]\b", r"\bmonth 3\b", r"\bfirst trimester\b", r"\b12.week scan\b"],
    "pregnancy_m4": [r"\bweek 1[3-6]\b", r"\bmonth 4\b", r"\bsecond trimester\b"],
    "pregnancy_m5": [r"\bweek 1[7-9]\b", r"\bweek 20\b", r"\bmonth 5\b", r"\banatomy scan\b", r"\b20.week\b"],
    "pregnancy_m6": [r"\bweek 2[1-4]\b", r"\bmonth 6\b", r"\bglucose\b", r"\bgtt\b"],
    "pregnancy_m7": [r"\bweek 2[5-8]\b", r"\bmonth 7\b", r"\bthird trimester\b"],
    "pregnancy_m8": [r"\bweek 2[9-9]\b", r"\bweek 3[0-2]\b", r"\bmonth 8\b"],
    "pregnancy_m9": [r"\bweek 3[3-9]\b", r"\bweek 40\b", r"\bmonth 9\b", r"\bterm\b", r"\blabou?r\b", r"\bdue date\b"],
    # Postpartum months
    "postpartum_m0": [r"\bnewborn\b", r"\bday [1-9]\b", r"\bfirst week\b", r"\bapgar\b", r"\bu1\b", r"\bu2\b"],
    "postpartum_m1": [r"\b[23456] weeks? (old|postpartum|after)\b", r"\bwochenbett\b", r"\b6.week check\b"],
    "postpartum_m2": [r"\b2 months?\b", r"\b8 weeks?\b", r"\bu3\b"],
    "postpartum_m3": [r"\b3 months?\b", r"\b12 weeks?\b", r"\bu4\b"],
    "postpartum_m4": [r"\b4 months?\b", r"\bsleep regression\b"],
    "postpartum_m6": [r"\b6 months?\b", r"\bweaning\b", r"\bfirst food\b", r"\bsolids\b", r"\bu5\b"],
    "postpartum_m9": [r"\b9 months?\b", r"\bcrawl\b", r"\bpull.to.stand\b"],
    "postpartum_m12": [r"\b12 months?\b", r"\bone year\b", r"\bfirst birthday\b", r"\bfirst step\b", r"\bu6\b"],
    "postpartum_m18": [r"\b18 months?\b", r"\btoddler\b", r"\btantrum\b"],
    "postpartum_m24": [r"\b24 months?\b", r"\b2 years?\b", r"\bfirst step\b", r"\bu7\b"],
}

_COMPILED_TOPICS  = {t: [re.compile(p, re.IGNORECASE) for p in pats] for t, pats in TOPIC_PATTERNS.items()}
_COMPILED_PERIODS = {p: [re.compile(r, re.IGNORECASE) for r in pats] for p, pats in PERIOD_PATTERNS.items()}


def detect_topic(text: str) -> str:
    """Return the best matching topic for a chunk of text."""
    q      = text[:1000]  # scan first 1000 chars
    scores = {t: sum(1 for p in pats if p.search(q)) for t, pats in _COMPILED_TOPICS.items()}
    scores = {t: s for t, s in scores.items() if s > 0}
    return max(scores, key=scores.__getitem__) if scores else "medical_board"


def detect_period(text: str, week: int = None, postpartum_weeks: int = None) -> str:
    """Return the best matching period for a chunk. Explicit week/pp_weeks take priority."""
    if week and 1 <= week <= 40:
        return week_to_pregnancy_month(week)
    if postpartum_weeks is not None and postpartum_weeks >= 0:
        return postpartum_weeks_to_period(postpartum_weeks)

    q      = text[:1000]
    scores = {p: sum(1 for r in pats if r.search(q)) for p, pats in _COMPILED_PERIODS.items()}
    scores = {p: s for p, s in scores.items() if s > 0}
    return max(scores, key=scores.__getitem__) if scores else "all"

# ── Query strings per topic — used by retrieve_by_topic() ────────────────────
# Pre-written queries give better embedding alignment than the raw topic key.
 
TOPIC_COLLECTION_MAP_QUERIES: dict[str, str] = {
    "pregnancy_symptoms":  "common pregnancy symptoms nausea fatigue back pain heartburn what to expect",
    "fetal_movement":      "fetal movement baby kicks quickening how much movement is normal week by week development",
    "mental_health":       "mental health anxiety depression during pregnancy postnatal depression PND emotional support",
    "diet_advice":         "diet nutrition safe foods supplements during pregnancy what to eat and avoid",
    "pregnancy_workout":   "exercise safe activities workout during pregnancy pelvic floor yoga swimming",
    "informed_choices":    "informed choices birth plan antenatal screening tests options rights decisions",
    "labor":               "labour birth contractions pain relief stages delivery what to expect",
    "breastfeeding":       "breastfeeding latch milk supply colostrum technique how to start engorgement",
    "for_partner":         "partner dad support during pregnancy how to help what she is going through",
    "medical_board":       "pregnancy complications preeclampsia gestational diabetes danger signs medical conditions",
    "first_weeks":         "first weeks after birth newborn fourth trimester what to expect postnatal",
    "baby_care_guide":     "newborn baby care bathing nappy feeding winding settling sleep safe",
    "baby_development":    "baby development milestones 0 to 24 months growth crawling walking talking",
    "clothing":            "baby clothing what to buy newborn essentials size guide safe sleep layers",
}


def detect_section(topic: str) -> str:
    """Return 'pregnancy', 'postpartum', or 'both'."""
    in_preg = topic in PREGNANCY_TOPICS
    in_post = topic in POSTPARTUM_TOPICS
    if in_preg and in_post:
        return "both"
    if in_post:
        return "postpartum"
    return "pregnancy"
"""
backend/routers/articles_router.py — v4
-----------------------------------------
Clean, direct flow:

  Frontend sends topic label ("Fetal Movement", "Labor", etc.)
      ↓
  Router maps label → internal key ("fetal_movement", "labor")
      ↓
  retrieve_by_topic(topic=key, period=period)
      ↓
  ChromaDB filters: {topic: key, source_type: [pdf, youtube]}
      ↓
  Cohere reranker → top 6 cards returned

That's it. No TOPIC_QUERIES in the router. No extra logic.
The query string used for similarity search lives in taxonomy.py
under TOPIC_COLLECTION_MAP_QUERIES — one source of truth.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/articles", tags=["articles"])


# ── Label → internal key ───────────────────────────────────────────────────────
# Frontend sends the display label exactly as shown in the UI.
# We translate it to the internal taxonomy key used in ChromaDB metadata.

LABEL_TO_KEY: dict[str, str] = {
    # Pregnancy section
    "pregnancy":              "fetal_movement",
    "symptoms":               "pregnancy_symptoms",
    "fetal movement":         "fetal_movement",
    "mental health":          "mental_health",
    "diet advice":            "diet_advice",
    "pregnancy workout":      "pregnancy_workout",
    "informed choices":       "informed_choices",
    "labor":                  "labor",
    "breastfeeding guide":    "breastfeeding",
    "for you as a partner":   "for_partner",
    "medical board":          "medical_board",
    # Childcare section
    "trimester":              "first_weeks",
    "first weeks":            "first_weeks",
    "baby care guide":        "baby_care_guide",
    "baby 0-24 months":       "baby_development",
    "baby development":       "baby_development",
    "clothing":               "clothing",
}

def label_to_key(label: str) -> str:
    """'Fetal Movement' → 'fetal_movement'"""
    return LABEL_TO_KEY.get(label.lower().strip(), label.lower().replace(" ", "_"))


# ── Retriever singleton ────────────────────────────────────────────────────────

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from rag.rag_system_universal_v2 import UniversalBabyOSRetriever
        _retriever = UniversalBabyOSRetriever(
            k_candidates=20,
            k_final=6,
            use_hyde=False,
            use_reranker=True,
        )
    return _retriever


# ── Models ─────────────────────────────────────────────────────────────────────

class ArticlesRequest(BaseModel):
    topic:   str                  # frontend label e.g. "Fetal Movement"
    section: str  = "pregnancy"   # "pregnancy" or "childcare"
    week:    Optional[int]  = None
    period:  Optional[str]  = None
    limit:   int = 6

class ArticleCard(BaseModel):
    id:           str
    title:        str
    body:         str
    source_type:  str             # "pdf" or "youtube"
    source_name:  str
    topic:        str
    period:       str
    youtube_url:  Optional[str]  = None
    rerank_score: Optional[float] = None
    pdf_url:      Optional[str]  = None


# ── Card builder ───────────────────────────────────────────────────────────────

def _build_card(doc, idx: int, topic_key: str) -> ArticleCard:
    print("dfg", doc.metadata)
    meta        = doc.metadata
    source_type = meta.get("source_type", "")
    source_name = meta.get("source_name", "")
    week        = meta.get("week")
    timestamp   = meta.get("timestamp")
    channel     = meta.get("channel", source_name)

    if source_type == "youtube":
        title = f"📹 {channel}" + (f" — {timestamp}" if timestamp else "")
    elif source_type == "pdf":
        title = f"📖 {source_name}"
    else:
        title = source_name or "BabyOS"
    
    print("meta", meta)
    
    pdf_url = (
    f"/api/articles/pdf/{urllib.parse.quote(meta.get('source_file', ''))}"
    if source_type == "pdf"
    else None
    )
    print("articleInfo", pdf_url)

    return ArticleCard(
        id=           f"{topic_key}_{idx}",
        title=        title,
        body=         doc.page_content[:800].strip(),
        source_type=  source_type,
        source_name=  source_name,
        topic=        meta.get("topic", topic_key),
        period=       meta.get("period", "all"),
        youtube_url=  meta.get("youtube_url"),
        rerank_score= meta.get("rerank_score"),
        pdf_url=      meta.get('source_file', ''),
    )


# ── Main endpoint ──────────────────────────────────────────────────────────────

@router.post("", response_model=list[ArticleCard])
async def get_articles(req: ArticlesRequest):
    """
    Called by the frontend when a topic chip is tapped.

    Flow:
      1. Translate frontend label → taxonomy key
      2. Resolve week → period if provided
      3. Call retrieve_by_topic(key, period)  ← all logic lives in rag_system.py
      4. Filter to pdf + youtube only
      5. Return ArticleCard array
    """
    topic_key = label_to_key(req.topic)

    # Resolve period from week number if provided
    period = req.period
    if not period and req.week:
        from rag.taxonomy import week_to_pregnancy_month
        period = week_to_pregnancy_month(req.week)

    retriever = get_retriever()
    try:
        docs = retriever.retrieve_by_topic(
            topic=topic_key,
            period=period,
            k=req.limit,
        )
    except Exception as e:
        print(f"[articles] '{topic_key}' failed: {e}")
        return []

    # Only show PDF and YouTube content
    docs = [d for d in docs if d.metadata.get("source_type") in ("pdf", "youtube")]

    return [_build_card(doc, i, topic_key) for i, doc in enumerate(docs)]


# ── Topic / period lists for frontend ─────────────────────────────────────────

@router.get("/topics")
async def list_topics(section: Optional[str] = Query(None)):
    """Returns topics matching your frontend arrays, with internal keys attached."""
    all_topics = [
        # Pregnancy
        {"key": "fetal_movement",     "label": "Pregnancy",            "emoji": "🤰", "section": "pregnancy"},
        {"key": "pregnancy_symptoms", "label": "Symptoms",             "emoji": "💊", "section": "pregnancy"},
        {"key": "fetal_movement",     "label": "Fetal Movement",       "emoji": "👶", "section": "pregnancy"},
        {"key": "mental_health",      "label": "Mental Health",        "emoji": "🧠", "section": "pregnancy"},
        {"key": "diet_advice",        "label": "Diet Advice",          "emoji": "🥗", "section": "pregnancy"},
        {"key": "pregnancy_workout",  "label": "Pregnancy Workout",    "emoji": "🏃‍♀️", "section": "pregnancy"},
        {"key": "informed_choices",   "label": "Informed Choices",     "emoji": "✅", "section": "pregnancy"},
        {"key": "labor",              "label": "Labor",                "emoji": "🏥", "section": "pregnancy"},
        {"key": "breastfeeding",      "label": "Breastfeeding Guide",  "emoji": "🤱", "section": "pregnancy"},
        {"key": "for_partner",        "label": "For You as a Partner", "emoji": "💑", "section": "pregnancy"},
        {"key": "medical_board",      "label": "Medical Board",        "emoji": "🩺", "section": "pregnancy"},
        # Childcare
        {"key": "first_weeks",        "label": "Trimester",            "emoji": "📅", "section": "childcare"},
        {"key": "first_weeks",        "label": "First Weeks",          "emoji": "🌱", "section": "childcare"},
        {"key": "baby_care_guide",    "label": "Baby Care Guide",      "emoji": "🍼", "section": "childcare"},
        {"key": "baby_development",   "label": "Baby 0-24 Months",     "emoji": "👶", "section": "childcare"},
        {"key": "breastfeeding",      "label": "Breastfeeding Guide",  "emoji": "🤱", "section": "childcare"},
        {"key": "diet_advice",        "label": "Diet Advice",          "emoji": "🥗", "section": "childcare"},
        {"key": "mental_health",      "label": "Mental Health",        "emoji": "🧠", "section": "childcare"},
        {"key": "clothing",           "label": "Clothing",             "emoji": "👗", "section": "childcare"},
        {"key": "medical_board",      "label": "Medical Board",        "emoji": "🩺", "section": "childcare"},
    ]
    if section:
        return [t for t in all_topics if t["section"] == section]
    return all_topics


@router.get("/periods")
async def list_periods(section: Optional[str] = Query(None)):
    """Month-by-month period list for timeline filtering."""
    from rag.taxonomy import PREGNANCY_PERIODS, POSTPARTUM_PERIODS, PERIOD_LABELS

    periods = []
    if not section or section == "pregnancy":
        for p in PREGNANCY_PERIODS:
            periods.append({"key": p, "label": PERIOD_LABELS[p], "section": "pregnancy"})
    if not section or section in ("childcare", "postpartum"):
        for p in POSTPARTUM_PERIODS:
            periods.append({"key": p, "label": PERIOD_LABELS[p], "section": "childcare"})
    return periods


# ── Get actual pdf ─────────────────────────────────────────
from fastapi.responses import FileResponse
from fastapi import HTTPException
import urllib.parse

# Where your PDFs live on disk — adjust to your actual path
PDF_DIR = Path(__file__).parent/ "data" / "raw" / "books"

@router.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """
    Serve a PDF file by name.
    Called by frontend when user taps 'Open PDF' on an article card.
    
    Usage: GET /api/articles/pdf/some_book.pdf
    """
    # Decode and sanitize — prevent path traversal
    print("FN", filename)
    safe_name = Path(urllib.parse.unquote(filename)).name  # strips any ../
    print("SN", safe_name)
    pdf_path  = PDF_DIR / safe_name
    print("PDFSN", pdf_path)

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {safe_name}")
    if not pdf_path.suffix.lower() == ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are served here")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=safe_name,
        headers={"Content-Disposition": f"inline; filename={safe_name}"},  # inline = open in viewer, not download
    )

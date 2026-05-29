# 🤰 BabyOS — AI Pregnancy Intelligence Hub

An AI-powered pregnancy companion built with LangChain, LangGraph, ChromaDB,
and Streamlit. Supports mom, dad, and partner roles with week-aware responses,
RAG over medical and Germany-specific content, and multi-agent architecture.

---

## Project Structure

```
babyos/
├── corpus/
│   ├── corpus_fetcher.py     # Step 1: Scrape NHS/WHO web content
│   └── ingest.py             # Step 2: Chunk, embed, store in ChromaDB
├── data/
│   ├── raw/                  # Source documents (markdown, json, scraped txt)
│   │   ├── fetal_development.json
│   │   ├── medical_reference.md
│   │   ├── germany_pregnancy_guide.md
│   │   ├── dad_partner_guide.md
│   │   ├── pregnancy_faqs.md
│   │   └── web_scraped/      # Created by corpus_fetcher.py
│   └── chroma_db/            # Created by ingest.py
├── rag/
│   └── rag_system.py         # BabyOSRetriever + RAG chain factory
├── agents/                   # (Day 3) LangGraph multi-agent system
├── utils/                    # (Day 4) User profile, session state
├── app.py                    # (Day 4) Streamlit UI entry point
├── .env.example              # Copy to .env and fill in keys
└── requirements.txt
```

---

## ChromaDB Collections
============================================================
  INGESTION SUMMARY — babyos_universal
============================================================
  Total chunks: 11263

  By source type:
    pdf           8719
    youtube       2032
    web            412
    markdown        88
    json            12

  By topic (top 15):
    medical_board                   2844  ██████████████████████████████
    labor                           1949  ██████████████████████████████
    diet_advice                     1728  ██████████████████████████████
    informed_choices                 889  ██████████████████████████████
    pregnancy_symptoms               771  ██████████████████████████████
    pregnancy_workout                770  ██████████████████████████████
    fetal_movement                   646  ██████████████████████████████
    mental_health                    606  ██████████████████████████████
    clothing                         383  ██████████████████████████████
    first_weeks                      198  ██████████████████████████████
    for_partner                      169  ██████████████████████████████
    baby_development                 117  ███████████████████████
    breastfeeding                    102  ████████████████████
    baby_care_guide                   91  ██████████████████

  By period (pregnancy months):
    pregnancy_m1            29
    pregnancy_m2           104
    pregnancy_m3           160
    pregnancy_m4           103
    pregnancy_m5            52
    pregnancy_m6           114
    pregnancy_m7            96
    pregnancy_m8            29
    pregnancy_m9          1568

  By period (postpartum, key months):
    postpartum_m0          590
    postpartum_m1            6
    postpartum_m3           53
    postpartum_m6           60
    postpartum_m12          70
    postpartum_m24           9
    all                   8173

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (and optionally LANGCHAIN_API_KEY)
```

### 3. Fetch web corpus (optional — adds NHS/WHO content)
```bash
python corpus/corpus_fetcher.py
```
This takes ~5 minutes and is polite to the servers (1.5s delay between requests).
You can skip it and ingest only the local files first.

### 4. Run ingestion
```bash
# First time
python corpus/ingest.py

# Reset and rebuild from scratch
python corpus/ingest.py --reset
```
Expected output:
```
babyos_development          12 chunks
babyos_medical              28 chunks
babyos_germany              22 chunks
babyos_dad                  31 chunks
babyos_faqs                 19 chunks
babyos_web                 180 chunks   (if fetched)
TOTAL                       292 chunks
```

### 5. Smoke test the RAG system
```bash
python rag/rag_system.py
```

### 6. Run the app (once built)
```bash
streamlit run app.py
```

---

## Architecture

```
User Input
    │
    ▼
Streamlit UI (app.py)
    │
    ▼
LangGraph Supervisor
    │
    ├── Medical Q&A Agent      ──► babyos_medical + babyos_web
    ├── Baby Tracker Agent     ──► babyos_development
    ├── Emotional Support Agent──► user context + memory
    ├── Dad/Partner Agent      ──► babyos_dad + babyos_development
    └── Germany Info Agent     ──► babyos_germany
         │
         ▼
    BabyOSRetriever (rag_system.py)
         │
         ▼
    ChromaDB (6 collections)
         │
         ▼
    GPT-4o / GPT-4o-mini
         │
         ▼
    LangSmith (tracing + evaluation)
```

---

## Deployment

### Streamlit Cloud (recommended for demo)
1. Push repo to GitHub
2. Go to share.streamlit.io → New app
3. Add secrets in Streamlit Cloud dashboard (copy from .env)
4. Deploy

### HuggingFace Spaces
- Use Streamlit SDK
- Note: ChromaDB persistence requires paid storage; use in-memory for demo

### Docker
```bash
docker build -t babyos .
docker run -p 8501:8501 --env-file .env babyos
```

---

## LangSmith Evaluation

Set `LANGCHAIN_TRACING_V2=true` in your `.env`.  
All RAG retrievals and agent responses are automatically traced.

To run evaluations:
```bash
python eval/run_evals.py    # (created on Day 5)
```

---

## Important Notes

- This app is **not a substitute for professional medical advice**.
  Every response from the Medical Q&A agent includes a disclaimer.
- Danger signs trigger an **immediate escalation response** recommending
  the user contact their doctor or call emergency services.
- Scan image analysis uses GPT-4o Vision for descriptive summaries only —
  it does not provide diagnostic interpretations.

"""
backend/main.py
---------------
BabyOS FastAPI Backend

Routes:
  POST /api/chat          — standard chat
  POST /api/chat/stream   — SSE streaming chat
  POST /api/documents/analyze — vision analysis
  GET  /api/profile       — get user profile
  PUT  /api/profile       — upsert user profile
  GET  /api/timeline      — list timeline logs
  POST /api/timeline      — create timeline log
  GET  /api/chat/history  — conversation history
  GET  /auth/callback     — Supabase OAuth callback

Auth: Supabase JWT — every protected route calls verify_token()
DB:   Supabase (Postgres under the hood, accessed via supabase-py)
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from router.articles_router import router as articles_router
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BabyOS API",
    description="Born Together — AI pregnancy and parenting companion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     "http://localhost:3000",           # React dev
    #     "http://localhost:19006",          # Expo dev
    #     os.getenv("FRONTEND_URL", ""),     # Production
    # ],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase ──────────────────────────────────────────────────────────────────

SUPABASE_URL     = os.getenv("SUPABASE_URL","")
SUPABASE_SERVICE = os.getenv("SUPABASE_SERVICE_KEY","")  # service key — server only

if not SUPABASE_URL or not SUPABASE_SERVICE:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE)

# ── Articles Router ──────────────────────────────────────────────────────────────────────
app.include_router(articles_router)

# ── Auth ──────────────────────────────────────────────────────────────────────

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate Supabase JWT and return the user dict."""
    token = credentials.credentials
    try:
        sb    = get_supabase()
        user  = sb.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"id": user.user.id, "email": user.user.email}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")

# ── LangGraph agent (lazy-loaded) ─────────────────────────────────────────────

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        import sys
        sys.path.insert(0, str(__file__).replace("backend/main.py", ""))
        from graphv2 import get_graph as _get
        _graph = _get()
    return _graph

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:       str
    image_base64:  Optional[str] = None
    image_type:    Optional[str] = None
    session_id:    Optional[str] = None

class ChatResponse(BaseModel):
    response:    str
    agent_name:  str
    sources:     list[str]
    danger_flag: bool
    session_id:  str
    document_analysis: Optional[dict] = None

class ProfileUpsert(BaseModel):
    name:                 Optional[str]  = None
    role:                 Optional[str]  = None
    partner_name:         Optional[str]  = None
    mom_name:             Optional[str]  = None
    baby_name:            Optional[str]  = None
    current_week:         Optional[int]  = None
    postpartum_weeks:     Optional[int]  = None
    due_date:             Optional[str]  = None
    birth_date:           Optional[str]  = None
    lmp_date:             Optional[str]  = None
    pregnancy_type:       Optional[str]  = None
    age_mom:              Optional[int]  = None
    conditions:           Optional[list[str]] = None
    medications:          Optional[list[str]] = None
    blood_type_mom:       Optional[str]  = None
    previous_pregnancies: Optional[int]  = None
    is_ivf:               Optional[bool] = None
    hospital_name:        Optional[str]  = None
    hebamme_name:         Optional[str]  = None
    krankenkasse:         Optional[str]  = None
    language:             Optional[str]  = None
    units:                Optional[str]  = None

class TimelineLogCreate(BaseModel):
    phase:             str
    week:              Optional[int]   = None
    postpartum_weeks:  Optional[int]   = None
    baby_age_months:   Optional[float] = None
    symptoms:          list[str]       = []
    mood_score:        Optional[int]   = None
    weight_kg:         Optional[float] = None
    blood_pressure:    Optional[str]   = None
    doctor_notes:      Optional[str]   = None
    baby_weight_g:     Optional[float] = None
    baby_height_cm:    Optional[float] = None
    baby_milestones:   list[str]       = []
    questions_for_doctor: list[str]    = []

class DocumentAnalyzeRequest(BaseModel):
    image_base64: str
    image_type:   str = "other"

# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_or_create_profile(sb: Client, user_id: str, email: str) -> dict:
    """Fetch profile from Supabase; create default if first login."""
    print("profileasd", user_id, email)
    if(True):
        res=None
        res = sb.table("profiles").select("*").eq("id", user_id).execute()
        if res.data:
            return res.data[0]

        print("could not fetch profile details")
        print("qwe", user_id, email)
    # First login — create minimal profile
        profile = {
            "id":                   user_id,
            "email":                email,
            "name":                 email.split("@")[0].title(),
            "role":                 "mom",
            "current_week":         20,
            "postpartum_weeks":     0,
            "phase":                "T2",
            "pregnancy_type":       "singleton",
            "conditions":           [],
            "medications":          [],
            "previous_pregnancies": 0,
            "is_ivf":               False,
            "language":             "en",
            "units":                "metric",
            "created_at":           datetime.now(timezone.utc).isoformat(),
            "updated_at":           datetime.now(timezone.utc).isoformat(),
        }
        print("creating new", profile)
        sb.table("profiles").insert(profile).execute()
        print("creating new2", profile)
        return profile


def _build_agent_state(profile: dict, messages: list[dict]) -> dict:
    """Convert Supabase profile row to BabyOSState-compatible dict."""
    from agents.state import week_to_phase
    from langchain_core.messages import HumanMessage, AIMessage
    print(123, profile)
    week = profile.get("current_week", 20)
    pp   = profile.get("postpartum_weeks", 0)

    lc_messages = []
    for m in messages[-20:]:   # last 20 messages for context window
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    return {
        "messages":         lc_messages,
        "user_profile":     profile,
        "current_query":    "",
        "current_week":     week,
        "postpartum_weeks": pp,
        "current_phase":    week_to_phase(week, pp),
        "user_role":        profile.get("role", "mom"),
        "next_agent":       "FINISH",
        "retrieved_context":  "",
        "retrieved_sources":  [],
        "agent_response":     "",
        "agent_name":         "",
        "danger_flag":        False,
        "danger_reason":      None,
        "uploaded_image_b64": None,
        "uploaded_image_type": None,
        "last_document_analysis": None,
        "timeline_logs":    [],
        "run_metadata":     {},
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "BabyOS API"}


# ── Profile ───────────────────────────────────────────────────────────────────

@app.get("/api/profile")
async def get_profile(user: dict = Depends(verify_token)):
    sb      = get_supabase()
    profile = _get_or_create_profile(sb, user["id"], user["email"])
    return profile


@app.put("/api/profile")
async def upsert_profile(
    data:  ProfileUpsert,
    user:  dict = Depends(verify_token),
):
    sb      = get_supabase()
    payload = {k: v for k, v in data.model_dump().items() if v is not None}
    payload["id"] = user["id"]

    # Recompute phase if week or pp changed
    if "current_week" in payload or "postpartum_weeks" in payload:
        from agents.state import week_to_phase
        week = payload.get("current_week", 20)
        pp   = payload.get("postpartum_weeks", 0)
        payload["phase"] = week_to_phase(week, pp)

    res = sb.table("profiles").upsert(payload).execute()
    return res.data[0] if res.data else payload


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: dict = Depends(verify_token)):
    sb        = get_supabase()
    profile   = _get_or_create_profile(sb, user["id"], user["email"])
    session_id = req.session_id or user["id"]

    # Load recent messages
    hist_res = (
        sb.table("messages")
          .select("*")
          .eq("session_id", session_id)
          .order("created_at", desc=True)
          .limit(20)
          .execute()
    )
    history   = list(reversed(hist_res.data or []))
    state     = _build_agent_state(profile, history)

    # Inject image if present
    if req.image_base64:
        state["uploaded_image_b64"]  = req.image_base64
        state["uploaded_image_type"] = req.image_type or "other"

    # Add human message to state
    from langchain_core.messages import HumanMessage
    state["messages"].append(HumanMessage(content=req.message))

    # Run graph
    from graphv2 import chat as agent_chat
    response_text, result_state = agent_chat(
        message=req.message,
        state=state,
        thread_id=session_id,
        image_b64=req.image_base64,
        image_type=req.image_type,
    )

    # Persist messages to Supabase
    now = datetime.now(timezone.utc).isoformat()
    print("qwe1",result_state.get("danger_flag", False))
    sb.table("messages").insert([
        {
            "session_id":  session_id,
            "user_id":     user["id"],
            "role":        "user",
            "content":     req.message,
            "created_at":  now,
            "danger_flag": False,
        },
        {
            "session_id":  session_id,
            "user_id":     user["id"],
            "role":        "assistant",
            "content":     response_text,
            "agent_name":  result_state.get("agent_name", ""),
            "sources":     result_state.get("retrieved_sources", []),
            "danger_flag": result_state.get("danger_flag", False),
            "created_at":  now,
        },
    ]).execute()

    # Persist document analysis if present
    doc_analysis = result_state.get("last_document_analysis")
    if doc_analysis:
        sb.table("documents").insert({
            "user_id":     user["id"],
            **{k: v for k, v in doc_analysis.items() if v is not None},
        }).execute()

    return ChatResponse(
        response=response_text,
        agent_name=result_state.get("agent_name", ""),
        sources=result_state.get("retrieved_sources", []),
        danger_flag=result_state.get("danger_flag", False),
        session_id=session_id,
        document_analysis=doc_analysis,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, user: dict = Depends(verify_token)):
    """Server-Sent Events streaming endpoint."""

    async def event_generator() -> AsyncGenerator[str, None]:
        sb        = get_supabase()
        profile   = _get_or_create_profile(sb, user["id"], user["email"])
        session_id = req.session_id or user["id"]

        hist_res = (
            sb.table("messages")
              .select("*")
              .eq("session_id", session_id)
              .order("created_at", desc=True)
              .limit(20)
              .execute()
        )
        history = list(reversed(hist_res.data or []))
        state   = _build_agent_state(profile, history)

        if req.image_base64:
            state["uploaded_image_b64"]  = req.image_base64
            state["uploaded_image_type"] = req.image_type or "other"

        from langchain_core.messages import HumanMessage
        state["messages"].append(HumanMessage(content=req.message))

        # Stream via LangGraph .stream()
        from graphv2 import get_graph
        graph      = get_graph()
        full_text  = ""
        agent_name = ""
        sources    = []
        danger     = False

        config = {"configurable": {"thread_id": session_id}}

        try:
            for chunk in graph.stream(state, config=config):
                for node_name, node_output in chunk.items():
                    if node_name == "output_node":
                        response = node_output.get("agent_response", "")
                        agent_name = node_output.get("run_metadata", {}).get("agent_name", "")
                        danger     = node_output.get("danger_flag", False)

                        # Stream word by word for natural feel
                        words = response.split(" ")
                        for i, word in enumerate(words):
                            segment = word + (" " if i < len(words) - 1 else "")
                            full_text += segment
                            yield f"data: {json.dumps({'chunk': segment})}\n\n"
                            await asyncio.sleep(0.015)  # ~65 words/sec

        except Exception as e:
            yield f"data: {json.dumps({'chunk': f'Error: {e}'})}\n\n"

        # Final metadata event
        yield f"data: {json.dumps({'done': True, 'agent_name': agent_name, 'sources': sources, 'danger_flag': danger, 'session_id': session_id})}\n\n"
        yield "data: [DONE]\n\n"

        # Persist asynchronously
        now = datetime.now(timezone.utc).isoformat()
        sb.table("messages").insert([
            {"session_id": session_id, "user_id": user["id"], "role": "user",
             "content": req.message, "created_at": now},
            {"session_id": session_id, "user_id": user["id"], "role": "assistant",
             "content": full_text, "agent_name": agent_name,
             "sources": sources, "danger_flag": danger, "created_at": now},
        ]).execute()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat/history")
async def chat_history(
    session_id: Optional[str] = None,
    limit:      int            = 50,
    user:       dict           = Depends(verify_token),
):
    sb  = get_supabase()
    sid = session_id or user["id"]
    res = (
        sb.table("messages")
          .select("*")
          .eq("session_id", sid)
          .eq("user_id", user["id"])
          .order("created_at")
          .limit(limit)
          .execute()
    )
    return {"messages": res.data or [], "session_id": sid}


# ── Documents ─────────────────────────────────────────────────────────────────

@app.post("/api/documents/analyze")
async def analyze_document(
    req:  DocumentAnalyzeRequest,
    user: dict = Depends(verify_token),
):
    sb      = get_supabase()
    profile = _get_or_create_profile(sb, user["id"], user["email"])
    from agents.state import make_initial_state
    from agents.vision import vision_node

    state = make_initial_state(profile)
    state["uploaded_image_b64"]  = req.image_base64
    state["uploaded_image_type"] = req.image_type

    result = vision_node(state)
    analysis = result.get("last_document_analysis")

    if analysis:
        sb.table("documents").insert({
            "user_id": user["id"], **analysis
        }).execute()

    return analysis or {"error": "Analysis failed"}


@app.get("/api/documents")
async def list_documents(user: dict = Depends(verify_token)):
    sb  = get_supabase()
    res = (
        sb.table("documents")
          .select("*")
          .eq("user_id", user["id"])
          .order("uploaded_at", desc=True)
          .limit(50)
          .execute()
    )
    return res.data or []


# ── Timeline ──────────────────────────────────────────────────────────────────

@app.get("/api/timeline")
async def list_timeline(user: dict = Depends(verify_token)):
    sb  = get_supabase()
    res = (
        sb.table("timeline_logs")
          .select("*")
          .eq("user_id", user["id"])
          .order("logged_at", desc=True)
          .execute()
    )
    return res.data or []


@app.post("/api/timeline")
async def create_timeline_log(
    log:  TimelineLogCreate,
    user: dict = Depends(verify_token),
):
    sb      = get_supabase()
    payload = {
        "user_id":    user["id"],
        "logged_at":  datetime.now(timezone.utc).isoformat(),
        **log.model_dump(),
    }
    res = sb.table("timeline_logs").insert(payload).execute()
    return res.data[0] if res.data else payload


# ── Auth callback ─────────────────────────────────────────────────────────────

@app.get("/auth/callback")
async def auth_callback():
    """Supabase redirects here after Google OAuth. Frontend handles the rest."""
    return {"message": "Auth complete — return to app"}

"""
rag/rag_system.py — v4  (Production RAG)
-----------------------------------------
Pipeline per query:
  1. HyDE  : generate hypothetical answer → embed it
  2. Dense : vector similarity top-20 per collection
  3. Sparse: BM25 keyword retrieval on same corpus
  4. Ensemble merge dense + sparse
  5. Reranker: Cohere cross-encoder → top-5
  6. Books: topic-filtered MMR in parallel
  7. YouTube: semantic search with timestamp citations
  8. Dedup and return
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever

load_dotenv()

BASE_DIR   = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

COLLECTION_DESCRIPTIONS = {
    "babyos_development": "fetal size and weight by week, organ milestones, baby movements, toddler milestones",
    "babyos_medical":     "danger signs, symptoms, food/drug safety, blood test results, GDM, preeclampsia",
    "babyos_postpartum":  "fourth trimester, newborn care, breastfeeding, baby sleep, PND, weaning, toddler",
    "babyos_germany":     "Mutterpass, Vorsorgeuntersuchungen, Hebamme, Elterngeld, Kindergeld, Kita, Standesamt",
    "babyos_dad":         "dad and partner support, hospital bag, labour support, paternal PND, bonding",
    "babyos_faqs":        "common questions: exercise, travel, weight gain, Braxton Hicks, birth plan, miscarriage",
    "babyos_web":         "NHS and WHO clinical guidelines, antenatal care, newborn screening, labour procedures",
    "babyos_books":       "authoritative books: nutrition, breastfeeding, birth positions, infant development, complications",
    "babyos_youtube":     "video transcripts: NHS, Tommy's, BZgA — birth, breastfeeding, baby care, exercise",
}

AGENT_COLLECTION_GUARANTEES = {
    "medical_agent":   ["babyos_medical", "babyos_web"],
    "tracker_agent":   ["babyos_development"],
    "emotional_agent": ["babyos_postpartum"],
    "parent_agent":    ["babyos_dad"],
    "hebamme_agent":   ["babyos_medical", "babyos_germany"],
    "germany_agent":   ["babyos_germany"],
    "default":         [],
}

BOOK_TOPIC_KEYWORDS = {
    "nutrition":     ["food","eat","diet","nutrient","vitamin","iron","folic","supplement","calcium","omega"],
    "labour":        ["labour","labor","birth","contraction","pushing","epidural","delivery","crowning"],
    "newborn":       ["newborn","nappy","diaper","umbilical","jaundice","apgar","neonatal","cord"],
    "breastfeeding": ["breastfeed","latch","milk supply","engorgement","nipple","formula","colostrum"],
    "development":   ["milestone","development","crawl","walk","speech","language","motor","cognitive"],
    "mental_health": ["depression","anxiety","pnd","postnatal","postpartum","baby blues","stress"],
    "complications": ["preeclampsia","gestational diabetes","preterm","miscarriage","ectopic","placenta"],
    "germany":       ["mutterpass","hebamme","elterngeld","vorsorge","kita","krankenkasse","bzga"],
}


def _detect_book_topic(query: str) -> Optional[str]:
    q      = query.lower()
    scores = {t: sum(1 for kw in kws if kw in q) for t, kws in BOOK_TOPIC_KEYWORDS.items()}
    scores = {t: s for t, s in scores.items() if s > 0}
    return max(scores, key=scores.__getitem__) if scores else None


# ── HyDE ──────────────────────────────────────────────────────────────────────

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a pregnancy expert writing a knowledge base entry. "
     "Write 3-5 factual sentences that directly answer the question, "
     "using the same vocabulary found in medical documents. "
     "Do not say 'here is an answer'. Just write the paragraph."),
    ("human", "{question}"),
])

def generate_hyde_embedding(query: str, llm: ChatOpenAI) -> str:
    try:
        return (_HYDE_PROMPT | llm | StrOutputParser()).invoke({"question": query})
    except Exception:
        return query


# ── Cohere Reranker ───────────────────────────────────────────────────────────

def rerank_documents(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    if not COHERE_API_KEY or not docs:
        return docs[:top_n]
    try:
        import cohere
        co      = cohere.Client(COHERE_API_KEY)
        results = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[d.page_content for d in docs],
            top_n=min(top_n, len(docs)),
        )
        reranked = []
        for r in results.results:
            doc = docs[r.index]
            doc.metadata["rerank_score"] = round(r.relevance_score, 4)
            reranked.append(doc)
        return reranked
    except Exception as e:
        print(f"[Reranker] Failed: {e}")
        return docs[:top_n]


# ── Main Retriever ─────────────────────────────────────────────────────────────

class BabyOSRetriever:

    def __init__(self, k_candidates: int = 20, k_final: int = 5,
                 use_hyde: bool = True, use_reranker: bool = True):
        self.k_candidates = k_candidates
        self.k_final      = k_final
        self.use_hyde     = use_hyde
        self.use_reranker = use_reranker
        self._embeddings  = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._stores: dict[str, Chroma] = {}
        self._load_collections()

    def _load_collections(self) -> None:
        loaded = []
        for name in COLLECTION_DESCRIPTIONS:
            path = str(CHROMA_DIR / name)
            if not Path(path).exists():
                continue
            try:
                self._stores[name] = Chroma(
                    collection_name=name,
                    embedding_function=self._embeddings,
                    persist_directory=path,
                )
                loaded.append(name)
            except Exception as e:
                print(f"[RAG] Could not load '{name}': {e}")
        if not loaded:
            raise RuntimeError("No ChromaDB collections found. Run corpus/ingest.py first.")
        print(f"[RAG] Loaded: {', '.join(loaded)}")

    def route_query(self, query: str, role: str, agent_name: str, phase: str) -> list[str]:
        available = {n: d for n, d in COLLECTION_DESCRIPTIONS.items()
                     if n in self._stores and n not in ("babyos_books", "babyos_youtube")}
        col_list  = "\n".join(f"  {n}: {d}" for n, d in available.items())

        prompt = (
            f"Select 1-3 collections for this query.\n\n"
            f"RULES:\n"
            f"- babyos_development = how the baby grows, size, milestones — NOT medical symptoms\n"
            f"- babyos_medical = symptoms, danger signs, food/drug safety, lab results ONLY\n"
            f"- babyos_postpartum = anything after birth\n"
            f"- babyos_germany = German system, admin, vocabulary ONLY\n\n"
            f"Collections:\n{col_list}\n\n"
            f"Role: {role} | Phase: {phase} | Query: \"{query}\"\n\n"
            f"Return ONLY comma-separated names:"
        )
        result  = self._llm.invoke(prompt).content.strip()
        routed  = {c.strip() for c in result.split(",") if c.strip() in self._stores}
        routed |= {c for c in AGENT_COLLECTION_GUARANTEES.get(agent_name, []) if c in self._stores}
        return list(routed - {"babyos_books", "babyos_youtube"})

    def _retrieve_from_collection(self, embed_q: str, raw_q: str,
                                  col: str, week: Optional[int] = None) -> list[Document]:
        store  = self._stores[col]
        kwargs = {"k": self.k_candidates}
        if col == "babyos_development" and week:
            kwargs["filter"] = {"week": {"$gte": max(1, week-3), "$lte": min(42, week+3)}}

        dense = store.similarity_search(embed_q, **kwargs)

        try:
            bm25   = BM25Retriever.from_documents(dense, k=min(10, len(dense)))
            sparse = bm25.invoke(raw_q)
            seen   = {d.page_content[:200] for d in dense}
            for d in sparse:
                if d.page_content[:200] not in seen:
                    dense.append(d)
                    seen.add(d.page_content[:200])
        except Exception:
            pass

        return dense

    def _retrieve_books(self, query: str, k: int = 3) -> list[Document]:
        if "babyos_books" not in self._stores:
            return []
        store = self._stores["babyos_books"]
        topic = _detect_book_topic(query)
        try:
            kwargs = {"k": k, "fetch_k": k * 6}
            if topic:
                kwargs["filter"] = {"topic_tag": topic}
            docs = store.max_marginal_relevance_search(query, **kwargs)
            if not docs:
                docs = store.max_marginal_relevance_search(query, k=k, fetch_k=k*6)
            for d in docs:
                d.metadata.update({"retrieved_from": "babyos_books", "book_topic": topic or "general"})
            return docs
        except Exception as e:
            print(f"[RAG] Books: {e}")
            return []

    def _retrieve_youtube(self, query: str, k: int = 2) -> list[Document]:
        if "babyos_youtube" not in self._stores:
            return []
        try:
            docs = self._stores["babyos_youtube"].similarity_search(query, k=k)
            for d in docs:
                d.metadata["retrieved_from"] = "babyos_youtube"
                vid = d.metadata.get("video_id", "")
                ts  = d.metadata.get("start_seconds", 0)
                if vid:
                    d.metadata["youtube_url"] = f"https://youtube.com/watch?v={vid}&t={ts}s"
            return docs
        except Exception as e:
            print(f"[RAG] YouTube: {e}")
            return []

    def retrieve(self, query: str, week: Optional[int] = None,
                 role: str = "mom", phase: str = "", agent_name: str = "default",
                 collections: Optional[list[str]] = None) -> list[Document]:
        embed_q     = generate_hyde_embedding(query, self._llm) if self.use_hyde else query
        target_cols = collections or self.route_query(query, role, agent_name, phase)

        candidates: list[Document] = []
        seen: set[str] = set()

        for col in target_cols:
            if col not in self._stores:
                continue
            try:
                for d in self._retrieve_from_collection(embed_q, query, col, week):
                    fp = d.page_content[:200].strip()
                    if fp not in seen:
                        seen.add(fp)
                        d.metadata["retrieved_from"] = col
                        candidates.append(d)
            except Exception as e:
                print(f"[RAG] {col}: {e}")

        reranked = rerank_documents(query, candidates, top_n=self.k_final)

        for d in self._retrieve_books(query, k=3):
            fp = d.page_content[:200].strip()
            if fp not in seen:
                seen.add(fp)
                reranked.append(d)

        for d in self._retrieve_youtube(query, k=2):
            fp = d.page_content[:200].strip()
            if fp not in seen:
                seen.add(fp)
                reranked.append(d)

        return reranked

    def retrieve_week_info(self, week: int) -> Optional[Document]:
        if "babyos_development" not in self._stores:
            return None
        results = self._stores["babyos_development"].similarity_search(
            f"pregnancy week {week} fetal development", k=1, filter={"week": week},
        )
        return results[0] if results else None

    def format_context(self, docs: list[Document]) -> str:
        if not docs:
            return "No relevant information found."
        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            col    = doc.metadata.get("retrieved_from", "")
            score  = doc.metadata.get("rerank_score", "")
            yt_url = doc.metadata.get("youtube_url", "")
            score_str = f" [score:{score}]" if score else ""
            yt_str    = f"\n📹 {yt_url}" if yt_url else ""
            sections.append(f"[Source {i} — {source} ({col}){score_str}]{yt_str}:\n{doc.page_content}")
        return "\n\n---\n\n".join(sections)

    def debug_retrieve(self, query: str, week=None, role="mom", phase="", agent_name="default"):
        print(f"\n{'='*60}\n  QUERY : {query}")
        if self.use_hyde:
            hyp = generate_hyde_embedding(query, self._llm)
            print(f"  HyDE  : {hyp[:120]}...")
        routed = self.route_query(query, role, agent_name, phase)
        print(f"  Routes: {routed}")
        docs   = self.retrieve(query, week=week, role=role, phase=phase, agent_name=agent_name)
        by_col: dict[str, list] = {}
        for d in docs:
            by_col.setdefault(d.metadata.get("retrieved_from","?"), []).append(d)
        print(f"  Total : {len(docs)} chunks")
        for col, col_docs in by_col.items():
            print(f"  [{col}] {len(col_docs)} chunks")
            for d in col_docs[:2]:
                score = d.metadata.get("rerank_score","n/a")
                print(f"    [{score}] {d.page_content[:80].replace(chr(10),' ')}...")
        print()


if __name__ == "__main__":
    r = BabyOSRetriever(k_candidates=20, k_final=5)
    tests = [
        ("How big is the baby at week 20?", 20, "mom", "T2", "tracker_agent"),
        ("Is salmon safe to eat?", None, "mom", "T2", "medical_agent"),
        ("What is the Mutterpass?", None, "dad", "T1", "germany_agent"),
        ("When do babies start walking?", None, "mom", "PP_12M_24M", "tracker_agent"),
    ]
    for q, w, role, phase, agent in tests:
        r.debug_retrieve(q, week=w, role=role, phase=phase, agent_name=agent)
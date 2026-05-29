"""
rag_system.py — v3
------------------
BabyOS RAG System

Fixes in this version:

  FIX 1 — babyos_medical dominance
    - Removed ALWAYS_SEARCH. Medical is no longer forced into every query.
    - Router now gets explicit examples of when NOT to pick medical.
    - Per-agent collection hints: tracker_agent always gets development,
      dad_agent always gets dad, etc. — no LLM needed for those.
    - Danger-sign queries still force medical via supervisor, not retriever.

  FIX 2 — babyos_books never retrieved
    - Books collection re-indexed with fine-grained topic tags at ingest time
      (see ingest_books.py v2 — tags: nutrition, labour, newborn, development,
       mental_health, germany, breastfeeding, complications, general)
    - Router now uses per-topic sub-descriptions for books
    - New retrieve_from_books() does topic-filtered search WITHIN babyos_books
    - Books are searched in parallel to other collections, not as a fallback
    - MMR (Maximal Marginal Relevance) used for books to avoid all chunks
      coming from the same chapter of the same book
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
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

BASE_DIR   = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
DEBUG_MODE = os.getenv("DEBUG_MODE")


# ── Collection descriptions — used by LLM router ─────────────────────────────
# Deliberately specific so the router can distinguish them cleanly.

COLLECTION_DESCRIPTIONS = {
    "babyos_development": (
        "fetal size and weight by week, organ development milestones, baby movements, "
        "what the baby looks like this week, heartbeat, limb development, brain growth, "
        "toddler motor milestones, baby growth charts, first steps, first words"
    ),
    "babyos_medical": (
        "danger signs requiring hospital, specific symptoms (bleeding, pain, swelling, "
        "headache, reduced movement), safe and unsafe foods, medication safety, "
        "blood test interpretation, GDM, preeclampsia, anaemia, UTI, Group B Strep"
    ),
    "babyos_postpartum": (
        "fourth trimester, newborn care, breastfeeding, formula feeding, baby sleep, "
        "postnatal depression, postpartum recovery, weaning, starting solids, "
        "separation anxiety, toddler behaviour, Kita start, return to work"
    ),
    "babyos_germany": (
        "Mutterpass, Vorsorgeuntersuchungen, Hebamme, Elterngeld, Kindergeld, "
        "Krankenkasse, U-Untersuchungen, Kita registration, Standesamt, "
        "German medical vocabulary, hospital registration in Germany, BZgA"
    ),
    "babyos_dad": (
        "dad and partner support, what to do each trimester, hospital bag checklist, "
        "how to support during labour, paternal postnatal depression, "
        "bonding with baby, practical tasks, paternity leave"
    ),
    "babyos_faqs": (
        "common questions: exercise in pregnancy, sex in pregnancy, travel, "
        "weight gain, when to tell people, miscarriage causes, movement timeline, "
        "anatomy scan, Braxton Hicks, mucus plug, waters breaking, birth plan"
    ),
    "babyos_web": (
        "NHS and WHO clinical guidelines, evidence-based recommendations, "
        "antenatal care schedule, newborn screening, postnatal checks, "
        "labour and birth procedures, infant feeding policy"
    ),
    "babyos_books": (
        "in-depth guidance from authoritative books and official guidelines — "
        "search here for: detailed nutrition advice, breastfeeding technique, "
        "birth positions, postpartum mental health, infant development theory, "
        "complications management, German BZgA parenting guides"
    ),
}

# ── Agent → collections that should ALWAYS be searched for that agent ─────────
# This replaces the old ALWAYS_SEARCH = {babyos_medical} which caused dominance.
# Each agent type has its own guaranteed collections — no LLM call needed.

AGENT_COLLECTION_GUARANTEES: dict[str, list[str]] = {
    "medical_agent":   ["babyos_medical", "babyos_web"],
    "tracker_agent":   ["babyos_development"],
    "emotional_agent": ["babyos_postpartum"],
    "parent_agent":    ["babyos_dad"],
    "hebamme_agent":   ["babyos_medical", "babyos_germany"],
    "germany_agent":   ["babyos_germany"],
    "default":         [],    # pure router decision — no forced collections
}

# ── Book topic tags — used for filtered search inside babyos_books ────────────
# These match the topic_tag metadata field written by ingest_books.py v2
BOOK_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "nutrition":       ["food", "eat", "diet", "nutrient", "vitamin", "iron", "folic", "supplement",
                        "calcium", "omega", "dha", "weight gain", "calorie", "safe to eat"],
    "labour":          ["labour", "labor", "birth", "contractions", "pushing", "delivery",
                        "epidural", "pain relief", "waters", "crowning", "episiotomy"],
    "newborn":         ["newborn", "baby", "nappy", "diaper", "cord", "jaundice", "APGAR",
                        "umbilical", "fontanelle", "reflex", "cry", "sleep baby"],
    "breastfeeding":   ["breastfeed", "breast feed", "latch", "milk supply", "engorgement",
                        "nipple", "colostrum", "formula", "weaning", "bottle"],
    "development":     ["milestone", "development", "crawl", "walk", "talk", "language",
                        "motor", "cogniti", "toddler", "growth"],
    "mental_health":   ["depression", "anxiety", "PND", "postnatal", "postpartum", "stress",
                        "mood", "baby blues", "identity", "matrescence"],
    "complications":   ["complication", "preeclampsia", "gestational diabetes", "preterm",
                        "miscarriage", "ectopic", "placenta", "IUGR", "stillbirth"],
    "germany":         ["mutterpass", "hebamme", "elterngeld", "vorsorge", "kita",
                        "krankenkasse", "german", "bzga", "kindergeld"],
    "general":         [],   # fallback — no filter
}


def _detect_book_topic(query: str) -> Optional[str]:
    """
    Fast keyword scan to find the most relevant book topic tag.
    Returns the topic key, or None (use general search).
    """
    if(DEBUG_MODE): print("Detecting Book Topic for query: ",query)
    q = query.lower()
    scores: dict[str, int] = {}
    for topic, keywords in BOOK_TOPIC_KEYWORDS.items():
        if topic == "general":
            continue
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[topic] = score
    if not scores:
        return None
    return max(scores, key=scores.__getitem__)


class BabyOSRetriever:
    """
    Central retriever for the BabyOS RAG system — v3.

    Key changes from v2:
      - No forced collections — per-agent guarantees instead
      - Books retrieved via topic-filtered MMR search in parallel
      - Router given explicit negative examples to prevent medical dominance
      - retrieve() accepts agent_name to apply per-agent guarantees
    """

    def __init__(self, k: int = 4, use_multi_query: bool = True):
        self.k = k
        self.use_multi_query = use_multi_query
        self._embeddings = _make_embeddings()
        self._stores: dict[str, Chroma] = {}
        self._load_all_collections()
        self._llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all_collections(self) -> None:
        loaded = []
        for name in COLLECTION_DESCRIPTIONS:
            persist_path = str(CHROMA_DIR / name)
            if not Path(persist_path).exists():
                continue
            try:
                self._stores[name] = Chroma(
                    collection_name=name,
                    embedding_function=self._embeddings,
                    persist_directory=persist_path,
                )
                loaded.append(name)
            except Exception as e:
                print(f"[RAG] Warning: could not load '{name}': {e}")

        if not loaded:
            raise RuntimeError("No ChromaDB collections found. Run corpus/ingest.py first.")
        print(f"[RAG] Loaded: {', '.join(loaded)}")

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_query(
        self,
        query: str,
        role: str = "mom",
        agent_name: str = "default",
        phase: str = "",
    ) -> list[str]:
        """
        Decide which non-book collections to search.
        Books are handled separately via retrieve_from_books().
        Per-agent guarantees are merged in after LLM routing.
        """
        if(DEBUG_MODE): print("Routing for query: ",query)
        available = {
            name: desc for name, desc in COLLECTION_DESCRIPTIONS.items()
            if name in self._stores and name != "babyos_books"
        }
        if not available:
            return []

        collection_list = "\n".join(
            f"  {name}: {desc}" for name, desc in available.items()
        )

        prompt = f"""You are a routing assistant for BabyOS, a pregnancy and parenting app.
Select the 1-3 most relevant collections to search for this query.

IMPORTANT RULES:
- babyos_development is for questions about HOW THE BABY is growing — size, organs, milestones
- babyos_medical is for SYMPTOMS, DANGER SIGNS, food safety, test results, medications
- Do NOT pick babyos_medical for general development questions like "how big is baby at week 20"
- Do NOT pick babyos_medical unless there is a health concern, symptom, or safety question
- babyos_postpartum is for anything about life AFTER birth — baby care, feeding, PND, toddler
- babyos_germany is ONLY for German system questions — Mutterpass, Hebamme, Elterngeld etc.
- babyos_faqs covers general common questions about pregnancy milestones and experiences

Collections:
{collection_list}

Role: {role}
Phase: {phase or "unknown"}
Query: "{query}"

Return ONLY a comma-separated list of collection names. No explanation.
Example: babyos_development, babyos_faqs
Output:"""

        result = self._llm.invoke(prompt).content.strip()
        routed = [c.strip() for c in result.split(",")]

        valid   = set(self._stores.keys()) - {"babyos_books"}
        selected = {c for c in routed if c in valid}

        # Merge per-agent guarantees
        for col in AGENT_COLLECTION_GUARANTEES.get(agent_name, []):
            if col in self._stores and col != "babyos_books":
                selected.add(col)

        return list(selected)

    # ── Books retrieval (topic-filtered MMR) ──────────────────────────────────

    def retrieve_from_books(self, query: str, k_books: int = 3) -> list[Document]:
        """
        Search babyos_books with topic-filtered MMR.

        MMR (Maximal Marginal Relevance) ensures we don't get 3 chunks
        from the same chapter of the same book — it balances relevance
        with diversity across different sources.

        Topic filter narrows the search to the relevant section of the
        books collection before MMR runs, improving precision.
        """
        if(DEBUG_MODE): print("Retrieving Book Topic for query: ",k_books, query)
        if "babyos_books" not in self._stores:
            return []

        store  = self._stores["babyos_books"]
        topic  = _detect_book_topic(query)

        try:
            if topic and topic != "general":
                # Filtered MMR — search only chunks tagged with this topic
                docs = store.max_marginal_relevance_search(
                    query,
                    k=k_books,
                    fetch_k=k_books * 5,   # fetch more, then diversity-select
                    filter={"topic_tag": topic},
                )
                # Fallback to unfiltered if filter returns nothing
                if not docs:
                    docs = store.max_marginal_relevance_search(
                        query, k=k_books, fetch_k=k_books * 5,
                    )
            else:
                docs = store.max_marginal_relevance_search(
                    query, k=k_books, fetch_k=k_books * 5,
                )

            for doc in docs:
                doc.metadata["retrieved_from"] = "babyos_books"
                doc.metadata["book_topic"]     = topic or "general"

            return docs

        except Exception as e:
            print(f"[RAG] Books retrieval failed: {e}")
            return []

    # ── Core retrieval ────────────────────────────────────────────────────────

    def _get_base_retriever(self, collection_name: str, week: Optional[int] = None):
        store = self._stores[collection_name]

        if collection_name == "babyos_development" and week:
            return store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.k,
                    "filter": {"week": {"$gte": max(1, week - 4), "$lte": min(42, week + 4)}},
                },
            )
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )

    def _multi_query_retrieve(
        self,
        query: str,
        collection_name: str,
        week: Optional[int] = None,
    ) -> list[Document]:
        base = self._get_base_retriever(collection_name, week)
        mq   = MultiQueryRetriever.from_llm(retriever=base, llm=self._llm)
        return mq.invoke(query)

    # ── Main interface ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        week: Optional[int] = None,
        role: str = "mom",
        phase: str = "",
        agent_name: str = "default",
        collections: Optional[list[str]] = None,
        include_books: bool = True,
    ) -> list[Document]:
        """
        Main retrieval interface.

        Args:
            query:         User question
            week:          Pregnancy week (for development filter)
            role:          mom | dad | hebamme
            phase:         Current phase string (PRE, T1, T2 … PP_12M_24M)
            agent_name:    Which agent is calling (applies per-agent guarantees)
            collections:   Explicit override — skips routing if provided
            include_books: Whether to run parallel book search (default True)

        Returns:
            Deduplicated, source-tagged list of Documents
        """
        # 1. Route to non-book collections
        target_cols = collections or self.route_query(query, role, agent_name, phase)

        all_docs:     list[Document] = []
        seen_content: set[str]       = set()

        # 2. Retrieve from non-book collections
        for col in target_cols:
            if col not in self._stores:
                continue
            try:
                if self.use_multi_query:
                    docs = self._multi_query_retrieve(query, col, week)
                else:
                    docs = self._get_base_retriever(col, week).invoke(query)

                for doc in docs:
                    fp = doc.page_content[:200].strip()
                    if fp not in seen_content:
                        seen_content.add(fp)
                        doc.metadata["retrieved_from"] = col
                        all_docs.append(doc)
            except Exception as e:
                print(f"[RAG] Warning — '{col}': {e}")

        # 3. Parallel book search — always runs if babyos_books exists
        if include_books and "babyos_books" in self._stores:
            book_docs = self.retrieve_from_books(query, k_books=3)
            for doc in book_docs:
                fp = doc.page_content[:200].strip()
                if fp not in seen_content:
                    seen_content.add(fp)
                    all_docs.append(doc)

        return all_docs

    # ── Direct week fetch ─────────────────────────────────────────────────────

    def retrieve_week_info(self, week: int) -> Optional[Document]:
        """Directly fetch the exact week document — bypasses routing."""
        if(DEBUG_MODE): print("Retreiving Week info")
        if "babyos_development" not in self._stores:
            return None
        store   = self._stores["babyos_development"]
        results = store.similarity_search(
            f"pregnancy week {week} fetal development",
            k=1,
            filter={"week": week},
        )
        return results[0] if results else None

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_context(self, docs: list[Document]) -> str:
        if not docs:
            return "No relevant information found in knowledge base."
        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            col    = doc.metadata.get("retrieved_from", "")
            topic  = doc.metadata.get("book_topic", "")
            label  = f"{col}/{topic}" if topic else col
            sections.append(f"[Source {i} — {source} ({label})]:\n{doc.page_content}")
        return "\n\n---\n\n".join(sections)

    # ── Debug helper ──────────────────────────────────────────────────────────

    def debug_retrieve(self, query: str, week: int = None, role: str = "mom",
                       phase: str = "", agent_name: str = "default") -> None:
        """Print a detailed breakdown of what was retrieved and from where."""
        print(f"\n{'='*60}")
        print(f"  QUERY : {query}")
        print(f"  WEEK  : {week}   ROLE: {role}   PHASE: {phase}")
        print(f"  AGENT : {agent_name}")
        print(f"{'='*60}")

        routed = self.route_query(query, role, agent_name, phase)
        print(f"\n  Router selected: {routed}")

        topic = _detect_book_topic(query)
        print(f"  Book topic tag:  {topic or 'none (general search)'}")

        docs = self.retrieve(query, week=week, role=role, phase=phase,
                             agent_name=agent_name, include_books=True)

        by_collection: dict[str, list] = {}
        for doc in docs:
            col = doc.metadata.get("retrieved_from", "unknown")
            by_collection.setdefault(col, []).append(doc)

        print(f"\n  Total chunks retrieved: {len(docs)}")
        for col, col_docs in by_collection.items():
            print(f"\n  [{col}] — {len(col_docs)} chunk(s)")
            for doc in col_docs:
                src = doc.metadata.get("source_file", "?")
                preview = doc.page_content[:100].replace("\n", " ").strip()
                print(f"    {src}: {preview}...")
        print()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_rag_chain(collection_name: str, system_prompt: str,
                    llm: Optional[ChatOpenAI] = None):
    """Build a standalone RAG chain for a single collection."""
    embeddings   = _make_embeddings()
    persist_path = str(CHROMA_DIR / collection_name)
    store        = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_path,
    )
    retriever = store.as_retriever(search_kwargs={"k": 4})
    _llm      = llm or ChatOpenAI(
        model="gpt-4o-mini", temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext:\n{context}"),
        ("human",  "{question}"),
    ])
    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | _llm | StrOutputParser()
    )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("BabyOS RAG System v3 — Debug Test\n")
    r = BabyOSRetriever(k=3, use_multi_query=False)

    tests = [
        ("How big is the baby at week 20?",               20,   "mom",  "T2",  "tracker_agent"),
        ("Is it safe to eat salmon during pregnancy?",    None,  "mom",  "T2",  "medical_agent"),
        ("What should I do as a dad this week?",          28,   "dad",  "T3",  "parent_agent"),
        ("What is the Mutterpass?",                       None,  "dad",  "T1",  "germany_agent"),
        ("I'm feeling really anxious about the birth",   36,   "mom",  "T3",  "emotional_agent"),
        ("What are the stages of labour?",               38,   "mom",  "T3",  "medical_agent"),
        ("How should I support breastfeeding?",          None,  "dad",  "PP_0_6W", "parent_agent"),
        ("When do babies start walking?",                None,  "mom",  "PP_12M_24M", "tracker_agent"),
    ]

    for query, week, role, phase, agent in tests:
        r.debug_retrieve(query, week=week, role=role, phase=phase, agent_name=agent)
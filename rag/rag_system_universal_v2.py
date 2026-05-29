"""
rag/rag_system_universal.py
---------------------------
Universal BabyOS RAG System

Changes from original:
  1. Tiered filtering (topic+period → topic → section → none)
     instead of all-or-nothing $and filter
  2. Source diversity enforcement in final results
  3. BM25 runs on a broader unfiltered pool, not just dense results
  4. Soft topic/period score bonus applied alongside SOURCE_WEIGHTS
  5. metadata_filter = None hardcode removed — filtering is re-enabled
     properly via the tiered cascade
  6. format_context() emits structured source metadata the frontend
     can parse for citations + YouTube embeds
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, List
from collections import defaultdict


from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from .taxonomy import _COMPILED_TOPICS, detect_topic

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "babyos_universal"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Source weights: higher = more trusted. Applied as multiplier on rerank score.
SOURCE_WEIGHTS = {
    "WHO":         0.99,
    "NHS":         0.98,
    "CDC":         0.95,
    "ACOG":        0.95,
    "AAP":         0.92,
    "UNICEF":      0.90,
    "EFSA":        0.90,
    "Mayo Clinic": 0.88,
    "pdf":         0.85,
    "json":        0.82,
    "markdown":    0.78,
    "web":         0.72,
    "youtube":     1.00,
}

DEFAULT_SOURCE_WEIGHT = 0.70

# How many candidates to fetch at each filter tier before falling back
TIER_K = 20  # candidates per tier attempt

# Minimum chunks per tier before accepting results (don't accept 1-2 flukes)
MIN_RESULTS_THRESHOLD = 3

# Source diversity: guarantee at least this many source types in final results
# when enough candidates exist. Set to 1 to disable.
MIN_SOURCE_TYPES = 2

# ──────────────────────────────────────────────────────────────────────────────
# TOPIC INFERENCE (query-side, not chunk-side)
# ──────────────────────────────────────────────────────────────────────────────

# TOPIC_KEYWORDS = {
#     "nutrition":       ["food", "diet", "vitamin", "iron", "folic", "supplement", "nutrition", "eat"],
#     "labor":           ["labour", "labor", "contraction", "delivery", "birth", "epidural", "pushing"],
#     "breastfeeding":   ["breastfeed", "latch", "milk", "formula", "colostrum", "engorgement"],
#     "mental_health":   ["depression", "anxiety", "stress", "baby blues", "postpartum depression"],
#     "baby_development":["crawl", "walk", "speech", "milestone", "motor", "cognitive"],
#     "fetal_movement":  ["kick", "movement", "baby moving", "fetal movement"],
#     "medical_board":   ["preeclampsia", "gestational diabetes", "miscarriage", "bleeding", "placenta"],
#     "germany":         ["mutterpass", "hebamme", "elterngeld", "krankenkasse", "vorsorge"],
# }


def infer_topic(query: str) -> Optional[str]:
    """Infer topic from query text using the same regex patterns as detect_topic()."""
    scores = {
        t: sum(1 for p in pats if p.search(query))
        for t, pats in _COMPILED_TOPICS.items()
    }
    scores = {t: s for t, s in scores.items() if s > 0}
    return max(scores, key=scores.get) if scores else None


# ──────────────────────────────────────────────────────────────────────────────
# HYDE
# ──────────────────────────────────────────────────────────────────────────────

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical retrieval assistant. "
        "Write a concise factual paragraph (3-5 sentences) as if it were from a "
        "medical textbook or trusted health guide. Use correct medical terminology. "
        "Avoid speculation."
    ),
    ("human", "{question}")
])


def generate_hyde(query: str, llm: ChatOpenAI) -> str:
    try:
        return (_HYDE_PROMPT | llm | StrOutputParser()).invoke({"question": query})
    except Exception:
        return query


# ──────────────────────────────────────────────────────────────────────────────
# RERANKING
# ──────────────────────────────────────────────────────────────────────────────

def rerank_documents(query: str, docs: List[Document], top_n: int = 6) -> List[Document]:
    if not docs:
        return []
    if not COHERE_API_KEY:
        return docs[:top_n]
    try:
        import cohere
        client  = cohere.Client(COHERE_API_KEY)
        results = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[d.page_content for d in docs],
            top_n=min(top_n, len(docs)),
        )
        reranked = []
        for result in results.results:
            doc = docs[result.index]
            doc.metadata["rerank_score"] = round(result.relevance_score, 4)
            reranked.append(doc)
        return reranked
    except Exception as e:
        print(f"[Reranker Error] {e}")
        return docs[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN RETRIEVER
# ──────────────────────────────────────────────────────────────────────────────

class UniversalBabyOSRetriever:

    def __init__(
        self,
        k_candidates: int = 25,
        k_final:      int = 6,
        use_hyde:     bool = True,
        use_reranker: bool = True,
    ):
        self.k_candidates = k_candidates
        self.k_final      = k_final
        self.use_hyde     = use_hyde
        self.use_reranker = use_reranker

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )
        self.store = self._load_vectorstore()

    # ──────────────────────────────────────────────────────────────────────────

    def _load_vectorstore(self) -> Chroma:
        persist_path = str(CHROMA_DIR / COLLECTION_NAME)
        if not Path(persist_path).exists():
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run universal_ingest.py first."
            )
        print(f"[RAG] Loading: {COLLECTION_NAME}")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=persist_path,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TIERED FILTER BUILDER
    # Produces a cascade of filters from most to least specific.
    # retrieve() walks this list and stops at the first tier that returns
    # enough results.
    # ──────────────────────────────────────────────────────────────────────────

    def _build_filter_tiers(
        self,
        topic:       Optional[str] = None,
        period:      Optional[str] = None,
        section:     Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Optional[dict]]:
        """
        Returns a list of filters ordered from most to least restrictive.
        The last entry is always None (no filter = full collection search).

        Tier 0:  topic + period  (most specific)
        Tier 1:  topic only
        Tier 2:  section only    (pregnancy / postpartum)
        Tier 3:  None            (full collection)

        Period=all is excluded from Tier 0 because it means "unperiodized"
        and adding it as a filter would REDUCE results, not increase them.
        """
        tiers: List[Optional[dict]] = []

        # Tier 0: topic + period (skip if period is "all" or missing)
        if topic and period and period != "all":
            tiers.append({"$and": [{"topic": topic}, {"period": period}]})

        # Tier 1: topic only
        if topic:
            tiers.append({"topic": topic})

        # Tier 2: section only
        if section and section in ("pregnancy", "postpartum"):
            tiers.append({"section": section})

        # Tier 3: no filter (always last)
        tiers.append(None)

        return tiers

    # ──────────────────────────────────────────────────────────────────────────

    def _dense_retrieve(
        self,
        query:           str,
        metadata_filter: Optional[dict] = None,
        k:               int = TIER_K,
    ) -> List[Document]:
        kwargs = {"k": k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        return self.store.similarity_search(query, **kwargs)

    # ──────────────────────────────────────────────────────────────────────────

    def _sparse_retrieve(self, query: str, docs: List[Document]) -> List[Document]:
        """BM25 over the provided doc pool. Returns empty list on failure."""
        if not docs:
            return []
        try:
            bm25 = BM25Retriever.from_documents(docs, k=min(10, len(docs)))
            return bm25.invoke(query)
        except Exception as e:
            print(f"[BM25 Error] {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        seen, unique = set(), []
        for doc in docs:
            digest = hashlib.sha256(
                doc.page_content.strip().lower().encode("utf-8")
            ).hexdigest()
            if digest not in seen:
                seen.add(digest)
                unique.append(doc)
        return unique

    # ──────────────────────────────────────────────────────────────────────────

    def _apply_scores(
        self,
        docs:       List[Document],
        topic_hint: Optional[str] = None,
        period:     Optional[str] = None,
    ) -> List[Document]:
        """
        Compute final_score = rerank_score × source_weight × soft_relevance_bonus.

        soft_relevance_bonus:
          +0.05  if chunk topic matches the routed topic
          +0.03  if chunk period matches the query period (and neither is "all")
        These are small nudges — they don't override reranking, just break ties
        in favour of topically aligned chunks.
        """
        for doc in docs:
            source_name  = doc.metadata.get("source_name", "")
            source_type  = doc.metadata.get("source_type", "")
            rerank_score = doc.metadata.get("rerank_score", 0.5)

            weight = SOURCE_WEIGHTS.get(
                source_name,
                SOURCE_WEIGHTS.get(source_type, DEFAULT_SOURCE_WEIGHT),
            )

            # Soft topic/period bonus
            bonus = 0.0
            if topic_hint and doc.metadata.get("topic") == topic_hint:
                bonus += 0.05
            if (
                period
                and period != "all"
                and doc.metadata.get("period") == period
            ):
                bonus += 0.03

            final_score = (rerank_score + bonus) * weight

            doc.metadata["source_weight"] = weight
            doc.metadata["topic_bonus"]   = round(bonus, 3)
            doc.metadata["final_score"]   = round(final_score, 4)

        docs.sort(key=lambda d: d.metadata.get("final_score", 0), reverse=True)
        return docs

    # ──────────────────────────────────────────────────────────────────────────

    def _enforce_source_diversity(
        self,
        docs:    List[Document],
        k_final: int,
    ) -> List[Document]:
        """
        Ensure the final result set contains chunks from at least
        MIN_SOURCE_TYPES different source_type values when possible.

        Strategy:
          1. Take the top-scoring chunk from each source type (guaranteed slots)
          2. Fill remaining slots from the ranked list
        This means YouTube chunks won't be completely drowned by PDFs even
        when they score slightly lower.
        """
        if not docs:
            return docs

        # Collect one guaranteed representative per source type
        seen_types: dict[str, Document] = {}
        remainder:  List[Document]      = []

        for doc in docs:
            stype = doc.metadata.get("source_type", "unknown")
            if stype not in seen_types:
                seen_types[stype] = doc
            else:
                remainder.append(doc)

        # We have enough source types — build diversity-aware result
        guaranteed = list(seen_types.values())  # one per type

        if len(seen_types) >= MIN_SOURCE_TYPES:
            # Start with one from each type, then fill from remainder by score
            combined = guaranteed + remainder
            # Re-sort so guaranteed slots don't break score ordering too badly
            combined.sort(key=lambda d: d.metadata.get("final_score", 0), reverse=True)
            # But ensure each guaranteed type appears at least once
            final    = []
            included = set()
            for doc in guaranteed:
                final.append(doc)
                included.add(id(doc))
            for doc in combined:
                if len(final) >= k_final:
                    break
                if id(doc) not in included:
                    final.append(doc)
                    included.add(id(doc))
            return final[:k_final]

        # Not enough types to diversify — just return top k
        return docs[:k_final]

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN RETRIEVE
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:       str,
        topic:       Optional[str] = None,
        period:      Optional[str] = None,
        source_type: Optional[str] = None,
        week:        Optional[int] = None,
        section:     Optional[str] = None,
    ) -> List[Document]:
        """
        Tiered retrieval pipeline:

        1. Build embed query (HyDE or raw)
        2. Walk filter tiers until MIN_RESULTS_THRESHOLD docs found
        3. Expand with unfiltered BM25 on the same embed query
        4. Deduplicate
        5. Rerank with Cohere
        6. Score with source weights + soft topic/period bonus
        7. Enforce source type diversity
        8. Return top k_final
        """

        # ── 1. Topic inference ─────────────────────────────────────────────────
        effective_topic = topic or infer_topic(query)

        # ── 2. Embed query ─────────────────────────────────────────────────────
        embed_query = (
            generate_hyde(query, self.llm)
            if self.use_hyde
            else query
        )

        # ── 3. Tiered dense retrieval ──────────────────────────────────────────
        tiers = self._build_filter_tiers(
            topic=effective_topic,
            period=period,
            section=section,
        )

        dense_docs  = []
        used_tier   = len(tiers) - 1  # default: no filter

        for tier_idx, metadata_filter in enumerate(tiers):
            candidates = self._dense_retrieve(
                embed_query,
                metadata_filter=metadata_filter,
                k=TIER_K,
            )
            tier_name = (
                str(metadata_filter)[:60]
                if metadata_filter
                else "no filter"
            )
            print(f"[RAG] Tier {tier_idx} ({tier_name}): {len(candidates)} docs")

            if len(candidates) >= MIN_RESULTS_THRESHOLD:
                dense_docs = candidates
                used_tier  = tier_idx
                break

        # ── 4. BM25 on unfiltered pool ─────────────────────────────────────────
        # Always run BM25 unfiltered so it can rescue relevant chunks that
        # dense retrieval missed (especially YouTube and small markdown files)
        unfiltered_pool = self._dense_retrieve(embed_query, metadata_filter=None, k=40)
        sparse_docs     = self._sparse_retrieve(query, unfiltered_pool)

        # ── 5. Combine + deduplicate ───────────────────────────────────────────
        combined = self._deduplicate(dense_docs + sparse_docs)

        # ── 6. Rerank ──────────────────────────────────────────────────────────
        if self.use_reranker and combined:
            combined = rerank_documents(query, combined, top_n=self.k_final * 2)

        # ── 7. Score (source weight + soft bonuses) ────────────────────────────
        combined = self._apply_scores(
            combined,
            topic_hint=effective_topic,
            period=period,
        )

        # ── 8. Enforce source diversity ────────────────────────────────────────
        final = self._enforce_source_diversity(combined, self.k_final)

        print(
            f"[RAG] Final: {len(final)} docs "
            f"(tier={used_tier}, topic={effective_topic}, period={period})"
        )

        return final

    # ──────────────────────────────────────────────────────────────────────────
    # FORMAT CONTEXT
    # ──────────────────────────────────────────────────────────────────────────

    def format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved docs into a string for the LLM prompt.

        Each block includes:
          - Source metadata (name, type, topic, period, score)
          - YouTube URL on its own line when available (frontend can embed it)
          - Page content

        The frontend can parse [Source N] blocks to build citation UI.
        """
        if not docs:
            return "No relevant information found."

        sections = []

        for i, doc in enumerate(docs, 1):
            meta        = doc.metadata
            source      = meta.get("source_name", "Unknown")
            source_type = meta.get("source_type", "")
            topic       = meta.get("topic", "")
            period      = meta.get("period", "")
            score       = meta.get("final_score", "")
            youtube_url = meta.get("youtube_url", "")
            timestamp   = meta.get("timestamp", "")
            video_label = meta.get("video_label", "")  # human-readable video title

            header = (
                f"[Source {i}] "
                f"{source} | "
                f"{source_type} | "
                f"topic={topic} | "
                f"period={period} | "
                f"score={score}"
            )

            if youtube_url:
                yt_label = video_label or source
                header += f"\n📹 {yt_label} @ {timestamp} → {youtube_url}"

            sections.append(f"{header}\n\n{doc.page_content}")

        return "\n\n---\n\n".join(sections)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Topic-browsing retrieval for the articles router.
    # Uses the pre-written semantic query from TOPIC_COLLECTION_MAP_QUERIES
    # instead of a user query, so results are topically representative
    # rather than query-specific.

    # Called by: backend/routers/articles_router.py
    
    def retrieve_by_topic(
    self,
    topic:  str,
    period: Optional[str] = None,
    k:      int = 6,
) -> List[Document]:
        from .taxonomy import TOPIC_COLLECTION_MAP_QUERIES

    # Use the pre-written query for this topic as the embed query
    # Falls back to the topic key itself if not found
        query = TOPIC_COLLECTION_MAP_QUERIES.get(topic, topic)

    # Temporarily override k_final for this call
        original_k = self.k_final
        self.k_final = k

        try:
            docs = self.retrieve(
                query=query,
                topic=topic,
                period=period,
            )
        finally:
            self.k_final = original_k  # always restore

        return docs
    
    # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # DEBUG
    # ──────────────────────────────────────────────────────────────────────────

    def debug_query(self, query: str, **kwargs) -> List[Document]:
        print("=" * 80)
        print(f"QUERY: {query}")

        docs = self.retrieve(query, **kwargs)

        print(f"\nRetrieved: {len(docs)} docs\n")

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            print(f"[{i}] Topic: {meta.get('topic')}")
            print(f"     Period: {meta.get('period')}")
            print(f"     Source: {meta.get('source_name')} ({meta.get('source_type')})")
            print(f"     Score : {meta.get('final_score')}  "
                  f"(rerank={meta.get('rerank_score')}, "
                  f"weight={meta.get('source_weight')}, "
                  f"bonus={meta.get('topic_bonus')})")
            if meta.get("youtube_url"):
                print(f"     YT URL: {meta.get('youtube_url')}")
            print(f"     Text  : {doc.page_content[:200]}")
            print("-" * 80)

        return docs


# ──────────────────────────────────────────────────────────────────────────────
# TESTING
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = UniversalBabyOSRetriever()

    tests = [
        ("Is it normal to feel short of breath during pregnancy?",    {"section": "pregnancy"}),
        ("How big is the baby at week 20?",                           {"week": 20}),
        ("Is salmon safe during pregnancy?",                          {}),
        ("What are signs of postpartum depression?",                  {"section": "postpartum"}),
        ("What is the Mutterpass?",                                   {}),
        ("How often should I feel baby movements?",                   {}),
        ("What pain relief options are available during labour?",     {"topic": "labor"}),
    ]

    for query, params in tests:
        retriever.debug_query(query, **params)
        print()
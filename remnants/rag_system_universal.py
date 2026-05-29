
"""
rag/rag_system_universal.py
---------------------------
Universal BabyOS RAG System (Refactored for unified architecture)

Architecture:
- ONE collection: babyos_universal
- Metadata-driven retrieval
- Hybrid dense + BM25 retrieval
- Cohere reranking
- Topic + period filtering
- Stable retrieval pipeline

Key improvements:
✓ Works with universal_ingest.py
✓ Unified metadata schema
✓ Metadata filtering instead of collection routing
✓ Stable retrieval strategy
✓ Better deduplication
✓ Source weighting
✓ Universal source compatibility
✓ YouTube/book/web/pdf support in same collection
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
from taxonomy import ALL_TOPICS, detect_topic

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "babyos_universal"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

SOURCE_WEIGHTS = {
    "WHO": 1.0,
    "NHS": 0.98,
    "CDC": 0.95,
    "ACOG": 0.95,
    "AAP": 0.92,
    "UNICEF": 0.90,
    "EFSA": 0.90,
    "Mayo Clinic": 0.88,
    "book": 0.85,
    "youtube": 0.65,
    "web": 0.70,
    "markdown": 0.75,
    "json": 0.80,
}

DEFAULT_SOURCE_WEIGHT = 0.70

# ──────────────────────────────────────────────────────────────────────────────
# TOPIC DETECTION
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_KEYWORDS = {
    "nutrition": [
        "food", "diet", "vitamin", "iron", "folic",
        "supplement", "nutrition", "eat"
    ],
    "labor": [
        "labour", "labor", "contraction", "delivery",
        "birth", "epidural", "pushing"
    ],
    "breastfeeding": [
        "breastfeed", "latch", "milk", "formula",
        "colostrum", "engorgement"
    ],
    "mental_health": [
        "depression", "anxiety", "stress",
        "baby blues", "postpartum depression"
    ],
    "baby_development": [
        "crawl", "walk", "speech", "milestone",
        "motor", "cognitive"
    ],
    "fetal_movement": [
        "kick", "movement", "baby moving",
        "fetal movement"
    ],
    "complications": [
        "preeclampsia", "gestational diabetes",
        "miscarriage", "bleeding", "placenta"
    ],
    "germany": [
        "mutterpass", "hebamme", "elterngeld",
        "krankenkasse", "vorsorge"
    ]
}


def infer_topic(query: str) -> Optional[str]:
    q = query.lower()

    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[topic] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


# ──────────────────────────────────────────────────────────────────────────────
# HYDE
# ──────────────────────────────────────────────────────────────────────────────

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical retrieval assistant. "
        "Write a concise factual paragraph using medical terminology. "
        "Avoid speculation or unsupported claims."
    ),
    ("human", "{question}")
])


def generate_hyde(query: str, llm: ChatOpenAI) -> str:
    try:
        return (
            _HYDE_PROMPT
            | llm
            | StrOutputParser()
        ).invoke({"question": query})

    except Exception:
        return query


# ──────────────────────────────────────────────────────────────────────────────
# RERANKING
# ──────────────────────────────────────────────────────────────────────────────

def rerank_documents(
    query: str,
    docs: List[Document],
    top_n: int = 6
) -> List[Document]:

    if not docs:
        return []

    if not COHERE_API_KEY:
        return docs[:top_n]

    try:
        import cohere

        client = cohere.Client(COHERE_API_KEY)

        results = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[d.page_content for d in docs],
            top_n=min(top_n, len(docs))
        )

        reranked = []

        for result in results.results:
            doc = docs[result.index]
            doc.metadata["rerank_score"] = round(
                result.relevance_score,
                4
            )

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
        k_final: int = 6,
        use_hyde: bool = True,
        use_reranker: bool = True,
    ):

        self.k_candidates = k_candidates
        self.k_final = k_final
        self.use_hyde = use_hyde
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

    # ──────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────

    def _build_filter(
        self,
        topic: Optional[str] = None,
        period: Optional[str] = None,
        source_type: Optional[str] = None,
        week: Optional[int] = None,
        section=None,
    ) -> Optional[dict]:

        filters = []
        
        if section:
            filters.append({"section": section})

        if topic:
            filters.append({"topic": topic})

        if period:
            filters.append({"period": period})

        if source_type:
            filters.append({"source_type": source_type})

        if week is not None:
            filters.append({
                 "$and": [
                            {"week": {"$gte": max(1, week - 2)}},
                            {"week": {"$lte": min(42, week + 2)}}
                            ]
            })

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    # ──────────────────────────────────────────────────────────────────

    def _dense_retrieve(
        self,
        query: str,
        metadata_filter: Optional[dict] = None
    ) -> List[Document]:

        kwargs = {"k": self.k_candidates}

        if metadata_filter:
            kwargs["filter"] = metadata_filter

        return self.store.similarity_search(
            query,
            **kwargs
        )

    # ──────────────────────────────────────────────────────────────────

    def _sparse_retrieve(
        self,
        query: str,
        docs: List[Document]
    ) -> List[Document]:

        if not docs:
            return []

        try:
            bm25 = BM25Retriever.from_documents(
                docs,
                k=min(10, len(docs))
            )

            return bm25.invoke(query)

        except Exception as e:
            print(f"[BM25 Error] {e}")
            return []

    # ──────────────────────────────────────────────────────────────────

    def _deduplicate(
        self,
        docs: List[Document]
    ) -> List[Document]:

        unique = []
        seen = set()

        for doc in docs:

            normalized = (
                doc.page_content
                .strip()
                .lower()
            )

            digest = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if digest not in seen:
                seen.add(digest)
                unique.append(doc)

        return unique

    # ──────────────────────────────────────────────────────────────────

    def _apply_source_weights(
        self,
        docs: List[Document]
    ) -> List[Document]:

        weighted = []

        for doc in docs:

            source_name = doc.metadata.get(
                "source_name",
                ""
            )

            source_type = doc.metadata.get(
                "source_type",
                ""
            )

            weight = SOURCE_WEIGHTS.get(
                source_name,
                SOURCE_WEIGHTS.get(
                    source_type,
                    DEFAULT_SOURCE_WEIGHT
                )
            )

            rerank_score = doc.metadata.get(
                "rerank_score",
                0.5
            )

            final_score = rerank_score * weight

            doc.metadata["source_weight"] = weight
            doc.metadata["final_score"] = round(
                final_score,
                4
            )

            weighted.append(doc)

        weighted.sort(
            key=lambda d: d.metadata.get(
                "final_score",
                0
            ),
            reverse=True
        )

        return weighted

    # ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        topic: Optional[str] = None,
        period: Optional[str] = None,
        source_type: Optional[str] = None,
        week: Optional[int] = None,
        section: Optional[int] = None,
        
    ) -> List[Document]:

        inferred_topic = topic or infer_topic(query)

        # metadata_filter  = self._build_filter(
        #     topic=inferred_topic,
        #     period=period,
        #     source_type=source_type,
        #     week=week,
        # )
        
        metadata_filter  = None
        

        embed_query = (
            generate_hyde(query, self.llm)
            if self.use_hyde
            else query
        )

        # Dense retrieval
        dense_docs = self._dense_retrieve(
            embed_query,
            metadata_filter
        )

        # Sparse retrieval
        sparse_docs = self._sparse_retrieve(
            query,
            dense_docs
        )

        combined = dense_docs + sparse_docs

        # Deduplicate
        combined = self._deduplicate(combined)

        # Rerank
        if self.use_reranker:
            combined = rerank_documents(
                query,
                combined,
                top_n=self.k_final * 2
            )

        # Apply source weighting
        combined = self._apply_source_weights(combined)

        return combined[:self.k_final]

    # ──────────────────────────────────────────────────────────────────

    def format_context(
        self,
        docs: List[Document]
    ) -> str:

        if not docs:
            return "No relevant information found."

        sections = []

        for i, doc in enumerate(docs, 1):

            meta = doc.metadata

            source = meta.get("source_name", "Unknown")
            source_type = meta.get("source_type", "")
            topic = meta.get("topic", "")
            period = meta.get("period", "")
            score = meta.get("final_score", "")
            youtube_url = meta.get("youtube_url", "")

            header = (
                f"[Source {i}] "
                f"{source} | "
                f"{source_type} | "
                f"{topic} | "
                f"{period} | "
                f"score={score}"
            )

            if youtube_url:
                header += f"\n📹 {youtube_url}"

            sections.append(
                f"{header}\n\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(sections)

    # ──────────────────────────────────────────────────────────────────

    def debug_query(
        self,
        query: str,
        **kwargs
    ):

        print("=" * 80)
        print(f"QUERY: {query}")

        docs = self.retrieve(
            query,
            **kwargs
        )

        print(f"Retrieved: {len(docs)} docs\n")
        print(docs)

        for i, doc in enumerate(docs, 1):

            meta = doc.metadata

            print(f"[{i}]")
            print(f"Topic: {meta.get('topic')}")
            print(f"Period: {meta.get('period')}")
            print(f"Source: {meta.get('source_name')}")
            print(f"Score : {meta.get('final_score')}")
            print(doc.page_content[:250])
            print("-" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# TESTING
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    retriever = UniversalBabyOSRetriever()

    tests = [
        ("How big is the baby at week 20?", {"week": 20}),
        ("Is salmon safe during pregnancy?", {}),
        ("What are signs of postpartum depression?", {}),
        ("What is the Mutterpass?", {}),
        ("How often should I feel baby movements?", {}),
    ]

    for query, params in tests:

        retriever.debug_query(
            query,
            **params
        )

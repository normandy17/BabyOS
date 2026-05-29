"""
ingest.py
---------
BabyOS RAG Ingestion Pipeline

Loads all corpus documents, chunks them with metadata-aware splitting,
embeds using OpenAI, and stores in a persistent ChromaDB collection.

Collections created:
  - babyos_medical     : medical reference, symptoms, danger signs
  - babyos_development : fetal development by week (structured JSON)
  - babyos_germany     : Germany-specific pregnancy system info
  - babyos_dad         : dad/partner guides and checklists
  - babyos_faqs        : pregnancy FAQs
  - babyos_web         : scraped NHS/WHO content

Run:  python ingest.py
      python ingest.py --reset   (wipe and rebuild from scratch)
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# ── Embedding model ─────────────────────────────────────────────────────────-
EMBEDDING_MODEL = "text-embedding-3-small"   # cheap, fast, good quality

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 800   # tokens ≈ characters / 4
CHUNK_OVERLAP = 150

# ── Collection name → source file(s) mapping ──────────────────────────────────
COLLECTIONS: dict[str, dict] = {
    "babyos_medical": {
        "files": ["medical_reference.md"],
        "description": "Medical reference: danger signs, tests, safe foods, supplements, GDM, preeclampsia",
    },
    "babyos_germany": {
        "files": ["germany_pregnancy_guide.md"],
        "description": "Germany-specific: Mutterpass, Vorsorgeuntersuchungen, Hebamme, Elterngeld, vocabulary",
    },
    "babyos_dad": {
        "files": ["dad_partner_guide.md"],
        "description": "Dad and partner guide: trimester support, hospital bag, postpartum, admin tasks",
    },
    "babyos_faqs": {
        "files": ["pregnancy_faqs.md"],
        "description": "Pregnancy FAQs: general, trimester-specific, postpartum questions and answers",
    },
}

TOPIC_PATTERNS: dict[str, list[str]] = {
    "nutrition": [
        r"\bnutrition\b", r"\bdiet\b", r"\bfolic acid\b", r"\biron\b",
        r"\bvitamin\b", r"\bsupplement\b", r"\bcalcium\b", r"\bomega.3\b",
        r"\bdha\b", r"\bfood\b", r"\beat\b", r"\bcalorie\b", r"\bweight gain\b",
    ],
    "labour": [
        r"\blabour\b", r"\blabor\b", r"\bbirth\b", r"\bdelivery\b",
        r"\bcontractions?\b", r"\bpushing\b", r"\bcrowning\b",
        r"\bepidural\b", r"\bpain relief\b", r"\bcaesarean\b", r"\b\bsection\b",
        r"\bwaters? broke?\b", r"\bepisiotomy\b", r"\bforceps\b",
    ],
    "newborn": [
        r"\bnewborn\b", r"\bnappy\b", r"\bdiaper\b", r"\bumbilic\b",
        r"\bjaundice\b", r"\bapgar\b", r"\bfontanelle\b", r"\breflex\b",
        r"\bneonatal\b", r"\binfant\b", r"\bcolostrum\b",
    ],
    "breastfeeding": [
        r"\bbreastfeed\b", r"\bbreast.?feed\b", r"\blatch\b",
        r"\bmilk supply\b", r"\bengorge\b", r"\bnipple\b",
        r"\bformula\b", r"\bwean\b", r"\bbottle.?feed\b",
    ],
    "development": [
        r"\bmilestone\b", r"\bdevelopment\b", r"\bcrawl\b", r"\bwalk\b",
        r"\bspeech\b", r"\blanguage\b", r"\bmotor\b", r"\bcogniti\b",
        r"\btoddler\b", r"\bfirst words?\b", r"\bgrowth chart\b",
    ],
    "mental_health": [
        r"\bdepression\b", r"\banxiety\b", r"\bpnd\b", r"\bpostnatal\b",
        r"\bpostpartum\b", r"\bbaby blues\b", r"\bstress\b",
        r"\bidentity\b", r"\bmatrescence\b", r"\bpatrescence\b",
        r"\bmental health\b", r"\btherapy\b", r"\bcbt\b",
    ],
    "complications": [
        r"\bpreeclampsia\b", r"\bgestational diabetes\b", r"\bpreterm\b",
        r"\bmiscarriage\b", r"\bectopic\b", r"\bplacenta praevia\b",
        r"\biugr\b", r"\bstillbirth\b", r"\bcomplication\b",
        r"\bhypertension\b", r"\bhyperemesis\b",
    ],
    "germany": [
        r"\bmutterpass\b", r"\bhebamme\b", r"\belterngeld\b",
        r"\bvorsorge\b", r"\bkita\b", r"\bkrankenkasse\b",
        r"\bstandesamt\b", r"\bbzga\b", r"\bkindergeld\b",
        r"\bgeburtshaus\b", r"\bkrei.?saal\b",
    ],
    "dad_partner": [
        r"\bdad\b", r"\bpartner\b", r"\bfather\b", r"\bspouse\b",
        r"\bparenting\b", r"\bchecklist\b", r"\bpostpartum\b", r"\broles?\b",
        r"\bfeeding support\b", r"\bmental health\b",
    ],
}

_COMPILED = {
    topic: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for topic, patterns in TOPIC_PATTERNS.items()
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def tag_chunk(text: str) -> str:
    scores: dict[str, int] = {}
    for topic, patterns in _COMPILED.items():
        score = sum(1 for pattern in patterns if pattern.search(text))
        if score:
            scores[topic] = score
    if not scores:
        return "general"
    return max(scores, key=scores.__getitem__)


def make_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)


def make_splitter(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )


def load_markdown(filepath: Path, collection: str) -> list[Document]:
    """Load a markdown file and tag each chunk with metadata."""
    loader = TextLoader(str(filepath), encoding="utf-8")
    docs = loader.load()
    splitter = make_splitter()
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source_file": filepath.name,
            "collection":  collection,
            "chunk_index": i,
            "doc_type":    "markdown",
            "topic_tag":   tag_chunk(chunk.page_content),
        })
    return chunks


def load_fetal_development() -> list[Document]:
    """
    Load fetal_development.json and create one rich Document per week.
    Each week becomes a single chunk — they are already the right size
    and keeping them atomic means retrieval returns complete week info.
    """
    json_path = RAW_DIR / "fetal_development.json"
    data: list[dict] = json.loads(json_path.read_text(encoding="utf-8"))

    docs = []
    for week_data in data:
        week = week_data["week"]
        trimester = week_data["trimester"]

        # Build a rich human-readable text block for embedding
        text = f"""PREGNANCY WEEK {week} — TRIMESTER {trimester}

Baby size: {week_data['size_comparison']} ({week_data['size_cm']} cm, {week_data['weight_g']} g)

FETAL DEVELOPMENT:
{week_data['development']}

BABY MILESTONES THIS WEEK:
{", ".join(week_data['baby_milestones'])}

WHAT MOM IS EXPERIENCING:
{week_data['mom_changes']}

COMMON SYMPTOMS: {", ".join(week_data['mom_symptoms'])}

FOR DAD / PARTNER:
{week_data['dad_partner_tips']}

UPCOMING APPOINTMENTS: {", ".join(week_data['appointments'])}

NUTRITION FOCUS: {", ".join(week_data['nutrition_focus'])}

DANGER SIGNS — SEEK MEDICAL HELP IMMEDIATELY IF:
{", ".join(week_data['danger_signs'])}
"""

        doc = Document(
            page_content=text,
            metadata={
                "week":         week,
                "trimester":    trimester,
                "size_cm":      week_data["size_cm"],
                "weight_g":     week_data["weight_g"],
                "size_compare": week_data["size_comparison"],
                "source_file":  "fetal_development.json",
                "collection":   "babyos_development",
                "doc_type":     "structured_json",
                "topic_tag":    "development",
                "chunk_index":  week,
            }
        )
        docs.append(doc)

    return docs


def load_web_scraped() -> list[Document]:
    """Load all scraped .txt files from data/raw/web_scraped/."""
    web_dir = RAW_DIR / "web_scraped"
    if not web_dir.exists() or not list(web_dir.glob("*.txt")):
        print("  ⚠️  No web-scraped files found. Run corpus_fetcher.py first.")
        return []

    loader = DirectoryLoader(
        str(web_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    splitter = make_splitter()
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        fname = Path(chunk.metadata.get("source", "unknown")).name
        source_type = "nhs" if "nhs" in fname else "who" if "who" in fname else "web"
        chunk.metadata.update({
            "source_file": fname,
            "collection":  "babyos_web",
            "chunk_index": i,
            "doc_type":    "web_scraped",
            "source_type": source_type,
            "topic_tag":   tag_chunk(chunk.page_content),
        })
    return chunks


def build_collection(
    name: str,
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    reset: bool = False,
) -> Chroma:
    """Create or update a Chroma collection from a list of Documents."""
    persist_path = str(CHROMA_DIR / name)

    if reset:
        import shutil
        if Path(persist_path).exists():
            shutil.rmtree(persist_path)
            print(f"  🗑️  Wiped existing collection: {name}")

    print(f"  📥 Indexing {len(docs)} chunks into '{name}'...")
    
    # Batch in groups of 100 to avoid rate limits
    batch_size = 100
    vectorstore = None

    for i in tqdm(range(0, len(docs), batch_size), desc=f"  {name}"):
        batch = docs[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=name,
                persist_directory=persist_path,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"  ✅ '{name}' — {len(docs)} chunks indexed.\n")
    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

def main(reset: bool = False) -> None:
    print("=" * 60)
    print("  BabyOS RAG Ingestion Pipeline")
    print("=" * 60)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = make_embeddings()

    # ── 1. Markdown collections ──────────────────────────────────────────────
    for collection_name, config in COLLECTIONS.items():
        print(f"\n[{collection_name}]")
        all_chunks: list[Document] = []
        for fname in config["files"]:
            fpath = RAW_DIR / fname
            if not fpath.exists():
                print(f"  ⚠️  File not found: {fpath} — skipping.")
                continue
            chunks = load_markdown(fpath, collection_name)
            print(f"  📄 {fname} → {len(chunks)} chunks")
            all_chunks.extend(chunks)

        if all_chunks:
            build_collection(collection_name, all_chunks, embeddings, reset)

    # ── 2. Structured JSON — fetal development ───────────────────────────────
    print("\n[babyos_development]")
    dev_docs = load_fetal_development()
    print(f"  📄 fetal_development.json → {len(dev_docs)} week documents")
    build_collection("babyos_development", dev_docs, embeddings, reset)

    # ── 3. Web-scraped content ───────────────────────────────────────────────
    print("\n[babyos_web]")
    web_docs = load_web_scraped()
    if web_docs:
        build_collection("babyos_web", web_docs, embeddings, reset)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    
    all_collections = list(COLLECTIONS.keys()) + ["babyos_development"]
    if web_docs:
        all_collections.append("babyos_web")

    total_chunks = 0
    for name in all_collections:
        persist_path = str(CHROMA_DIR / name)
        try:
            vs = Chroma(
                collection_name=name,
                embedding_function=embeddings,
                persist_directory=persist_path,
            )
            count = vs._collection.count()
            total_chunks += count
            print(f"  {name:<30} {count:>5} chunks")
        except Exception:
            print(f"  {name:<30} (not found)")

    print(f"\n  {'TOTAL':<30} {total_chunks:>5} chunks")
    print(f"\n  ChromaDB stored at: {CHROMA_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BabyOS RAG Ingestion")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all existing collections and rebuild from scratch",
    )
    args = parser.parse_args()
    main(reset=args.reset)

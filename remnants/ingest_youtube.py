"""
corpus/ingest_youtube.py
------------------------
BabyOS YouTube Transcript Ingestion

Fetches transcripts from curated, trusted pregnancy/parenting channels.
Stores in babyos_youtube ChromaDB collection with video_id + timestamp metadata
so retrieved chunks link back to the exact moment in the video.

No API key needed — uses youtube-transcript-api (public transcripts only).

Run:
  python corpus/ingest_youtube.py
  python corpus/ingest_youtube.py --reset
  python corpus/ingest_youtube.py --channel nhs   (single channel)
"""

import os
import re
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm
import requests
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR   = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION = "babyos_youtube"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ── Curated video whitelist ────────────────────────────────────────────────────
# Only trusted, evidence-based sources. Quality > quantity.
# Format: { channel_id: { "name": str, "topic": str, "video_ids": [str] } }

CURATED_VIDEOS: dict[str, dict] = {
    "nhs": {
        "name":     "NHS (UK National Health Service)",
        "topic":    "clinical",
        "video_ids": [
            "XiRkCMmMgKE",  # Your pregnancy — week by week
            "6TJ3HoHQSdM",  # Signs of labour
            "dqnfpvIzVVQ",  # What to expect at your antenatal appointments
            "LGEM2PqTRzY",  # Breastfeeding — getting started
            "g3EmPv9w5Hc",  # Postnatal depression
        ],
    },
    "tommys": {
        "name":     "Tommy's Pregnancy Charity",
        "topic":    "pregnancy",
        "video_ids": [
            "D_2BFEA3OaU",  # What is a birth plan?
            "nZs_2VBz9is",  # Pelvic floor exercises in pregnancy
            "zZJ7Jy2ETSE",  # Preparing for labour
            "fKE0dNoRtNY",  # Coping with pregnancy anxiety
            "bK-VxAx6j_8",  # Movement in pregnancy
        ],
    },
    "what_to_expect": {
        "name":     "What to Expect",
        "topic":    "pregnancy_parenting",
        "video_ids": [
            "HpFcmcJhOx0",  # First trimester symptoms
            "YNQ-8LVXZ3c",  # Second trimester what to expect
            "5NhSFv5v7Jg",  # Third trimester tips
            "j5qRwOozZmU",  # Newborn care basics
            "n5TDSijm14k",  # Breastfeeding tips for new moms
            "4kzJAjWPd8s",  # Baby milestones 0-3 months
            "yqWuMcqBMhM",  # Baby milestones 6-9 months
        ],
    },
    "unicef": {
        "name":     "UNICEF",
        "topic":    "baby_care",
        "video_ids": [
            "tKvZJTD1MFE",  # Breastfeeding: the first hour
            "LfBW8mG0RWk",  # Responsive feeding
            "N-qpQMCWBr4",  # Skin to skin contact
        ],
    },
    "peanut": {
        "name":     "Peanut App (evidence-based parenting)",
        "topic":    "parenting",
        "video_ids": [
            "fh0MuZY1TT0",  # Postpartum recovery what to expect
            "hI4dqGCJL4E",  # Fourth trimester explained
            "EQBqMZv4Hxc",  # Postnatal depression signs
        ],
    },
    "dad_specific": {
        "name":     "Dad University",
        "topic":    "dad_partner",
        "video_ids": [
            "mW3xJhI89EY",  # How to support your partner in labour
            "0Ns8M0gfmRA",  # New dad survival guide
            "kZ5LMlOsFOo",  # Paternity leave tips
        ],
    },
}

# ── Chunk config ───────────────────────────────────────────────────────────────
WINDOW_SECONDS  = 120   # each chunk = 2-minute window of transcript
OVERLAP_SECONDS = 20    # overlap between windows

YOUTUBE_SEARCH_TEMPLATES: dict[str, list[str]] = {
    "labour": [
        "signs of labour", "what to expect during labour", "labour pain relief",
        "early labour symptoms", "birth plan preparation"
    ],
    "pregnancy": [
        "pregnancy week by week", "first trimester symptoms", "second trimester questions",
        "third trimester tips", "antenatal appointments explained"
    ],
    "postpartum": [
        "postnatal depression", "postpartum recovery", "postpartum care for new moms",
        "fourth trimester explained"
    ],
    "baby_care": [
        "newborn care basics", "breastfeeding tips", "skin to skin contact",
        "responsive feeding", "newborn milestones"
    ],
    "dad_partner": [
        "how to support your partner in labour", "new dad survival guide", "paternity leave tips",
        "partner support in pregnancy"
    ],
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
    ],
}

_COMPILED = {
    topic: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for topic, patterns in TOPIC_PATTERNS.items()
}


def tag_chunk(text: str) -> str:
    scores: dict[str, int] = {}
    for topic, patterns in _COMPILED.items():
        score = sum(1 for pattern in patterns if pattern.search(text))
        if score:
            scores[topic] = score
    if not scores:
        return "general"
    return max(scores, key=scores.__getitem__)


def parse_iso8601_duration(duration: str) -> int:
    hours = minutes = seconds = 0
    for part in re.findall(r"(\d+)([HMS])", duration):
        value, unit = int(part[0]), part[1]
        if unit == "H":
            hours = value
        elif unit == "M":
            minutes = value
        elif unit == "S":
            seconds = value
    return hours * 3600 + minutes * 60 + seconds


def build_search_queries(topic: str) -> list[str]:
    return YOUTUBE_SEARCH_TEMPLATES.get(topic, [])


def search_youtube(query: str, max_results: int = 15) -> list[dict]:
    if not YOUTUBE_API_KEY:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "order": "relevance",
        "videoDuration": "medium",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        items = response.json().get("items", [])
        return [item for item in items if item.get("id", {}).get("videoId")]
    except Exception as e:
        print(f"    ⚠ YouTube search failed for '{query}': {e}")
        return []


def fetch_video_details(video_ids: list[str]) -> list[dict]:
    if not video_ids or not YOUTUBE_API_KEY:
        return []

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json().get("items", [])
    except Exception as e:
        print(f"    ⚠ YouTube details request failed: {e}")
        return []


def score_video(video: dict) -> float:
    snippet = video.get("snippet", {})
    stats = video.get("statistics", {})
    views = int(stats.get("viewCount", 0))
    published = snippet.get("publishedAt", "1970-01-01T00:00:00Z")
    try:
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(published.replace("Z", "+00:00"))).days
    except Exception:
        age_days = 365
    recency_score = max(0.0, 1.0 - min(age_days / 365.0, 1.0))
    return views * 0.00001 + recency_score


def select_ranked_videos(topic: str, max_videos: int = 10) -> list[dict]:
    queries = build_search_queries(topic)
    seen = set()
    results: list[dict] = []

    for query in queries:
        search_items = search_youtube(query, max_results=10)
        for item in search_items:
            video_id = item["id"]["videoId"]
            if video_id in seen:
                continue
            seen.add(video_id)
            results.append(item)

    if not results and not YOUTUBE_API_KEY:
        return []

    video_ids = [item["id"]["videoId"] for item in results]
    details = {video["id"]: video for video in fetch_video_details(video_ids)}

    enriched = []
    for item in results:
        video_id = item["id"]["videoId"]
        video = details.get(video_id)
        if not video:
            continue
        duration = parse_iso8601_duration(video.get("contentDetails", {}).get("duration", "PT0S"))
        if duration < 120 or duration > 1800:
            continue
        video["score"] = score_video(video)
        enriched.append(video)

    return sorted(enriched, key=lambda v: v["score"], reverse=True)[:max_videos]


def get_search_video_ids(topic: str, max_videos: int = 10) -> list[str]:
    if not YOUTUBE_API_KEY:
        return []

    ranked = select_ranked_videos(topic, max_videos=max_videos)
    return [video["id"] for video in ranked]


# ── Transcript fetching ────────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> list[dict] | None:
    """
    Fetch transcript for a YouTube video.
    Returns list of {text, start, duration} dicts, or None if unavailable.
    Tries English first, then auto-generated English.
    """
    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for lang in ["en", "en-GB", "en-US"]:
                try:
                    return transcript_list.find_manually_created_transcript([lang]).fetch()
                except Exception:
                    pass
            try:
                return transcript_list.find_generated_transcript(["en"]).fetch()
            except Exception:
                pass
            transcript = transcript_list.find_transcript(
                transcript_list._transcripts.keys()
            )
            return transcript.translate("en").fetch()

        return YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-GB", "en-US"])

    except (NoTranscriptFound, TranscriptsDisabled):
        return None
    except Exception as e:
        print(f"    ⚠ Transcript error for {video_id}: {e}")
        return None


def transcript_to_chunks(
    transcript: list[dict],
    video_id: str,
    channel_name: str,
    topic: str,
    window: int = WINDOW_SECONDS,
    overlap: int = OVERLAP_SECONDS,
) -> list[Document]:
    """
    Convert a timestamped transcript into overlapping Document chunks.
    Each chunk covers `window` seconds with `overlap` seconds of context.
    Metadata includes start_seconds so we can link to the exact timestamp.
    """
    if not transcript:
        return []

    chunks: list[Document] = []
    total_duration = transcript[-1]["start"] + transcript[-1].get("duration", 0)
    start_time     = 0.0

    while start_time < total_duration:
        end_time = start_time + window

        # Gather transcript segments in this window
        segment_texts = []
        for entry in transcript:
            t = entry["start"]
            if start_time <= t < end_time:
                text = entry["text"].strip()
                # Clean up auto-generated transcript artifacts
                text = re.sub(r"\[.*?\]", "", text)     # remove [Music], [Applause]
                text = re.sub(r"\s+", " ", text)
                if text:
                    segment_texts.append(text)

        if segment_texts:
            content = " ".join(segment_texts)
            # Skip very short or low-content chunks
            if len(content) > 80:
                start_mm  = int(start_time) // 60
                start_ss  = int(start_time) % 60
                timestamp = f"{start_mm}:{start_ss:02d}"

                chunks.append(Document(
                    page_content=f"[{channel_name} — {timestamp}] {content}",
                    metadata={
                        "video_id":      video_id,
                        "channel_name":  channel_name,
                        "topic_tag":     tag_chunk(content),
                        "youtube_topic": topic,
                        "start_seconds": int(start_time),
                        "timestamp":     timestamp,
                        "youtube_url":   f"https://youtube.com/watch?v={video_id}&t={int(start_time)}s",
                        "collection":    COLLECTION,
                        "doc_type":      "youtube_transcript",
                        "source_file":   f"youtube_{video_id}",
                        "ingested_at":   datetime.now(timezone.utc).isoformat(),
                    },
                ))

        start_time += window - overlap   # advance with overlap

    return chunks


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest_youtube(
    reset: bool = False,
    channel_filter: str | None = None,
    search: bool = False,
    search_topic: str | None = None,
    max_videos: int = 10,
) -> None:
    embeddings   = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    persist_path = str(CHROMA_DIR / COLLECTION)

    if reset:
        import shutil
        if Path(persist_path).exists():
            shutil.rmtree(persist_path)
            print(f"🗑️  Wiped '{COLLECTION}'.")

    if channel_filter and channel_filter not in CURATED_VIDEOS:
        print(f"Channel '{channel_filter}' not found. Available: {list(CURATED_VIDEOS.keys())}")
        return

    all_chunks: list[Document] = []
    stats: dict[str, dict]     = {}
    source_keys = [channel_filter] if channel_filter else list(CURATED_VIDEOS.keys())

    if search and not YOUTUBE_API_KEY:
        print("⚠ YOUTUBE_API_KEY not set. Falling back to curated video ids only.")
        search = False

    topic_keys = [search_topic] if search_topic else list(YOUTUBE_SEARCH_TEMPLATES.keys())

    for source_key in source_keys:
        channel_info = CURATED_VIDEOS[source_key]
        name         = channel_info["name"]
        channel_topic = channel_info["topic"]
        channel_videos: list[str] = []
        skipped      = 0
        channel_chunks = 0

        if search:
            print(f"\n🔎 Searching YouTube for {name} and topics: {topic_keys}")
            for topic in topic_keys:
                video_items = select_ranked_videos(topic, max_videos=max_videos)
                channel_videos.extend([item["id"] for item in video_items])
            channel_videos = list(dict.fromkeys(channel_videos))
        else:
            channel_videos = channel_info["video_ids"]

        print(f"\n📺  {name} ({len(channel_videos)} videos)")
        for vid_id in channel_videos:
            print(f"    Fetching: {vid_id} ... ", end="", flush=True)
            transcript = fetch_transcript(vid_id)

            if not transcript:
                print("❌ no transcript")
                skipped += 1
                time.sleep(1)
                continue

            chunks = transcript_to_chunks(transcript, vid_id, name, channel_topic)
            all_chunks.extend(chunks)
            channel_chunks += len(chunks)
            print(f"✅ {len(chunks)} chunks ({len(transcript)} segments)")
            time.sleep(1.5)   # polite delay

        stats[source_key] = {"chunks": channel_chunks, "skipped": skipped}

    if not all_chunks:
        print("\nNo chunks to index.")
        return

    print(f"\n📥  Indexing {len(all_chunks)} transcript chunks into '{COLLECTION}'...")

    batch_size  = 100
    vectorstore = None

    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i: i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION,
                persist_directory=persist_path,
            )
        else:
            vectorstore.add_documents(batch)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n✅  '{COLLECTION}' complete\n")
    print(f"  {'Source':<30} {'Chunks':>7}  {'Skipped':>7}")
    print(f"  {'-'*48}")
    for key, s in stats.items():
        name = CURATED_VIDEOS[key]["name"][:28]
        print(f"  {name:<30} {s['chunks']:>7}  {s['skipped']:>7}")
    print(f"\n  Total chunks: {len(all_chunks)}")
    print(f"\n  ℹ  Retrieved chunks will include YouTube links with exact timestamps.")
    print(f"     Format in citations: https://youtube.com/watch?v=VIDEO_ID&t=SECONDSs\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BabyOS YouTube Transcript Ingestion")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild")
    parser.add_argument("--channel", type=str, default=None,
                        help=f"Ingest single channel: {list(CURATED_VIDEOS.keys())}")
    parser.add_argument("--search", action="store_true",
                        help="Search YouTube by topic and ingest only ranked videos")
    parser.add_argument("--topic", type=str, default=None,
                        help=f"Search a single topic from: {list(YOUTUBE_SEARCH_TEMPLATES.keys())}")
    parser.add_argument("--max-videos", type=int, default=10,
                        help="Maximum ranked videos to ingest per topic")
    args = parser.parse_args()
    ingest_youtube(
        reset=args.reset,
        channel_filter=args.channel,
        search=args.search,
        search_topic=args.topic,
        max_videos=args.max_videos,
    )

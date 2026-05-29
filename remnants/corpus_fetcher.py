"""
corpus_fetcher.py
-----------------
Fetches additional pregnancy content from public web sources to supplement
the local corpus. Run this once before ingestion.

Sources targeted:
  - NHS Pregnancy Week by Week (weeks 4-42)
  - WHO Antenatal care recommendations summary
  - BabyCenter fetal development pages (selected weeks)

Output: data/raw/web_scraped/ folder with .txt files per page
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "data" / "raw" / "web_scraped"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BabyOS-research-bot/1.0; "
        "educational project)"
    )
}


def clean_text(soup: BeautifulSoup, selectors_to_remove: list[str]) -> str:
    """Remove nav/footer/ads and extract clean text."""
    for sel in selectors_to_remove:
        for tag in soup.select(sel):
            tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def fetch_nhs_week_pages() -> None:
    """Fetch NHS pregnancy week-by-week pages."""
    print("\n--- Fetching NHS week-by-week pages ---")
    base = "https://www.nhs.uk/pregnancy/week-by-week/1-to-12/4-weeks/"
    
    # NHS week paths
    week_paths = {
        4:  "1-to-12/4-weeks/",
        5:  "1-to-12/5-weeks/",
        6:  "1-to-12/6-weeks/",
        7:  "1-to-12/7-weeks/",
        8:  "1-to-12/8-weeks/",
        9:  "1-to-12/9-weeks/",
        10: "1-to-12/10-weeks/",
        11: "1-to-12/11-weeks/",
        12: "1-to-12/12-weeks/",
        13: "13-to-27/13-weeks/",
        14: "13-to-27/14-weeks/",
        16: "13-to-27/16-weeks/",
        18: "13-to-27/18-weeks/",
        20: "13-to-27/20-weeks/",
        22: "13-to-27/22-weeks/",
        24: "13-to-27/24-weeks/",
        26: "13-to-27/26-weeks/",
        28: "28-to-40/28-weeks/",
        30: "28-to-40/30-weeks/",
        32: "28-to-40/32-weeks/",
        34: "28-to-40/34-weeks/",
        36: "28-to-40/36-weeks/",
        38: "28-to-40/38-weeks/",
        40: "28-to-40/40-weeks/",
    }

    remove = ["nav", "footer", "header", ".breadcrumb", ".nhsuk-navigation",
              ".nhsuk-header", ".nhsuk-footer", "script", "style"]

    for week, path in week_paths.items():
        url = f"https://www.nhs.uk/pregnancy/week-by-week/{path}"
        out_file = OUTPUT_DIR / f"nhs_week_{week:02d}.txt"
        if out_file.exists():
            print(f"  Week {week}: already fetched, skipping.")
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = clean_text(soup, remove)
            out_file.write_text(
                f"SOURCE: NHS Pregnancy Week {week}\nURL: {url}\n\n{text}",
                encoding="utf-8"
            )
            print(f"  Week {week}: saved ({len(text)} chars)")
            time.sleep(1.5)  # polite crawl delay
        except Exception as e:
            print(f"  Week {week}: FAILED — {e}")


def fetch_nhs_topics() -> None:
    """Fetch key NHS pregnancy topic pages."""
    print("\n--- Fetching NHS pregnancy topic pages ---")
    topics = {
        "signs_of_labour":        "https://www.nhs.uk/pregnancy/labour-and-birth/signs-of-labour/",
        "antenatal_checks":       "https://www.nhs.uk/pregnancy/your-pregnancy-care/antenatal-checks/",
        "common_symptoms":        "https://www.nhs.uk/pregnancy/related-conditions/common-symptoms/",
        "foods_to_avoid":         "https://www.nhs.uk/pregnancy/keeping-well/foods-to-avoid/",
        "vitamins_supplements":   "https://www.nhs.uk/pregnancy/keeping-well/vitamins-supplements-and-nutrition/",
        "exercise":               "https://www.nhs.uk/pregnancy/keeping-well/exercise/",
        "mental_health":          "https://www.nhs.uk/pregnancy/keeping-well/mental-health/",
        "your_antenatal_team":    "https://www.nhs.uk/pregnancy/your-pregnancy-care/your-antenatal-care-team/",
        "gestational_diabetes":   "https://www.nhs.uk/conditions/gestational-diabetes/",
        "preeclampsia":           "https://www.nhs.uk/conditions/pre-eclampsia/",
        "miscarriage":            "https://www.nhs.uk/conditions/miscarriage/",
        "postnatal_depression":   "https://www.nhs.uk/mental-health/conditions/post-natal-depression/overview/",
        "breastfeeding":          "https://www.nhs.uk/conditions/baby/breastfeeding-and-bottle-feeding/breastfeeding/",
        "newborn_checks":         "https://www.nhs.uk/conditions/baby/newborn-screening/checks-and-tests/",
    }

    remove = ["nav", "footer", "header", "script", "style",
              ".nhsuk-navigation", ".nhsuk-header", ".nhsuk-footer",
              ".nhsuk-breadcrumb"]

    for name, url in topics.items():
        out_file = OUTPUT_DIR / f"nhs_{name}.txt"
        if out_file.exists():
            print(f"  {name}: already fetched, skipping.")
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = clean_text(soup, remove)
            out_file.write_text(
                f"SOURCE: NHS — {name.replace('_', ' ').title()}\nURL: {url}\n\n{text}",
                encoding="utf-8"
            )
            print(f"  {name}: saved ({len(text)} chars)")
            time.sleep(1.5)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")


def fetch_who_guidelines() -> None:
    """Fetch WHO antenatal care recommendations overview."""
    print("\n--- Fetching WHO guidelines ---")
    urls = {
        "who_antenatal_overview": "https://www.who.int/news-room/fact-sheets/detail/pregnancy",
        "who_maternal_health":    "https://www.who.int/health-topics/maternal-health",
    }
    remove = ["nav", "footer", "header", "script", "style"]

    for name, url in urls.items():
        out_file = OUTPUT_DIR / f"{name}.txt"
        if out_file.exists():
            print(f"  {name}: already fetched, skipping.")
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = clean_text(soup, remove)
            out_file.write_text(
                f"SOURCE: WHO — {name}\nURL: {url}\n\n{text}",
                encoding="utf-8"
            )
            print(f"  {name}: saved ({len(text)} chars)")
            time.sleep(2)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")


def report() -> None:
    files = list(OUTPUT_DIR.glob("*.txt"))
    total_chars = sum(f.stat().st_size for f in files)
    print(f"\n✅ Corpus fetch complete.")
    print(f"   Files saved: {len(files)}")
    print(f"   Total size:  {total_chars / 1024:.1f} KB")
    print(f"   Location:    {OUTPUT_DIR}")


if __name__ == "__main__":
    fetch_nhs_week_pages()
    fetch_nhs_topics()
    fetch_who_guidelines()
    report()

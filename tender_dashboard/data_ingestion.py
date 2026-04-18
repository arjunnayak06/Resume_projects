"""
data_ingestion.py
-----------------
Step 1 & 2 of the pipeline:
  • Reads the Google Sheet (requires gspread + service-account credentials, OR
    the sheet must be publicly shared so a simple URL export works).
  • Falls back to local CSV files (tenders_raw.csv / bidders_raw.csv) when
    Google Sheets is not accessible.
  • Scrapes bidder lists from tender247.com for rows that have a Tender_URL
    but an empty Bidder_List.

Environment variables (optional):
  GOOGLE_SHEET_ID    – the spreadsheet ID
  GOOGLE_SHEET_GID   – the worksheet gid (default: 26798947)
  GSPREAD_CREDS_JSON – path to service-account JSON key file
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Google Sheets ────────────────────────────────────────────────────────────

SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "1v85LFfZ9uP5Sojm7cti5oUFSFyhSaRJt7QFH45GhlBA")
SHEET_GID = os.getenv("GOOGLE_SHEET_GID", "26798947")
CREDS_PATH = os.getenv("GSPREAD_CREDS_JSON", "")

_DATA_DIR = Path(__file__).parent


def load_sheet_via_public_export() -> Optional[pd.DataFrame]:
    """Download a *publicly shared* Google Sheet as CSV."""
    url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
        f"/export?format=csv&gid={SHEET_GID}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        logger.info("Loaded %d rows from Google Sheets (public export).", len(df))
        return df
    except Exception as exc:
        logger.warning("Public export failed: %s", exc)
        return None


def load_sheet_via_gspread() -> Optional[pd.DataFrame]:
    """Download using a service-account key (private sheets)."""
    if not CREDS_PATH or not Path(CREDS_PATH).exists():
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        ws = sh.get_worksheet_by_id(int(SHEET_GID))
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        logger.info("Loaded %d rows via gspread.", len(df))
        return df
    except Exception as exc:
        logger.warning("gspread failed: %s", exc)
        return None


def load_local_fallback() -> pd.DataFrame:
    """Load pre-generated CSV files when Google Sheets is unavailable."""
    tenders_path = _DATA_DIR / "tenders_raw.csv"
    bidders_path = _DATA_DIR / "bidders_raw.csv"

    if tenders_path.exists() and bidders_path.exists():
        tenders = pd.read_csv(tenders_path)
        bidders = pd.read_csv(bidders_path)
        logger.info(
            "Loaded %d tenders + %d bidder rows from local CSV files.",
            len(tenders),
            len(bidders),
        )
        return tenders, bidders

    # Generate synthetic data on the fly
    logger.info("Generating synthetic data …")
    import sys

    sys.path.insert(0, str(_DATA_DIR))
    from generate_sample_data import generate_tenders, generate_bidders

    tenders = generate_tenders(800)
    bidders = generate_bidders(tenders)
    tenders.to_csv(tenders_path, index=False)
    bidders.to_csv(bidders_path, index=False)
    return tenders, bidders


# ── tender247.com scraper ────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def scrape_bidders_from_tender247(url: str, retries: int = 2) -> list[str]:
    """
    Attempt to extract the bidder/participant list from a tender247.com page.

    Returns a list of bidder name strings (may be empty if page is
    inaccessible, behind a paywall, or structure has changed).
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            bidders: list[str] = []

            # Pattern 1: table rows with class containing "bidder"
            for row in soup.select("tr"):
                cells = row.find_all("td")
                text_in_row = " ".join(c.get_text(" ", strip=True) for c in cells).lower()
                if any(
                    kw in text_in_row
                    for kw in ("bidder", "participant", "tenderer", "vendor")
                ):
                    name_cell = cells[0].get_text(" ", strip=True) if cells else ""
                    if name_cell and len(name_cell) > 3:
                        bidders.append(name_cell)

            # Pattern 2: named sections / divs
            if not bidders:
                for div in soup.find_all(
                    ["div", "section", "li"],
                    string=re.compile(r"(?i)(bidder|participant|tenderer)"),
                ):
                    text = div.get_text(" ", strip=True)
                    if text and len(text) > 3:
                        bidders.append(text)

            return list(dict.fromkeys(bidders))  # de-dup preserving order
        except requests.RequestException as exc:
            logger.warning("Attempt %d scraping %s: %s", attempt + 1, url, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return []


def enrich_tenders_with_scraped_bidders(
    tenders: pd.DataFrame,
    bidders: pd.DataFrame,
    max_scrape: int = 50,
) -> pd.DataFrame:
    """
    For tenders that have a URL but no bidder entry, scrape tender247.com
    and append the results to the bidders DataFrame.
    """
    existing_tids = set(bidders["Tender_ID"].unique())
    missing = tenders[
        tenders["Tender_URL"].notna()
        & ~tenders["Tender_ID"].isin(existing_tids)
    ].head(max_scrape)

    new_rows = []
    for _, row in missing.iterrows():
        scraped = scrape_bidders_from_tender247(row["Tender_URL"])
        if not scraped:
            continue
        logger.info("  Scraped %d bidders for %s", len(scraped), row["Tender_ID"])
        for rank, name in enumerate(scraped, start=1):
            new_rows.append(
                {
                    "Tender_ID": row["Tender_ID"],
                    "State": row["State"],
                    "Year": row["Year"],
                    "Work_Category": row.get("Work_Category", ""),
                    "Bidder_Name": name,
                    "Bid_Rank": rank,
                    "Is_Winner": name == row.get("Winner_Name", ""),
                    "Tender_Value_INR": row.get("Estimated_Value_INR", 0),
                }
            )
        time.sleep(1)  # polite crawl delay

    if new_rows:
        extra = pd.DataFrame(new_rows)
        bidders = pd.concat([bidders, extra], ignore_index=True)

    return bidders


# ── public API ───────────────────────────────────────────────────────────────

def ingest(scrape: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (tenders_df, bidders_df).

    Priority order:
      1. Google Sheet via public CSV export
      2. Google Sheet via gspread service account
      3. Local CSV files  /  synthetic generation
    """
    sheet_df = load_sheet_via_public_export() or load_sheet_via_gspread()

    if sheet_df is not None:
        # Normalise column names from the real sheet
        sheet_df.columns = [c.strip() for c in sheet_df.columns]
        col_map = {
            "State": "State",
            "Tender ID": "Tender_ID",
            "Tender URL": "Tender_URL",
            "Year": "Year",
            "Winner": "Winner_Name",
            "Winner Name": "Winner_Name",
            "Bidder List": "Bidder_List",
            "Bidders": "Bidder_List",
            "Value": "Estimated_Value_INR",
            "Estimated Value": "Estimated_Value_INR",
        }
        rename_map = {k: v for k, v in col_map.items() if k in sheet_df.columns}
        sheet_df = sheet_df.rename(columns=rename_map)

        # Build bidders table from Bidder_List column
        if "Bidder_List" not in sheet_df.columns:
            sheet_df["Bidder_List"] = ""

        bidder_rows = []
        for _, row in sheet_df.iterrows():
            raw = str(row.get("Bidder_List", "") or "")
            names = [n.strip() for n in re.split(r"[|,;\n]+", raw) if n.strip()]
            for rank, name in enumerate(names, start=1):
                bidder_rows.append(
                    {
                        "Tender_ID": row.get("Tender_ID", ""),
                        "State": row.get("State", ""),
                        "Year": row.get("Year", ""),
                        "Work_Category": row.get("Work_Category", ""),
                        "Bidder_Name": name,
                        "Bid_Rank": rank,
                        "Is_Winner": name == row.get("Winner_Name", ""),
                        "Tender_Value_INR": row.get("Estimated_Value_INR", 0),
                    }
                )
        tenders = sheet_df
        bidders = pd.DataFrame(bidder_rows) if bidder_rows else pd.DataFrame(
            columns=["Tender_ID", "State", "Year", "Work_Category",
                     "Bidder_Name", "Bid_Rank", "Is_Winner", "Tender_Value_INR"]
        )
    else:
        tenders, bidders = load_local_fallback()

    if scrape and "Tender_URL" in tenders.columns:
        bidders = enrich_tenders_with_scraped_bidders(tenders, bidders)

    return tenders, bidders

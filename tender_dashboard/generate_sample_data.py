"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic Indian-tender dataset that mirrors the
expected Google Sheet schema when real credentials are not available.

Run:
    python generate_sample_data.py
Writes:
    tenders_raw.csv   – one row per tender
    bidders_raw.csv   – one row per (tender, bidder)
"""

import random
import string
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ── static reference lists ──────────────────────────────────────────────────

STATES = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab",
    "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal",
]

DEPARTMENTS = [
    "Public Works Department", "Rural Development", "National Highway Authority",
    "Ministry of Water Resources", "State Electricity Board",
    "Urban Development Authority", "Ministry of Railways",
    "Department of Health", "Municipal Corporation", "CPWD",
]

BIDDER_POOL = [
    "Larsen & Toubro Ltd", "Tata Projects Ltd", "Afcons Infrastructure",
    "NCC Limited", "HCC Ltd", "Simplex Infrastructures", "KEC International",
    "Dilip Buildcon Ltd", "J Kumar Infraprojects", "PNC Infratech",
    "Ashoka Buildcon", "Gayatri Projects", "IRCON International",
    "RITES Ltd", "BEML Limited", "Megha Engineering", "GR Infraprojects",
    "Patel Engineering", "MEIL", "Shapoorji Pallonji",
    "Era Infra Engineering", "ITD Cementation", "Punj Lloyd",
    "Soma Enterprise", "Nagarjuna Construction", "Gammon India",
    "Hindustan Construction", "Oriental Structural Engineers",
    "Tantia Constructions", "Vishwa Infrastructures",
    "Welspun Enterprises", "Sadbhav Engineering", "KNR Constructions",
    "Apco Infratech", "Capacite Infraprojects", "Ahluwalia Contracts",
    "NBCC India", "Engineers India Ltd", "Sterlite Power",
    "Torrent Construction", "Ramky Infrastructure", "Pratibha Industries",
]

WORK_CATEGORIES = [
    "Road Construction", "Bridge Construction", "Water Supply",
    "Sewerage", "Building Construction", "Electrical Works",
    "IT Infrastructure", "Equipment Procurement", "Consultancy",
    "Operation & Maintenance",
]

# ── helper functions ─────────────────────────────────────────────────────────


def random_date(start_year: int, end_year: int) -> date:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def tender_id(state: str, year: int, seq: int) -> str:
    state_code = state[:2].upper()
    return f"{state_code}/{year}/{seq:04d}"


def tender_url(tid: str) -> str:
    slug = tid.replace("/", "-").lower()
    return f"https://www.tender247.com/tender/{slug}"


# ── main generation ──────────────────────────────────────────────────────────

def generate_tenders(n: int = 800) -> pd.DataFrame:
    rows = []
    seq_counter: dict[str, int] = {}
    for _ in range(n):
        state = random.choice(STATES)
        year = random.choices(
            range(2018, 2025),
            weights=[5, 8, 12, 15, 18, 20, 22],  # more recent tenders
        )[0]
        dept = random.choice(DEPARTMENTS)
        category = random.choice(WORK_CATEGORIES)
        issued_date = random_date(year, year)
        estimated_value = round(np.random.lognormal(mean=16, sigma=1.5))  # INR

        key = f"{state}-{year}"
        seq_counter[key] = seq_counter.get(key, 0) + 1
        tid = tender_id(state, year, seq_counter[key])

        n_bidders = random.choices(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
            weights=[2, 5, 10, 15, 18, 16, 12, 8, 5, 3, 3, 2, 1],
        )[0]
        # bias bidders toward states – dominant firms win more often
        bidder_pool_for_state = BIDDER_POOL.copy()
        random.shuffle(bidder_pool_for_state)
        bidders = bidder_pool_for_state[:n_bidders]
        winner = bidders[0]

        rows.append(
            {
                "Tender_ID": tid,
                "Tender_URL": tender_url(tid),
                "State": state,
                "Year": year,
                "Issued_Date": issued_date.isoformat(),
                "Department": dept,
                "Work_Category": category,
                "Estimated_Value_INR": estimated_value,
                "Num_Bidders": n_bidders,
                "Winner_Name": winner,
                "Bidder_List": "|".join(bidders),
            }
        )
    return pd.DataFrame(rows)


def generate_bidders(tenders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tenders.iterrows():
        bidders = row["Bidder_List"].split("|")
        for rank, bidder in enumerate(bidders, start=1):
            rows.append(
                {
                    "Tender_ID": row["Tender_ID"],
                    "State": row["State"],
                    "Year": row["Year"],
                    "Work_Category": row["Work_Category"],
                    "Bidder_Name": bidder,
                    "Bid_Rank": rank,
                    "Is_Winner": bidder == row["Winner_Name"],
                    "Tender_Value_INR": row["Estimated_Value_INR"],
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tenders = generate_tenders(800)
    bidders = generate_bidders(tenders)

    tenders.to_csv("tenders_raw.csv", index=False)
    bidders.to_csv("bidders_raw.csv", index=False)

    print(f"✓ tenders_raw.csv  → {len(tenders):,} rows")
    print(f"✓ bidders_raw.csv  → {len(bidders):,} rows")
    print(f"  States covered   : {tenders['State'].nunique()}")
    print(f"  Year range       : {tenders['Year'].min()} – {tenders['Year'].max()}")
    print(f"  Unique bidders   : {bidders['Bidder_Name'].nunique()}")

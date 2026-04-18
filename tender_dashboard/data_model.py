"""
data_model.py
-------------
Step 3 & 4 of the pipeline: data cleaning, normalisation, and analytical
table construction.

Public API
----------
build_model(tenders, bidders) -> dict[str, pd.DataFrame]

Returns a dictionary with keys:
  master_bidder     – one row per (Bidder, State, Year)
  yearly_summary    – aggregated stats by year
  state_summary     – aggregated stats by state
  top_bidders       – bidder ranking (overall)
  state_bidder_rank – bidder ranking per state
  heatmap_data      – pivot: State × Year → bidder count
  winner_share      – winner vs non-winner share
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

# ── normalisation helpers ─────────────────────────────────────────────────────

# Common abbreviation expansions (add more as needed)
_ABBR = {
    r"\bltd\.?\b": "Limited",
    r"\bpvt\.?\b": "Private",
    r"\binfra\b": "Infrastructure",
    r"\bconst\b": "Construction",
    r"\beng\b": "Engineering",
    r"\bcorp\b": "Corporation",
    r"\bco\.?\b": "Company",
    r"\bl&t\b": "Larsen & Toubro",
    r"\bncc\b": "NCC Limited",
}


def _normalise_name(name: str) -> str:
    """Lowercase, strip, expand abbreviations, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = name.title()  # Title-case first pass
    # Expand known abbreviations
    for pattern, replacement in _ABBR.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    # Collapse internal whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _coerce_year(val) -> Optional[int]:
    """Convert various year representations to int, return None if invalid."""
    if pd.isna(val):
        return None
    try:
        y = int(str(val)[:4])
        return y if 2000 <= y <= 2030 else None
    except (ValueError, TypeError):
        return None


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean_tenders(tenders: pd.DataFrame) -> pd.DataFrame:
    df = tenders.copy()
    df["Year"] = df["Year"].apply(_coerce_year)
    df = df.dropna(subset=["Tender_ID", "Year"])
    df["State"] = df["State"].str.strip().str.title()
    df["Winner_Name"] = df["Winner_Name"].apply(_normalise_name)
    df["Tender_ID"] = df["Tender_ID"].astype(str).str.strip()
    if "Estimated_Value_INR" in df.columns:
        df["Estimated_Value_INR"] = pd.to_numeric(
            df["Estimated_Value_INR"], errors="coerce"
        ).fillna(0)
    return df.drop_duplicates(subset=["Tender_ID"])


def clean_bidders(bidders: pd.DataFrame) -> pd.DataFrame:
    df = bidders.copy()
    df["Year"] = df["Year"].apply(_coerce_year)
    df = df.dropna(subset=["Bidder_Name", "Tender_ID", "Year"])
    df["Bidder_Name"] = df["Bidder_Name"].apply(_normalise_name)
    df = df[df["Bidder_Name"].str.len() > 2]  # drop single-char artefacts
    df["State"] = df["State"].str.strip().str.title()
    df["Is_Winner"] = df["Is_Winner"].astype(bool)
    df["Tender_Value_INR"] = pd.to_numeric(
        df.get("Tender_Value_INR", 0), errors="coerce"
    ).fillna(0)
    return df.drop_duplicates(subset=["Tender_ID", "Bidder_Name"])


# ── master bidder table ───────────────────────────────────────────────────────

def build_master_bidder(bidders: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (Bidder_Name, State, Year) with participation & win counts.
    """
    grp = (
        bidders.groupby(["Bidder_Name", "State", "Year"])
        .agg(
            Tenders_Participated=("Tender_ID", "nunique"),
            Tenders_Won=("Is_Winner", "sum"),
            Total_Value_INR=("Tender_Value_INR", "sum"),
        )
        .reset_index()
    )
    grp["Win_Rate_Pct"] = (
        grp["Tenders_Won"] / grp["Tenders_Participated"] * 100
    ).round(1)
    return grp.sort_values(
        ["State", "Year", "Tenders_Participated"], ascending=[True, True, False]
    ).reset_index(drop=True)


# ── yearly aggregation ────────────────────────────────────────────────────────

def build_yearly_summary(tenders: pd.DataFrame, bidders: pd.DataFrame) -> pd.DataFrame:
    t_year = (
        tenders.groupby("Year")
        .agg(Total_Tenders=("Tender_ID", "nunique"))
        .reset_index()
    )
    b_year = (
        bidders.groupby("Year")
        .agg(
            Total_Unique_Bidders=("Bidder_Name", "nunique"),
            Total_Participations=("Bidder_Name", "count"),
        )
        .reset_index()
    )
    summary = t_year.merge(b_year, on="Year", how="outer").sort_values("Year")
    summary["Avg_Bidders_Per_Tender"] = (
        summary["Total_Participations"] / summary["Total_Tenders"]
    ).round(2)

    # Top bidder per year
    top = (
        bidders.groupby(["Year", "Bidder_Name"])["Tender_ID"]
        .nunique()
        .reset_index(name="Count")
        .sort_values(["Year", "Count"], ascending=[True, False])
        .groupby("Year")
        .first()
        .reset_index()
        .rename(columns={"Bidder_Name": "Top_Bidder", "Count": "Top_Bidder_Count"})
    )
    summary = summary.merge(top, on="Year", how="left")
    return summary


# ── state aggregation ─────────────────────────────────────────────────────────

def build_state_summary(tenders: pd.DataFrame, bidders: pd.DataFrame) -> pd.DataFrame:
    t_state = (
        tenders.groupby("State")
        .agg(Total_Tenders=("Tender_ID", "nunique"))
        .reset_index()
    )
    b_state = (
        bidders.groupby("State")
        .agg(
            Total_Unique_Bidders=("Bidder_Name", "nunique"),
            Total_Participations=("Bidder_Name", "count"),
        )
        .reset_index()
    )
    summary = t_state.merge(b_state, on="State", how="outer")

    # HHI – concentration index: lower = more competitive
    def hhi(group):
        counts = group["Tender_ID"].value_counts(normalize=True)
        return round((counts ** 2).sum() * 10000, 1)

    hhi_vals = (
        bidders.groupby("State")
        .apply(hhi)
        .reset_index(name="HHI_Concentration")
    )
    summary = summary.merge(hhi_vals, on="State", how="left")
    summary["Competitiveness"] = pd.cut(
        summary["HHI_Concentration"],
        bins=[0, 1000, 2500, 5000, 10001],
        labels=["High", "Moderate", "Low", "Very Low"],
    )
    return summary.sort_values("Total_Tenders", ascending=False).reset_index(drop=True)


# ── bidder rankings ───────────────────────────────────────────────────────────

def build_top_bidders(bidders: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    rank_df = (
        bidders.groupby("Bidder_Name")
        .agg(
            States_Active=("State", "nunique"),
            Total_Participations=("Tender_ID", "nunique"),
            Total_Wins=("Is_Winner", "sum"),
            Years_Active=("Year", "nunique"),
            First_Year=("Year", "min"),
            Last_Year=("Year", "max"),
        )
        .reset_index()
    )
    rank_df["Win_Rate_Pct"] = (
        rank_df["Total_Wins"] / rank_df["Total_Participations"] * 100
    ).round(1)
    rank_df["Overall_Rank"] = rank_df["Total_Participations"].rank(
        ascending=False, method="min"
    ).astype(int)
    return rank_df.sort_values("Total_Participations", ascending=False).head(n).reset_index(drop=True)


def build_state_bidder_rank(bidders: pd.DataFrame) -> pd.DataFrame:
    """Bidder ranking within each state."""
    rank_df = (
        bidders.groupby(["State", "Bidder_Name"])
        .agg(
            Participations=("Tender_ID", "nunique"),
            Wins=("Is_Winner", "sum"),
        )
        .reset_index()
    )
    rank_df["State_Rank"] = rank_df.groupby("State")["Participations"].rank(
        ascending=False, method="min"
    ).astype(int)
    rank_df["Win_Rate_Pct"] = (rank_df["Wins"] / rank_df["Participations"] * 100).round(1)
    return rank_df.sort_values(["State", "State_Rank"]).reset_index(drop=True)


# ── heatmap pivot ─────────────────────────────────────────────────────────────

def build_heatmap_data(bidders: pd.DataFrame) -> pd.DataFrame:
    """State × Year → unique bidder count pivot table."""
    pivot = (
        bidders.groupby(["State", "Year"])["Bidder_Name"]
        .nunique()
        .unstack(fill_value=0)
    )
    return pivot


# ── winner share ──────────────────────────────────────────────────────────────

def build_winner_share(bidders: pd.DataFrame) -> pd.DataFrame:
    """How many bidders have won at least once vs. never won."""
    bidder_stats = (
        bidders.groupby("Bidder_Name")["Is_Winner"].sum().reset_index(name="Total_Wins")
    )
    bidder_stats["Category"] = bidder_stats["Total_Wins"].apply(
        lambda w: "Winner (≥1 win)" if w > 0 else "Participant (0 wins)"
    )
    return bidder_stats.groupby("Category").size().reset_index(name="Count")


# ── emerging bidders ──────────────────────────────────────────────────────────

def build_emerging_bidders(bidders: pd.DataFrame) -> pd.DataFrame:
    """
    Bidders who appeared for the first time in the last two available years
    and already have ≥2 participations.
    """
    max_year = int(bidders["Year"].max())
    recent_years = {max_year - 1, max_year}

    first_year = bidders.groupby("Bidder_Name")["Year"].min().reset_index(name="First_Year")
    total = (
        bidders.groupby("Bidder_Name")["Tender_ID"]
        .nunique()
        .reset_index(name="Total_Participations")
    )
    wins = bidders.groupby("Bidder_Name")["Is_Winner"].sum().reset_index(name="Total_Wins")

    emerging = (
        first_year[first_year["First_Year"].isin(recent_years)]
        .merge(total, on="Bidder_Name")
        .merge(wins, on="Bidder_Name")
        .query("Total_Participations >= 2")
        .sort_values("Total_Participations", ascending=False)
        .reset_index(drop=True)
    )
    return emerging


# ── repeat winners ────────────────────────────────────────────────────────────

def build_repeat_winners(bidders: pd.DataFrame, min_wins: int = 3) -> pd.DataFrame:
    rw = (
        bidders.groupby("Bidder_Name")
        .agg(
            Total_Wins=("Is_Winner", "sum"),
            Total_Participations=("Tender_ID", "nunique"),
        )
        .reset_index()
        .query(f"Total_Wins >= {min_wins}")
        .sort_values("Total_Wins", ascending=False)
        .reset_index(drop=True)
    )
    rw["Win_Rate_Pct"] = (rw["Total_Wins"] / rw["Total_Participations"] * 100).round(1)
    return rw


# ── public API ────────────────────────────────────────────────────────────────

def build_model(
    tenders: pd.DataFrame, bidders: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Run the full modelling pipeline and return a dict of analysis DataFrames.
    """
    tenders_clean = clean_tenders(tenders)
    bidders_clean = clean_bidders(bidders)

    return {
        "tenders": tenders_clean,
        "bidders": bidders_clean,
        "master_bidder": build_master_bidder(bidders_clean),
        "yearly_summary": build_yearly_summary(tenders_clean, bidders_clean),
        "state_summary": build_state_summary(tenders_clean, bidders_clean),
        "top_bidders": build_top_bidders(bidders_clean),
        "state_bidder_rank": build_state_bidder_rank(bidders_clean),
        "heatmap_data": build_heatmap_data(bidders_clean),
        "winner_share": build_winner_share(bidders_clean),
        "emerging_bidders": build_emerging_bidders(bidders_clean),
        "repeat_winners": build_repeat_winners(bidders_clean),
    }

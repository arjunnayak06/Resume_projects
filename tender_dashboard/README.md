# Indian Tender Intelligence Dashboard

A **Streamlit + Plotly** dashboard for analysing Indian public-procurement tender data — year-wise bidder participation, state-wise dominance, win-rate analysis, trends, and more.

---

## Quick Start

```bash
cd tender_dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Architecture

```
tender_dashboard/
├── app.py                  # Streamlit dashboard (main entry point)
├── data_ingestion.py       # Google Sheets reader + tender247.com scraper
├── data_model.py           # Data cleaning, normalisation, analytical tables
├── generate_sample_data.py # Synthetic Indian-tender dataset generator
├── requirements.txt
└── README.md
```

---

## Data Sources

### 1. Google Sheet (primary)
Set these environment variables before running:

| Variable              | Purpose                                      |
|-----------------------|----------------------------------------------|
| `GOOGLE_SHEET_ID`     | Spreadsheet ID (in the URL)                  |
| `GOOGLE_SHEET_GID`    | Worksheet GID (default: `26798947`)          |
| `GSPREAD_CREDS_JSON`  | Path to service-account JSON key (optional)  |

If the sheet is publicly shared, the app will download it automatically as CSV — no credentials needed.

### 2. Local CSV fallback
If Google Sheets is unreachable, place `tenders_raw.csv` and `bidders_raw.csv` in this folder (see expected schema below), or run:
```bash
python generate_sample_data.py
```
to create a synthetic dataset automatically.

### 3. Web scraping (tender247.com)
Set `ENABLE_SCRAPING=1` to scrape bidder lists from tender URLs found in the sheet.  
The scraper is polite (1-second delay between requests) and retries on failure.

---

## Expected Google Sheet Schema

| Column Name       | Description                              |
|-------------------|------------------------------------------|
| `State`           | Indian state name                        |
| `Tender ID`       | Unique tender identifier                 |
| `Tender URL`      | Full URL on tender247.com                |
| `Year`            | Year tender was issued                   |
| `Winner Name`     | Name of the winning bidder               |
| `Bidder List`     | `|`-separated list of all bidders        |
| `Estimated Value` | Tender value in INR (optional)           |
| `Work Category`   | Type of work (optional)                  |

Column aliases (e.g. `Winner`, `Bidders`) are handled automatically.

---

## Dashboard Features

### KPI Cards
- Total Tenders · Unique Bidders · Top Bidder · Most Competitive State · Avg Bidders/Tender

### Tab 1 — Overview
- Year-wise bidder + tender growth (dual-axis line + bar)
- Winner vs Participant share (donut chart)
- State × Year bidder participation heatmap

### Tab 2 — Bidder Analysis
- Top-N bidders by participations (coloured by win rate)
- Win Rate vs Participation scatter plot
- Emerging bidders (new entrants with early traction)
- Repeat winners table

### Tab 3 — State Analysis
- State-wise unique bidder count
- HHI Concentration Index per state (with competitiveness label)
- Top bidders within a selected state
- Automatic concentration-risk alerts

### Tab 4 — Trends
- Average bidders per tender over time
- Top bidder each year
- Year-over-year multi-state comparison
- Treemap: State × Bidder participation volume

### Tab 5 — Raw Tables
- All analytical tables viewable and downloadable as CSV

### Sidebar Filters
- State · Year · Highlight Bidder · Chart theme · Top-N setting

---

## Analytical Tables Produced

| Table                  | Description                                      |
|------------------------|--------------------------------------------------|
| `master_bidder`        | (Bidder, State, Year) → participations, wins, win rate |
| `yearly_summary`       | Total tenders/bidders per year, top bidder       |
| `state_summary`        | Totals per state + HHI concentration index       |
| `top_bidders`          | Global bidder ranking by participations          |
| `state_bidder_rank`    | Bidder ranking within each state                 |
| `heatmap_data`         | State × Year pivot of unique bidder counts       |
| `winner_share`         | Winners vs never-won bidder count                |
| `emerging_bidders`     | New entrants with ≥2 participations              |
| `repeat_winners`       | Bidders with ≥3 wins                             |

---

## Deployment

### Streamlit Cloud
1. Push the `tender_dashboard/` folder to your repo.
2. Set the **Main file path** to `tender_dashboard/app.py`.
3. Add environment variables in the Secrets section.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY tender_dashboard/ .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

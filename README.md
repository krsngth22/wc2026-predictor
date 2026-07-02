# World Cup 2026 Prediction Model

Predicts match scores, corners, and player stats for the 2026 World Cup,
with a Streamlit dashboard on top. Built over 5 days.

## Status
- [x] Day 1: Project scaffold, historical data pipeline, Elo rating engine
- [ ] Day 2: Score prediction model (Dixon-Coles Poisson)
- [ ] Day 3: Corners model + player stats model
- [ ] Day 4: Streamlit dashboard
- [ ] Day 5: Backtesting, calibration, polish

## Setup

```bash
cd wc2026_predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Day 1 — run this now

Pulls ~49,000 historical international matches (1872-present, free/no-auth dataset)
and computes Elo strength ratings for every national team.

```bash
python -m src.data_collection
python -m src.elo_ratings
```

Expected output: `data/processed/elo_ratings_current.csv` (team strength rankings)
and `data/processed/elo_history.csv` (match-by-match rating history, used as
model features from Day 2 onward).

Sanity check: the script prints how many 2026 World Cup matches it found in the
historical dataset. If that GitHub dataset hasn't caught up with this tournament's
live results yet, that's fine — Day 2 pulls fresh 2026 fixtures/results separately.

## Day 2 preview — what you'll need

Get a **free API-Football key** (RapidAPI) for live 2026 fixtures, corners, and
player box scores — this is the one paid-optional step:
1. https://rapidapi.com/api-sports/api/api-football → "Subscribe" to the free plan
2. Copy your key, then set it as an environment variable:
   ```bash
   export API_FOOTBALL_KEY="your_key_here"     # Windows: set API_FOOTBALL_KEY=your_key_here
   ```
Free tier gives ~100 requests/day, which is enough for daily fixture/stats pulls
during the tournament.

## Project structure

```
wc2026_predictor/
├── config.py                  # paths, API keys, Elo constants
├── requirements.txt
├── data/
│   ├── raw/                   # untouched downloads
│   └── processed/             # cleaned CSVs used by models
├── src/
│   ├── data_collection.py     # Day 1: historical results
│   ├── elo_ratings.py         # Day 1: team strength ratings
│   ├── live_data.py           # Day 2: API-Football live fixtures/stats
│   ├── score_model.py         # Day 2: Dixon-Coles score predictor
│   ├── corners_model.py       # Day 3
│   ├── player_model.py        # Day 3
│   └── backtest.py            # Day 5
└── app.py                     # Day 4: Streamlit dashboard
```

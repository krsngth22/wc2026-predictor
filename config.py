"""
Central configuration for the World Cup 2026 prediction model.
Fill in API_FOOTBALL_KEY once you've grabbed a free key (see README.md, Day 2 section).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env if present, so API_FOOTBALL_KEY doesn't need to be exported manually
except ImportError:
    pass

# ---- Paths ----
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ---- Data sources ----
# Free, no-auth historical international results (1872-present), maintained on GitHub.
HIST_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
HIST_GOALSCORERS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
HIST_SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

# SofaScore (RapidAPI, provider "apidojo") - needed for live 2026 fixtures, corners,
# lineups, player stats. https://rapidapi.com/apidojo/api/sofascore
SOFASCORE_API_KEY = os.environ.get("SOFASCORE_API_KEY", "")
SOFASCORE_HOST = "sofascore.p.rapidapi.com"
SOFASCORE_BASE = "https://sofascore.p.rapidapi.com"

# SofaScore internal IDs for this tournament - confirmed via the playground:
# it's labeled "World Championship" internally, not "FIFA World Cup"
SOFASCORE_TOURNAMENT_ID = 16
SOFASCORE_SEASON_ID = 58210   # World Cup 2026

# 2026 World Cup competition metadata
WC2026_YEAR = 2026
WC2026_HOSTS = ["United States", "Mexico", "Canada"]

# ---- Elo model constants (World Football Elo Ratings methodology) ----
ELO_INITIAL = 1500
ELO_HOME_ADVANTAGE = 100  # rating points added to home team when not on neutral ground
ELO_TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60,
    "Copa América": 50,
    "UEFA Euro": 50,
    "FIFA World Cup qualification": 40,
    "Confederations Cup": 40,
    "African Cup of Nations": 40,
    "AFC Asian Cup": 40,
    "CONCACAF Championship": 40,
    "Friendly": 20,
}
ELO_DEFAULT_WEIGHT = 30  # for tournaments not explicitly listed above

# ---- Team name reconciliation ----
# The historical dataset (Day 1, from GitHub) and SofaScore (Day 3, live data) don't
# always use the same team names. This maps SofaScore's naming -> the historical
# dataset's naming, so a team's real Elo/attack/defense rating is used instead of
# silently falling back to a flat league-average. Confirmed/likely mismatches for
# 2026 World Cup teams - run `python -m src.check_team_names` to verify the full list
# against your actual data and catch anything missing here.
TEAM_NAME_ALIASES = {
    "USA": "United States",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curacao",
    "South Korea": "South Korea",
    "DR Congo": "DR Congo",
}

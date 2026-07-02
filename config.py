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

# API-Football (RapidAPI) - needed for live 2026 fixtures, corners, lineups, player stats.
# Free tier: https://rapidapi.com/api-sports/api/api-football  (~100 req/day free)
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_FOOTBALL_HOST = "api-football-v1.p.rapidapi.com"
API_FOOTBALL_BASE = "https://api-football-v1.p.rapidapi.com/v3"

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

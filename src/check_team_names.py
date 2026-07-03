"""
Diagnostic - verifies every 2026 World Cup team name (from SofaScore) resolves
correctly to a team in the historical training data (from the Day 1 dataset).

Run this any time you're unsure whether a prediction is using a team's real
rating or silently falling back to league-average due to a naming mismatch:

    python -m src.check_team_names

If it finds unmatched teams, it suggests close matches (via fuzzy string
matching) so you can add them to TEAM_NAME_ALIASES in config.py.
"""
import sys
import difflib
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, TEAM_NAME_ALIASES
from src.score_model import DixonColesModel


def main():
    model_path = DATA_PROCESSED / "dixon_coles_model.json"
    fixtures_path = DATA_PROCESSED / "wc2026_fixtures.csv"

    if not model_path.exists():
        raise FileNotFoundError("Run `python -m src.score_model` first.")
    if not fixtures_path.exists():
        raise FileNotFoundError("Run `python -m src.live_data` first.")

    model = DixonColesModel.load(model_path)
    historical_teams = set(model.teams)

    fixtures = pd.read_csv(fixtures_path)
    # only real national teams, not placeholders like "W85" for undecided knockout slots
    real_fixtures = fixtures[~fixtures["home_team_disabled"] & ~fixtures["away_team_disabled"]]
    live_teams = sorted(set(real_fixtures["home_team"]) | set(real_fixtures["away_team"]))

    print(f"Checking {len(live_teams)} live 2026 World Cup team names against "
          f"{len(historical_teams)} teams in the historical training data...\n")

    unmatched = []
    for team in live_teams:
        if team in historical_teams:
            continue
        alias = TEAM_NAME_ALIASES.get(team)
        if alias and alias in historical_teams:
            continue
        unmatched.append(team)

    if not unmatched:
        print("All teams resolve correctly. No naming mismatches found.")
        return

    print(f"{len(unmatched)} team(s) NOT resolving to their real historical rating "
          f"(currently falling back to league-average):\n")
    for team in unmatched:
        suggestions = difflib.get_close_matches(team, historical_teams, n=3, cutoff=0.6)
        if suggestions:
            print(f"  '{team}' -> possible matches: {suggestions}")
        else:
            print(f"  '{team}' -> no close match found in historical data "
                  f"(may genuinely be a new/rare team name)")

    print("\nTo fix: add the correct mapping to TEAM_NAME_ALIASES in config.py, e.g.")
    if unmatched:
        example = unmatched[0]
        print(f'    "{example}": "<correct historical name>",')


if __name__ == "__main__":
    main()

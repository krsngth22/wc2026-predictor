"""
One-time patch: restores a fixture that got orphaned by the pagination-window
bug (fixed in live_data.py) before the fix was applied. This uses real data
already confirmed from the API earlier - no new request needed.

Run once:
    python patch_fixture.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from config import DATA_PROCESSED

RECOVERED_FIXTURE = {
    "fixture_id": 15186908,
    "date": "2026-06-25T20:00:00",
    "status": "finished",
    "round": "",
    "home_team": "Curaçao",
    "away_team": "Côte d'Ivoire",
    "home_team_disabled": False,
    "away_team_disabled": False,
    "home_score": 0,
    "away_score": 2,
}


def main():
    path = DATA_PROCESSED / "wc2026_fixtures.csv"
    df = pd.read_csv(path)

    if RECOVERED_FIXTURE["fixture_id"] in df["fixture_id"].values:
        print("Fixture already present - nothing to patch.")
        return

    df = pd.concat([df, pd.DataFrame([RECOVERED_FIXTURE])], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"Patched. Finished count now: {(df['status'] == 'finished').sum()}")
    print("Re-run `python -m src.live_data` next - it'll find this fixture's "
          "stats/lineups already cached and fold it into your CSVs at zero API cost.")


if __name__ == "__main__":
    main()

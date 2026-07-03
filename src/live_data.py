"""
Live 2026 World Cup data - Day 3 (SofaScore API).

Pulls this tournament's actual fixtures, team match statistics (incl. corners),
and player match statistics. Requires a free SofaScore API key (RapidAPI,
provider "apidojo") set as SOFASCORE_API_KEY in your .env file.

Key efficiency win confirmed against the real API: matches/get-lineups returns
EVERY player's full match stats in ONE call, so we never need to query
player-by-player (which would burn through any free-tier quota instantly on
a ~22-player match). Cost per finished match: 2 requests total
(team stats + lineups).

This script caches every response to disk and is safe to re-run repeatedly -
it always picks up right where it left off.

Usage:
    python -m src.live_data                  # default budget: 85 requests this run
    python -m src.live_data --budget 50
"""
import sys
import time
import json
import argparse
from pathlib import Path

import requests
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import (DATA_RAW, DATA_PROCESSED, SOFASCORE_API_KEY, SOFASCORE_HOST,
                    SOFASCORE_BASE, SOFASCORE_TOURNAMENT_ID, SOFASCORE_SEASON_ID)

REQUEST_DELAY = 3.0  # seconds between calls - adjust up if you hit rate-limit errors

FIXTURE_STATS_DIR = DATA_RAW / "fixture_stats"
FIXTURE_LINEUPS_DIR = DATA_RAW / "fixture_lineups"
FIXTURE_STATS_DIR.mkdir(parents=True, exist_ok=True)
FIXTURE_LINEUPS_DIR.mkdir(parents=True, exist_ok=True)


def _headers():
    if not SOFASCORE_API_KEY:
        raise RuntimeError(
            "SOFASCORE_API_KEY not set. Copy .env.example to .env and paste your "
            "RapidAPI key in, then try again."
        )
    return {"x-rapidapi-host": SOFASCORE_HOST, "x-rapidapi-key": SOFASCORE_API_KEY}


def _get(endpoint: str, params: dict, max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{SOFASCORE_BASE}/{endpoint}", headers=_headers(),
                                params=params, timeout=25)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                wait = 3 * attempt
                print(f"    [{endpoint} attempt {attempt} failed ({type(e).__name__}) - "
                      f"retrying in {wait}s...]")
                time.sleep(wait)
    raise RuntimeError(f"Request to {endpoint} failed after {max_retries} attempts: {last_error}")


def _flatten_event(ev: dict) -> dict:
    home, away = ev["homeTeam"], ev["awayTeam"]
    hs, as_ = ev.get("homeScore", {}) or {}, ev.get("awayScore", {}) or {}
    return {
        "fixture_id": ev["id"],
        "date": pd.to_datetime(ev["startTimestamp"], unit="s").isoformat(),
        "status": ev["status"]["type"],  # "finished", "notstarted", "inprogress"
        "round": (ev.get("roundInfo") or {}).get("name", ""),
        "home_team": home["name"], "away_team": away["name"],
        # disabled=True means a placeholder like "W85" (winner of a match not yet played)
        "home_team_disabled": home.get("disabled", False),
        "away_team_disabled": away.get("disabled", False),
        "home_score": hs.get("current"), "away_score": as_.get("current"),
    }


def fetch_fixtures() -> pd.DataFrame:
    print("Fetching finished + upcoming fixtures...")
    rows = []
    for endpoint in ("tournaments/get-last-matches", "tournaments/get-next-matches"):
        page = 0
        while page < 10:  # safety cap - should only need a handful of pages
            print(f"  {endpoint}: requesting page {page}...")
            t0 = time.time()
            data = _get(endpoint, {"tournamentId": SOFASCORE_TOURNAMENT_ID,
                                   "seasonId": SOFASCORE_SEASON_ID, "page": page})
            events = data.get("events", [])
            print(f"    -> got {len(events)} events in {time.time()-t0:.1f}s")
            if not events:
                break
            rows.extend(_flatten_event(e) for e in events)
            time.sleep(REQUEST_DELAY)
            if not data.get("hasNextPage", False):
                break
            page += 1

    df = pd.DataFrame(rows).drop_duplicates(subset="fixture_id")
    out_path = DATA_PROCESSED / "wc2026_fixtures.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> {len(df)} unique fixtures saved to {out_path}")
    print(f"  -> {(df['status'] == 'finished').sum()} finished so far")
    return df


def fetch_fixture_stats(fixture_id: int) -> dict:
    data = _get("matches/get-statistics", {"matchId": fixture_id})
    (FIXTURE_STATS_DIR / f"{fixture_id}.json").write_text(json.dumps(data, indent=2))
    time.sleep(REQUEST_DELAY)
    return data


def fetch_fixture_lineups(fixture_id: int) -> dict:
    data = _get("matches/get-lineups", {"matchId": fixture_id})
    (FIXTURE_LINEUPS_DIR / f"{fixture_id}.json").write_text(json.dumps(data, indent=2))
    time.sleep(REQUEST_DELAY)
    return data


def parse_team_stats(fixture_id: int, stats_data: dict, fdate: str,
                     home_team: str, away_team: str) -> list:
    """Flattens the 'ALL' period across every stat group into one row per team."""
    all_period = next((p for p in stats_data.get("statistics", []) if p.get("period") == "ALL"), None)
    if not all_period:
        return []

    stat_map = {}
    for group in all_period.get("groups", []):
        for item in group.get("statisticsItems", []):
            stat_map[item["key"]] = (item.get("homeValue"), item.get("awayValue"))

    def _row(team_name, idx):
        return {
            "fixture_id": fixture_id, "date": fdate, "team": team_name,
            "corners": stat_map.get("cornerKicks", (None, None))[idx],
            "shots_total": stat_map.get("totalShotsOnGoal", (None, None))[idx],
            "shots_on_target": stat_map.get("shotsOnGoal", (None, None))[idx],
            "possession": stat_map.get("ballPossession", (None, None))[idx],
            "fouls": stat_map.get("fouls", (None, None))[idx],
            "yellow_cards": stat_map.get("yellowCards", (None, None))[idx],
            "red_cards": stat_map.get("redCards", (None, None))[idx],
        }

    return [_row(home_team, 0), _row(away_team, 1)]


def parse_player_stats(fixture_id: int, lineup_data: dict, fdate: str,
                       home_team: str, away_team: str) -> list:
    rows = []
    # Note: each player's "teamId" field in this API is their CLUB team, not their
    # national team - so we get the national team from the home/away grouping instead.
    for side, team_name in (("home", home_team), ("away", away_team)):
        for p in lineup_data.get(side, {}).get("players", []):
            info, stats = p["player"], p.get("statistics", {}) or {}
            minutes = stats.get("minutesPlayed")
            if not minutes:
                continue  # did not play - stats block is empty/meaningless
            rows.append({
                "fixture_id": fixture_id, "date": fdate, "team": team_name,
                "player": info["name"], "minutes": minutes,
                "position": info.get("position"), "rating": stats.get("rating"),
                "shots_total": stats.get("totalShots", 0),
                "shots_on_target": stats.get("onTargetScoringAttempt", 0),
                "goals": stats.get("goals", 0), "assists": stats.get("goalAssist", 0),
                "yellow_cards": stats.get("yellowCard", 0), "red_cards": stats.get("redCard", 0),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=85,
                        help="Max stats/lineup requests to spend this run (default 85)")
    args = parser.parse_args()

    fixtures_df = fetch_fixtures()
    finished = fixtures_df[fixtures_df["status"] == "finished"].sort_values("date")
    print(f"\n{len(finished)} finished matches so far.")

    team_stats_rows, player_stats_rows = [], []
    spent = 0
    failed_fixtures = []

    for _, frow in finished.iterrows():
        fid = int(frow["fixture_id"])
        fdate, home, away = frow["date"], frow["home_team"], frow["away_team"]
        stats_cache = FIXTURE_STATS_DIR / f"{fid}.json"
        lineups_cache = FIXTURE_LINEUPS_DIR / f"{fid}.json"

        if not stats_cache.exists():
            if spent >= args.budget:
                break
            print(f"  Fetching stats for fixture {fid} ({home} vs {away})...")
            try:
                fetch_fixture_stats(fid)
            except RuntimeError as e:
                print(f"    [SKIPPING fixture {fid} stats - {e}]")
                failed_fixtures.append(fid)
                spent += 1
                continue
            spent += 1
        if not lineups_cache.exists():
            if spent >= args.budget:
                break
            print(f"  Fetching lineups for fixture {fid}...")
            try:
                fetch_fixture_lineups(fid)
            except RuntimeError as e:
                print(f"    [SKIPPING fixture {fid} lineups - {e}]")
                failed_fixtures.append(fid)
                spent += 1
                continue
            spent += 1

        if stats_cache.exists():
            team_stats_rows += parse_team_stats(fid, json.loads(stats_cache.read_text()), fdate, home, away)
        if lineups_cache.exists():
            player_stats_rows += parse_player_stats(fid, json.loads(lineups_cache.read_text()), fdate, home, away)

    team_df = pd.DataFrame(team_stats_rows)
    player_df = pd.DataFrame(player_stats_rows)
    team_df.to_csv(DATA_PROCESSED / "wc2026_team_match_stats.csv", index=False)
    player_df.to_csv(DATA_PROCESSED / "wc2026_player_match_stats.csv", index=False)

    pulled = len({p.stem for p in FIXTURE_STATS_DIR.glob("*.json")})
    print(f"\nSpent ~{spent} stats/lineup requests this run (plus a few for the fixture list).")
    print(f"Team match stats: {len(team_df)} rows -> data/processed/wc2026_team_match_stats.csv")
    print(f"Player match stats: {len(player_df)} rows -> data/processed/wc2026_player_match_stats.csv")
    if failed_fixtures:
        print(f"\n{len(failed_fixtures)} fixture(s) failed after retries and were skipped "
              f"(will retry automatically next run): {failed_fixtures}")
    remaining = len(finished) - pulled
    if remaining > 0:
        print(f"\n{remaining} finished fixtures still need pulling - re-run this script to continue "
              f"(check your RapidAPI dashboard for remaining quota if you're on a daily-limited plan).")
    else:
        print("\nAll finished fixtures pulled. Re-run periodically as new matches complete.")


if __name__ == "__main__":
    main()

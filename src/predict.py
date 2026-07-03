"""
Get a full prediction for any matchup: score, corners, and top player props,
all in one command. This is the fastest way to check a game right now -
no dashboard needed.

Usage:
    python -m src.predict "Portugal" "Croatia"
    python -m src.predict "USA" "Mexico" --home-advantage      # only if NOT neutral venue
    python -m src.predict "France" "England" --top-players 5   # show more players per team
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, TEAM_NAME_ALIASES
from src.score_model import DixonColesModel, matrix_to_markets
from src.corners_model import CornersModel
from src.player_model import predict_player_stats


def _resolves_in_training_data(team: str, model: DixonColesModel) -> bool:
    return team in model.attack or TEAM_NAME_ALIASES.get(team) in model.attack


def print_score_section(home_team: str, away_team: str, neutral: bool):
    model_path = DATA_PROCESSED / "dixon_coles_model.json"
    if not model_path.exists():
        print("\nSCORE: not available - run `python -m src.score_model` first.")
        return

    model = DixonColesModel.load(model_path)
    for team in (home_team, away_team):
        if not _resolves_in_training_data(team, model):
            print(f"Note: '{team}' not found in training data (check spelling) - "
                  f"using league-average fallback for that team.")

    matrix, lam, mu_ = model.predict_score_matrix(home_team, away_team, neutral=neutral)
    markets = matrix_to_markets(matrix)

    print("\nSCORE")
    print(f"  Expected goals: {lam:.2f} - {mu_:.2f}")
    print(f"  1X2: {home_team} {markets['home_win']:.1%} | Draw {markets['draw']:.1%} "
          f"| {away_team} {markets['away_win']:.1%}")
    print(f"  Over 2.5: {markets['over_2_5']:.1%} | Under 2.5: {markets['under_2_5']:.1%} "
          f"| BTTS Yes: {markets['btts_yes']:.1%}")
    print("  Most likely scorelines:")
    for (h, a), p in markets["top_scorelines"]:
        print(f"    {h}-{a}: {p:.1%}")


def print_corners_section(home_team: str, away_team: str):
    model_path = DATA_PROCESSED / "corners_model.json"
    if not model_path.exists():
        print("\nCORNERS: not available yet - run `python -m src.live_data` "
              "then `python -m src.corners_model`.")
        return

    model = CornersModel.load(model_path)
    pred, probs = model.over_under_probs(home_team, away_team)

    print("\nCORNERS")
    print(f"  Expected corners: {pred['expected_home_corners']} - {pred['expected_away_corners']} "
          f"(total {pred['expected_total_corners']})")
    if pred["home_matches_sample"] < 3 or pred["away_matches_sample"] < 3:
        print("  (note: limited sample so far for one or both teams - leaning on tournament average)")
    for k, v in probs.items():
        print(f"  {k}: {v:.1%}")


def print_player_props_section(home_team: str, away_team: str, top_n: int, assumed_minutes: float):
    rates_path = DATA_PROCESSED / "player_rates.csv"
    if not rates_path.exists():
        print("\nPLAYER PROPS: not available yet - run `python -m src.live_data` "
              "then `python -m src.player_model`.")
        return

    rates = pd.read_csv(rates_path)
    for team in (home_team, away_team):
        team_players = rates[rates["team"] == team].sort_values(
            "shots_on_target_per90_adj", ascending=False)
        print(f"\nPLAYER PROPS - {team} (top {top_n} by shots-on-target rate, "
              f"assuming {assumed_minutes:.0f} mins)")
        if team_players.empty:
            print("  No player data yet for this team.")
            continue
        for _, row in team_players.head(top_n).iterrows():
            p = predict_player_stats(rates, row["player"], expected_minutes=assumed_minutes)
            print(f"  {p['player']} ({p['position']}): "
                  f"shots {p['expected_shots_total']} (on target {p['expected_shots_on_target']}, "
                  f"{p['prob_at_least_1_shots_on_target']:.0%} chance of 1+) | "
                  f"goals {p['expected_goals']} ({p['prob_at_least_1_goals']:.0%} to score)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--home-advantage", action="store_true",
                        help="Set this if the match is NOT on a neutral venue "
                             "(most World Cup matches ARE neutral except host nations)")
    parser.add_argument("--top-players", type=int, default=3,
                        help="How many top players per team to show props for (default 3)")
    parser.add_argument("--minutes", type=float, default=75,
                        help="Assumed minutes played for player props (default 75)")
    args = parser.parse_args()
    neutral = not args.home_advantage

    print("=" * 64)
    print(f"{args.home_team} vs {args.away_team}  "
          f"({'neutral venue' if neutral else 'home advantage applied'})")
    print("=" * 64)

    print_score_section(args.home_team, args.away_team, neutral)
    print_corners_section(args.home_team, args.away_team)
    print_player_props_section(args.home_team, args.away_team, args.top_players, args.minutes)


if __name__ == "__main__":
    main()

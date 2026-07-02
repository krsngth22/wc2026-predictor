"""
Elo rating engine - Day 1.

Implements the World Football Elo Ratings methodology (as used by eloratings.net):
- K factor scaled by competition importance (World Cup weighted highest)
- Goal-difference multiplier (bigger wins move rating more)
- Home advantage bonus (skipped for neutral-venue matches)

This produces a single "team strength" number per team that feeds directly into
the Day 2 score model as the attack/defense prior.

Usage:
    python -m src.elo_ratings
"""
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, ELO_INITIAL, ELO_HOME_ADVANTAGE, ELO_TOURNAMENT_WEIGHT, ELO_DEFAULT_WEIGHT


def goal_diff_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8


def expected_score(elo_home: float, elo_away: float, neutral: bool) -> float:
    dr = elo_home - elo_away
    if not neutral:
        dr += ELO_HOME_ADVANTAGE
    return 1 / (10 ** (-dr / 400) + 1)


def build_elo_ratings(results: pd.DataFrame, start_rating: float = ELO_INITIAL) -> tuple[dict, pd.DataFrame]:
    """
    Runs the full match history through the Elo update rule chronologically.
    Returns (final_ratings_dict, history_dataframe) where history has a row per
    match with pre-match ratings for both teams (useful as model features later).
    """
    ratings = defaultdict(lambda: start_rating)
    history_rows = []

    for row in results.itertuples(index=False):
        home, away = row.home_team, row.away_team
        elo_home, elo_away = ratings[home], ratings[away]

        neutral = bool(getattr(row, "neutral", False))
        we_home = expected_score(elo_home, elo_away, neutral)

        gd = row.home_score - row.away_score
        w_home = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)

        weight = ELO_TOURNAMENT_WEIGHT.get(row.tournament, ELO_DEFAULT_WEIGHT)
        k = weight * goal_diff_multiplier(gd)

        delta = k * (w_home - we_home)
        ratings[home] = elo_home + delta
        ratings[away] = elo_away - delta

        history_rows.append({
            "date": row.date, "home_team": home, "away_team": away,
            "home_score": row.home_score, "away_score": row.away_score,
            "elo_home_pre": elo_home, "elo_away_pre": elo_away,
            "elo_home_post": ratings[home], "elo_away_post": ratings[away],
            "expected_home_result": we_home, "tournament": row.tournament,
        })

    history_df = pd.DataFrame(history_rows)
    return dict(ratings), history_df


def main():
    clean_path = DATA_PROCESSED / "international_results_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found. Run `python -m src.data_collection` first."
        )

    results = pd.read_csv(clean_path, parse_dates=["date"])
    ratings, history = build_elo_ratings(results)

    ratings_df = (
        pd.DataFrame(sorted(ratings.items(), key=lambda x: -x[1]), columns=["team", "elo"])
        .reset_index(drop=True)
    )
    ratings_df.index += 1
    ratings_df.index.name = "rank"

    ratings_out = DATA_PROCESSED / "elo_ratings_current.csv"
    history_out = DATA_PROCESSED / "elo_history.csv"
    ratings_df.to_csv(ratings_out)
    history.to_csv(history_out, index=False)

    print(f"Computed Elo ratings for {len(ratings_df)} teams -> {ratings_out}")
    print(f"Full match-by-match Elo history -> {history_out}\n")
    print("Top 20 teams by current Elo:")
    print(ratings_df.head(20).to_string())


if __name__ == "__main__":
    main()

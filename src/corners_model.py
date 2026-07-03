"""
Corners prediction model - Day 3.

Each team gets an empirical "corners won per match" and "corners conceded per
match" rate from this tournament's actual games so far (far more relevant than
old club-league data would be), shrunk toward the tournament-wide average to
handle the fact each team only has a handful of matches. Combined
multiplicatively - same idea as the score model's attack/defense ratios - to
get expected corners for any matchup, then modeled as Poisson for over/under
lines.

Run after src/live_data.py has pulled at least some match stats:
    python -m src.corners_model
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED

SHRINKAGE_K = 4  # pseudo-matches of league-average blended into each team's rate


def load_team_match_stats() -> pd.DataFrame:
    path = DATA_PROCESSED / "wc2026_team_match_stats.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.live_data` first to pull match stats.")
    df = pd.read_csv(path).dropna(subset=["corners"])
    df["corners"] = df["corners"].astype(float)
    return df


def add_opponent_corners(df: pd.DataFrame) -> pd.DataFrame:
    """For each team-match row, attach the opponent's corners from the same fixture."""
    out = df.copy()
    out["corners_conceded"] = np.nan
    for fid, group in out.groupby("fixture_id"):
        if len(group) != 2:
            continue
        i0, i1 = group.index.tolist()
        out.loc[i0, "corners_conceded"] = group.loc[i1, "corners"]
        out.loc[i1, "corners_conceded"] = group.loc[i0, "corners"]
    return out.dropna(subset=["corners_conceded"])


def build_team_rates(df: pd.DataFrame):
    league_avg_for = df["corners"].mean()
    league_avg_against = df["corners_conceded"].mean()

    grouped = df.groupby("team").agg(
        matches=("corners", "count"),
        avg_corners_for=("corners", "mean"),
        avg_corners_against=("corners_conceded", "mean"),
    ).reset_index()

    grouped["corners_for_adj"] = (
        grouped["matches"] * grouped["avg_corners_for"] + SHRINKAGE_K * league_avg_for
    ) / (grouped["matches"] + SHRINKAGE_K)
    grouped["corners_against_adj"] = (
        grouped["matches"] * grouped["avg_corners_against"] + SHRINKAGE_K * league_avg_against
    ) / (grouped["matches"] + SHRINKAGE_K)

    return grouped, league_avg_for, league_avg_against


class CornersModel:
    def __init__(self, team_rates: pd.DataFrame, league_avg_for: float, league_avg_against: float):
        self.rates = team_rates.set_index("team")
        self.league_avg_for = league_avg_for
        self.league_avg_against = league_avg_against

    def _team_rate(self, team: str):
        if team in self.rates.index:
            row = self.rates.loc[team]
            return row["corners_for_adj"], row["corners_against_adj"], int(row["matches"])
        return self.league_avg_for, self.league_avg_against, 0

    def predict(self, home_team: str, away_team: str) -> dict:
        h_for, h_against, h_n = self._team_rate(home_team)
        a_for, a_against, a_n = self._team_rate(away_team)

        exp_home = self.league_avg_for * (h_for / self.league_avg_for) * (a_against / self.league_avg_against)
        exp_away = self.league_avg_for * (a_for / self.league_avg_for) * (h_against / self.league_avg_against)

        return {
            "home_team": home_team, "away_team": away_team,
            "expected_home_corners": round(float(exp_home), 2),
            "expected_away_corners": round(float(exp_away), 2),
            "expected_total_corners": round(float(exp_home + exp_away), 2),
            "home_matches_sample": h_n, "away_matches_sample": a_n,
        }

    def over_under_probs(self, home_team: str, away_team: str, lines=(8.5, 9.5, 10.5, 11.5)):
        pred = self.predict(home_team, away_team)
        total_lambda = pred["expected_total_corners"]
        probs = {}
        for line in lines:
            threshold = int(np.floor(line))
            p_under_or_eq = poisson.cdf(threshold, total_lambda)
            probs[f"over_{line}"] = round(1 - p_under_or_eq, 3)
            probs[f"under_{line}"] = round(p_under_or_eq, 3)
        return pred, probs

    def save(self, path):
        payload = {"league_avg_for": self.league_avg_for, "league_avg_against": self.league_avg_against,
                   "team_rates": self.rates.reset_index().to_dict(orient="records")}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        return cls(pd.DataFrame(payload["team_rates"]), payload["league_avg_for"], payload["league_avg_against"])


def main():
    df = load_team_match_stats()
    df = add_opponent_corners(df)
    n_fixtures = df["fixture_id"].nunique()
    print(f"Building corners model from {n_fixtures} finished matches ({len(df)} team-match rows)")
    if n_fixtures < 10:
        print("Note: limited data so far - predictions lean heavily on the tournament-wide "
              "average until you've run live_data.py a few more times.")

    team_rates, league_for, league_against = build_team_rates(df)
    model = CornersModel(team_rates, league_for, league_against)

    out_path = DATA_PROCESSED / "corners_model.json"
    model.save(out_path)
    print(f"Model saved -> {out_path}")
    print(f"\nLeague-wide average: {league_for:.2f} corners per team per match "
          f"({league_for * 2:.1f} total per match)")

    print("\nTop 10 teams by adjusted corner-winning rate:")
    print(team_rates.sort_values("corners_for_adj", ascending=False)
          [["team", "matches", "avg_corners_for", "corners_for_adj"]].head(10).to_string(index=False))

    if len(team_rates) >= 2:
        t1, t2 = team_rates.iloc[0]["team"], team_rates.iloc[1]["team"]
        pred, probs = model.over_under_probs(t1, t2)
        print(f"\nExample: {t1} vs {t2}")
        print(f"  Expected corners: {pred['expected_home_corners']} - {pred['expected_away_corners']} "
              f"(total {pred['expected_total_corners']})")
        for k, v in probs.items():
            print(f"  {k}: {v:.1%}")


if __name__ == "__main__":
    main()

"""
Player stats prediction model - Day 3.

For each player: per-90-minute rates for shots, shots on target, goals, and
cards, computed from this tournament's actual matches and shrunk toward their
position's average (since most players only have a handful of WC 2026 matches
so far). Converts to expected values and simple prop probabilities (e.g.
"at least 1 shot on target") for any assumed minutes played.

Run after src/live_data.py has pulled at least some match stats:
    python -m src.player_model
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import poisson

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED

SHRINKAGE_MINUTES = 180  # ~2 full matches of position-average blended into each player's rate
STAT_COLUMNS = ["shots_total", "shots_on_target", "goals", "yellow_cards"]


def load_player_match_stats() -> pd.DataFrame:
    path = DATA_PROCESSED / "wc2026_player_match_stats.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.live_data` first to pull match stats.")
    df = pd.read_csv(path)
    df = df.dropna(subset=["minutes"])
    df = df[df["minutes"] > 0].copy()
    for col in STAT_COLUMNS:
        df[col] = df[col].fillna(0)
    df["position"] = df["position"].fillna("Unknown")
    return df


def build_player_rates(df: pd.DataFrame) -> pd.DataFrame:
    agg_kwargs = {"matches": ("fixture_id", "nunique"), "minutes": ("minutes", "sum")}
    for c in STAT_COLUMNS:
        agg_kwargs[f"{c}_total"] = (c, "sum")
    agg = df.groupby(["player", "team", "position"]).agg(**agg_kwargs).reset_index()

    for col in STAT_COLUMNS:
        league_per90 = agg[f"{col}_total"].sum() / agg["minutes"].sum() * 90
        pos_totals = agg.groupby("position").agg(tot=(f"{col}_total", "sum"), mins=("minutes", "sum"))
        pos_totals["rate"] = (pos_totals["tot"] / pos_totals["mins"] * 90).fillna(league_per90)
        pos_map = pos_totals["rate"].to_dict()

        agg[f"{col}_pos_avg"] = agg["position"].map(pos_map).fillna(league_per90)
        raw_per90 = agg[f"{col}_total"] / agg["minutes"] * 90
        agg[f"{col}_per90_adj"] = (
            agg["minutes"] * raw_per90 + SHRINKAGE_MINUTES * agg[f"{col}_pos_avg"]
        ) / (agg["minutes"] + SHRINKAGE_MINUTES)

    return agg


def predict_player_stats(player_rates: pd.DataFrame, player_name: str, expected_minutes: float = 75):
    row = player_rates[player_rates["player"] == player_name]
    if row.empty:
        return None
    row = row.iloc[0]
    result = {"player": player_name, "team": row["team"], "position": row["position"],
             "matches_played_so_far": int(row["matches"]), "assumed_minutes": expected_minutes}
    for col in STAT_COLUMNS:
        expected = float(row[f"{col}_per90_adj"]) * (expected_minutes / 90)
        result[f"expected_{col}"] = round(expected, 2)
        result[f"prob_at_least_1_{col}"] = round(1 - poisson.pmf(0, expected), 3)
    return result


def main():
    df = load_player_match_stats()
    n_fixtures = df["fixture_id"].nunique()
    print(f"Building player stats model from {n_fixtures} finished matches ({df['player'].nunique()} players)")
    if n_fixtures < 10:
        print("Note: limited data so far - rates lean heavily on position averages until "
              "you've run live_data.py a few more times.")

    player_rates = build_player_rates(df)
    out_path = DATA_PROCESSED / "player_rates.csv"
    player_rates.to_csv(out_path, index=False)
    print(f"Player rates saved -> {out_path}")

    print("\nTop 10 players by adjusted shots-on-target per 90:")
    print(player_rates.sort_values("shots_on_target_per90_adj", ascending=False)
          [["player", "team", "position", "matches", "shots_on_target_per90_adj"]]
          .head(10).to_string(index=False))

    if len(player_rates) > 0:
        top = player_rates.sort_values("shots_on_target_per90_adj", ascending=False).iloc[0]["player"]
        pred = predict_player_stats(player_rates, top, expected_minutes=90)
        print(f"\nExample prediction for {top} (assuming 90 mins):")
        for k, v in pred.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

"""
Data collection - Day 1.

Pulls the free, no-auth historical international football results dataset
(49k+ matches, 1872-present) and saves a cleaned local copy.

Run this once locally (needs internet):
    python -m src.data_collection
"""
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED, HIST_RESULTS_URL, HIST_SHOOTOUTS_URL


def download_csv(url: str, dest: Path) -> pd.DataFrame:
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    df = pd.read_csv(dest)
    print(f"  -> saved {len(df):,} rows to {dest}")
    return df


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, parse dates, drop incomplete rows."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["result"] = df["goal_diff"].apply(lambda d: "H" if d > 0 else ("A" if d < 0 else "D"))
    df = df.sort_values("date").reset_index(drop=True)
    return df


def merge_shootouts(results: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    """Flag matches decided on penalties (useful later for knockout-stage modeling)."""
    shootouts = shootouts.copy()
    shootouts["date"] = pd.to_datetime(shootouts["date"])
    key_cols = ["date", "home_team", "away_team"]
    shootouts_slim = shootouts[key_cols + ["winner"]].rename(columns={"winner": "penalty_winner"})
    merged = results.merge(shootouts_slim, on=key_cols, how="left")
    merged["went_to_penalties"] = merged["penalty_winner"].notna()
    return merged


def main():
    results_raw = download_csv(HIST_RESULTS_URL, DATA_RAW / "results.csv")
    shootouts_raw = download_csv(HIST_SHOOTOUTS_URL, DATA_RAW / "shootouts.csv")

    results = clean_results(results_raw)
    results = merge_shootouts(results, shootouts_raw)

    out_path = DATA_PROCESSED / "international_results_clean.csv"
    results.to_csv(out_path, index=False)
    print(f"\nClean dataset: {len(results):,} matches, {results['date'].min().date()} to {results['date'].max().date()}")
    print(f"Saved to {out_path}")

    # Quick sanity check: recent World Cup matches present?
    wc_recent = results[(results["tournament"] == "FIFA World Cup") & (results["date"] >= "2026-06-01")]
    print(f"\n2026 World Cup matches found in dataset: {len(wc_recent)}")
    if len(wc_recent) > 0:
        print(wc_recent[["date", "home_team", "away_team", "home_score", "away_score"]].tail(10).to_string(index=False))
    else:
        print("(Dataset may not be updated yet with live 2026 results - that's expected, "
              "Day 2 will pull those separately via API-Football.)")


if __name__ == "__main__":
    main()

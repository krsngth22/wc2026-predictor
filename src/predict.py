"""
Quick CLI to test the fitted model on any matchup.

Usage:
    python -m src.predict "Portugal" "Croatia"
    python -m src.predict "USA" "Mexico" --home-advantage    # if not a neutral venue
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED
from src.score_model import DixonColesModel, matrix_to_markets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--home-advantage", action="store_true",
                        help="Set this if the match is NOT on a neutral venue "
                             "(most World Cup matches ARE neutral except host nations)")
    args = parser.parse_args()

    model_path = DATA_PROCESSED / "dixon_coles_model.json"
    if not model_path.exists():
        raise FileNotFoundError("Model not found - run `python -m src.score_model` first.")

    model = DixonColesModel.load(model_path)
    neutral = not args.home_advantage

    for team in (args.home_team, args.away_team):
        if team not in model.attack:
            print(f"Note: '{team}' not found in training data (check spelling) - "
                  f"will use league-average fallback.")

    matrix, lam, mu_ = model.predict_score_matrix(args.home_team, args.away_team, neutral=neutral)
    markets = matrix_to_markets(matrix)

    print(f"\n{args.home_team} vs {args.away_team}  ({'neutral venue' if neutral else 'home advantage applied'})")
    print(f"Expected goals: {lam:.2f} - {mu_:.2f}")
    print(f"\n1X2:")
    print(f"  {args.home_team} win : {markets['home_win']:.1%}")
    print(f"  Draw          : {markets['draw']:.1%}")
    print(f"  {args.away_team} win : {markets['away_win']:.1%}")
    print(f"\nGoals:")
    print(f"  Over 2.5  : {markets['over_2_5']:.1%}")
    print(f"  Under 2.5 : {markets['under_2_5']:.1%}")
    print(f"  BTTS Yes  : {markets['btts_yes']:.1%}")
    print(f"\nMost likely scorelines:")
    for (h, a), p in markets["top_scorelines"]:
        print(f"  {h}-{a}: {p:.1%}")


if __name__ == "__main__":
    main()

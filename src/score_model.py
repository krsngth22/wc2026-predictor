"""
Score prediction model - Day 2.

Dixon-Coles bivariate Poisson model: the standard approach for football score
prediction. Each team gets an attack rating and a defense rating, fit by
maximum likelihood on historical results. Recent matches and higher-stakes
tournaments (World Cup > continental > qualifiers > friendlies) are weighted
more heavily. A small low-score correlation correction (the "tau" adjustment
from Dixon & Coles, 1997) fixes the tendency of plain independent-Poisson
models to misprice 0-0, 1-0, 0-1, 1-1 results.

Run after Day 1:
    python -m src.score_model
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED

XI = 0.0018            # time-decay rate per day (~1 year half-life)
REG_LAMBDA = 0.08       # L2 regularization on attack/defense (identifiability + shrinkage for thin data)
MAX_GOALS = 8           # scoreline grid size for prediction output

TOURNAMENT_IMPORTANCE = {
    "FIFA World Cup": 1.0,
    "FIFA World Cup qualification": 0.65,
    "Copa América": 0.85,
    "UEFA Euro": 0.85,
    "UEFA Euro qualification": 0.55,
    "UEFA Nations League": 0.6,
    "Confederations Cup": 0.7,
    "African Cup of Nations": 0.75,
    "AFC Asian Cup": 0.75,
    "CONCACAF Championship": 0.6,
    "Friendly": 0.35,
}
DEFAULT_IMPORTANCE = 0.5


def load_training_data(years_back: int = 12) -> pd.DataFrame:
    path = DATA_PROCESSED / "international_results_clean.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=365 * years_back)
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    return df


def compute_weights(df: pd.DataFrame) -> np.ndarray:
    latest = df["date"].max()
    days_ago = (latest - df["date"]).dt.days.values
    time_weight = np.exp(-XI * days_ago)
    importance = df["tournament"].map(TOURNAMENT_IMPORTANCE).fillna(DEFAULT_IMPORTANCE).values
    return time_weight * importance


class DixonColesModel:
    def __init__(self):
        self.teams = []
        self.attack = {}
        self.defense = {}
        self.home_adv = 0.0
        self.rho = 0.0
        self.mu = 0.0

    def fit(self, df: pd.DataFrame, weights: np.ndarray):
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        home_idx = df["home_team"].map(team_idx).astype(int).values
        away_idx = df["away_team"].map(team_idx).astype(int).values
        hg = df["home_score"].values.astype(float)
        ag = df["away_score"].values.astype(float)
        neutral = df["neutral"].fillna(False).astype(bool).values
        home_flag = (~neutral).astype(float)

        # Precompute masks for the tau low-score correction (fully vectorized -
        # this matters a lot: it's evaluated on every optimizer function call)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        def unpack(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv, rho, mu = params[2 * n], params[2 * n + 1], params[2 * n + 2]
            return attack, defense, home_adv, rho, mu

        def neg_log_likelihood(params):
            attack, defense, home_adv, rho, mu = unpack(params)
            log_lam = mu + attack[home_idx] + defense[away_idx] + home_adv * home_flag
            log_mu_ = mu + attack[away_idx] + defense[home_idx]
            lam = np.exp(log_lam)
            mu_ = np.exp(log_mu_)

            ll = weights * (poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu_))

            tau_vals = np.ones_like(lam)
            tau_vals[m00] = 1 - lam[m00] * mu_[m00] * rho
            tau_vals[m01] = 1 + lam[m01] * rho
            tau_vals[m10] = 1 + mu_[m10] * rho
            tau_vals[m11] = 1 - rho
            tau_vals = np.clip(tau_vals, 1e-6, None)
            ll += weights * np.log(tau_vals)

            reg = REG_LAMBDA * (np.sum(attack ** 2) + np.sum(defense ** 2))
            return -np.sum(ll) + reg

        x0 = np.zeros(2 * n + 3)
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                           options={"maxiter": 300})

        attack, defense, home_adv, rho, mu = unpack(result.x)
        self.teams = teams
        self.attack = dict(zip(teams, attack.tolist()))
        self.defense = dict(zip(teams, defense.tolist()))
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.mu = float(mu)
        return result

    def _strength(self, team: str):
        if team in self.attack:
            return self.attack[team], self.defense[team]
        return 0.0, 0.0  # unseen team -> league-average fallback

    def predict_score_matrix(self, home_team: str, away_team: str, neutral: bool = True,
                              max_goals: int = MAX_GOALS):
        a_h, d_h = self._strength(home_team)
        a_a, d_a = self._strength(away_team)
        home_bonus = 0.0 if neutral else self.home_adv
        lam = float(np.exp(self.mu + a_h + d_a + home_bonus))
        mu_ = float(np.exp(self.mu + a_a + d_h))

        goals = np.arange(0, max_goals + 1)
        p_home = poisson.pmf(goals, lam)
        p_away = poisson.pmf(goals, mu_)
        matrix = np.outer(p_home, p_away)

        for x in (0, 1):
            for y in (0, 1):
                if x == 0 and y == 0:
                    matrix[x, y] *= (1 - lam * mu_ * self.rho)
                elif x == 0 and y == 1:
                    matrix[x, y] *= (1 + lam * self.rho)
                elif x == 1 and y == 0:
                    matrix[x, y] *= (1 + mu_ * self.rho)
                elif x == 1 and y == 1:
                    matrix[x, y] *= (1 - self.rho)
        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()
        return matrix, lam, mu_

    def save(self, path):
        payload = {"teams": self.teams, "attack": self.attack, "defense": self.defense,
                   "home_adv": self.home_adv, "rho": self.rho, "mu": self.mu}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        m = cls()
        m.teams, m.attack, m.defense = payload["teams"], payload["attack"], payload["defense"]
        m.home_adv, m.rho, m.mu = payload["home_adv"], payload["rho"], payload["mu"]
        return m


def matrix_to_markets(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())

    totals = {}
    for i in range(n):
        for j in range(n):
            totals[i + j] = totals.get(i + j, 0.0) + matrix[i, j]
    over_2_5 = float(sum(p for t, p in totals.items() if t > 2.5))

    btts_yes = float(sum(matrix[i, j] for i in range(1, n) for j in range(1, n)))

    flat = sorted([((i, j), float(matrix[i, j])) for i in range(n) for j in range(n)],
                  key=lambda x: -x[1])

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over_2_5": over_2_5, "under_2_5": 1 - over_2_5,
        "btts_yes": btts_yes, "btts_no": 1 - btts_yes,
        "top_scorelines": flat[:5],
    }


def main():
    df = load_training_data(years_back=12)
    weights = compute_weights(df)
    print(f"Training Dixon-Coles model on {len(df):,} matches "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    print("Fitting... (this can take 1-3 minutes given the number of teams)")

    model = DixonColesModel()
    result = model.fit(df, weights)
    print(f"Converged: {result.success} | home advantage: {model.home_adv:.3f} "
          f"| rho: {model.rho:.3f}")

    model_path = DATA_PROCESSED / "dixon_coles_model.json"
    model.save(model_path)
    print(f"Model saved -> {model_path}")

    print("\n--- Example: Argentina vs France (neutral venue) ---")
    matrix, lam, mu_ = model.predict_score_matrix("Argentina", "France", neutral=True)
    markets = matrix_to_markets(matrix)
    print(f"Expected goals: {lam:.2f} - {mu_:.2f}")
    print(f"1X2: Home {markets['home_win']:.1%} | Draw {markets['draw']:.1%} | Away {markets['away_win']:.1%}")
    print(f"Over 2.5 goals: {markets['over_2_5']:.1%} | BTTS Yes: {markets['btts_yes']:.1%}")
    print("Most likely scorelines:")
    for (h, a), p in markets["top_scorelines"]:
        print(f"  {h}-{a}: {p:.1%}")


if __name__ == "__main__":
    main()

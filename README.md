# NBA Player Prop Predictor

An end-to-end machine learning system that predicts NBA player props (points, rebounds, assists) using XGBoost, with a PostgreSQL data pipeline, FastAPI backend, React dashboard, and AWS deployment.

## Live Demo

**URL**: http://3.18.81.118

**Demo credentials**:
- Username: `demo`
- Password: `nba2025`

## Tech Stack

- **Data**: Python, pandas, nba_api, PostgreSQL, Docker
- **ML**: XGBoost, scikit-learn, SHAP, MLflow, Optuna
- **Backend**: FastAPI, SQLAlchemy, Redis, JWT auth
- **Frontend**: React, TypeScript, Tailwind CSS, Recharts
- **DevOps**: Docker, AWS (EC2, RDS, ECR), GitHub Actions

## Project Structure

    nba-props/
    ├── src/
    │   ├── ingestion/      # ETL pipeline (fetch, transform, load, validate)
    │   ├── models/         # ML model training and inference
    │   └── api/            # FastAPI backend
    ├── frontend/           # React + TypeScript dashboard
    ├── tests/              # pytest test suite (64 tests)
    ├── docs/               # Architecture diagram and model card
    ├── notebooks/          # Jupyter exploration notebooks
    ├── data/               # Model artifacts
    └── logs/               # Pipeline log files

## Architecture

```
React Dashboard ──▶ Nginx ──▶ FastAPI ──▶ PostgreSQL (RDS)
                                    └──▶ Redis (cache)
                                    └──▶ XGBoost Models
```

See [docs/architecture.md](docs/architecture.md) for the full system diagram.

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+
- Node.js 20+
- WSL (Windows) or Linux/Mac

### Setup

1. Clone the repo:

```bash
git clone https://github.com/krsngth22/nba-props-predictor.git
cd nba-props-predictor
```

2. Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Start the database:

```bash
make up
```

4. Apply the schema:

```bash
python src/ingestion/schema.py
```

5. Seed the data:

```bash
make pipeline
```

6. Start the API:

```bash
uvicorn src.api.main:app --reload --port 8000
```

7. Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

## Make Commands

| Command | Description |
|---|---|
| `make up` | Start Docker containers |
| `make down` | Stop Docker containers |
| `make pipeline` | Run full data pipeline |
| `make pipeline-test` | Run pipeline with 3 players |
| `make test` | Run pytest suite |
| `make health` | Database health check |
| `make logs` | Tail pipeline logs |
| `make clean` | Remove cache files |

## Pipeline Architecture

```
NBA API → fetcher.py → transformer.py → validator.py → loader.py → PostgreSQL
```

- **fetcher.py** — pulls game logs from the NBA stats API with retry logic
- **transformer.py** — cleans and normalizes raw data into structured DataFrames
- **validator.py** — validates data quality before database insertion
- **loader.py** — bulk upserts data into PostgreSQL via psycopg2

## ML Architecture

```
PostgreSQL → features.py → trainer.py → tuner.py → predict.py
```

- **features.py** — engineers 39 features (rolling averages, lag features, efficiency metrics, opponent ratings)
- **trainer.py** — trains XGBoost models with TimeSeriesSplit cross-validation
- **tuner.py** — Optuna hyperparameter search (30 trials per model)
- **predict.py** — inference module for serving predictions
- **explainer.py** — SHAP feature importance and prediction explanations
- **backtest.py** — holdout evaluation simulating real prop betting

## Model Performance

| Metric | Points | Rebounds | Assists |
|---|---|---|---|
| MAE | 2.061 | 2.158 | 0.652 |
| Within 3 | 77.5% | 76.2% | 96.3% |
| Within 5 | 90.9% | 92.3% | 99.3% |
| Within 10 | 98.6% | 99.4% | 100.0% |
| Bet accuracy | 91.5% | 76.4% | 85.0% |

Models trained on 2 seasons of NBA data (44,142 player-game records) across 809 active players.
Tuned with Optuna (30 trials), tracked with MLflow, explained with SHAP.

### Top Features (Points Model)
1. `points_per_minute` — scoring efficiency
2. `minutes_played_roll_10` — 10-game average minutes
3. `minutes_played_roll_20` — 20-game average minutes
4. `points_roll_20` — 20-game scoring average
5. `true_shooting_pct` — shooting efficiency

## CI/CD

Every push to `main` automatically:
1. Runs the pytest test suite (64 tests)
2. Builds Docker images
3. Pushes to AWS ECR
4. Deploys to EC2

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/token` | Login and get JWT token |
| GET | `/health` | API health check |
| GET | `/players` | List players (search supported) |
| GET | `/players/{id}` | Get player by ID |
| GET | `/players/{id}/stats` | Get player game history |
| GET | `/predictions/{id}` | Get ML predictions for player |
| GET | `/predictions/{id}/explain/{target}` | Get SHAP explanation |

Full documentation available at `/docs` (Swagger UI).

## Author

[@krsngth22](https://github.com/krsngth22)

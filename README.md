# FinOpsia

FinOpsia is an API-first financial operations backend that ingests transaction data, classifies cash activity, persists account history, auto-trains account forecasters, and exposes the full workflow through a FastAPI service and CLI.

This project is not just a CRUD API with a model attached. It implements a proper model lifecycle:

- models are checked in memory first for fast reuse
- missing models are loaded from persisted Joblib artifacts on disk
- account forecasters can auto-train themselves from stored history when absent
- newly trained models are saved to disk and cached in memory for subsequent requests
- categorization stays intentionally strict and refuses silent retraining in production paths

## What The System Does

- Reads transaction CSVs and validates schema and business rules
- Persists users, accounts, and transactions to a local SQLite database
- Categorizes transactions with a trained CatBoost classifier
- Builds account-level balance forecasts from persisted transaction history
- Auto-trains missing forecasters when a forecast is requested
- Exposes auth, transactions, analytics, forecasting, model management, import/export, admin, and health APIs
- Supports scheduled automation with retry and backoff behavior
- Writes structured logs for operational visibility

## What Is Actually Impressive Here

- Policy-driven model persistence: categorizer and forecaster follow different runtime rules, enforced in one persistence layer instead of scattered ad hoc logic
- Multi-stage model resolution: in-memory cache first, then disk artifact lookup, then controlled lazy training when policy allows it
- Account-scoped forecasting: forecasters are isolated per account and tied back to verified account ownership
- Self-bootstrapping inference flow: forecast requests can create a missing forecaster from real historical data and persist it for later reuse
- Explicit production safety boundary: the categorizer never silently trains during normal inference because classification is treated as a controlled decision system
- API hardening: JWT auth, admin gating, request validation, rate limiting, security headers, and restricted schema access are already wired into the app
- Reproducible local environment: seeded demo users, sample transactions, local SQLite DB, persisted artifacts, and a CLI path for retraining and pipeline runs
- Operational plumbing included: scheduler, retry decorator with exponential backoff, health endpoints, and log access

## Core Architecture

FinOpsia is organized around five cooperating layers:

- ingestion: reads and validates incoming transaction data
- persistence: stores transactions and model artifacts
- machine learning: categorizer, forecaster, and model lifecycle management
- API and auth: secure FastAPI surface for users and admins
- automation and monitoring: scheduled tasks, retry logic, and structured logging

## Model Lifecycle

The model lifecycle is one of the strongest parts of the project.

### Resolution flow

When the app needs a model, it follows this order:

1. check the in-memory cache
2. if missing, check the saved artifact on disk
3. if still missing, enforce the policy for that model type
4. if lazy training is allowed, train the model, save it, and place it in memory

This behavior lives in the shared persistence layer, so the app does not duplicate model-loading rules across routes and commands.

### Categorizer policy

- global model
- explicitly retrained only
- if the artifact is missing, inference fails fast with a clear error

### Forecaster policy

- account-scoped model
- can auto-train from account transaction history
- saved as a per-account artifact
- cached in memory after load or training

That gives the backend a good balance:

- strict where decisions affect categorization quality
- flexible where advisory forecasting benefits from self-bootstrapping behavior

## Current Stack

- Python 3.11
- FastAPI + Uvicorn
- Pandas
- CatBoost
- Joblib
- APScheduler
- Loguru
- SQLite
- JWT auth with `python-jose`

## Project Layout

```text
FinOpsia/
|-- src/
|   |-- api/              # FastAPI app, auth, routes, schemas, rate limiting
|   |-- automation/       # scheduler, retry logic, background tasks
|   |-- core/             # shared helpers and env loading
|   |-- db/               # repository layer and DB access
|   |-- ingestion/        # CSV reading, validation, loading
|   |-- ml/               # categorizer, forecaster, model persistence
|   |-- monitoring/       # logging and alert utilities
|   |-- __main__.py       # CLI entrypoint
|   `-- runner.py         # end-to-end pipeline runner
|-- data/
|   |-- raw/              # sample input CSVs
|   |-- processed/        # generated processed CSV output
|   `-- finopsia.db       # local SQLite database
|-- logs/                 # runtime logs, including finopsia.log
|-- scripts/
|   `-- setup_db.py       # database bootstrap and demo seeding
|-- tests/
|-- requirements.txt
`-- README.md
```

## Quick Start

### 1. Create and activate the environment

```powershell
conda create -n finopsia python=3.11 -y
conda activate finopsia
pip install -r requirements.txt
```

### 2. Seed the local database

```powershell
python scripts/setup_db.py
```

This creates `data/finopsia.db` and seeds demo users, accounts, and transactions.

Demo credentials:

- `admin / AdminPass123!`
- `alice / AlicePass123!`
- `bob / BobPass123!`

### 3. Run the API

```powershell
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs:

- `http://127.0.0.1:8000/api/v1/docs`

### 4. Run the CLI pipeline

```powershell
python -m src run --csv data/raw/transactions.csv --user-id 2
```

Run with forecasting:

```powershell
python -m src run --csv data/raw/transactions.csv --user-id 2 --forecast --account-id 1 --forecast-days 7
```

## Retraining Commands

Train the categorizer from labeled data:

```powershell
python -m src retrain categorizer --csv path\to\labeled_transactions.csv
```

Train a forecaster for a specific account:

```powershell
python -m src retrain forecaster --account-id 1 --user-id 2
```

Persisted model artifacts live under `src/ml/artifacts/`.

## Auto-Train Forecasting

Forecasting is designed to stay operational with minimal manual setup:

- if a forecaster already exists in memory, it is reused immediately
- if it is not in memory but exists on disk, it is loaded and cached
- if it does not exist and policy allows it, the app trains it from account history
- once trained, the model is saved and reused on future requests

This means the system can move from "no forecaster exists for this account" to "forecast served from persisted artifact" within the normal application flow.

## API Surface

The FastAPI app currently exposes routes for:

- authentication and user management
- accounts
- transactions
- categorization
- forecasting
- model management
- analytics
- import/export
- admin operations
- health and log access

Useful endpoint:

- `GET /api/v1/health/logs`

## Security And Operations

The backend already includes operational and security controls that are worth calling out:

- JWT-based authentication
- admin-only route protection
- role-aware and ownership-aware access checks
- request body limits
- CORS configuration
- security headers
- rate limiting
- standardized error responses
- structured file and console logging
- scheduler support and retry with exponential backoff

## Important Local Files

Files the app depends on locally:

- `data/finopsia.db`
- `data/raw/transactions.csv`
- `src/ml/artifacts/categorizer.joblib` if you want categorization to work without retraining

Files that are generated and can usually be recreated:

- `logs/`
- `data/processed/transactions_processed.csv`
- `src/ml/artifacts/forecaster_*.joblib`
- pytest and Python cache directories

## Testing

Run the full suite:

```powershell
pytest -v
```

Run a focused module:

```powershell
pytest tests/test_forecaster.py -v
```

## Notes

- Environment variables are loaded from the project `.env` file by `src/core/env.py`
- Logs are written to `logs/finopsia.log` unless overridden
- The current repository is API-first; the older dashboard files are no longer part of the active runtime

## License

This project is licensed under the MIT License. See `LICENSE`.

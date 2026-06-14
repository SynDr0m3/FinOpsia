# FinOpsia

Financial operations backend that ingests transactions, classifies them with ML, and generates forecasts through a production-grade model lifecycle system.

---

## Features

- **Transaction ingestion** – CSV validation and persistence
- **ML categorization** – CatBoost classifier for transaction classification
- **Account forecasting** – Auto-trained Prophet models per account
- **JWT auth** – User management with role-based access control
- **API-first** – FastAPI with auto-docs at `/api/v1/docs`
- **Model lifecycle** – In-memory cache → disk artifacts → lazy training
- **Structured logging** – Full observability of decisions and errors
- **Retry with backoff** – Background task automation

---

## Architecture

Five layers handling data flow and model lifecycle:

| Layer           | Purpose                                      |
| --------------- | -------------------------------------------- |
| **Ingestion**   | CSV reading, validation                      |
| **Persistence** | Model caching, disk artifacts, lazy training |
| **ML**          | Categorizer and forecaster implementations   |
| **API**         | FastAPI routes, auth, rate limiting          |
| **Automation**  | Scheduler, retry logic, background tasks     |

Key design: Unified persistence layer enforces model policies in one place.

---

## Quick Start

### 1. Install

```bash
conda create -n finopsia python=3.11 -y
conda activate finopsia
pip install -r requirements.txt
```

### 2. Seed Database

```bash
python scripts/setup_db.py
```

Demo credentials: `admin / AdminPass123!`, `alice / AlicePass123!`, `bob / BobPass123!`

### 3. Run API

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://127.0.0.1:8000/api/v1/docs`

### 4. Run Pipeline

```bash
python -m src run --csv data/raw/transactions.csv --user-id 2 --forecast --account-id 1 --forecast-days 7
```

---

## Project Structure

```
src/
├── api/          # FastAPI routes, auth, schemas
├── ml/           # Categorizer, forecaster, model persistence
├── db/           # Data access layer
├── ingestion/    # CSV reading, validation
├── automation/   # Scheduler, retry logic
├── monitoring/   # Logging utilities
└── core/         # Config, environment

data/
├── raw/          # Input CSVs
└── finopsia.db   # SQLite database
```

---

## Testing

```bash
pytest -v
```

---

## License

MIT License – See `LICENSE`

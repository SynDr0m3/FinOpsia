"""
FinOpsia CLI entrypoint.

Available commands:

run:
    End-to-end inference pipeline:
        ingest → categorize → persist → (optional) forecast
    NOTE: Categorizer model MUST exist. No training allowed.

retrain:
    Backend-only model retraining:
        categorizer: Global model (requires labeled CSV)
        forecaster: Per-account model (uses DB history)

Golden Rule:
    Training is a decision. Inference is a service.
    Never mix the two silently.
"""

import argparse
from pathlib import Path
from loguru import logger

from runner import run_pipeline
from ml.persistence import train_and_save_model


def main():
    parser = argparse.ArgumentParser(
        prog="finopsia",
        description="FinOpsia Financial Operations Platform",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # =========================================================
    # run command
    # =========================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run ingestion, categorization, and optional forecasting",
    )

    run_parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/transactions.csv"),
        help="Path to raw transactions CSV",
    )

    run_parser.add_argument(
        "--forecast",
        action="store_true",
        help="Enable balance forecasting",
    )

    run_parser.add_argument(
        "--account-id",
        type=str,
        help="Account ID (required if forecasting)",
    )

    run_parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        help="Number of days to forecast",
    )

    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without persisting data",
    )

    # =========================================================
    # retrain command (BACKEND-ONLY)
    # =========================================================
    # NOTE: Retraining is a controlled backend operation.
    # Users should NOT retrain models directly.

    retrain_parser = subparsers.add_parser(
        "retrain",
        help="[Backend-only] Retrain ML models explicitly",
    )

    retrain_subparsers = retrain_parser.add_subparsers(
        dest="model",
        required=True,
    )

    # -------- categorizer retrain --------
    cat_parser = retrain_subparsers.add_parser(
        "categorizer",
        help="Retrain the global transaction categorizer (requires labeled data)",
    )

    cat_parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV containing human-labeled training data (description, category)",
    )

    # -------- forecaster retrain --------
    fore_parser = retrain_subparsers.add_parser(
        "forecaster",
        help="Retrain account-specific balance forecaster (uses DB history)",
    )

    fore_parser.add_argument(
        "--account-id",
        type=str,
        required=True,
        help="Account ID for which to retrain the forecaster",
    )

    args = parser.parse_args()

    # =========================================================
    # Command dispatch
    # =========================================================
    if args.command == "run":
        run_pipeline(
            csv_path=args.csv,
            forecast=args.forecast,
            account_id=args.account_id,
            forecast_days=args.forecast_days,
            dry_run=args.dry_run,
        )

    elif args.command == "retrain":
        if args.model == "categorizer":
            logger.info("[Backend] Retraining global categorizer model")

            # Load labeled training data from CSV
            import pandas as pd
            training_df = pd.read_csv(args.csv)
            logger.info(f"Loaded {len(training_df)} labeled samples from {args.csv}")

            train_and_save_model(
                model_type="categorizer",
                df=training_df,
            )
            logger.success("Categorizer retraining completed")

        elif args.model == "forecaster":
            logger.info(
                f"[Backend] Retraining forecaster for account {args.account_id}"
            )
            train_and_save_model(
                model_type="forecaster",
                account_id=args.account_id,
            )
            logger.success(f"Forecaster retraining completed for account {args.account_id}")


if __name__ == "__main__":
    main()

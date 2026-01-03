"""
Test file for the balance forecaster.

The forecaster uses Prophet to predict future account balances.
Unlike the categorizer, the forecaster CAN lazy-train if no model exists.

Run with: pytest tests/test_forecaster.py -v

Note: These tests require:
1. Database with transactions table populated (run test_classification.py first)
2. Account metadata table with accounts 1-4

Test flow:
1. Run test_train_categorizer.py to create the classifier model
2. Run test_classification.py to classify and save transactions to DB
3. Run this file to test forecasting for accounts 1-4
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.forecaster import (
    train_model,
    forecast_balance,
    build_daily_balance_series,
    InsufficientDataError,
    MIN_DAYS_REQUIRED,
)
from ml.persistence import get_model, train_and_save_model, MODEL_DIR, _MODEL_CACHE
from db.repositories import DB_PATH


# Test all account IDs that exist in transactions.csv
TEST_ACCOUNT_IDS = ["1", "2", "3", "4"]


@pytest.fixture
def check_database():
    """Ensure database exists before running tests."""
    if not DB_PATH.exists():
        pytest.skip(
            f"Database not found at {DB_PATH}. "
            "Run test_classification.py first to populate the database."
        )
    return True


@pytest.fixture
def clear_forecaster_cache():
    """Clear the model cache before tests to ensure fresh lazy training."""
    # Clear any cached forecaster models
    keys_to_remove = [k for k in _MODEL_CACHE if k.startswith("forecaster:")]
    for key in keys_to_remove:
        del _MODEL_CACHE[key]
    return True


class TestBuildBalanceSeries:
    """Tests for building the daily balance time series."""

    @pytest.mark.parametrize("account_id", TEST_ACCOUNT_IDS)
    def test_build_series_for_all_accounts(self, check_database, account_id):
        """Test building daily balance series for each account."""
        try:
            df = build_daily_balance_series(account_id)

            # Should have ds (date) and y (balance) columns
            assert "ds" in df.columns
            assert "y" in df.columns

            # Should have data
            assert len(df) > 0

            print(f"\nAccount {account_id} balance series:")
            print(f"  Shape: {df.shape}")
            print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")
            print(f"  Balance range: {df['y'].min():.2f} to {df['y'].max():.2f}")

        except InsufficientDataError:
            pytest.skip(f"Insufficient data for account {account_id}")


class TestForecasterTraining:
    """Tests for forecaster model training."""

    @pytest.mark.parametrize("account_id", TEST_ACCOUNT_IDS)
    def test_train_model_for_all_accounts(self, check_database, account_id):
        """Test training a forecaster model for each account."""
        try:
            model = train_model(account_id)

            # Should be a Prophet model
            assert hasattr(model, "predict")
            assert hasattr(model, "make_future_dataframe")

            print(f"\n✓ Forecaster for account {account_id} trained successfully")

        except InsufficientDataError:
            pytest.skip(f"Insufficient data for account {account_id}")

    @pytest.mark.parametrize("account_id", TEST_ACCOUNT_IDS)
    def test_train_and_save_all_accounts(self, check_database, account_id):
        """Test training and saving models for all accounts."""
        try:
            model = train_and_save_model(
                model_type="forecaster",
                account_id=account_id,
            )

            # Verify model was saved
            model_path = MODEL_DIR / f"forecaster_{account_id}.joblib"
            assert model_path.exists(), f"Model not saved to {model_path}"

            print(f"\n✓ Forecaster for account {account_id} saved to {model_path}")

        except InsufficientDataError:
            pytest.skip(f"Insufficient data for account {account_id}")


class TestForecasting:
    """Tests for balance forecasting."""

    @pytest.mark.parametrize("account_id", TEST_ACCOUNT_IDS)
    def test_forecast_7_days_all_accounts(self, check_database, account_id):
        """Test 7-day balance forecast for all accounts."""
        try:
            model = get_model("forecaster", account_id=account_id)
            forecast = forecast_balance(model, days_ahead=7)

            # Should have forecast columns
            assert "ds" in forecast.columns
            assert "yhat" in forecast.columns
            assert "yhat_lower" in forecast.columns
            assert "yhat_upper" in forecast.columns

            # Should have 7 rows
            assert len(forecast) == 7

            print(f"\nAccount {account_id} - 7-Day Forecast:")
            print("-" * 50)
            for _, row in forecast.iterrows():
                print(
                    f"  {row['ds'].strftime('%Y-%m-%d')}: "
                    f"{row['yhat']:,.2f} "
                    f"[{row['yhat_lower']:,.2f} - {row['yhat_upper']:,.2f}]"
                )

        except InsufficientDataError:
            pytest.skip(f"Insufficient data for account {account_id}")

    @pytest.mark.parametrize("account_id", TEST_ACCOUNT_IDS)
    def test_forecast_30_days_all_accounts(self, check_database, account_id):
        """Test 30-day balance forecast for all accounts."""
        try:
            model = get_model("forecaster", account_id=account_id)
            forecast = forecast_balance(model, days_ahead=30)

            assert len(forecast) == 30

            print(f"\nAccount {account_id} - 30-Day Forecast Summary:")
            print(f"  Start: {forecast['yhat'].iloc[0]:,.2f}")
            print(f"  End: {forecast['yhat'].iloc[-1]:,.2f}")
            print(f"  Min: {forecast['yhat'].min():,.2f}")
            print(f"  Max: {forecast['yhat'].max():,.2f}")

        except InsufficientDataError:
            pytest.skip(f"Insufficient data for account {account_id}")


class TestLazyTraining:
    """
    Test lazy training behavior for forecaster.
    
    This verifies that:
    1. Forecaster models are created automatically when missing
    2. Each account gets its own model
    3. Models are persisted to disk
    """

    def test_lazy_train_all_accounts(self, check_database, tmp_path, monkeypatch):
        """
        Forecaster should automatically train for ALL accounts when missing.
        This tests the core lazy training mechanism.
        """
        import ml.persistence as persistence

        # Use temp directory to simulate missing models
        monkeypatch.setattr(persistence, "MODEL_DIR", tmp_path)
        monkeypatch.setattr(persistence, "_MODEL_CACHE", {})

        trained_accounts = []
        skipped_accounts = []

        for account_id in TEST_ACCOUNT_IDS:
            try:
                # This should trigger lazy training
                model = get_model("forecaster", account_id=account_id)

                # Model should exist now
                model_path = tmp_path / f"forecaster_{account_id}.joblib"
                assert model_path.exists(), f"Model not saved for account {account_id}"

                trained_accounts.append(account_id)
                print(f"\n✓ Account {account_id}: Lazy training created model at {model_path}")

            except InsufficientDataError:
                skipped_accounts.append(account_id)
                print(f"\n⚠ Account {account_id}: Skipped (insufficient data)")

        # Summary
        print("\n" + "=" * 60)
        print("Lazy Training Summary:")
        print("=" * 60)
        print(f"  Successfully trained: {trained_accounts}")
        print(f"  Skipped (insufficient data): {skipped_accounts}")
        print(f"  Models created in: {tmp_path}")

        # At least one account should have been trained
        assert len(trained_accounts) > 0, "At least one account should have sufficient data"

    def test_models_are_cached_after_lazy_train(self, check_database, tmp_path, monkeypatch):
        """Verify models are cached in memory after lazy training."""
        import ml.persistence as persistence

        monkeypatch.setattr(persistence, "MODEL_DIR", tmp_path)
        fresh_cache = {}
        monkeypatch.setattr(persistence, "_MODEL_CACHE", fresh_cache)

        account_id = TEST_ACCOUNT_IDS[0]

        try:
            # First call - should lazy train
            model1 = get_model("forecaster", account_id=account_id)

            # Check cache
            cache_key = f"forecaster:{account_id}"
            assert cache_key in fresh_cache, "Model should be cached"

            # Second call - should hit cache
            model2 = get_model("forecaster", account_id=account_id)

            # Should be same object
            assert model1 is model2, "Second call should return cached model"

            print(f"\n✓ Model caching works correctly for account {account_id}")

        except InsufficientDataError:
            pytest.skip("Insufficient data")


class TestForecasterValidation:
    """Validation tests for forecaster."""

    def test_requires_account_id(self):
        """Forecaster requires an account_id."""
        with pytest.raises(ValueError, match="account_id is required"):
            get_model("forecaster", account_id=None)

    def test_invalid_account_raises_error(self, check_database):
        """Invalid account should raise appropriate error."""
        with pytest.raises(Exception):
            # Should fail during data fetch or training
            train_model("nonexistent_account_xyz")


class TestFullIntegrationAllAccounts:
    """
    Full integration test: Train and forecast for all accounts 1-4.
    
    This is the main test to verify the entire forecasting pipeline works.
    """

    def test_train_and_forecast_all_accounts(self, check_database, clear_forecaster_cache):
        """
        End-to-end test for all 4 accounts:
        1. Get/train model for each account (lazy training)
        2. Generate 7-day forecast
        3. Verify model files are created
        """
        results = {}

        print("\n" + "=" * 70)
        print("FULL INTEGRATION TEST: Forecasting for Accounts 1-4")
        print("=" * 70)

        for account_id in TEST_ACCOUNT_IDS:
            print(f"\n--- Account {account_id} ---")

            try:
                # Get model (will lazy train if needed)
                model = get_model("forecaster", account_id=account_id)

                # Verify model file exists
                model_path = MODEL_DIR / f"forecaster_{account_id}.joblib"
                model_exists = model_path.exists()

                # Generate forecast
                forecast = forecast_balance(model, days_ahead=7)

                results[account_id] = {
                    "status": "success",
                    "model_saved": model_exists,
                    "forecast_days": len(forecast),
                    "forecast_start": forecast["yhat"].iloc[0],
                    "forecast_end": forecast["yhat"].iloc[-1],
                }

                print(f"  ✓ Model loaded/trained")
                print(f"  ✓ Model file exists: {model_exists}")
                print(f"  ✓ 7-day forecast generated")
                print(f"    Start: {forecast['yhat'].iloc[0]:,.2f}")
                print(f"    End: {forecast['yhat'].iloc[-1]:,.2f}")

            except InsufficientDataError as e:
                results[account_id] = {
                    "status": "skipped",
                    "reason": "insufficient data",
                }
                print(f"  ⚠ Skipped: {e}")

            except Exception as e:
                results[account_id] = {
                    "status": "failed",
                    "error": str(e),
                }
                print(f"  ✗ Failed: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        successful = [k for k, v in results.items() if v["status"] == "success"]
        skipped = [k for k, v in results.items() if v["status"] == "skipped"]
        failed = [k for k, v in results.items() if v["status"] == "failed"]

        print(f"  Successful: {successful}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")

        # Assertions
        assert len(failed) == 0, f"Some accounts failed: {failed}"
        assert len(successful) > 0, "At least one account should succeed"

        print(f"\n✓ Integration test passed for {len(successful)} accounts")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test file for the full classification pipeline.

This tests the categorization process using both:
1. Rule-based categorization (keyword matching)
2. ML model fallback (CatBoost classifier)

The model must be trained first using test_train_categorizer.py

Run with: pytest tests/test_classification.py -v

Integration test flow:
1. Load transactions.csv and DROP the category column
2. Classify using rules + ML model
3. Save classified transactions to database
4. This prepares data for forecaster tests (accounts 1-4)
"""

import pytest
import pandas as pd
import sqlite3
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.categorizer import rule_based_category, predict
from ml.persistence import get_model, ModelNotFoundError, MODEL_DIR
from ingestion.loader import load_transactions, init_db, DB_PATH
from ingestion.validator import normalize_types, derive_features


# Paths
RAW_CSV = Path("data/raw/transactions.csv")
TEST_USER_ID = "test_user"

# Sample transactions WITHOUT category (for unit tests)
SAMPLE_TRANSACTIONS = [
    # Should match rules
    {"description": "Monthly Salary Payment", "direction": "outflow"},
    {"description": "Electricity Bill", "direction": "outflow"},
    {"description": "Office Rent", "direction": "outflow"},
    {"description": "Cash Sale", "direction": "inflow"},
    {"description": "Facebook Ads Campaign", "direction": "outflow"},
    {"description": "Inventory Restock", "direction": "outflow"},
    # May need ML fallback
    {"description": "Equipment repair", "direction": "outflow"},
    {"description": "Client consultation", "direction": "inflow"},
    {"description": "Annual subscription", "direction": "outflow"},
    # Edge cases
    {"description": "", "direction": "inflow"},  # Empty - should default
    {"description": None, "direction": "outflow"},  # None - should default
]


@pytest.fixture
def sample_df():
    """Create a DataFrame of sample transactions without categories."""
    return pd.DataFrame(SAMPLE_TRANSACTIONS)


@pytest.fixture
def categorizer_model():
    """
    Load the trained categorizer model.
    Skips test if model doesn't exist.
    """
    model_path = MODEL_DIR / "categorizer.joblib"
    if not model_path.exists():
        pytest.skip(
            "Categorizer model not found. "
            "Run test_train_categorizer.py first to train the model."
        )

    return get_model("categorizer")


@pytest.fixture
def unlabeled_transactions():
    """
    Load transactions from CSV and DROP the category column.
    This simulates real-world uncategorized transactions.
    """
    if not RAW_CSV.exists():
        pytest.skip(f"Transaction data not found at {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    # Drop category column to simulate uncategorized transactions
    if "category" in df.columns:
        df = df.drop(columns=["category"])

    return df


class TestRuleBasedCategorization:
    """Tests for rule-based categorization."""

    def test_salary_keywords(self):
        """Test salary-related keyword matching."""
        result = rule_based_category("monthly salary payment", "outflow")
        assert result == "Salaries"

        result = rule_based_category("bonus payment", "outflow")
        assert result == "Salaries"

    def test_rent_keywords(self):
        """Test rent-related keyword matching."""
        result = rule_based_category("office rent", "outflow")
        assert result == "Rent"

        result = rule_based_category("shop rent", "outflow")
        assert result == "Rent"

    def test_utilities_keywords(self):
        """Test utilities-related keyword matching."""
        result = rule_based_category("electricity bill", "outflow")
        assert result == "Utilities"

        result = rule_based_category("internet service", "outflow")
        assert result == "Utilities"

    def test_revenue_keywords(self):
        """Test revenue-related keyword matching."""
        result = rule_based_category("cash sale", "inflow")
        assert result == "Revenue"

        result = rule_based_category("pos transaction", "inflow")
        assert result == "Revenue"

    def test_empty_description_inflow(self):
        """Empty description with inflow should default to Revenue."""
        result = rule_based_category("", "inflow")
        assert result == "Revenue"

        result = rule_based_category(None, "inflow")
        assert result == "Revenue"

    def test_empty_description_outflow(self):
        """Empty description with outflow should default to Miscellaneous."""
        result = rule_based_category("", "outflow")
        assert result == "Miscellaneous"

        result = rule_based_category(None, "outflow")
        assert result == "Miscellaneous"

    def test_no_rule_match(self):
        """Unrecognized descriptions should return None (ML fallback)."""
        result = rule_based_category("random unknown expense", "outflow")
        assert result is None  # Will need ML fallback


class TestMLCategorization:
    """Tests for ML-based categorization."""

    def test_model_loads(self, categorizer_model):
        """Test that the categorizer model loads correctly."""
        assert categorizer_model is not None
        assert hasattr(categorizer_model, "predict")

    def test_model_predicts(self, categorizer_model):
        """Test that the model can make predictions."""
        test_df = pd.DataFrame({"description": ["salary payment"]})
        predictions = categorizer_model.predict(test_df)
        assert len(predictions) == 1
        print(f"\nPrediction for 'salary payment': {predictions[0]}")

    def test_model_batch_predict(self, categorizer_model):
        """Test batch predictions."""
        descriptions = ["rent payment", "electricity", "office supplies"]
        test_df = pd.DataFrame({"description": descriptions})
        predictions = categorizer_model.predict(test_df)
        assert len(predictions) == 3
        print(f"\nBatch predictions: {list(zip(descriptions, predictions))}")


class TestFullClassificationPipeline:
    """Tests for the complete classification pipeline (rules + ML)."""

    def test_predict_function(self, sample_df, categorizer_model):
        """Test the full predict function with rules + ML fallback."""
        result_df = predict(sample_df, categorizer_model)

        # Should have category column added
        assert "category" in result_df.columns

        # All rows should have a category
        assert result_df["category"].notna().all()

        # Print results for inspection
        print("\n" + "=" * 60)
        print("Classification Results:")
        print("=" * 60)
        for _, row in result_df.iterrows():
            desc = row["description"] if row["description"] else "(empty)"
            print(f"  {desc[:40]:<40} -> {row['category']}")

    def test_rules_take_priority(self, sample_df, categorizer_model):
        """Verify that rule-based categorization takes priority over ML."""
        # Create a transaction that definitely matches a rule
        test_df = pd.DataFrame([
            {"description": "monthly salary payment", "direction": "outflow"},
        ])

        result_df = predict(test_df, categorizer_model)

        # Should match the "Salaries" rule
        assert result_df.iloc[0]["category"] == "Salaries"

    def test_ml_fallback_works(self, sample_df, categorizer_model):
        """Test that ML fallback works for non-rule descriptions."""
        # Create a transaction that won't match any rule
        test_df = pd.DataFrame([
            {"description": "miscellaneous business expense xyz", "direction": "outflow"},
        ])

        result_df = predict(test_df, categorizer_model)

        # Should still get a category (from ML)
        assert result_df.iloc[0]["category"] is not None
        print(f"\nML fallback prediction: {result_df.iloc[0]['category']}")


class TestModelNotFoundBehavior:
    """Test behavior when model is missing."""

    def test_categorizer_raises_error_when_missing(self, tmp_path, monkeypatch):
        """
        Categorizer should raise ModelNotFoundError when model is missing.
        (It should NEVER lazy-train)
        """
        # Point to empty directory
        import ml.persistence as persistence
        monkeypatch.setattr(persistence, "MODEL_DIR", tmp_path)
        monkeypatch.setattr(persistence, "_MODEL_CACHE", {})

        with pytest.raises(ModelNotFoundError):
            get_model("categorizer")


class TestIntegrationClassifyAndPersist:
    """
    Integration tests: Load CSV → Remove category → Classify → Save to DB.
    
    This prepares data for forecaster tests with accounts 1-4.
    """

    def test_classify_full_dataset_and_save_to_db(
        self, unlabeled_transactions, categorizer_model
    ):
        """
        Full integration test:
        1. Load transactions.csv WITHOUT category
        2. Classify using rules + ML
        3. Save to database
        
        This sets up data for forecaster tests.
        """
        df = unlabeled_transactions

        # Verify category was dropped
        assert "category" not in df.columns, "Category should be removed"

        # Check we have transactions for accounts 1-4
        account_ids = df["account_id"].unique()
        print(f"\nAccounts in dataset: {sorted(account_ids)}")
        for acc_id in [1, 2, 3, 4]:
            assert acc_id in account_ids, f"Account {acc_id} not found in data"

        # Step 1: Classify transactions
        print(f"\nClassifying {len(df)} transactions...")
        classified_df = predict(df, categorizer_model)

        # Verify all rows got categories
        assert "category" in classified_df.columns
        assert classified_df["category"].notna().all()

        # Show category distribution
        category_counts = classified_df["category"].value_counts()
        print(f"\nCategory distribution after classification:")
        print(category_counts)

        # Step 2: Prepare for DB (normalize types and convert amounts)
        from ingestion.validator import normalize_types, derive_features

        # Add currency if missing (for amount conversion)
        if "currency" not in classified_df.columns:
            classified_df["currency"] = "NGN"

        classified_df = normalize_types(classified_df)
        classified_df = derive_features(classified_df)

        # Drop currency column (not in DB schema)
        classified_df = classified_df.drop(columns=["currency"], errors="ignore")

        # Step 3: Initialize DB and save
        init_db()

        # Also set up accounts table for forecaster
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    account_number TEXT UNIQUE NOT NULL CHECK(length(account_number) = 10 AND account_number GLOB '[0-9]*'),
                    starting_balance INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    last_updated DATE,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Insert sample accounts for forecaster tests
            sample_accounts = [
                ("1", 1, "0449371890", 50000000, "NGN", "2025-01-01"),
                ("2", 2, "0449371891", 75000000, "NGN", "2025-01-01"),
                ("3", 3, "0449371892", 100000000, "NGN", "2025-01-01"),
                ("4", 4, "0449371893", 25000000, "NGN", "2025-01-01"),
            ]
            conn.executemany("""
                INSERT OR REPLACE INTO accounts
                (account_id, user_id, account_number, starting_balance, currency, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, sample_accounts)
            conn.commit()
            print("\n✓ Account metadata table set up for accounts 1-4")

        # Clear existing transactions to avoid duplicates
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM transactions")
            conn.commit()
            print("Cleared existing transactions from DB")

        # Save classified transactions
        load_transactions(classified_df, user_id=TEST_USER_ID)
        print(f"Saved {len(classified_df)} classified transactions to DB")

        # Step 4: Verify data in DB
        with sqlite3.connect(DB_PATH) as conn:
            db_count = pd.read_sql(
                "SELECT COUNT(*) as count FROM transactions", conn
            ).iloc[0]["count"]

            # Check each account has data
            for acc_id in [1, 2, 3, 4]:
                acc_count = pd.read_sql(
                    f"SELECT COUNT(*) as count FROM transactions WHERE account_id = '{acc_id}'",
                    conn
                ).iloc[0]["count"]
                print(f"Account {acc_id}: {acc_count} transactions")
                assert acc_count > 0, f"Account {acc_id} should have transactions"

        assert db_count == len(classified_df)
        print(f"\n✓ Total transactions in DB: {db_count}")

    def test_verify_account_data_for_forecaster(self, categorizer_model):
        """
        Verify DB has sufficient data for forecaster tests.
        Run after test_classify_full_dataset_and_save_to_db.
        """
        if not DB_PATH.exists():
            pytest.skip("Database not found. Run classification test first.")

        with sqlite3.connect(DB_PATH) as conn:
            for acc_id in [1, 2, 3, 4]:
                df = pd.read_sql(
                    f"""
                    SELECT transaction_date, amount, direction, category
                    FROM transactions
                    WHERE account_id = '{acc_id}'
                    ORDER BY transaction_date
                    """,
                    conn
                )

                if len(df) == 0:
                    pytest.fail(f"No data for account {acc_id}")

                date_range = pd.to_datetime(df["transaction_date"])
                days_span = (date_range.max() - date_range.min()).days

                print(f"\nAccount {acc_id}:")
                print(f"  Transactions: {len(df)}")
                print(f"  Date range: {date_range.min().date()} to {date_range.max().date()}")
                print(f"  Days span: {days_span}")

                # Forecaster needs at least 90 days
                if days_span < 90:
                    print(f"  ⚠ Warning: Less than 90 days of data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

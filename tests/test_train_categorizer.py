"""
Test file for training the transaction categorizer model.

This script trains the CatBoost classifier using labeled transaction data
and saves it as a .joblib file in the ml/artifacts directory.

Run with: pytest tests/test_train_categorizer.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.categorizer import train_model
from ml.persistence import train_and_save_model, MODEL_DIR


RAW_CSV = Path("data/raw/transactions.csv")
TEST_USER_ID = "test_user"


@pytest.fixture
def labeled_training_data():
    """
    Load labeled transactions from raw CSV for training.
    The CSV must contain 'description' and 'category' columns.
    """
    if not RAW_CSV.exists():
        pytest.skip(f"Training data not found at {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    required_cols = {"description", "category"}
    if not required_cols.issubset(df.columns):
        pytest.fail(
            f"Training CSV must contain columns: {required_cols}. "
            f"Found: {df.columns.tolist()}"
        )

    # Filter out rows with missing values
    df = df.dropna(subset=["description", "category"])

    return df


def test_train_categorizer_model(labeled_training_data):
    """
    Train the categorizer model and verify it's saved correctly.
    """
    df = labeled_training_data

    # Train and save using persistence layer
    model = train_and_save_model(
        model_type="categorizer",
        df=df[["description", "category"]],
    )

    # Verify model was saved
    model_path = MODEL_DIR / "categorizer.joblib"
    assert model_path.exists(), f"Model not saved to {model_path}"

    # Verify model can make predictions (pass as DataFrame with 'description' column)
    test_descriptions = ["salary payment", "electricity bill", "office rent"]
    test_df = pd.DataFrame({"description": test_descriptions})
    predictions = model.predict(test_df)

    assert len(predictions) == 3, "Model should return 3 predictions"
    print(f"\nTest predictions: {list(zip(test_descriptions, predictions))}")


def test_train_categorizer_direct(labeled_training_data):
    """
    Test the train_model function directly (without persistence).
    """
    df = labeled_training_data[["description", "category"]]

    model = train_model(df)

    # Model should be a CatBoostClassifier
    assert hasattr(model, "predict"), "Model must have predict method"
    assert hasattr(model, "fit"), "Model must have fit method"

    # Test prediction (pass as DataFrame with 'description' column)
    test_df = pd.DataFrame({"description": ["monthly payroll"]})
    pred = model.predict(test_df)[0]
    print(f"\nPrediction for 'monthly payroll': {pred}")


def test_training_data_quality(labeled_training_data):
    """
    Validate training data quality before training.
    """
    df = labeled_training_data

    # Check we have enough data
    assert len(df) >= 100, f"Need at least 100 samples, got {len(df)}"

    # Check category distribution
    category_counts = df["category"].value_counts()
    print(f"\nCategory distribution:\n{category_counts}")

    # Ensure we have multiple categories
    assert len(category_counts) >= 2, "Need at least 2 categories for training"

    # Check for empty descriptions
    empty_desc = df["description"].isna().sum() + (df["description"] == "").sum()
    print(f"Empty descriptions: {empty_desc}")


if __name__ == "__main__":
    # Allow running directly for quick training
    import pandas as pd
    from ml.persistence import train_and_save_model

    print("Loading training data...")
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["description", "category"])

    print(f"Training with {len(df)} samples...")
    model = train_and_save_model(
        model_type="categorizer",
        df=df[["description", "category"]],
    )

    print("Categorizer model trained and saved successfully!")

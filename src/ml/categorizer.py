from typing import Optional
import pandas as pd
from loguru import logger
from catboost import CatBoostClassifier


# -----------------------------
# Rule-based categorization
# -----------------------------

KEYWORD_RULES = {
    "salary": ["salary", "wages", "payroll"],
    "rent": ["rent", "lease"],
    "transport": ["fuel", "diesel", "petrol", "uber", "bolt"],
    "utilities": ["electricity", "power", "water", "internet"],
}

EMPTY_DESCRIPTIONS = {"", "nil", "nill", "n/a", "none", "-"}


def normalize_text(text: str) -> str:
    return text.strip().lower()


def rule_based_category(
    description: Optional[str],
    direction: str,
) -> Optional[str]:
    """
    Apply rule-based categorization.
    Returns category if rule matches, else None.
    """
    if not description or normalize_text(description) in EMPTY_DESCRIPTIONS:
        if direction == "inflow":
            return "revenue"
        return "miscellaneous"

    desc = normalize_text(description)

    for category, keywords in KEYWORD_RULES.items():
        if any(keyword in desc for keyword in keywords):
            return category

    return None


# -----------------------------
# ML Model
# -----------------------------

def train_model(df: pd.DataFrame) -> CatBoostClassifier:
    """
    Train categorizer using labeled data.
    Expects columns: description, category
    """
    logger.info("Training transaction categorizer model")

    required = {"description", "category"}
    if not required.issubset(df.columns):
        raise ValueError("Training data must contain description and category")

    X = df["description"].astype(str)
    y = df["category"].astype(str)

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        verbose=False,
    )

    model.fit(X, y)
    logger.success("Categorizer training completed")

    return model


def predict(
    df: pd.DataFrame,
    model: CatBoostClassifier,
) -> pd.DataFrame:
    """
    Predict categories for transactions.
    Rules first, ML fallback.
    """
    logger.info("Starting transaction categorization")

    categories = []

    for _, row in df.iterrows():
        rule_category = rule_based_category(
            description=row.get("description"),
            direction=row.get("direction"),
        )

        if rule_category is not None:
            categories.append(rule_category)
        else:
            pred = model.predict([row["description"]])[0]
            categories.append(pred)

    df = df.copy()
    df["category"] = categories

    logger.success("Transaction categorization completed")
    return df

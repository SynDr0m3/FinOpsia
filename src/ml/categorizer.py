from typing import Optional
import pandas as pd
from monitoring.logger import logger
from catboost import CatBoostClassifier


# -----------------------------
# Rule-based categorization
# -----------------------------

KEYWORD_RULES = {
    # Salaries - employee compensation
    "Salaries": ["salary", "wages", "payroll", "bonus payment", "staff wages", "driver salary", "manager salary", "admin staff", "contractor fee"],
    
    # Rent - property/space payments
    "Rent": ["rent", "lease", "store rental", "shop rent", "co-working", "storage unit", "kiosk", "office rent"],
    
    # Utilities - services/bills
    "Utilities": ["electricity", "power", "water bill", "internet", "broadband", "dstv", "energy cost", "heating bill", "waste disposal", "sewage"],
    
    # Revenue - money coming in (be specific to avoid false matches)
    "Revenue": ["cash sale", "pos transaction", "retail sales", "customer transfer", "sales revenue", "online order", "consulting revenue", "refund received"],
    
    # Inventory - stock purchases
    "Inventory": ["goods received", "vendor purchase", "inventory restock", "supplier payment", "goods acquisition", "wholesale goods", "bulk product", "retail inventory"],
    
    # Marketing - advertising/promotion
    "Marketing": ["marketing", "banner design", "radio ad", "product launch", "pr services", "facebook ads", "brand campaign", "email marketing"],
    
    # Supplies - equipment/consumables
    "Supplies": ["tools & hardware", "merchandise restock", "medical supplies", "consumables", "safety equipment", "equipment purchase"],
    
    # Miscellaneous - catch-all (checked last)
    "Miscellaneous": ["accounting", "freight", "permit fee", "customs duty", "conference fee", "shipping cost", "vehicle maintenance", "fuel cost"],
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
            return "Revenue"
        return "Miscellaneous"

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

    # Prepare features as DataFrame with text column
    X = df[["description"]].astype(str)
    y = df["category"].astype(str)

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        verbose=False,
    )

    # Tell CatBoost that 'description' is a text feature
    model.fit(X, y, text_features=["description"])
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
            # Pass as DataFrame with 'description' column to match training format
            pred_input = pd.DataFrame({"description": [str(row["description"])]})
            pred = model.predict(pred_input)[0]
            categories.append(pred)

    df = df.copy()
    df["category"] = categories

    logger.success("Transaction categorization completed")
    return df

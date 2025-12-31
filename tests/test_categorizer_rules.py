import pytest

from ml.categorizer import rule_based_category


# -----------------------------
# Empty / Nil descriptions
# -----------------------------

def test_empty_description_inflow():
    assert rule_based_category("", "inflow") == "revenue"


def test_nil_description_outflow():
    assert rule_based_category("nil", "outflow") == "miscellaneous"


def test_none_description_outflow():
    assert rule_based_category(None, "outflow") == "miscellaneous"


def test_na_description_inflow():
    assert rule_based_category("N/A", "inflow") == "revenue"


# -----------------------------
# Keyword-based categorization
# -----------------------------

def test_salary_keyword():
    desc = "Staff salary for May"
    assert rule_based_category(desc, "outflow") == "salary"


def test_rent_keyword():
    desc = "Paid office rent"
    assert rule_based_category(desc, "outflow") == "rent"


def test_transport_keyword():
    desc = "Fuel purchase at station"
    assert rule_based_category(desc, "outflow") == "transport"


def test_utilities_keyword():
    desc = "Internet subscription payment"
    assert rule_based_category(desc, "outflow") == "utilities"


# -----------------------------
# No rule match
# -----------------------------

def test_no_rule_match_returns_none():
    desc = "Random transaction xyz"
    assert rule_based_category(desc, "outflow") is None

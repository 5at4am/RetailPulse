"""The stdlib loader and pandas must agree on the promotion column.

Previously the loader returned the string "None" while pd.read_csv returned NaN, so
`r["promotion"] is None` was False for 91.6% of rows depending on which loader you used.
"""
from collections import Counter

import pandas as pd

import retail_pulse as rp


def test_nulls_agree_between_loader_and_pandas(sales):
    df = pd.read_csv(rp.SALES_CSV, parse_dates=["date"])
    assert sum(1 for r in sales if r["promotion"] is None) == int(df["promotion"].isna().sum())


def test_loader_value_counts_match_pandas(sales):
    df = pd.read_csv(rp.SALES_CSV, parse_dates=["date"])
    loader_counts = Counter(r["promotion"] for r in sales)
    assert loader_counts[None] == int(df["promotion"].isna().sum())
    assert set(loader_counts) - {None} == set(df["promotion"].dropna().unique())


def test_is_promotion_agrees_with_promotion_under_both_loaders(sales):
    df = pd.read_csv(rp.SALES_CSV, parse_dates=["date"])
    assert all(r["is_promotion"] == (0 if r["promotion"] is None else 1) for r in sales)
    assert (df["is_promotion"] == df["promotion"].notna().astype(int)).all()


def test_tuple_mode_is_documented_as_uncast(sales):
    tuples = rp.load_sales(as_dicts=False)
    i = rp.SALES_COLUMNS.index("promotion")
    assert tuples[0][i] == "None", "tuple mode keeps the literal string, by design"
    assert isinstance(tuples[0][0], str), "tuple mode does not cast dates either"
    assert len(tuples) == len(sales)


def test_full_contract_validation_still_passes(sales):
    assert rp.validate_sales(sales) is True
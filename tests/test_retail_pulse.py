"""Negative tests for retail_pulse.validate_sales.

A validator that always passes is worthless, so each documented contract is broken
deliberately and the validator is required to catch it.

Each case mutates a row that genuinely satisfies the violation's precondition --
otherwise the mutation is a no-op and the test proves nothing. Only the rows up to and
including the mutation are validated, which keeps eight full-frame copies off a
250,000-row list.
"""
import pytest

import retail_pulse as rp


def _mutated(rows, idx, field, value):
    """A list holding rows[:idx] plus a broken copy of rows[idx]."""
    bad = rows[: idx + 1]
    bad[idx] = dict(rows[idx])
    bad[idx][field] = value
    return bad


@pytest.fixture(scope="module")
def index(sales):
    """The first row satisfying each violation's precondition."""
    return {
        "promo_flag": next(i for i, r in enumerate(sales) if r["is_promotion"] == 0),
        "stockout": next(i for i, r in enumerate(sales) if r["quantity_sold"] == 0),
        "plain": next(i for i, r in enumerate(sales) if r["promotion"] is None),
    }


def test_baseline_data_satisfies_every_contract(sales):
    assert rp.validate_sales(sales, sample=1000) is True


@pytest.mark.parametrize(
    "label,field,value,precondition",
    [
        ("is_promotion=1 but promotion=None", "is_promotion", 1, "promo_flag"),
        ("qty=0 but inventory>0", "inventory_level", 9, "stockout"),
        ("discount on a non-promoted line", "discount", 10.0, "plain"),
        ("sales_amount wrong", "sales_amount", 999.99, None),
        ("discount > unit_price", "discount", 99999.0, None),
        ("discount_pct inconsistent", "discount_pct", 55.0, None),
        ("negative quantity", "quantity_sold", -3, None),
        ("negative inventory", "inventory_level", -1, None),
    ],
)
def test_validator_catches_broken_contract(sales, index, label, field, value,
                                           precondition):
    idx = index[precondition] if precondition else 0
    with pytest.raises(AssertionError):
        rp.validate_sales(_mutated(sales, idx, field, value))
"""Pytest bootstrap: make the project root importable, and load the big frames once.

`tests/` imports the root-level `retail_pulse` module and `src.config`. Neither is on
`sys.path` by default once the tests live in their own directory, so add it here rather
than duplicating a sys.path hack in every test module.

The sales file is 250,000 rows and the panel 780,000, so they are session-scoped
fixtures -- loaded once for the whole run, and only if something asks for them.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sales():
    """All 250,000 sales line-items as a list of dicts."""
    import retail_pulse as rp
    return rp.load_sales()


@pytest.fixture(scope="session")
def panel():
    """All 780,000 weekly panel rows as a list of dicts."""
    import retail_pulse as rp
    return rp.load_panel()
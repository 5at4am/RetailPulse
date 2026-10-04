"""The committed screenshots have to be evidence, not decoration.

A folder of PNGs that nobody checks is worth exactly as much as whatever is wrong with it. The
failure this file exists to prevent is the one that actually happened twice while building it:

`chrome --screenshot` returned five byte-identical images of Streamlit's loading skeleton. Every
per-file check passed -- valid PNG, correct 1440x2400 dimensions, 200 kB of "content", no
exception on the page. The captures were reported as 5/5 usable. Only comparing the files
against each other showed that real pages differ, and identical captures mean nothing rendered.

So two rules are enforced here. The committed files must all differ from one another, and each
must contain text unique to its own page. The second rule matters more than it looks: Streamlit
resolves an unregistered route to Home instead of returning a 404, so a page that is simply
missing still renders completely and passes every check that looks at one file at a time. It
took unregistering all four pages to make that failure visible, and a single missing page would
have shipped unnoticed.

The captures themselves are not regenerated here. They come from `python -m src.screenshots`
against a running server, which needs a real browser; this file only holds the committed set to
the standard the script claims to meet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src import config

SCREENSHOT_DIR = config.REPORTS / "screenshots"
MANIFEST = SCREENSHOT_DIR / "screenshots.json"
PNG_MAGIC = bytes([137, 80, 78, 71, 13, 10, 26, 10])

EXPECTED_PAGES = (
    "Home",
    "Demand_Forecasting",
    "Customer_Segments",
    "Churn_Risk",
    "Inventory_Recommendations",
)


def load_manifest() -> dict:
    if not MANIFEST.exists():
        pytest.fail(
            f"{MANIFEST.relative_to(config.ROOT)} is missing. Regenerate with: "
            "python -m src.screenshots (needs `streamlit run app/Home.py` running)"
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


class TestManifest:
    def test_covers_all_five_pages(self):
        assert [p["page"] for p in load_manifest()["pages"]] == list(EXPECTED_PAGES)

    def test_every_capture_is_marked_usable(self):
        broken = [(p["page"], p["problem"]) for p in load_manifest()["pages"] if not p["ok"]]
        assert not broken, f"unusable captures: {broken}"

    def test_each_page_saw_its_own_text(self):
        # The manifest records this at capture time; re-asserting it here means a manifest
        # written by an older version of the script, before the check existed, cannot pass as
        # if it had been verified.
        missing = [p["page"] for p in load_manifest()["pages"]
                   if not p.get("saw_expected_text")]
        assert not missing, (
            f"no content verification recorded for {missing}; regenerate the screenshots "
            "with a script that checks each page rendered its own content"
        )

    def test_expected_text_is_not_left_blank(self):
        for entry in load_manifest()["pages"]:
            assert entry.get("expected_text"), (
                f"{entry['page']} has no expected text, so its content cannot be checked"
            )


class TestImages:
    def test_files_exist_and_are_real_pngs(self):
        for entry in load_manifest()["pages"]:
            path = SCREENSHOT_DIR / entry["file"]
            assert path.exists(), f"{entry['file']} is named in the manifest but absent"
            assert path.read_bytes()[:8] == PNG_MAGIC, f"{entry['file']} is not a PNG"

    def test_sizes_are_plausible_for_a_rendered_page(self):
        # The skeletons were ~5 kB. A real page with charts and a Plotly canvas is far larger,
        # so this floor catches a regression to the loading state without needing an image
        # decoder as a test dependency.
        for entry in load_manifest()["pages"]:
            assert entry["bytes"] > 50_000, (
                f"{entry['file']} is only {entry['bytes']} bytes, which is closer to a "
                "loading skeleton than to a rendered page"
            )

    def test_no_two_pages_are_byte_identical(self):
        # The check that would have caught the original bug. Identical files across pages mean
        # at most one page actually rendered.
        digests: dict[str, list[str]] = {}
        for entry in load_manifest()["pages"]:
            path = SCREENSHOT_DIR / entry["file"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digests.setdefault(digest, []).append(entry["file"])
        duplicates = {d: files for d, files in digests.items() if len(files) > 1}
        assert not duplicates, (
            "these captures are byte-identical, so the pages did not render differently: "
            f"{duplicates}"
        )

    def test_recorded_digests_match_the_files_on_disk(self):
        # Stops the manifest from being left describing an older set of images, which is what
        # makes a screenshot folder quietly untrustworthy.
        for entry in load_manifest()["pages"]:
            path = SCREENSHOT_DIR / entry["file"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert entry["sha256"] in (actual, actual[:len(entry["sha256"])]), (
                f"{entry['file']} no longer matches the digest recorded in the manifest; "
                "regenerate the screenshots"
            )


class TestThePagesActuallyExist:
    """The screenshots are only worth committing if the pages behind them are registered.

    A stale-but-valid PNG set outlives the bug that produced it. This is the check that says the
    navigation those five routes depend on is still declared, so nobody can delete it and leave
    five correct-looking images behind.
    """

    def test_home_declares_every_route(self):
        text = (config.ROOT / "app" / "Home.py").read_text(encoding="utf-8")
        assert "st.navigation" in text, (
            "app/Home.py no longer calls st.navigation, so the other four routes resolve to "
            "Home again -- the screenshots would then be five copies of one page"
        )
        for page in EXPECTED_PAGES[1:]:
            assert page in text, (
                f"{page}.py is not registered as a page in app/Home.py"
            )

    def test_pages_registered_as_files_exist(self):
        text = (config.ROOT / "app" / "Home.py").read_text(encoding="utf-8")
        for page in EXPECTED_PAGES[1:]:
            assert (config.ROOT / "app" / f"{page}.py").exists(), (
                f"app/{page}.py is registered but does not exist"
            )
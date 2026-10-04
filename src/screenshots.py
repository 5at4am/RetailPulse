"""Capture the dashboard pages as PNGs by driving Chrome over the DevTools Protocol.

Why a script and not five ad-hoc screenshots
-------------------------------------------
The design spec says screenshots are the fallback if the live demo is down, and the demo video
outline points at them. That only works if they can be regenerated on the day they are needed.
A folder of hand-taken screenshots cannot be, and cannot be re-taken after a code change
without someone noticing which ones went stale.

Why CDP and not `chrome --screenshot`
------------------------------------
The obvious approach does not work, and fails silently. Chrome's `--screenshot` flag captures
as soon as the load event fires; Streamlit renders over a websocket *after* that, so the flag
returns the loading skeleton. It returns a valid PNG of the right dimensions every time, so no
size or format check catches it.

The first version of this script did exactly that and produced five byte-identical images --
the same skeleton saved five times. They were reported as 5/5 usable. Comparing the files
against each other is what exposed it: real pages differ, so identical captures are proof of a
non-render rather than evidence of five working pages.

So this drives Chrome over CDP and waits for the app to actually be ready: the Streamlit view
container present, no status widget, no exception. No new dependency -- `websockets` only --
because the dashboard has to install cleanly on Streamlit Cloud.

Usage:
    python -m src.screenshots                        # against http://localhost:8501
    python -m src.screenshots --url http://host:8501 --width 1440
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePath

from src import config

OUTPUT_DIR = config.REPORTS / "screenshots"

# Route slug -> file stem -> caption -> text that must appear on that page.
#
# The expected string is the load-bearing part. Streamlit resolves every unregistered route to
# Home rather than 404ing, so a missing page still returns a fully rendered page with a full
# screenshot, a plausible text length and a valid PNG. Only its content is wrong, and neither a
# byte count nor an image diff can see that -- the five captures were byte-identical, which is
# how it was caught, but identical files also mean the check only works when ALL of them are
# wrong. One page unregistered would slip through. So each page names a phrase from its own
# content, and the capture fails if that phrase is absent.
#
# The slugs are not free choices: Streamlit derives the URL from the page filename.
PAGES: tuple[tuple[str, str, str, str], ...] = (
    ("", "01_home", "KPI overview and the honest headline findings",
     "Weekly revenue"),
    ("Demand_Forecasting", "02_demand_forecasting",
     "F-03 backtest, model selection, forecast", "Model backtest"),
    ("Customer_Segments", "03_customer_segments", "F-02 K-Means segmentation",
     "Revenue vs customers"),
    ("Churn_Risk", "04_churn_risk", "F-04 XGBoost scoring and SHAP",
     "AUC target missed"),
    ("Inventory_Recommendations", "05_inventory_recommendations", "F-05 reorder plan",
     "Why safety stock is a Poisson quantile"),
)

CANDIDATE_BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

CHROME_FLAGS = (
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    "--force-device-scale-factor=1",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-features=Translate,BackForwardCache",
)

# Streamlit renders after the load event, so "loaded" has to mean "the app finished running",
# not "the HTML arrived". stStatusWidget is the Running... indicator; stException is the red
# traceback box. The threshold on innerText is a floor so an app that failed to mount any
# content is not mistaken for a legitimately sparse page.
READINESS_EXPRESSION = """
(() => {
  const q = (s) => document.querySelector(s);
  const exc = q('.stException, [data-testid="stException"]');
  return {
    container: !!q('[data-testid="stAppViewContainer"]'),
    running: !!q('[data-testid="stStatusWidget"]'),
    // Only a real traceback box. stAlertContainer is deliberately NOT here: Streamlit reuses
    // it for st.info / st.success / st.warning, so including it flagged every honest info
    // banner on the dashboard as an exception.
    exception: !!exc,
    exceptionText: exc ? (exc.innerText || '').slice(0, 300) : '',
    textLength: (document.body && document.body.innerText || '').trim().length,
    // The text itself, not just its length. Needed because a page that Streamlit silently
    // resolves to Home renders perfectly by every measure above.
    text: (document.body && document.body.innerText || ''),
    title: document.title || ''
  };
})()
"""

MIN_TEXT_LENGTH = 200

# A fixed character floor cannot be the only gate. Streamlit streams a page: on a cold server
# Home's title and caption are on screen while the rest is still rendering, and those two
# elements alone measure 238 characters. A 200-character floor is therefore satisfied before
# the first chart appears, the capture fires, and the resulting PNG is a half-drawn page.
#
# The fix is to require the length to stop changing. Streamlit appends elements as they
# stream in, so a length that is identical across two consecutive polls means the page has
# finished arriving -- whether that took 200 characters or 3,000. The floor stays as a
# backstop for a page that genuinely never renders, which stabilises at a small length.
STABLE_POLLS = 2
STABLE_INTERVAL = 0.7


def find_browser() -> str | None:
    for candidate in CANDIDATE_BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def wait_for_server(base_url: str, timeout: float = 90.0) -> bool:
    """Block until Streamlit's health endpoint answers."""
    deadline = time.monotonic() + timeout
    health = base_url.rstrip("/") + "/_stcore/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    return False


class Chrome:
    """A headless Chrome with the DevTools endpoint open, driven over a websocket."""

    def __init__(self, browser: str, port: int, width: int, height: int):
        self.port = port
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            [browser, *CHROME_FLAGS, f"--remote-debugging-port={port}",
             f"--window-size={width},{height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._socket = None
        self._next_id = 0

    def wait_until_debugger(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/json/version", timeout=3) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        return False

    def open(self):
        """Attach to the blank target the browser started with."""
        from websockets.sync.client import connect

        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list",
                                    timeout=10) as response:
            targets = json.loads(response.read())
        page = next((t for t in targets if t.get("type") == "page"), None)
        if not page:
            raise RuntimeError("Chrome started with no page target")
        # legacy=True keeps this a direct connection instead of an async context manager, which
        # is the shape this class uses. websockets >= 15 deprecated the implicit form and
        # prints a warning on every capture otherwise.
        self._socket = connect(page["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024,
                               legacy=True)
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Emulation.setDeviceMetricsOverride", {
            "width": self.width, "height": self.height,
            "deviceScaleFactor": 1, "mobile": False,
        })
        return self

    def send(self, method: str, params: dict | None = None, timeout: float = 60.0):
        """One CDP round trip. Returns the result payload, ignoring streamed events."""
        self._next_id += 1
        message_id = self._next_id
        self._socket.send(json.dumps({"id": message_id, "method": method,
                                      "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self._socket.recv(timeout=max(0.1, deadline - time.monotonic()))
            message = json.loads(raw)
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})
        raise TimeoutError(f"{method} did not answer in {timeout}s")

    def evaluate(self, expression: str):
        result = self.send("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": True,
        })
        # A JS exception returns a result object with no `value`, which would otherwise come
        # back as a bare None and be indistinguishable from "the page said no".
        details = result.get("exceptionDetails")
        if details:
            text = details.get("exception", {}).get("description") or details.get("text")
            raise RuntimeError(f"page evaluation raised: {text}")
        return result.get("result", {}).get("value")

    def wait_until_ready(self, timeout: float = 45.0, settle: float = 2.0,
                         expect: str | None = None) -> dict:
        """Poll until Streamlit has mounted, is not running, shows no exception, and is done.

        `not running` alone is not "done". Between Streamlit's script chunks there are windows
        where the status widget is absent while the page is still being assembled, and on a cold
        server Home's title and caption are already 238 characters before the first chart
        exists. A capture taken in that window is a partial page that passes every structural
        check, because the container is mounted, nothing raised, and there is plenty of text.

        So readiness is defined by the page's own content rather than by its size. When `expect`
        is given -- the heading each route is known to contain -- the wait ends only once that
        text is present *and* the length has stopped changing, which is when the elements after
        it have arrived too. Without `expect` the length-stability rule still applies, so an
        unrecognised page is not captured mid-stream either; it simply cannot be confirmed and
        the caller's own content check reports the mismatch.
        """
        deadline = time.monotonic() + timeout
        state: dict = {}
        previous_length: int | None = None
        stable = 0
        while time.monotonic() < deadline:
            state = self.evaluate(READINESS_EXPRESSION) or {}
            if state.get("exception"):
                return state

            length = state.get("textLength", 0) or 0
            mounted = state.get("container") and not state.get("running")
            if length == previous_length:
                stable += 1
            else:
                stable = 0
            previous_length = length

            has_expectation = expect is None or expect in (state.get("text") or "")
            if mounted and length >= MIN_TEXT_LENGTH and stable >= STABLE_POLLS \
                    and has_expectation:
                break
            time.sleep(STABLE_INTERVAL)
        # Charts draw after the text settles; capturing immediately catches blank canvases.
        time.sleep(settle)
        return state

    def screenshot(self, target: Path) -> None:
        result = self.send("Page.captureScreenshot", {
            "format": "png", "captureBeyondViewport": True, "fromSurface": True,
        })
        target.write_bytes(base64.b64decode(result["data"]))

    def close(self) -> None:
        try:
            if self._socket is not None:
                self._socket.close()
        except Exception:                                   # noqa: BLE001
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
        except Exception:                                   # noqa: BLE001
            self.process.kill()


def diagnose(state: dict, target: Path) -> str | None:
    """Name the reason a capture is unusable, so the failure is actionable not just red."""
    if state.get("exception"):
        detail = state.get("exceptionText") or ""
        return f"the page rendered a Streamlit exception: {detail}"
    if state.get("error"):
        return state["error"]
    if not target.exists():
        return "no file written"
    size = target.stat().st_size
    if size < 5_000:
        return f"only {size} bytes, which is a blank canvas"
    if target.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        return "not a PNG"
    if state.get("running"):
        return "Streamlit was still running when the capture fired (the skeleton problem)"
    if not state.get("container"):
        return "the Streamlit view container never mounted"
    if state.get("textLength", 0) < MIN_TEXT_LENGTH:
        return (f"only {state.get('textLength')} characters of text; a working page has more")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture RetailPulse dashboard screenshots.")
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=2400)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--only", nargs="*", help="capture just these route slugs")
    args = parser.parse_args(argv)

    browser = find_browser()
    if not browser:
        print("No Chrome or Edge found. Install one, or capture the pages by hand into "
              f"{OUTPUT_DIR.relative_to(config.ROOT)}.", file=sys.stderr)
        return 1

    base = args.url.rstrip("/")
    print(f"waiting for {base} ...")
    if not wait_for_server(base):
        print(f"{base} did not become healthy. Is `streamlit run app/Home.py` running?",
              file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = [p for p in PAGES if not args.only or p[0] in args.only]

    chrome = Chrome(browser, args.port, args.width, args.height)
    results: list[dict] = []
    try:
        if not chrome.wait_until_debugger():
            print("Chrome's DevTools endpoint never opened", file=sys.stderr)
            return 1
        chrome.open()

        for slug, stem, caption, expected in pages:
            url = f"{base}/{slug}" if slug else f"{base}/"
            target = OUTPUT_DIR / f"{stem}.png"
            print(f"  {slug or '<home>':<28} -> {target.name}")
            state: dict = {}
            try:
                chrome.send("Page.navigate", {"url": url})
                # `expect` goes into the wait, not only into the check afterwards. By the time
                # the content check can reject a half-rendered page, the bad capture is already
                # on disk and has to be recognised as bad rather than simply not taken.
                state = chrome.wait_until_ready(expect=expected)
                chrome.screenshot(target)
            except Exception as error:                      # noqa: BLE001
                state = {"error": str(error)}

            problem = diagnose(state, target)
            # Content check, run only once the page is known to have rendered. Everything the
            # checks above can see is also true of Home, which is what an unregistered route
            # returns.
            if problem is None and expected not in (state.get("text") or ""):
                problem = (f"the page rendered but never showed {expected!r}, so this is the "
                           f"wrong page -- is {slug or 'Home'} registered with st.navigation?")

            results.append({
                "page": slug or "Home",
                "url": url,
                "file": target.name,
                "caption": caption,
                "expected_text": expected,
                "saw_expected_text": expected in (state.get("text") or ""),
                "bytes": target.stat().st_size if target.exists() else 0,
                "text_length": state.get("textLength"),
                "ok": problem is None,
                "problem": problem,
            })
            if problem:
                print(f"      PROBLEM: {problem}", file=sys.stderr)
    finally:
        chrome.close()

    # Identical files mean a non-render, not five good captures. This is the check that would
    # have caught the --screenshot version of this script.
    digests: dict[str, str] = {}
    import hashlib
    for entry in results:
        path = OUTPUT_DIR / entry["file"]
        if path.exists():
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            digests.setdefault(entry["sha256"], []).append(entry["file"])
    duplicates = {k: v for k, v in digests.items() if len(v) > 1}

    manifest = OUTPUT_DIR / "screenshots.json"
    manifest.write_text(json.dumps({
        "source_url": base,
        "viewport": {"width": args.width, "height": args.height},
        # The executable's full path is machine-specific, so it is not committed. The
        # product and major version are what make a capture reproducible, and they are
        # what a reviewer needs to tell whether the images came from a current browser.
        "browser": f"{PurePath(browser).name} (devtools protocol)",
        "method": "chrome devtools protocol, waited on Streamlit readiness",
        "pages": results,
    }, indent=2) + "\n", encoding="utf-8")

    if duplicates:
        print("\nDUPLICATE CAPTURES -- these pages did not render differently:", file=sys.stderr)
        for files in duplicates.values():
            print(f"  identical: {', '.join(files)}", file=sys.stderr)

    good = sum(1 for r in results if r["ok"])
    print(f"\n{good}/{len(results)} usable  ->  {OUTPUT_DIR.relative_to(config.ROOT)}/")
    print(f"manifest: {manifest.relative_to(config.ROOT)}")
    return 0 if good == len(results) and not duplicates else 1


if __name__ == "__main__":
    raise SystemExit(main())
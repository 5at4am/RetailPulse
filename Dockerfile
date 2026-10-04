# RetailPulse — container image for the Streamlit dashboard.
#
# Tier 3 of the design spec: authored and statically validated, NOT deployed. The spec says
# this verbatim in the report, and that sentence has to stay true.
#
# Two stages. The builder installs only what the dashboard needs, which is the whole of the
# root requirements.txt: the app reads committed aggregates from data/processed/ and never
# imports Prophet, TensorFlow or XGBoost. The modelling stack lives in requirements-ml.txt and
# has no business in a runtime image -- baking 2 GB of ML into it is how a demo URL stops
# responding. The same split is what lets Streamlit Community Cloud install three packages
# instead of a quarter-gigabyte toolchain.
#
#   docker build -t retailpulse .
#   docker run --rm -p 8501:8501 retailpulse

FROM python:3.11-slim AS base

# Streamlit needs a writable home for its config and cache. Without this it warns on every
# session and, on some base images, fails outright.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/appuser

# curl is here for the HEALTHCHECK below. git is needed because Streamlit Cloud and most CI
# runners resolve dependencies from the repo.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --------------------------------------------------------------------------- dependencies
# The root requirements.txt is the app set (streamlit, pandas, numpy) -- the same three
# Streamlit Community Cloud installs, so the container and the public demo run identical code
# against an identical dependency set.
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------- app
COPY app/ ./app/
COPY src/config.py ./src/
# NOTE: the dashboard's data-access module is `app/dashboard_data.py` and it arrives with
# `COPY app/ ./app/` above. There is deliberately no `src/dashboard_data.py` copy: Dockerfile
# COPY has no shell, so `2>/dev/null || true` is not a "skip if missing" idiom -- it is a
# parse error, and a copy of a path that does not exist fails the build regardless.
COPY .streamlit/ ./.streamlit/
COPY data/processed/ ./data/processed/

# Create the unprivileged user after the copies, then chown, so the image layer does not
# invalidate when the app files change.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit's own health endpoint. Docker calls it every 30s with the interval below; three
# consecutive failures mark the container unhealthy, which is what an orchestrator needs to
# know before it routes traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# One worker. The app is stateless and reads only committed CSVs, so horizontal scaling is a
# replica count, not a worker count. More workers on a small container costs memory and buys
# nothing, because there is no session state to keep warm.
CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.maxUploadSize=1", "--browser.gatherUsageStats=false"]
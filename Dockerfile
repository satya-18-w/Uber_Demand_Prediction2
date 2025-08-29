# ---------- Stage 1: Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system deps (needed for numpy, scikit-learn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements-docker.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-docker.txt

# Copy app files
COPY ./data/external/plot_data.csv ./data/external/plot_data.csv
COPY ./data/processed/test_data.csv ./data/processed/test_data.csv
COPY ./reports/ ./reports/
COPY ./models/ ./models/
COPY ./app.py ./app.py
# COPY .env .env


# ---------- Stage 2: Final ----------
FROM python:3.12-slim AS final

WORKDIR /app

# Copy dependencies and app from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 5000

# ✅ Use ENTRYPOINT instead of CMD for reliability
ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]

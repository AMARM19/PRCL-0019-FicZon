# Build from the repository root:  docker build -f backend/Dockerfile -t ficzon-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=5000
WORKDIR /app

# XGBoost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Code + trained model only. No client data is copied into the image.
COPY src/ src/
COPY backend/app.py backend/
COPY models/lead_quality_pipeline.joblib models/model_metadata.json models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD gunicorn --chdir backend --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app

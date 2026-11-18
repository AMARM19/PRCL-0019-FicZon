# Build from the repository root:  docker build -f frontend/Dockerfile -t ficzon-ui .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8501 BACKEND_URL=http://localhost:5000
WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY frontend/app.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false

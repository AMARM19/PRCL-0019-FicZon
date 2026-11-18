"""Flask REST API for FicZon lead-quality prediction.

Endpoints
  GET  /health              liveness + model info
  GET  /api/v1/metadata     metrics, threshold, allowed input values, feature importance
  POST /api/v1/predict      JSON: one lead object, or {"leads": [...]} with up to MAX_BATCH leads

A lead object has these fields. REQUIRED_FIELDS must be present; the rest are optional (missing = 'Unknown').
Why required: in the training data, leads with no product/location were almost all fresh 'Open' leads, so the
model treats a blank product/location as a High signal. Blank inputs are therefore rejected instead of scored.
  created        "YYYY-MM-DD HH:MM"   when the lead came in
  product_id     e.g. "18"
  source         e.g. "Call", "Live Chat-Direct", "Website", "Referral" (see /api/v1/metadata choices)
  sales_agent    e.g. "Sales-Agent-4"
  location       e.g. "Bangalore", "International"
  delivery_mode  "Mode-1" ... "Mode-5"
  has_mobile     1 if the lead gave a mobile number, else 0
  email_type     "gmail" | "yahoo" | "microsoft" | "other" | "missing"
No personal data (names, phone numbers, e-mails) is needed or stored.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402

from ficzon.features import LEAD_FIELDS  # noqa: E402  (also makes make_features importable for unpickling)

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))
REQUIRED_FIELDS = ["product_id", "source", "location", "delivery_mode"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ficzon-api")


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.pipeline = joblib.load(model_dir / "lead_quality_pipeline.joblib")
        self.metadata = json.loads((model_dir / "model_metadata.json").read_text())
        self.threshold = float(self.metadata["threshold"])

    def predict(self, leads: list[dict]) -> list[dict]:
        df = pd.DataFrame([{k: lead.get(k) for k in LEAD_FIELDS} for lead in leads])
        p_high = self.pipeline.predict_proba(df)[:, 1]
        out = []
        for p in p_high:
            p = float(p)
            out.append({
                "prediction": "High potential" if p >= self.threshold else "Low potential",
                "probability_high": round(p, 4),
                "priority": "Hot" if p >= 0.65 else "Warm" if p >= self.threshold else "Cold",
            })
        return out


def create_app(predictor: Predictor | None = None) -> Flask:
    app = Flask(__name__)
    app.json.sort_keys = False
    app.predictor = predictor or Predictor()
    log.info("Loaded model '%s' (threshold %.2f)", app.predictor.metadata.get("model_name"), app.predictor.threshold)

    def error(msg, code):
        return jsonify({"error": msg}), code

    @app.get("/")
    def index():
        return jsonify({"service": "FicZon Lead Quality API",
                        "docs": "GET /health, GET /api/v1/metadata, POST /api/v1/predict"})

    @app.get("/health")
    def health():
        m = app.predictor.metadata
        return jsonify({"status": "ok", "model": m.get("model_name"), "threshold": app.predictor.threshold,
                        "test_roc_auc": round(m["test_metrics"]["roc_auc"], 4)})

    @app.get("/api/v1/metadata")
    def metadata():
        return jsonify(app.predictor.metadata)

    @app.post("/api/v1/predict")
    def predict():
        body = request.get_json(silent=True)
        if body is None:
            return error("Send a JSON body: one lead object or {\"leads\": [...]}", 400)
        batch = isinstance(body, list) or "leads" in body
        leads = body if isinstance(body, list) else body.get("leads", [body])
        if not isinstance(leads, list) or not leads or not all(isinstance(x, dict) for x in leads):
            return error("'leads' must be a non-empty list of objects", 400)
        if len(leads) > MAX_BATCH:
            return error(f"At most {MAX_BATCH} leads per request", 413)
        for i, x in enumerate(leads):
            missing = [f for f in REQUIRED_FIELDS if x.get(f) in (None, "", "Unknown")]
            if missing:
                where = f"lead {i}: " if batch else ""
                return error(f"{where}missing required field(s) {missing}", 400)
        unknown = sorted({k for x in leads for k in x} - set(LEAD_FIELDS) - {"id"})
        t0 = time.perf_counter()
        try:
            results = app.predictor.predict(leads)
        except Exception as exc:  # bad types etc.
            log.exception("prediction failed")
            return error(f"Could not score lead(s): {exc}", 422)
        for lead, r in zip(leads, results):
            if "id" in lead:
                r["id"] = lead["id"]
        latency = round((time.perf_counter() - t0) * 1000, 1)
        payload = {"results": results, "count": len(results)} if batch else results[0]
        payload.update({"threshold": app.predictor.threshold, "latency_ms": latency})
        if unknown:
            payload["ignored_fields"] = unknown
        return jsonify(payload)

    return app


app = create_app() if os.getenv("FICZON_NO_AUTOLOAD") != "1" else None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

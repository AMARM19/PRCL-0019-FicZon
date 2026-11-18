import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ["FICZON_NO_AUTOLOAD"] = "1"
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app  # noqa: E402

GOOD = {"created": "2018-11-10 11:30", "product_id": "27", "source": "Referral", "sales_agent": "Sales-Agent-9",
        "location": "Bangalore", "delivery_mode": "Mode-4", "has_mobile": 1, "email_type": "gmail"}
POOR = {"created": "2018-06-03 23:10", "product_id": "15", "source": "Website", "sales_agent": "Sales-Agent-5",
        "location": "Other Locations", "delivery_mode": "Mode-5", "has_mobile": 0, "email_type": "missing"}


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_metadata_has_choices(client):
    m = client.get("/api/v1/metadata").get_json()
    assert {"source", "location", "delivery_mode", "email_type"} <= set(m["choices"])
    assert 0 < m["threshold"] < 1


def test_single_prediction_ranks_sensibly(client):
    good = client.post("/api/v1/predict", json=GOOD).get_json()
    poor = client.post("/api/v1/predict", json=POOR).get_json()
    assert good["prediction"] == "High potential"
    assert poor["prediction"] == "Low potential"
    assert good["probability_high"] > poor["probability_high"]


def test_batch_and_ids(client):
    r = client.post("/api/v1/predict", json={"leads": [{**GOOD, "id": "a"}, {**POOR, "id": "b"}]}).get_json()
    assert r["count"] == 2 and [x["id"] for x in r["results"]] == ["a", "b"]


def test_unseen_values_and_optional_fields_are_safe(client):
    r = client.post("/api/v1/predict", json={"source": "TikTok", "product_id": 999, "location": "Mars",
                                             "delivery_mode": "Mode-9", "foo": 1})
    body = r.get_json()
    assert r.status_code == 200 and 0 <= body["probability_high"] <= 1
    assert body["ignored_fields"] == ["foo"]


def test_required_fields(client):
    assert client.post("/api/v1/predict", json={}).status_code == 400
    r = client.post("/api/v1/predict", json={"leads": [GOOD, {**POOR, "location": ""}]})
    assert r.status_code == 400 and "lead 1" in r.get_json()["error"]


def test_bad_requests(client):
    assert client.post("/api/v1/predict", data="x", content_type="text/plain").status_code == 400
    assert client.post("/api/v1/predict", json={"leads": []}).status_code == 400

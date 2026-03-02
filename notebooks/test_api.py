import json
from app.app import app

def test_train_endpoint():
    client = app.test_client()
    resp = client.post("/train", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "model_path" in data

def test_predict_endpoint():
    client = app.test_client()
    # ensure model exists
    client.post("/train", json={})

    resp = client.post("/predict", json={"country": "all", "date": "2019-09-15"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert isinstance(data["prediction"], (int, float))

def test_logs_endpoint():
    client = app.test_client()
    resp = client.get("/logs")
    assert resp.status_code == 200
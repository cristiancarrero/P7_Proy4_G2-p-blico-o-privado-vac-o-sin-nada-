import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "constructed_area": 80.0,
    "room_number": 3,
    "bath_number": 1,
    "latitude": 40.4168,
    "longitude": -3.7038,
    "distance_to_city_center": 2.5,
    "distance_to_metro": 0.3,
    "distance_to_castellana": 1.5,
    "cadmaxbuildingfloor": 5,
    "caddwellingcount": 20,
    "cadastralqualityid": 5.0,
    "floor_clean": 2.0,
    "has_terrace": 0,
    "has_lift": 1,
    "has_air_conditioning": 0,
    "has_parking_space": 0,
    "has_boxroom": 0,
    "has_wardrobe": 0,
    "has_swimming_pool": 0,
    "has_doorman": 0,
    "has_garden": 0,
    "is_duplex": 0,
    "is_studio": 0,
    "is_intopfloor": 0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_payload():
    resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert resp.status_code in (200, 503)


def test_predict_invalid_payload():
    payload = {"constructed_area": 80.0}  # faltan campos obligatorios
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_training_history():
    resp = client.get("/api/v1/training-history")
    assert resp.status_code == 200
    assert "runs" in resp.json()


def test_upload_non_csv():
    resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("test.txt", b"contenido", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_valid_csv():
    csv_content = b"PRICE,CONSTRUCTEDAREA,ROOMNUMBER\n300000,80,3\n"
    resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("test_data.csv", csv_content, "text/csv")},
    )
    assert resp.status_code == 200
    assert "subido" in resp.json()["message"]


def test_predict_empty_body():
    resp = client.post("/api/v1/predict", json={})
    assert resp.status_code == 422


def test_predict_negative_area():
    payload = {**VALID_PAYLOAD, "constructed_area": -10}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_amenity_value():
    payload = {**VALID_PAYLOAD, "has_terrace": 5}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_predict_response_format():
    resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    if resp.status_code == 200:
        data = resp.json()
        assert "precio_estimado" in data
        assert "modelo_usado" in data
        assert data["precio_estimado"] > 0


def test_retrain_missing_dataset():
    resp = client.post("/api/v1/retrain-with-dataset", params={"filename": "no_existe.csv"})
    assert resp.status_code == 404


def test_health_response_time():
    import time
    start = time.time()
    resp = client.get("/health")
    elapsed = time.time() - start
    assert resp.status_code == 200
    assert elapsed < 1.0

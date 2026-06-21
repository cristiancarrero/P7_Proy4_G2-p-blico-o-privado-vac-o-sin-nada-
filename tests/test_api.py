import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_payload():
    payload = {
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
    resp = client.post("/api/v1/predict", json=payload)
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

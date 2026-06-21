import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from mlflow.tracking import MlflowClient

from src.api.schemas import PredictionRequest, PredictionResponse, RetrainResponse
from src.ml.pipeline import ALL_FEATURES, train_baseline

router = APIRouter()

MODEL_PATH = Path("models/model.pkl")
ZONA_MEDIANS_PATH = Path("models/zona_medians.pkl")
GLOBAL_MEDIAN_PATH = Path("models/global_median.pkl")
BIN_EDGES_PATH = Path("models/bin_edges.pkl")
DATA_RAW_PATH = Path("data/raw")


def _load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible. Entrena primero.")
    return joblib.load(MODEL_PATH)


def _load_zone_medians():
    if ZONA_MEDIANS_PATH.exists() and GLOBAL_MEDIAN_PATH.exists():
        return joblib.load(ZONA_MEDIANS_PATH), joblib.load(GLOBAL_MEDIAN_PATH)
    raise HTTPException(status_code=503, detail="Medianas de zona no disponibles.")


def _load_bin_edges():
    if BIN_EDGES_PATH.exists():
        return joblib.load(BIN_EDGES_PATH)
    return None


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    pipe = _load_model()
    zone_medians, global_median = _load_zone_medians()

    # Calcular geo_zone del input usando bin edges del entrenamiento
    bin_edges = _load_bin_edges()
    if bin_edges:
        lat_bin = int(np.searchsorted(bin_edges["lat_edges"], req.latitude, side="right") - 1)
        lon_bin = int(np.searchsorted(bin_edges["lon_edges"], req.longitude, side="right") - 1)
        lat_bin = max(0, min(lat_bin, 19))
        lon_bin = max(0, min(lon_bin, 19))
    else:
        lat_bin = pd.cut([req.latitude], bins=20, labels=False)[0]
        lon_bin = pd.cut([req.longitude], bins=20, labels=False)[0]
    geo_zone = f"{lat_bin}_{lon_bin}"
    price_per_m2_zone = zone_medians.get(geo_zone, global_median)

    # Contar amenities
    amenities = [
        req.has_terrace, req.has_lift, req.has_air_conditioning,
        req.has_parking_space, req.has_boxroom, req.has_wardrobe,
        req.has_swimming_pool, req.has_doorman, req.has_garden,
        req.is_duplex, req.is_studio, req.is_intopfloor,
    ]
    n_amenities = sum(amenities)

    input_df = pd.DataFrame([{
        "CONSTRUCTEDAREA": req.constructed_area,
        "ROOMNUMBER": req.room_number,
        "BATHNUMBER": req.bath_number,
        "LATITUDE": req.latitude,
        "LONGITUDE": req.longitude,
        "DISTANCE_TO_CITY_CENTER": req.distance_to_city_center,
        "DISTANCE_TO_METRO": req.distance_to_metro,
        "DISTANCE_TO_CASTELLANA": req.distance_to_castellana,
        "CADMAXBUILDINGFLOOR": req.cadmaxbuildingfloor,
        "CADDWELLINGCOUNT": req.caddwellingcount,
        "CADASTRALQUALITYID": req.cadastralqualityid,
        "FLOORCLEAN": req.floor_clean,
        "price_per_m2_zone": price_per_m2_zone,
        "n_amenities": n_amenities,
        "HASTERRACE": req.has_terrace,
        "HASLIFT": req.has_lift,
        "HASAIRCONDITIONING": req.has_air_conditioning,
        "HASPARKINGSPACE": req.has_parking_space,
        "HASBOXROOM": req.has_boxroom,
        "HASWARDROBE": req.has_wardrobe,
        "HASSWIMMINGPOOL": req.has_swimming_pool,
        "HASDOORMAN": req.has_doorman,
        "HASGARDEN": req.has_garden,
        "ISDUPLEX": req.is_duplex,
        "ISSTUDIO": req.is_studio,
        "ISINTOPFLOOR": req.is_intopfloor,
    }])

    pred_log = pipe.predict(input_df[ALL_FEATURES])[0]
    precio = float(np.expm1(pred_log))

    return PredictionResponse(precio_estimado=round(precio, 2))


@router.get("/training-history")
def training_history():
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("idealista18_madrid_venta")
        if not experiment:
            return {"runs": []}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )

        history = []
        for run in runs:
            history.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
            })
        return {"runs": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".csv.gz")):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .csv o .csv.gz")

    dest = DATA_RAW_PATH / file.filename
    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    dest.write_bytes(content)

    return {"message": f"Dataset '{file.filename}' subido correctamente", "path": str(dest)}


@router.post("/retrain-with-dataset", response_model=RetrainResponse)
def retrain_with_dataset(background_tasks: BackgroundTasks, filename: str = "idealista18_madrid_sale (1).csv.gz"):
    filepath = DATA_RAW_PATH / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{filename}' no encontrado")

    background_tasks.add_task(_run_retrain, filepath)
    return RetrainResponse(status="accepted", message="Reentrenamiento iniciado en background")


def _run_retrain(filepath: Path):
    from src.data.ingestion import run_ingestion
    run_ingestion(path=filepath)
    train_baseline()

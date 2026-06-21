from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Madrid Housing Price Predictor",
    description="API de predicción de precios de vivienda en Madrid",
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    constructed_area: float = Field(..., gt=0, example=80.0)
    room_number: int = Field(..., ge=0, example=3)
    bath_number: int = Field(..., ge=1, example=1)
    latitude: float = Field(..., example=40.4168)
    longitude: float = Field(..., example=-3.7038)
    distance_to_city_center: float = Field(..., ge=0, example=2.5)
    distance_to_metro: float = Field(..., ge=0, example=0.3)
    distance_to_castellana: float = Field(..., ge=0, example=1.5)
    cadmaxbuildingfloor: int = Field(5, ge=0, example=5)
    caddwellingcount: int = Field(20, ge=0, example=20)
    cadastralqualityid: float = Field(5.0, example=5.0)
    floor_clean: float = Field(2.0, example=2.0)
    has_terrace: int = Field(0, ge=0, le=1)
    has_lift: int = Field(1, ge=0, le=1)
    has_air_conditioning: int = Field(0, ge=0, le=1)
    has_parking_space: int = Field(0, ge=0, le=1)
    has_boxroom: int = Field(0, ge=0, le=1)
    has_wardrobe: int = Field(0, ge=0, le=1)
    has_swimming_pool: int = Field(0, ge=0, le=1)
    has_doorman: int = Field(0, ge=0, le=1)
    has_garden: int = Field(0, ge=0, le=1)
    is_duplex: int = Field(0, ge=0, le=1)
    is_studio: int = Field(0, ge=0, le=1)
    is_intopfloor: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    precio_estimado: float
    modelo_usado: str = "xgboost-optuna-idealista18"


class RetrainResponse(BaseModel):
    status: str
    message: str
    run_id: Optional[str] = None

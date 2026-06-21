# 🏠 Predictor de Precio de Vivienda - Madrid

[![Tests](https://github.com/cristiancarrero/P7_Proy4_G2-p-blico-o-privado-vac-o-sin-nada-/actions/workflows/deploy.yml/badge.svg)](https://github.com/cristiancarrero/P7_Proy4_G2-p-blico-o-privado-vac-o-sin-nada-/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema **MLOps End-to-End** para la predicción del precio de viviendas en Madrid utilizando datos de Idealista18 (~94K registros). Pipeline completo desde la ingesta de datos hasta el despliegue en producción con monitoreo de data drift.

---

## 📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **R² (Test Holdout)** | 0.949 |
| **R² (CV 5-Fold)** | 0.945 |
| **RMSE** | 73,241 € |
| **Dataset** | 93,868 viviendas |
| **Algoritmo** | XGBoost + Optuna |

> El modelo supera ampliamente el benchmark mínimo aceptable de R² ≥ 0.85

---

## 🏗️ Arquitectura

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Streamlit     │────▶│     FastAPI      │────▶│   ML Pipeline    │
│    Frontend      │◀────│     Backend      │◀────│   (XGBoost)      │
│    :8501         │     │    :8000         │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         │                        ▼                        ▼
         │               ┌──────────────────┐     ┌──────────────────┐
         │               │     MLflow       │     │    Evidently     │
         │               │   (Tracking)     │     │  (Data Drift)    │
         │               └──────────────────┘     └──────────────────┘
         │
         ▼
┌──────────────────┐
│  Docker Compose  │
│  + GitHub Actions│
│    (CI/CD)       │
└──────────────────┘
```

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| **Machine Learning** | XGBoost, Optuna (bayesian optimization), scikit-learn |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | Streamlit, Plotly, Folium (mapas interactivos) |
| **MLOps** | MLflow (experiment tracking), Evidently AI (data drift) |
| **Infraestructura** | Docker Compose, GitHub Actions (CI/CD) |
| **Testing** | pytest, FastAPI TestClient |

---

## 📁 Estructura del Proyecto

```
├── src/
│   ├── api/                 # API REST
│   │   ├── main.py          # App FastAPI
│   │   ├── routes.py        # Endpoints (predict, history, upload, retrain)
│   │   └── schemas.py       # Modelos Pydantic (request/response)
│   ├── data/                # Ingesta y validación
│   │   └── ingestion.py     # Carga, limpieza y persistencia en parquet
│   ├── frontend/            # Interfaz de usuario
│   │   └── app.py           # Streamlit (predicción, mapa, comparativa, PDF)
│   └── ml/                  # Pipeline de Machine Learning
│       └── pipeline.py      # Features, preprocesamiento, Optuna, entrenamiento
├── tests/
│   ├── conftest.py          # Configuración de paths
│   └── test_api.py          # Tests de la API (12 tests)
├── scripts/
│   └── data_drift.py        # Detección de drift con Evidently + alertas Slack
├── data/
│   ├── raw/                 # Datos crudos (.csv, .csv.gz)
│   ├── processed/           # Datos procesados (.parquet)
│   └── geo/                 # Datos geoespaciales (GeoJSON distritos)
├── models/                  # Artefactos del modelo (.pkl, .json)
├── .github/workflows/
│   └── deploy.yml           # CI/CD pipeline
├── docker-compose.yml       # Orquestación multi-contenedor
├── Dockerfile.backend       # Imagen del backend
├── Dockerfile.frontend      # Imagen del frontend
├── .env.example             # Variables de entorno (template)
├── .dockerignore            # Exclusiones para Docker build
└── requirements.txt         # Dependencias Python
```

---

## 🚀 Quickstart

### Opción 1: Docker (recomendado)

```bash
# Clonar
git clone https://github.com/cristiancarrero/P7_Proy4_G2-p-blico-o-privado-vac-o-sin-nada-.git
cd P7_Proy4_G2-p-blico-o-privado-vac-o-sin-nada-

# Colocar dataset en data/raw/
# (idealista18_madrid_sale.csv.gz)

# Levantar
docker compose up -d --build
```

- 🖥️ **Frontend**: http://localhost:8501
- 📡 **API Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

### Opción 2: Local (desarrollo)

```bash
# Entorno virtual
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ingesta de datos
python -c "from src.data.ingestion import run_ingestion; run_ingestion()"

# Entrenamiento
python -c "from src.ml.pipeline import train_baseline; train_baseline()"

# Levantar API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Levantar Frontend (otra terminal)
streamlit run src/frontend/app.py
```

---

## 📡 API Endpoints

| Método | Ruta | Descripción | Body |
|--------|------|-------------|------|
| `GET` | `/health` | Health check | - |
| `POST` | `/api/v1/predict` | Predicción de precio | JSON (ver abajo) |
| `GET` | `/api/v1/training-history` | Historial de entrenamientos MLflow | - |
| `POST` | `/api/v1/datasets/upload` | Subir nuevo dataset CSV | multipart/form-data |
| `POST` | `/api/v1/retrain-with-dataset` | Reentrenar modelo en background | query: filename |

### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "constructed_area": 80,
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
    "is_intopfloor": 0
  }'
```

**Respuesta:**
```json
{
  "precio_estimado": 473829.12,
  "modelo_usado": "xgboost-optuna-idealista18"
}
```

---

## 🧠 Pipeline de Machine Learning

### Feature Engineering

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `CONSTRUCTEDAREA` | Numérica | Superficie construida (m²) |
| `LATITUDE`, `LONGITUDE` | Numérica | Coordenadas geográficas |
| `DISTANCE_TO_*` | Numérica | Distancias a puntos de interés |
| `price_per_m2_zone` | Engineered | Target encoding espacial (grid 20x20) |
| `n_amenities` | Engineered | Suma de todas las amenities binarias |
| `HAS*`, `IS*` | Binaria | Presencia de amenities |

### Proceso de Entrenamiento

1. **Ingesta**: Carga CSV → validación de esquema → limpieza de outliers (P99) → Parquet
2. **Target Encoding**: Grid geográfico 20x20 → mediana de precio/m² por celda (solo train)
3. **Preprocesamiento**: `SimpleImputer(median)` + `StandardScaler` para numéricas, passthrough para binarias
4. **Optimización**: Optuna (TPE sampler) → 10 trials × 5-Fold CV → minimizar RMSE
5. **Entrenamiento final**: XGBoost con mejores hiperparámetros
6. **Auto-swap**: Solo reemplaza el modelo en producción si el nuevo R² supera al anterior
7. **Tracking**: MLflow registra params, métricas y artefactos
8. **Serialización**: Pipeline completo → `models/model.pkl` + `latest_metrics.json`

### Variables objetivo

- **Target**: `log(1 + PRICE)` (transformación logarítmica para estabilizar varianza)
- **Inversión**: `expm1(prediction)` para obtener precio en euros

---

## 🖥️ Frontend

El frontend de Streamlit incluye 4 secciones:

| Tab | Funcionalidad |
|-----|--------------|
| 🔮 **Predicción** | Formulario interactivo + mapa Folium draggable + selector de barrios + precio/m² |
| 📊 **Historial** | Métricas del modelo + Feature Importance + historial MLflow con gráficos |
| 🏘️ **Comparativa** | Comparar precio de misma vivienda en múltiples barrios + gráfico de barras |
| ⚙️ **Reentrenamiento** | Upload CSV + preview + reentrenamiento en background |

**Features adicionales:**
- 📄 Exportar predicciones a PDF
- 📋 Historial de predicciones por sesión
- 🗺️ Mapa interactivo con click para seleccionar ubicación
- 🗺️ Capa GeoJSON con los 21 distritos de Madrid

---

## 🧪 Tests

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_health                      PASSED
tests/test_api.py::test_predict_valid_payload        PASSED
tests/test_api.py::test_predict_invalid_payload      PASSED
tests/test_api.py::test_training_history             PASSED
tests/test_api.py::test_upload_non_csv               PASSED
tests/test_api.py::test_upload_valid_csv             PASSED
tests/test_api.py::test_predict_empty_body           PASSED
tests/test_api.py::test_predict_negative_area        PASSED
tests/test_api.py::test_predict_invalid_amenity_value PASSED
tests/test_api.py::test_predict_response_format      PASSED
tests/test_api.py::test_retrain_missing_dataset      PASSED
tests/test_api.py::test_health_response_time         PASSED

======================== 12 passed ========================
```

---

## 🔄 CI/CD

El pipeline de GitHub Actions (`.github/workflows/deploy.yml`) ejecuta:

1. **CI (en cada push/PR a main)**:
   - Setup Python 3.11
   - Instalar dependencias
   - Ejecutar `pytest`

2. **CD (solo push a main)**:
   - SSH al VPS
   - `git pull origin main`
   - `docker compose up -d --build`
   - `docker image prune -f`

### Secrets necesarios

| Secret | Descripción |
|--------|-------------|
| `VPS_HOST` | IP o dominio del servidor |
| `VPS_USER` | Usuario SSH |
| `VPS_SSH_KEY` | Clave privada SSH |

---

## 📈 Monitoreo: Data Drift

Script semanal (`scripts/data_drift.py`) que:

1. Compara estadísticamente los datos de producción vs. el baseline de entrenamiento
2. Genera un reporte HTML con Evidently AI
3. Envía alerta a Slack si detecta drift

```bash
# Ejecución manual
python scripts/data_drift.py

# Cron semanal (lunes 9:00)
0 9 * * 1 cd /opt/P7_Proy4_G2 && python scripts/data_drift.py
```

**Variable de entorno necesaria**: `SLACK_WEBHOOK_URL`

---

## 🐳 Docker

```bash
# Configurar variables de entorno
cp .env.example .env

# Levantar
docker compose up -d --build

# Ver logs
docker compose logs -f

# Parar
docker compose down

# Rebuild tras cambios
docker compose up --build -d
```

| Servicio | Puerto | Imagen |
|----------|--------|--------|
| backend | 8000 | python:3.11-slim + FastAPI |
| frontend | 8501 | python:3.11-slim + Streamlit |

Los datos y modelos se montan como volúmenes para persistencia.

---

## 👥 Equipo

**Proyecto 4 - Grupo 2** | Bootcamp IA MAD P7

---

## 📄 Licencia

MIT

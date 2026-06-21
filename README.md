# 🏠 Predictor de Precio de Vivienda - Madrid

Sistema MLOps End-to-End para la predicción del precio de viviendas en Madrid usando datos de Idealista18.

## Arquitectura

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│  ML Pipeline│
│  Frontend   │◀────│   Backend   │◀────│  (XGBoost)  │
└─────────────┘     └─────────────┘     └─────────────┘
     :8501               :8000              MLflow
```

## Stack Tecnológico

- **ML**: XGBoost + Optuna (optimización bayesiana) + scikit-learn
- **API**: FastAPI + Pydantic
- **Frontend**: Streamlit + Plotly
- **MLOps**: MLflow (tracking) + Evidently AI (data drift)
- **Infra**: Docker Compose + GitHub Actions (CI/CD)

## Estructura del Proyecto

```
├── src/
│   ├── api/          # FastAPI endpoints
│   ├── data/         # Ingesta y validación
│   ├── frontend/     # Streamlit app
│   └── ml/           # Pipeline ML (features, training, optimización)
├── tests/            # Tests pytest
├── scripts/          # Data drift, utilidades
├── data/
│   ├── raw/          # CSVs originales
│   └── processed/    # Parquets limpios
├── models/           # Artefactos entrenados (.pkl)
├── .github/workflows/  # CI/CD
├── docker-compose.yml
├── Dockerfile.backend
└── Dockerfile.frontend
```

## Quickstart

### 1. Instalar dependencias

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Colocar datos

Poner el CSV de Idealista18 en `data/raw/`.

### 3. Entrenar modelo

```bash
python -c "from src.data.ingestion import run_ingestion; run_ingestion()"
python -c "from src.ml.pipeline import train_baseline; train_baseline()"
```

### 4. Levantar con Docker

```bash
docker compose up -d --build
```

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/predict` | Predicción de precio |
| GET | `/api/v1/training-history` | Historial MLflow |
| POST | `/api/v1/datasets/upload` | Subir nuevo dataset |
| POST | `/api/v1/retrain-with-dataset` | Reentrenar modelo |

## Métricas del Modelo

- **Dataset**: Idealista18 Madrid (93,868 viviendas)
- **Algoritmo**: XGBoost con optimización bayesiana (Optuna, 30 trials)
- **Validación**: 5-Fold Cross-Validation
- **Target**: log(1 + PRICE) para estabilizar varianza

## CI/CD

El pipeline de GitHub Actions ejecuta:
1. **CI**: `pytest` sobre los endpoints de la API
2. **CD**: Deploy automático al VPS vía SSH + Docker Compose

## Data Drift

Script semanal (`scripts/data_drift.py`) que usa Evidently AI para detectar degradación en datos de producción. Alertas vía Slack webhook.

## Equipo

Proyecto 4 - Grupo 2 | Bootcamp IA MAD P7

# Documentación Técnica Completa

## Índice

1. [Visión General del Sistema](#1-visión-general-del-sistema)
2. [Capa de Datos (Ingesta)](#2-capa-de-datos)
3. [Capa de Machine Learning](#3-capa-de-machine-learning)
4. [Capa de API (Backend)](#4-capa-de-api)
5. [Capa de Presentación (Frontend)](#5-capa-de-presentación)
6. [Infraestructura y Despliegue](#6-infraestructura)
7. [MLOps y Monitoreo](#7-mlops-y-monitoreo)
8. [Testing](#8-testing)
9. [Flujo de Datos End-to-End](#9-flujo-de-datos)

---

## 1. Visión General del Sistema

### 1.1 Objetivo

Sistema de predicción de precios de viviendas en Madrid mediante un pipeline MLOps completo. El sistema permite:

- Predecir el precio de venta de una vivienda dadas sus características físicas, ubicación geográfica y amenities.
- Reentrenar el modelo con nuevos datos sin interrumpir el servicio.
- Monitorear la degradación del modelo mediante detección de data drift.
- Visualizar predicciones, métricas y comparativas a través de una interfaz web interactiva.

### 1.2 Paradigma

- **Tipo de problema**: Regresión supervisada
- **Variable objetivo**: Precio de venta (€) transformado como log(1 + PRICE)
- **Algoritmo principal**: XGBoost (Gradient Boosting) con optimización bayesiana de hiperparámetros via Optuna
- **Métricas de evaluación**: R² (coeficiente de determinación), RMSE (Root Mean Squared Error)

### 1.3 Dataset

- **Fuente**: Idealista18 (portal inmobiliario español)
- **Registros**: 93,868 viviendas en Madrid (tras limpieza)
- **Periodo**: 2018 (4 trimestres: 201803, 201806, 201809, 201812)
- **Variables**: 41 columnas originales (numéricas, binarias, categóricas)
- **Target**: PRICE (precio de venta en euros, rango 21K - 2.2M€ tras filtrado)

### 1.4 Rendimiento

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| R² Test (Holdout 20%) | 0.949 | El modelo explica el 94.9% de la varianza |
| R² CV (5-Fold) | 0.945 ± 0.001 | Rendimiento estable sin overfitting |
| RMSE | 73,241 € | Error medio de predicción |
| Benchmark mínimo | 0.85 | Superado ampliamente |

---

## 2. Capa de Datos

### 2.1 Módulo: `src/data/ingestion.py`

Responsable de la extracción, validación y transformación de datos crudos a formato optimizado.

### 2.2 Flujo de Ingesta

```
CSV/CSV.GZ (data/raw/) → Validación de esquema → Limpieza → Parquet (data/processed/)
```

### 2.3 Validación de Esquema

Se define una lista blanca de columnas mínimas requeridas:

```python
EXPECTED_COLUMNS = [
    "PRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
    "LATITUDE", "LONGITUDE", "DISTANCE_TO_CITY_CENTER",
    "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA",
]
```

Si el dataset de entrada no contiene estas columnas, se lanza un `ValueError` inmediato.

### 2.4 Estrategia de Limpieza

1. **Filtrado de precios inválidos**: Se eliminan registros con PRICE ≤ 0.
2. **Remoción de outliers**: Se calcula el percentil 99 del precio y se descartan registros por encima de ese umbral. Esto elimina propiedades de lujo extremo que distorsionan el entrenamiento.

```python
q99 = df["PRICE"].quantile(0.99)  # ~2,207,000€
df = df[df["PRICE"] <= q99]
```

**Resultado**: De 94,815 filas originales → 93,868 filas tras limpieza (947 outliers eliminados).

### 2.5 Persistencia

Los datos limpios se guardan en formato Apache Parquet:
- **Ventajas**: Compresión columnar, lectura ~10x más rápida que CSV, tipado estricto.
- **Ruta**: `data/processed/idealista18_madrid.parquet`

### 2.6 Interfaz Pública

```python
from src.data.ingestion import run_ingestion

# Ejecutar ingesta completa
df = run_ingestion(source="idealista18", path=Path("data/raw/archivo.csv.gz"))
```

---

## 3. Capa de Machine Learning

### 3.1 Módulo: `src/ml/pipeline.py`

Contiene toda la lógica de feature engineering, preprocesamiento, optimización de hiperparámetros y entrenamiento.

### 3.2 Feature Engineering

#### 3.2.1 Features Numéricas Originales

| Feature | Descripción | Correlación con PRICE |
|---------|-------------|----------------------|
| CONSTRUCTEDAREA | Superficie construida (m²) | 0.859 |
| BATHNUMBER | Número de baños | 0.718 |
| ROOMNUMBER | Número de habitaciones | 0.465 |
| CADASTRALQUALITYID | Calidad catastral (1=mejor, 9=peor) | 0.458 |
| LATITUDE | Latitud geográfica | 0.247 |
| LONGITUDE | Longitud geográfica | - |
| DISTANCE_TO_CITY_CENTER | Distancia al centro (km) | - |
| DISTANCE_TO_METRO | Distancia al metro más cercano (km) | - |
| DISTANCE_TO_CASTELLANA | Distancia al Paseo de la Castellana (km) | 0.249 |
| CADMAXBUILDINGFLOOR | Plantas máximas del edificio | - |
| CADDWELLINGCOUNT | Número de viviendas en el edificio | - |
| FLOORCLEAN | Planta de la vivienda | - |

#### 3.2.2 Features Engineered

**`n_amenities`**: Suma de todas las variables binarias de amenities. Captura el "nivel de equipamiento" general de la vivienda.

```python
amenity_cols = [c for c in df.columns if c.startswith("HAS") or c in ["ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR"]]
df["n_amenities"] = df[amenity_cols].sum(axis=1)
```

**`price_per_m2_zone`**: Target encoding espacial. Se divide Madrid en un grid de 20x20 celdas geográficas y se calcula la mediana del precio/m² por celda. Esto captura el "valor de mercado de la zona" sin data leakage.

```python
df["lat_bin"] = pd.cut(df["LATITUDE"], bins=20, labels=False)
df["lon_bin"] = pd.cut(df["LONGITUDE"], bins=20, labels=False)
df["geo_zone"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

# Calculado SOLO en train
zone_medians = train.groupby("geo_zone")["PRICE"].median() / train.groupby("geo_zone")["CONSTRUCTEDAREA"].median()
```

#### 3.2.3 Features Binarias (12 variables)

| Feature | Significado |
|---------|------------|
| HASTERRACE | Tiene terraza |
| HASLIFT | Tiene ascensor |
| HASAIRCONDITIONING | Tiene aire acondicionado |
| HASPARKINGSPACE | Tiene plaza de parking |
| HASBOXROOM | Tiene trastero |
| HASWARDROBE | Tiene armarios empotrados |
| HASSWIMMINGPOOL | Tiene piscina comunitaria |
| HASDOORMAN | Tiene portero |
| HASGARDEN | Tiene jardín |
| ISDUPLEX | Es dúplex |
| ISSTUDIO | Es estudio |
| ISINTOPFLOOR | Es último piso |

### 3.3 Prevención de Data Leakage

El target encoding (`price_per_m2_zone`) se calcula exclusivamente sobre el conjunto de entrenamiento. El conjunto de test se mapea usando las medianas precalculadas, con fallback a la mediana global si la zona no existe en train.

```python
train_raw, test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)
zone_medians_train = train_raw.groupby("geo_zone")["PRICE"].median() / ...
# test_df usa zone_medians_train, NO calcula las suyas
```

### 3.4 Preprocesamiento

Se usa un `ColumnTransformer` de scikit-learn con dos ramas paralelas:

```python
ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERICAL),
    ("bin", "passthrough", BINARY),
])
```

- **Numéricas (14 features)**: Imputación de nulos con mediana + estandarización Z-score
- **Binarias (12 features)**: Sin transformación (passthrough), ya son 0/1

### 3.5 Transformación del Target

```python
y_train = np.log1p(train_df["PRICE"])  # log(1 + precio)
```

**Justificación**: La distribución de precios tiene cola larga (right-skewed). La transformación logarítmica:
- Reduce la influencia de propiedades caras
- Estabiliza la varianza
- Hace que el error del modelo sea proporcional al precio (no absoluto)

**Inversión en inferencia**:
```python
precio = np.expm1(prediccion_log)  # exp(pred) - 1
```

### 3.6 Optimización de Hiperparámetros

Se usa **Optuna** con el sampler **TPE (Tree-structured Parzen Estimator)** para explorar el espacio de hiperparámetros de forma bayesiana.

#### Espacio de búsqueda

| Hiperparámetro | Rango | Escala |
|----------------|-------|--------|
| n_estimators | [200, 800] | Lineal |
| max_depth | [4, 10] | Lineal |
| learning_rate | [0.01, 0.3] | Logarítmica |
| subsample | [0.6, 1.0] | Lineal |
| colsample_bytree | [0.6, 1.0] | Lineal |
| reg_alpha (L1) | [0.001, 10.0] | Logarítmica |
| reg_lambda (L2) | [0.001, 10.0] | Logarítmica |

#### Configuración

- **Trials**: 10 (balance entre tiempo y calidad)
- **Evaluación**: 5-Fold Cross-Validation por trial
- **Métrica objetivo**: RMSE (minimizar)
- **Seed**: 42 (reproducibilidad)

#### Mejores hiperparámetros encontrados

```python
{
    "n_estimators": 311,
    "max_depth": 10,
    "learning_rate": 0.1396,
    "subsample": 0.9758,
    "colsample_bytree": 0.9579,
    "reg_alpha": 0.2464,
    "reg_lambda": 4.8696,
}
```

### 3.7 Entrenamiento Final

1. Se entrena el pipeline completo (preprocesador + XGBoost) con los mejores hiperparámetros sobre todo el train set.
2. Se evalúa en el holdout test (20%).
3. Se serializa el pipeline completo con `joblib`.

### 3.8 Artefactos Generados

| Archivo | Contenido | Uso |
|---------|-----------|-----|
| `models/model.pkl` | Pipeline completo (preprocessor + XGBoost) | Inferencia en la API |
| `models/zona_medians.pkl` | Medianas de precio/m² por geo_zone | Target encoding en inferencia |
| `models/global_median.pkl` | Mediana global de precio/m² | Fallback para zonas desconocidas |

---

## 4. Capa de API

### 4.1 Módulo: `src/api/main.py`

Punto de entrada de la aplicación FastAPI.

```python
app = FastAPI(title="Madrid Room Rental Price Predictor")
app.include_router(router, prefix="/api/v1")
```

### 4.2 Módulo: `src/api/schemas.py`

Define los contratos de datos con Pydantic v2:

**PredictionRequest**: 24 campos validados con tipos, rangos y valores por defecto.
- Campos obligatorios: constructed_area, room_number, bath_number, latitude, longitude, distance_to_*
- Campos con default: amenities (default=0), datos de edificio (defaults razonables)

**PredictionResponse**: precio_estimado (float) + modelo_usado (str)

**RetrainResponse**: status + message + run_id opcional

### 4.3 Módulo: `src/api/routes.py`

#### Endpoint: POST `/api/v1/predict`

**Flujo de ejecución:**

1. Recibe el payload JSON validado por Pydantic
2. Carga el modelo serializado (`model.pkl`) y las medianas de zona
3. Calcula la geo_zone del input usando pd.cut con 20 bins
4. Busca el price_per_m2_zone en las medianas precalculadas
5. Cuenta el n_amenities sumando los campos binarios
6. Construye un DataFrame con las 26 features en el orden correcto
7. Ejecuta `pipe.predict(input_df)` que aplica preprocesamiento + XGBoost
8. Revierte la transformación logarítmica con `np.expm1()`
9. Retorna el precio estimado redondeado a 2 decimales

**Manejo de errores:**
- 503 si el modelo no existe (no se ha entrenado)
- 422 si el payload es inválido (Pydantic validation)

#### Endpoint: GET `/api/v1/training-history`

Conecta con `MlflowClient()` para:
1. Obtener el experimento por nombre
2. Buscar las últimas 20 runs ordenadas por fecha
3. Retornar run_id, nombre, status, métricas y parámetros

#### Endpoint: POST `/api/v1/datasets/upload`

1. Valida que el archivo sea .csv o .csv.gz
2. Guarda los bytes en `data/raw/{filename}`
3. Retorna confirmación con la ruta

#### Endpoint: POST `/api/v1/retrain-with-dataset`

1. Verifica que el archivo exista en data/raw/
2. Lanza `_run_retrain()` como BackgroundTask de FastAPI
3. Retorna inmediatamente con status "accepted"
4. En background: ejecuta ingesta → entrenamiento completo

**Nota sobre BackgroundTasks**: FastAPI ejecuta la tarea en un thread pool sin bloquear el event loop principal. El cliente recibe respuesta inmediata mientras el reentrenamiento puede tardar minutos.

---

## 5. Capa de Presentación

### 5.1 Módulo: `src/frontend/app.py`

Aplicación Streamlit con 4 tabs y funcionalidades avanzadas.

### 5.2 Configuración

- **Layout**: wide (aprovecha todo el ancho del navegador)
- **API_URL**: configurable via variable de entorno (Docker: `http://backend:8000/api/v1`, Local: `http://localhost:8000/api/v1`)

### 5.3 Tab 1: Predicción

**Componentes:**
- Selector de barrios predefinidos (17 distritos de Madrid con coordenadas reales)
- Formulario de características (superficie, habitaciones, baños, planta)
- Campos de ubicación (autocompletados por barrio o manuales)
- Checkboxes de amenities en 3 columnas
- Mapa Folium interactivo (click para mover marcador)
- Resultado con precio total + precio/m²
- Historial de predicciones en session_state

**Mapa interactivo:**
```python
folium.Marker([lat, lon], draggable=True)
# Click en mapa → st.rerun() con nuevas coordenadas
```

### 5.4 Tab 2: Historial & Métricas

- Tarjetas CSS con métricas del modelo actual (R², RMSE, dataset)
- Gráfico Plotly de Feature Importance (barras horizontales)
- Conexión a MLflow para historial de entrenamientos
- Gráfico de barras con línea roja en R²=0.85 (mínimo aceptable)

### 5.5 Tab 3: Comparativa de Barrios

- Configuración de vivienda base (m², habitaciones, baños, planta)
- Multiselect de barrios a comparar
- Barra de progreso mientras consulta la API por cada barrio
- Gráfico de barras coloreado (RdYlGn_r: rojo=caro, verde=barato)
- Tabla con precio y €/m² por barrio

### 5.6 Tab 4: Reentrenamiento

- File uploader para CSV/GZ
- Preview automático de las 5 primeras filas
- Botón que sube el archivo y lanza reentrenamiento en background
- Feedback visual con spinners y balloons

### 5.7 Exportación PDF

Genera un PDF con fpdf2 conteniendo:
- Cabecera con fecha y modelo usado
- Tabla con todas las predicciones de la sesión
- Función `_clean_text()` para sanitizar emojis incompatibles con fuentes PDF

### 5.8 CSS Custom

Estilos inyectados para:
- Header con gradiente azul
- Tarjetas de métricas con borde lateral
- Precio grande centrado
- Badges del modelo

---

## 6. Infraestructura

### 6.1 Docker

#### Dockerfile.backend

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Dockerfile.frontend

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### docker-compose.yml

```yaml
services:
  backend:
    build: { context: ., dockerfile: Dockerfile.backend }
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./mlruns:/app/mlruns
    environment:
      - MLFLOW_TRACKING_URI=file:///app/mlruns

  frontend:
    build: { context: ., dockerfile: Dockerfile.frontend }
    ports: ["8501:8501"]
    environment:
      - API_URL=http://backend:8000/api/v1
    depends_on: [backend]

networks:
  app-network:
    driver: bridge
```

**Decisiones de diseño:**
- `python:3.11-slim`: Imagen ligera (~150MB base) vs full (~900MB)
- Volúmenes para data/models/mlruns: persistencia entre rebuilds
- Red bridge: los contenedores se resuelven por nombre (frontend→backend)
- `depends_on`: frontend espera a que backend esté creado (no ready, solo created)

### 6.2 CI/CD (GitHub Actions)

**Archivo**: `.github/workflows/deploy.yml`

**Trigger**: push/PR a main

**Job CI:**
1. Checkout del código
2. Setup Python 3.11
3. `pip install -r requirements.txt`
4. `pytest tests/ -v`

**Job CD (solo push a main, tras CI exitoso):**
1. SSH al VPS usando `appleboy/ssh-action`
2. `git pull origin main`
3. `docker compose up -d --build`
4. `docker image prune -f` (limpiar imágenes huérfanas)

---

## 7. MLOps y Monitoreo

### 7.1 MLflow (Experiment Tracking)

**Experimento**: `idealista18_madrid_venta`

**Datos logueados por run:**
- Parámetros: todos los hiperparámetros de XGBoost + metadata (dataset, n_filas)
- Métricas: cv_r2_mean, cv_r2_std, test_r2, test_rmse_eur
- Artefactos: modelo XGBoost serializado

**Backend de almacenamiento**: SQLite local (`mlflow.db`) + filesystem (`mlruns/`)

### 7.2 Evidently AI (Data Drift)

**Script**: `scripts/data_drift.py`

**Funcionamiento:**
1. Carga el dataset baseline (entrenamiento) desde Parquet
2. Carga datos de producción (nuevos uploads) desde CSV
3. Ejecuta `DataDriftPreset()` de Evidently que aplica tests estadísticos por columna:
   - Kolmogorov-Smirnov para numéricas
   - Chi-squared para categóricas
4. Genera reporte HTML en `data/reports/drift_report.html`
5. Si detecta drift → envía alerta a Slack via webhook

**Ejecución recomendada**: Cron semanal (lunes 9:00)

---

## 8. Testing

### 8.1 Framework: pytest

**Configuración**: `tests/conftest.py` añade el root del proyecto al sys.path.

### 8.2 Tests Implementados

| Test | Qué valida |
|------|-----------|
| `test_health` | GET /health retorna 200 + {"status": "ok"} |
| `test_predict_valid_payload` | POST /predict con datos completos retorna 200 o 503 |
| `test_predict_invalid_payload` | POST /predict con datos incompletos retorna 422 |
| `test_training_history` | GET /training-history retorna 200 + campo "runs" |
| `test_upload_non_csv` | POST /upload con .txt retorna 400 |

### 8.3 Estrategia

- Se usa `FastAPI TestClient` (basado en httpx) para tests de integración sin levantar servidor real.
- `test_predict_valid_payload` acepta 200 (modelo existe) o 503 (modelo no entrenado) para funcionar en CI sin modelo.

---

## 9. Flujo de Datos End-to-End

### 9.1 Entrenamiento

```
CSV (data/raw/)
    │
    ▼
[Ingesta] Validación + Limpieza + Outliers
    │
    ▼
Parquet (data/processed/)
    │
    ▼
[Train/Test Split] 80/20, random_state=42
    │
    ▼
[Feature Engineering] Target encoding (solo train) + n_amenities
    │
    ▼
[Optuna] 10 trials × 5-Fold CV → mejores params
    │
    ▼
[Entrenamiento Final] Pipeline(preprocessor + XGBoost)
    │
    ▼
[MLflow] Log params + metrics + artifacts
    │
    ▼
[Serialización] model.pkl + zona_medians.pkl + global_median.pkl
```

### 9.2 Inferencia

```
Request JSON (24 campos)
    │
    ▼
[Pydantic] Validación de tipos y rangos
    │
    ▼
[Routes] Carga model.pkl + zona_medians.pkl
    │
    ▼
[Feature Calc] geo_zone → price_per_m2_zone, sum → n_amenities
    │
    ▼
[DataFrame] 26 features en orden correcto
    │
    ▼
[Pipeline.predict()] Imputer → Scaler → XGBoost
    │
    ▼
[expm1()] Inversión logarítmica
    │
    ▼
Response: {precio_estimado: float}
```

### 9.3 Reentrenamiento

```
Upload CSV → data/raw/
    │
    ▼
[BackgroundTask] (async, no bloquea API)
    │
    ▼
[Ingesta] Misma validación y limpieza
    │
    ▼
[train_baseline()] Pipeline completo
    │
    ▼
Nuevos .pkl sobreescriben los anteriores
    │
    ▼
Siguiente request usa modelo actualizado
```

---

## Apéndice A: Dependencias Principales

| Paquete | Versión | Uso |
|---------|---------|-----|
| pandas | ≥2.2.0 | Manipulación de datos |
| numpy | ≥1.26.4 | Operaciones numéricas |
| scikit-learn | ==1.7.2 | Preprocesamiento + pipeline (versión fijada por compatibilidad) |
| xgboost | ≥2.0.3 | Modelo de gradient boosting |
| optuna | ≥3.5.0 | Optimización bayesiana |
| mlflow | ≥2.11.3 | Experiment tracking |
| fastapi | ≥0.110.0 | Framework API REST |
| uvicorn | ≥0.27.1 | Servidor ASGI |
| streamlit | ≥1.31.1 | Framework frontend |
| evidently | ≥0.4.16 | Detección de data drift |
| plotly | ≥5.18.0 | Gráficos interactivos |
| folium | ≥0.15.0 | Mapas interactivos |
| fpdf2 | ≥2.7.0 | Generación de PDFs |
| httpx | ≥0.27.0 | Cliente HTTP async |
| joblib | ≥1.3.2 | Serialización de modelos |
| pyarrow | ≥15.0.0 | Soporte Parquet |

## Apéndice B: Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| API_URL | http://localhost:8000/api/v1 | URL del backend (frontend) |
| MLFLOW_TRACKING_URI | file:///app/mlruns | Backend de almacenamiento MLflow |
| SLACK_WEBHOOK_URL | (ninguno) | Webhook para alertas de drift |

## Apéndice C: Puertos

| Servicio | Puerto | Protocolo |
|----------|--------|-----------|
| Backend (FastAPI) | 8000 | HTTP |
| Frontend (Streamlit) | 8501 | HTTP |
| MLflow UI (opcional) | 5000 | HTTP |


---

## 10. Troubleshooting

### 10.1 Errores Comunes

#### Error: `ModuleNotFoundError: No module named 'src'`

**Causa**: Python no encuentra el paquete raíz del proyecto.

**Solución**:
- Para pytest: existe `tests/conftest.py` que añade el path automáticamente.
- Para ejecución directa: ejecutar siempre desde la raíz del proyecto.

```bash
cd /ruta/del/proyecto
python -c "from src.ml.pipeline import train_baseline; train_baseline()"
```

---

#### Error: `FileNotFoundError: data/processed/idealista18_madrid.parquet`

**Causa**: No se ha ejecutado la ingesta antes del entrenamiento.

**Solución**:
```bash
python -c "from src.data.ingestion import run_ingestion; run_ingestion()"
```

---

#### Error: `address already in use` (puerto 8000 o 8501)

**Causa**: Hay otro proceso ocupando el puerto.

**Solución**:
```bash
# Identificar y matar el proceso
kill $(lsof -t -i:8000)
kill $(lsof -t -i:8501)

# Relanzar
docker compose up -d
```

---

#### Error: `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`

**Causa**: Incompatibilidad de versiones de scikit-learn. El modelo fue entrenado con una versión diferente a la del entorno de inferencia.

**Solución**: Asegurar que scikit-learn sea exactamente la misma versión en todos los entornos:
```
scikit-learn==1.7.2
```
Si el error aparece en Docker, hacer rebuild:
```bash
docker compose up --build -d
```

---

#### Error: `FPDFUnicodeEncodingException` al exportar PDF

**Causa**: Caracteres emoji o Unicode no soportados por la fuente Helvetica de fpdf2.

**Solución**: Ya implementada. La función `_clean_text()` sanitiza emojis antes de escribir en el PDF. Si aparece con texto nuevo, verificar que se llama `_clean_text()` sobre cualquier string que vaya al PDF.

---

#### Error: `remote rejected - refusing to allow a Personal Access Token to create or update workflow`

**Causa**: El token de GitHub no tiene el scope `workflow`.

**Solución**: Regenerar el token en https://github.com/settings/tokens con los permisos `repo` + `workflow` marcados.

---

#### Error: `JSONDecodeError` en el frontend al predecir

**Causa**: La API devuelve una respuesta no-JSON (error 500 interno).

**Solución**:
1. Revisar logs del backend: `docker compose logs backend --tail=30`
2. Causas comunes:
   - Modelo no cargado → verificar que `models/model.pkl` existe
   - Versión de sklearn incompatible → verificar `requirements.txt`
   - Error en el cálculo de features → revisar que el payload tiene todos los campos

---

#### Docker: Frontend no conecta con backend

**Causa**: El frontend intenta conectar a `localhost` en vez del nombre del servicio Docker.

**Solución**: Verificar que la variable `API_URL` está configurada en docker-compose.yml:
```yaml
frontend:
  environment:
    - API_URL=http://backend:8000/api/v1
```

---

#### Entrenamiento tarda demasiado

**Causa**: Demasiados trials de Optuna o dataset muy grande.

**Solución**:
- Reducir trials en `src/ml/pipeline.py`:
  ```python
  study = optimizar_hiperparametros(X_train, y_train, n_trials=10)  # en vez de 30
  ```
- Ejecutar en background:
  ```bash
  nohup python3 -c 'from src.ml.pipeline import train_baseline; train_baseline()' > training.log 2>&1 &
  tail -f training.log
  ```
- Tiempo estimado: ~2 min por trial con 94K filas en CPU estándar.

---

### 10.2 Comandos Útiles de Diagnóstico

```bash
# Ver si los contenedores están corriendo
docker compose ps

# Logs en tiempo real
docker compose logs -f

# Logs solo del backend
docker compose logs backend --tail=50

# Verificar que la API responde
curl http://localhost:8000/health

# Verificar que el modelo existe dentro del contenedor
docker compose exec backend ls /app/models/

# Ver procesos Python corriendo
ps aux | grep python | grep -v grep

# Verificar versiones dentro del contenedor
docker compose exec backend python -c "import sklearn; print(sklearn.__version__)"

# Reconstruir sin cache
docker compose build --no-cache
docker compose up -d
```

---

### 10.3 Reset Completo

Si todo falla y quieres empezar de cero:

```bash
# Parar y eliminar contenedores, redes, volúmenes
docker compose down -v

# Eliminar imágenes del proyecto
docker rmi $(docker images | grep p7_proy4 | awk '{print $3}')

# Eliminar artefactos locales
rm -rf models/ mlruns/ data/processed/ mlflow.db training.log

# Re-ingesta + re-entrenamiento
source venv/bin/activate
python -c "from src.data.ingestion import run_ingestion; run_ingestion()"
python -c "from src.ml.pipeline import train_baseline; train_baseline()"

# Rebuild Docker
docker compose up --build -d
```

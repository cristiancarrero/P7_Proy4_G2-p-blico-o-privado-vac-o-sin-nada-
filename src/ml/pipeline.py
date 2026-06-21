import numpy as np
import pandas as pd

# Features para el modelo
NUMERICAL = [
    "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
    "LATITUDE", "LONGITUDE",
    "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA",
    "CADMAXBUILDINGFLOOR", "CADDWELLINGCOUNT", "CADASTRALQUALITYID",
    "FLOORCLEAN",
    # Engineered
    "price_per_m2_zone", "n_amenities",
]

BINARY = [
    "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING",
    "HASPARKINGSPACE", "HASBOXROOM", "HASWARDROBE",
    "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN",
    "ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR",
]

ALL_FEATURES = NUMERICAL + BINARY


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Número total de amenities
    amenity_cols = [c for c in df.columns if c.startswith("HAS") or c in ["ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR"]]
    df["n_amenities"] = df[amenity_cols].sum(axis=1)
    # Imputar nulls
    df["FLOORCLEAN"] = df["FLOORCLEAN"].fillna(df["FLOORCLEAN"].median())
    df["CADASTRALQUALITYID"] = df["CADASTRALQUALITYID"].fillna(df["CADASTRALQUALITYID"].median())
    return df


def _prepare_data():
    from sklearn.model_selection import train_test_split

    df_raw = pd.read_parquet("data/processed/idealista18_madrid.parquet")

    # Target encoding espacial: precio mediano por zona geográfica (grid)
    df_raw["lat_bin"] = pd.cut(df_raw["LATITUDE"], bins=20, labels=False)
    df_raw["lon_bin"] = pd.cut(df_raw["LONGITUDE"], bins=20, labels=False)
    df_raw["geo_zone"] = df_raw["lat_bin"].astype(str) + "_" + df_raw["lon_bin"].astype(str)

    # Guardar bin edges para reproducir en inferencia
    _, lat_edges = pd.cut(df_raw["LATITUDE"], bins=20, labels=False, retbins=True)
    _, lon_edges = pd.cut(df_raw["LONGITUDE"], bins=20, labels=False, retbins=True)

    train_raw, test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)

    # Target encoding calculado solo en train
    zone_medians_train = train_raw.groupby("geo_zone")["PRICE"].median() / train_raw.groupby("geo_zone")["CONSTRUCTEDAREA"].median()
    global_median = (train_raw["PRICE"] / train_raw["CONSTRUCTEDAREA"]).median()

    def add_zone_encoding(df, medians, fallback):
        df = df.copy()
        df["price_per_m2_zone"] = df["geo_zone"].map(medians).fillna(fallback)
        return df

    train_df = build_features(add_zone_encoding(train_raw, zone_medians_train, global_median))
    test_df = build_features(add_zone_encoding(test_raw, zone_medians_train, global_median))

    X_train = train_df[ALL_FEATURES]
    y_train = np.log1p(train_df["PRICE"])
    X_test = test_df[ALL_FEATURES]
    y_test = np.log1p(test_df["PRICE"])

    return X_train, X_test, y_train, y_test, zone_medians_train, global_median, lat_edges, lon_edges


def _build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.impute import SimpleImputer

    num_pipe = SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("num", num_pipe, NUMERICAL),
        ("bin", "passthrough", BINARY),
    ])


def optimizar_hiperparametros(X_train, y_train, n_trials=30):
    import optuna
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.INFO)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor = _build_preprocessor()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }
        pipe = Pipeline([("prep", preprocessor), ("model", XGBRegressor(**params))])
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def train_baseline():
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor
    import joblib
    import os
    import mlflow
    import mlflow.xgboost

    X_train, X_test, y_train, y_test, zone_medians, global_median, lat_edges, lon_edges = _prepare_data()
    preprocessor = _build_preprocessor()

    print("Optimizando hiperparametros con Optuna (10 trials)...")
    study = optimizar_hiperparametros(X_train, y_train, n_trials=10)
    print(f"Mejores params: {study.best_params}")

    best_params = {**study.best_params, "random_state": 42, "n_jobs": -1}
    pipe = Pipeline([("prep", preprocessor), ("model", XGBRegressor(**best_params))])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    print(f"CV R2 medio: {cv_r2_scores.mean():.3f} (+/- {cv_r2_scores.std():.3f})")

    pipe.fit(X_train, y_train)
    pred_test = pipe.predict(X_test)
    test_r2 = r2_score(y_test, pred_test)
    test_rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(pred_test))
    print(f"Test R2 (holdout): {test_r2:.3f}  RMSE: EUR{test_rmse:.0f}")

    mlflow.set_experiment("idealista18_madrid_venta")
    with mlflow.start_run(run_name="xgboost-optuna-idealista18"):
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_r2_mean", cv_r2_scores.mean())
        mlflow.log_metric("cv_r2_std", cv_r2_scores.std())
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_rmse_eur", test_rmse)
        mlflow.log_param("dataset_fuente", "idealista18")
        mlflow.log_param("n_filas_entrenamiento", len(X_train) + len(X_test))
        mlflow.xgboost.log_model(pipe.named_steps["model"], "xgboost_model")
        print("Run registrado en MLflow")

    # Auto-swap: solo reemplazar si el nuevo modelo es mejor
    import json
    from datetime import datetime

    os.makedirs("models", exist_ok=True)
    metrics_path = "models/latest_metrics.json"
    prev_r2 = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            prev_r2 = json.load(f).get("test_r2")

    if prev_r2 is not None and test_r2 <= prev_r2:
        print(f"⚠️ Nuevo modelo (R²={test_r2:.4f}) NO supera al actual (R²={prev_r2:.4f}). Modelo NO reemplazado.")
        return pipe, cv_r2_scores, test_r2

    if prev_r2 is not None:
        print(f"✅ Nuevo modelo (R²={test_r2:.4f}) supera al actual (R²={prev_r2:.4f}). Reemplazando...")

    joblib.dump(pipe, "models/model.pkl")
    joblib.dump(zone_medians, "models/zona_medians.pkl")
    joblib.dump(global_median, "models/global_median.pkl")
    joblib.dump({"lat_edges": lat_edges, "lon_edges": lon_edges}, "models/bin_edges.pkl")

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": "xgboost-optuna-idealista18",
        "cv_r2_mean": round(float(cv_r2_scores.mean()), 4),
        "cv_r2_std": round(float(cv_r2_scores.std()), 4),
        "test_r2": round(float(test_r2), 4),
        "test_rmse_eur": round(float(test_rmse), 2),
        "n_filas": len(X_train) + len(X_test),
        "best_params": best_params,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Modelo guardado en models/model.pkl")
    print("Métricas guardadas en models/latest_metrics.json")

    return pipe, cv_r2_scores, test_r2


if __name__ == "__main__":
    train_baseline()

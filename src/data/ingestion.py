import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/idealista18_madrid_sale (1).csv.gz")
PROCESSED_PATH = Path("data/processed/idealista18_madrid.parquet")

EXPECTED_COLUMNS = [
    "PRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
    "LATITUDE", "LONGITUDE", "DISTANCE_TO_CITY_CENTER",
    "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA",
]


def load_idealista18(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _validate_and_clean(df)


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")

    before = len(df)
    df = df[df["PRICE"] > 0].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[ingestion] Descartadas {dropped} filas con PRICE<=0")

    # Eliminar outliers extremos (top 1% de precio)
    q99 = df["PRICE"].quantile(0.99)
    before = len(df)
    df = df[df["PRICE"] <= q99].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[ingestion] Descartadas {dropped} filas outliers (PRICE > {q99:.0f})")

    return df


def run_ingestion(source: str = "idealista18", path: Path = RAW_PATH) -> pd.DataFrame:
    if source == "idealista18":
        df = load_idealista18(path)
    else:
        raise NotImplementedError(f"Fuente no soportada: {source}")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"[ingestion] {len(df)} filas guardadas en {PROCESSED_PATH}")
    return df

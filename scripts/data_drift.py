"""
Script de detección de Data Drift usando Evidently AI.
Ejecutar semanalmente via cron:
  0 9 * * 1 cd /opt/P7_Proy4_G2 && python scripts/data_drift.py
"""
import json
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

BASELINE_PATH = Path("data/processed/idealista18_madrid.parquet")
PRODUCTION_PATH = Path("data/raw/production_data.csv")
REPORT_PATH = Path("data/reports")

NUMERICAL_FEATURES = ["CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "PRICE"]


def run_drift_analysis():
    if not BASELINE_PATH.exists():
        print("[drift] No existe dataset baseline. Abortando.")
        return None

    if not PRODUCTION_PATH.exists():
        print("[drift] No hay datos de producción. Abortando.")
        return None

    baseline = pd.read_parquet(BASELINE_PATH)
    production = pd.read_csv(PRODUCTION_PATH)

    # Usar solo columnas comunes numéricas
    cols = [c for c in NUMERICAL_FEATURES if c in baseline.columns and c in production.columns]
    baseline = baseline[cols]
    production = production[cols]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline, current_data=production)

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH / "drift_report.html"))

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    print(f"[drift] Drift detectado: {drift_detected}")

    if drift_detected:
        _send_alert(result)

    return drift_detected


def _send_alert(result):
    """Enviar alerta via webhook (Slack)."""
    import os
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("[drift] SLACK_WEBHOOK_URL no configurado. Alerta no enviada.")
        return

    import httpx
    message = {
        "text": "⚠️ *Data Drift detectado* en el modelo de alquiler Madrid.\n"
                "Revisa el reporte en `data/reports/drift_report.html` y considera reentrenar."
    }
    httpx.post(webhook_url, json=message)
    print("[drift] Alerta enviada a Slack.")


if __name__ == "__main__":
    run_drift_analysis()

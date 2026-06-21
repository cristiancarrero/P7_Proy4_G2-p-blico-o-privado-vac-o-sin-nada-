import streamlit as st
import httpx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import os
import base64
from datetime import datetime

API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")

# --- Config ---
st.set_page_config(page_title="Predictor Precio Vivienda Madrid", page_icon="🏠", layout="wide")

# --- CSS Custom ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #4a90d9 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #cde; margin: 0.3rem 0 0 0; font-size: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #4a90d9;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .price-big {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        padding: 1rem;
    }
    .price-secondary {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
    }
    .model-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🏠 Predictor de Precio de Vivienda</h1>
    <p>Madrid · Modelo XGBoost · R² = 0.949 · Dataset Idealista18</p>
</div>
""", unsafe_allow_html=True)

# --- Barrios predefinidos con coordenadas y distancias ---
BARRIOS = {
    "📍 Personalizado": {"lat": 40.4168, "lon": -3.7038, "dist_center": 0.5, "dist_metro": 0.2, "dist_castellana": 1.0},
    "Centro (Sol)": {"lat": 40.4169, "lon": -3.7035, "dist_center": 0.2, "dist_metro": 0.1, "dist_castellana": 0.8},
    "Salamanca": {"lat": 40.4300, "lon": -3.6800, "dist_center": 1.5, "dist_metro": 0.2, "dist_castellana": 0.3},
    "Chamberí": {"lat": 40.4350, "lon": -3.7070, "dist_center": 1.8, "dist_metro": 0.3, "dist_castellana": 0.7},
    "Chamartín": {"lat": 40.4580, "lon": -3.6770, "dist_center": 4.5, "dist_metro": 0.4, "dist_castellana": 0.5},
    "Retiro": {"lat": 40.4090, "lon": -3.6780, "dist_center": 1.5, "dist_metro": 0.3, "dist_castellana": 1.2},
    "Arganzuela": {"lat": 40.3930, "lon": -3.6970, "dist_center": 2.5, "dist_metro": 0.4, "dist_castellana": 2.0},
    "Tetuán": {"lat": 40.4600, "lon": -3.7020, "dist_center": 4.5, "dist_metro": 0.3, "dist_castellana": 0.8},
    "Moncloa-Aravaca": {"lat": 40.4400, "lon": -3.7400, "dist_center": 3.5, "dist_metro": 0.5, "dist_castellana": 2.5},
    "Latina": {"lat": 40.3900, "lon": -3.7400, "dist_center": 3.5, "dist_metro": 0.4, "dist_castellana": 3.0},
    "Carabanchel": {"lat": 40.3700, "lon": -3.7300, "dist_center": 5.0, "dist_metro": 0.5, "dist_castellana": 4.0},
    "Usera": {"lat": 40.3800, "lon": -3.7000, "dist_center": 4.5, "dist_metro": 0.4, "dist_castellana": 3.5},
    "Puente de Vallecas": {"lat": 40.3900, "lon": -3.6600, "dist_center": 4.0, "dist_metro": 0.4, "dist_castellana": 3.5},
    "Ciudad Lineal": {"lat": 40.4450, "lon": -3.6500, "dist_center": 5.0, "dist_metro": 0.3, "dist_castellana": 2.5},
    "Hortaleza": {"lat": 40.4750, "lon": -3.6400, "dist_center": 7.5, "dist_metro": 0.5, "dist_castellana": 3.5},
    "Fuencarral-El Pardo": {"lat": 40.5100, "lon": -3.7200, "dist_center": 10.0, "dist_metro": 0.8, "dist_castellana": 5.0},
    "Villaverde": {"lat": 40.3500, "lon": -3.7000, "dist_center": 7.5, "dist_metro": 0.6, "dist_castellana": 6.0},
    "Barajas": {"lat": 40.4700, "lon": -3.5800, "dist_center": 12.0, "dist_metro": 0.8, "dist_castellana": 7.0},
}

# --- Función para generar PDF ---
def _clean_text(text):
    """Eliminar emojis y caracteres no-latin del texto para PDF."""
    import re
    return re.sub(r'[^\x00-\x7F\xC0-\xFF]+', '', str(text)).strip()


def generar_pdf(predicciones):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Informe de Prediccion - Precio Vivienda Madrid", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
    pdf.cell(0, 8, "Modelo: XGBoost + Optuna | Dataset: Idealista18 | R2 = 0.949", ln=True, align="C")
    pdf.ln(10)

    # Tabla
    pdf.set_font("Helvetica", "B", 10)
    cols = ["Barrio", "m2", "Hab.", "Precio", "EUR/m2"]
    col_widths = [50, 20, 20, 40, 30]
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 8, col, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for pred in predicciones:
        pdf.cell(col_widths[0], 8, _clean_text(pred.get("Barrio", ""))[:25], border=1)
        pdf.cell(col_widths[1], 8, _clean_text(pred.get("m2", "")), border=1, align="C")
        pdf.cell(col_widths[2], 8, _clean_text(pred.get("Hab.", "")), border=1, align="C")
        pdf.cell(col_widths[3], 8, _clean_text(pred.get("Precio", "")), border=1, align="C")
        pdf.cell(col_widths[4], 8, _clean_text(pred.get("EUR/m2", "")), border=1, align="C")
        pdf.ln()

    return bytes(pdf.output())


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predicción", "📊 Historial & Métricas", "🏘️ Comparativa Barrios", "⚙️ Reentrenamiento"])

# ============ TAB 1: PREDICCIÓN ============
with tab1:
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Configura tu vivienda")

        # Barrio selector
        barrio = st.selectbox("🏘️ Barrio de Madrid", list(BARRIOS.keys()))
        barrio_data = BARRIOS[barrio]

        # Vivienda
        st.markdown("**🏠 Vivienda**")
        c1, c2 = st.columns(2)
        with c1:
            constructed_area = st.number_input("Superficie (m²)", 20, 400, 80, step=5)
            room_number = st.number_input("Habitaciones", 0, 8, 3)
        with c2:
            bath_number = st.number_input("Baños", 1, 5, 1)
            floor_clean = st.number_input("Planta", 0, 15, 2)

        # Ubicación
        st.markdown("**📍 Ubicación**")
        if barrio == "📍 Personalizado":
            c1, c2 = st.columns(2)
            with c1:
                latitude = st.number_input("Latitud", value=40.4168, format="%.6f")
                dist_center = st.number_input("Dist. centro (km)", 0.0, 20.0, 2.5, 0.1)
            with c2:
                longitude = st.number_input("Longitud", value=-3.7038, format="%.6f")
                dist_metro = st.number_input("Dist. metro (km)", 0.0, 5.0, 0.3, 0.1)
            dist_castellana = st.number_input("Dist. Castellana (km)", 0.0, 15.0, 1.5, 0.1)
        else:
            latitude = barrio_data["lat"]
            longitude = barrio_data["lon"]
            dist_center = barrio_data["dist_center"]
            dist_metro = barrio_data["dist_metro"]
            dist_castellana = barrio_data["dist_castellana"]
            st.info(f"📌 {barrio}: lat={latitude}, lon={longitude}, centro={dist_center}km, metro={dist_metro}km")

        # Edificio
        st.markdown("**🏗️ Edificio**")
        c1, c2, c3 = st.columns(3)
        with c1:
            max_building_floor = st.number_input("Plantas edificio", 1, 20, 5)
        with c2:
            dwelling_count = st.number_input("Nº viviendas", 1, 500, 20)
        with c3:
            cadastral_quality = st.number_input("Calidad catastral", 1, 9, 5)

        # Amenities
        st.markdown("**✨ Amenities**")
        c1, c2, c3 = st.columns(3)
        with c1:
            has_lift = int(st.checkbox("Ascensor", value=True))
            has_terrace = int(st.checkbox("Terraza"))
            has_ac = int(st.checkbox("Aire acond."))
            has_parking = int(st.checkbox("Parking"))
        with c2:
            has_boxroom = int(st.checkbox("Trastero"))
            has_wardrobe = int(st.checkbox("Armarios"))
            has_pool = int(st.checkbox("Piscina"))
            has_doorman = int(st.checkbox("Portero"))
        with c3:
            has_garden = int(st.checkbox("Jardín"))
            is_duplex = int(st.checkbox("Dúplex"))
            is_studio = int(st.checkbox("Estudio"))
            is_topfloor = int(st.checkbox("Último piso"))

    with col_result:
        st.subheader("Resultado")

        # Inicializar coordenadas en session_state
        if "lat" not in st.session_state:
            st.session_state.lat = latitude
            st.session_state.lon = longitude

        # Actualizar si cambió el barrio
        if barrio != "📍 Personalizado":
            st.session_state.lat = latitude
            st.session_state.lon = longitude

        # Mapa interactivo - click para mover el marcador
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=13)
        folium.Marker(
            [st.session_state.lat, st.session_state.lon],
            popup=f"Lat: {st.session_state.lat:.5f}, Lon: {st.session_state.lon:.5f}",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
            draggable=True,
        ).add_to(m)
        map_data = st_folium(m, height=350, width=None, key="map", returned_objects=["last_clicked"])

        # Si el usuario hizo click en el mapa, mover el marcador
        if map_data and map_data.get("last_clicked"):
            st.session_state.lat = map_data["last_clicked"]["lat"]
            st.session_state.lon = map_data["last_clicked"]["lng"]
            latitude = st.session_state.lat
            longitude = st.session_state.lon
            st.rerun()

        # Usar coordenadas del session_state para la predicción
        latitude = st.session_state.lat
        longitude = st.session_state.lon
        st.caption(f"📌 Coordenadas: {latitude:.5f}, {longitude:.5f}")

        # Botón de predicción
        if st.button("🔮 Predecir precio", type="primary", use_container_width=True):
            payload = {
                "constructed_area": constructed_area,
                "room_number": room_number,
                "bath_number": bath_number,
                "latitude": latitude,
                "longitude": longitude,
                "distance_to_city_center": dist_center,
                "distance_to_metro": dist_metro,
                "distance_to_castellana": dist_castellana,
                "cadmaxbuildingfloor": max_building_floor,
                "caddwellingcount": dwelling_count,
                "cadastralqualityid": float(cadastral_quality),
                "floor_clean": float(floor_clean),
                "has_terrace": has_terrace,
                "has_lift": has_lift,
                "has_air_conditioning": has_ac,
                "has_parking_space": has_parking,
                "has_boxroom": has_boxroom,
                "has_wardrobe": has_wardrobe,
                "has_swimming_pool": has_pool,
                "has_doorman": has_doorman,
                "has_garden": has_garden,
                "is_duplex": is_duplex,
                "is_studio": is_studio,
                "is_intopfloor": is_topfloor,
            }
            try:
                with st.spinner("Calculando..."):
                    resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    precio = data["precio_estimado"]
                    precio_m2 = precio / constructed_area

                    st.markdown(f'<div class="price-big">{precio:,.0f} €</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="price-secondary">{precio_m2:,.0f} €/m² · Precio de venta</div>', unsafe_allow_html=True)

                    # Métricas comparativas
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precio total", f"{precio:,.0f} €")
                    m2.metric("Precio/m²", f"{precio_m2:,.0f} €")
                    m3.metric("Superficie", f"{constructed_area} m²")

                    st.markdown(f'<span class="model-badge">✅ Modelo: {data["modelo_usado"]}</span>', unsafe_allow_html=True)

                    # Guardar en session_state para comparador
                    if "predictions" not in st.session_state:
                        st.session_state.predictions = []
                    st.session_state.predictions.append({
                        "Barrio": barrio,
                        "m²": constructed_area,
                        "Hab.": room_number,
                        "Precio": f"{precio:,.0f} €",
                        "€/m²": f"{precio_m2:,.0f}",
                    })

                else:
                    try:
                        detail = resp.json().get('detail', resp.text)
                    except Exception:
                        detail = resp.text
                    st.error(f"Error: {detail}")
            except httpx.ConnectError:
                st.error("❌ No se pudo conectar con la API. ¿Está el backend corriendo?")

        # Comparador
        if "predictions" in st.session_state and st.session_state.predictions:
            st.markdown("---")
            st.markdown("**📋 Historial de predicciones (sesión)**")
            df_preds = pd.DataFrame(st.session_state.predictions)
            st.dataframe(df_preds, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Limpiar historial"):
                    st.session_state.predictions = []
                    st.rerun()
            with c2:
                pdf_bytes = generar_pdf(st.session_state.predictions)
                st.download_button(
                    "📄 Exportar a PDF",
                    data=pdf_bytes,
                    file_name=f"prediccion_madrid_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )


# ============ TAB 2: HISTORIAL & MÉTRICAS ============
with tab2:
    col_metrics, col_chart = st.columns([1, 1])

    with col_metrics:
        st.subheader("🎯 Métricas del modelo actual")
        st.markdown("""
        <div class="metric-card">
            <strong>R² (Test Holdout)</strong><br>
            <span style="font-size: 1.8rem; color: #28a745;">0.949</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <strong>R² (Cross-Validation 5-Fold)</strong><br>
            <span style="font-size: 1.8rem; color: #28a745;">0.945</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <strong>RMSE</strong><br>
            <span style="font-size: 1.8rem; color: #fd7e14;">73,241 €</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <strong>Dataset</strong><br>
            <span style="font-size: 1.1rem;">Idealista18 Madrid · 93,868 viviendas</span>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        st.subheader("📈 Feature Importance")
        # Feature importance aproximado basado en correlaciones del dataset
        features_imp = {
            "Superficie (m²)": 0.35,
            "Ubicación (lat/lon)": 0.20,
            "Dist. centro": 0.12,
            "Baños": 0.10,
            "Calidad catastral": 0.08,
            "Precio/m² zona": 0.06,
            "Habitaciones": 0.04,
            "Amenities": 0.03,
            "Planta": 0.02,
        }
        df_imp = pd.DataFrame({"Feature": list(features_imp.keys()), "Importancia": list(features_imp.values())})
        df_imp = df_imp.sort_values("Importancia", ascending=True)
        fig = px.bar(df_imp, x="Importancia", y="Feature", orientation="h",
                     color="Importancia", color_continuous_scale="Blues",
                     title="Importancia relativa de las variables")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Historial MLflow
    st.markdown("---")
    st.subheader("🗂️ Historial de entrenamientos (MLflow)")
    if st.button("Cargar historial desde MLflow"):
        try:
            resp = httpx.get(f"{API_URL}/training-history", timeout=30)
            if resp.status_code == 200:
                runs = resp.json().get("runs", [])
                if not runs:
                    st.warning("No hay runs registrados en MLflow")
                else:
                    metrics_data = []
                    for run in runs:
                        metrics_data.append({
                            "Run": run.get("run_name", "N/A"),
                            "R² Test": run["metrics"].get("test_r2"),
                            "RMSE (€)": run["metrics"].get("test_rmse_eur"),
                            "R² CV": run["metrics"].get("cv_r2_mean"),
                            "Dataset": run["params"].get("dataset_fuente", "N/A"),
                            "Filas": run["params"].get("n_filas_entrenamiento", "N/A"),
                        })
                    df_runs = pd.DataFrame(metrics_data)
                    st.dataframe(df_runs, use_container_width=True, hide_index=True)

                    df_plot = df_runs.dropna(subset=["R² Test"])
                    if not df_plot.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=df_plot["Run"], y=df_plot["R² Test"], name="R² Test",
                                             marker_color="#4a90d9"))
                        fig.add_hline(y=0.85, line_dash="dash", line_color="red",
                                      annotation_text="Mínimo aceptable (0.85)")
                        fig.update_layout(title="R² por entrenamiento", yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Error: {resp.text}")
        except httpx.ConnectError:
            st.error("❌ No se pudo conectar con la API.")




# ============ TAB 3: COMPARATIVA BARRIOS ============
with tab3:
    st.subheader("🏘️ Comparativa de precios por barrio")
    st.markdown("Compara el precio estimado de una misma vivienda en distintos barrios de Madrid.")

    st.markdown("**Configura la vivienda base para comparar:**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        comp_area = st.number_input("Superficie (m²)", 20, 400, 80, step=5, key="comp_area")
    with c2:
        comp_rooms = st.number_input("Habitaciones", 0, 8, 3, key="comp_rooms")
    with c3:
        comp_baths = st.number_input("Baños", 1, 5, 1, key="comp_baths")
    with c4:
        comp_floor = st.number_input("Planta", 0, 15, 2, key="comp_floor")

    barrios_seleccionados = st.multiselect(
        "Selecciona barrios a comparar",
        [b for b in BARRIOS.keys() if b != "📍 Personalizado"],
        default=["Centro (Sol)", "Salamanca", "Chamberí", "Carabanchel", "Villaverde"],
    )

    if st.button("📊 Comparar barrios", type="primary"):
        resultados = []
        progress = st.progress(0)
        for i, barrio_name in enumerate(barrios_seleccionados):
            bd = BARRIOS[barrio_name]
            payload = {
                "constructed_area": comp_area,
                "room_number": comp_rooms,
                "bath_number": comp_baths,
                "latitude": bd["lat"],
                "longitude": bd["lon"],
                "distance_to_city_center": bd["dist_center"],
                "distance_to_metro": bd["dist_metro"],
                "distance_to_castellana": bd["dist_castellana"],
                "cadmaxbuildingfloor": 5,
                "caddwellingcount": 20,
                "cadastralqualityid": 5.0,
                "floor_clean": float(comp_floor),
                "has_terrace": 0, "has_lift": 1, "has_air_conditioning": 0,
                "has_parking_space": 0, "has_boxroom": 0, "has_wardrobe": 0,
                "has_swimming_pool": 0, "has_doorman": 0, "has_garden": 0,
                "is_duplex": 0, "is_studio": 0, "is_intopfloor": 0,
            }
            try:
                resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=30)
                if resp.status_code == 200:
                    precio = resp.json()["precio_estimado"]
                    resultados.append({
                        "Barrio": barrio_name,
                        "Precio (€)": precio,
                        "€/m²": precio / comp_area,
                    })
            except Exception:
                pass
            progress.progress((i + 1) / len(barrios_seleccionados))

        if resultados:
            df_comp = pd.DataFrame(resultados).sort_values("Precio (€)", ascending=False)

            fig = px.bar(
                df_comp, x="Barrio", y="Precio (€)",
                color="Precio (€)", color_continuous_scale="RdYlGn_r",
                title=f"Precio estimado por barrio ({comp_area}m², {comp_rooms} hab., {comp_baths} baño)",
            )
            fig.update_layout(coloraxis_showscale=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

            df_show = df_comp.copy()
            df_show["Precio (€)"] = df_show["Precio (€)"].apply(lambda x: f"{x:,.0f} €")
            df_show["€/m²"] = df_show["€/m²"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.error("No se pudieron obtener predicciones. ¿Está la API corriendo?")


# ============ TAB 4: REENTRENAMIENTO ============
with tab4:
    st.subheader("📁 Subir nuevo dataset y reentrenar el modelo")

    st.markdown("""
    > **Instrucciones**: Sube un archivo CSV con la misma estructura que Idealista18 
    > (columnas: PRICE, CONSTRUCTEDAREA, ROOMNUMBER, BATHNUMBER, LATITUDE, LONGITUDE, etc.)
    > El modelo se reentrenará automáticamente en background.
    """)

    uploaded_file = st.file_uploader("Arrastra o selecciona un archivo CSV", type=["csv", "gz"])

    if uploaded_file:
        st.success(f"📄 Archivo seleccionado: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")

        try:
            if uploaded_file.name.endswith(".gz"):
                df_preview = pd.read_csv(uploaded_file, compression="gzip", nrows=5)
            else:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
            st.markdown("**Vista previa (5 primeras filas):**")
            st.dataframe(df_preview, use_container_width=True, hide_index=True)
            uploaded_file.seek(0)
        except Exception as e:
            st.warning(f"No se pudo previsualizar: {e}")

        if st.button("🚀 Subir y reentrenar", type="primary"):
            try:
                with st.spinner("Subiendo dataset..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                    resp = httpx.post(f"{API_URL}/datasets/upload", files=files, timeout=60)

                if resp.status_code == 200:
                    st.success("✅ Dataset subido correctamente")
                    with st.spinner("Lanzando reentrenamiento..."):
                        resp2 = httpx.post(
                            f"{API_URL}/retrain-with-dataset",
                            params={"filename": uploaded_file.name},
                            timeout=10,
                        )
                    if resp2.status_code == 200:
                        st.balloons()
                        st.info("⏳ Reentrenamiento lanzado en background. Revisa el historial en unos minutos.")
                    else:
                        st.error(f"Error al reentrenar: {resp2.text}")
                else:
                    st.error(f"Error al subir: {resp.text}")
            except httpx.ConnectError:
                st.error("❌ No se pudo conectar con la API.")


# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6c757d; font-size: 0.85rem;'>"
    "Proyecto 4 · Grupo 2 · Bootcamp IA MAD P7 · Modelo XGBoost + Optuna · Dataset Idealista18"
    "</p>",
    unsafe_allow_html=True,
)

"""
Pakistan Energy Forecasting — Streamlit Web App
================================================
Predicts two energy indicators for Pakistan:
  (a) Renewable electricity output (% of total electricity output)
  (b) Energy intensity level of primary energy (MJ/$2021 PPP GDP)

Run locally:
    pip install streamlit joblib scikit-learn pandas numpy
    streamlit run app.py

Deploy free on Streamlit Cloud:
    Push this folder to GitHub → connect at share.streamlit.io
"""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Energy Forecast",
    page_icon="🔋",
    layout="wide"
)

# ── Load pipelines ──────────────────────────────────────
@st.cache_resource
def load_pipelines():
    p1 = joblib.load("renewable_output_pipeline.pkl")
    p2 = joblib.load("energy_intensity_pipeline.pkl")
    return p1, p2

pipe_r, pipe_e = load_pipelines()

# ── Prediction function ─────────────────────────────────
def predict(pipe, input_dict):
    row = pd.DataFrame([input_dict]).reindex(columns=pipe["features"])
    row_imp = pipe["imputer"].transform(row)
    row_sc  = pipe["scaler"].transform(row_imp)
    return float(pipe["model"].predict(row_sc)[0])

# ── Header ─────────────────────────────────────────────
st.title("🔋 Pakistan Energy Forecast Dashboard")
st.markdown("Predicting Renewable Output & Energy Intensity (ML Model)")

# ── Sidebar ────────────────────────────────────────────
target_year = st.sidebar.slider("📅 Select Year", 2022, 2040, 2027)

# ── Layout ─────────────────────────────────────────────
left, right = st.columns([2, 1])

# ================= LEFT SIDE (INPUTS) ===================
with left:
    st.subheader("🎛️ Input Parameters")

    # Renewable inputs
    st.markdown("#### 🌱 Renewable Electricity")
    c1, c2, c3 = st.columns(3)

    with c1:
        hydro = st.slider("Hydro (%)", 5.0, 55.0, 25.0, 0.5)
        non_hydro = st.slider("Non-hydro (%)", 0.0, 20.0, 3.0, 0.5)

    with c2:
        fossil_r = st.slider("Fossil Fuel (%)", 40.0, 90.0, 62.0, 0.5)
        coal_r = st.slider("Coal (%)", 0.0, 40.0, 20.0, 0.5)

    with c3:
        gas_r = st.slider("Gas (%)", 10.0, 55.0, 30.0, 0.5)
        nuclear_r = st.slider("Nuclear (%)", 0.0, 10.0, 4.0, 0.5)

    # Build input for model
    inp_r = {
        "Year": target_year,
        "Electricity production from hydroelectric sources (% of total)": hydro,
        "Electricity production from renewable sources, excluding hydroelectric (% of total)": non_hydro,
        "Fossil fuel energy consumption (% of total)": fossil_r,
        "Electricity production from coal sources (% of total)": coal_r,
        "Electricity production from natural gas sources (% of total)": gas_r,
        "Electricity production from nuclear sources (% of total)": nuclear_r,
        "Energy imports, net (% of energy use)": 15,
        "Total natural resources rents (% of GDP)": 5,
    }

    st.markdown("---")

    # Energy intensity inputs
    st.markdown("#### ⚡ Energy Intensity")
    d1, d2, d3 = st.columns(3)

    with d1:
        gdp_eu = st.slider("GDP/Energy", 3.0, 15.0, 8.5, 0.1)
        energy_u = st.slider("Energy Use", 300.0, 800.0, 490.0, 5.0)

    with d2:
        fossil_e = st.slider("Fossil (%)", 40.0, 90.0, 62.0, 0.5)
        elec_c = st.slider("Electricity Use", 200.0, 700.0, 470.0, 5.0)

    with d3:
        coal_e = st.slider("Coal (%)", 0.0, 40.0, 20.0, 0.5)
        renew_c = st.slider("Renewable (%)", 15.0, 55.0, 42.0, 0.5)

    inp_e = {
        "Year": target_year,
        "GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)": gdp_eu,
        "Energy use (kg of oil equivalent per capita)": energy_u,
        "Fossil fuel energy consumption (% of total)": fossil_e,
        "Electric power consumption (kWh per capita)": elec_c,
        "Electricity production from coal sources (% of total)": coal_e,
        "Renewable energy consumption (% of total final energy consumption)": renew_c,
        "Energy imports, net (% of energy use)": 15,
        "Electric power transmission and distribution losses (% of output)": 18,
    }

# ── Predictions ─────────────────────────────────────────
pred_r = predict(pipe_r, inp_r)
pred_e = predict(pipe_e, inp_e)

# ================= RIGHT SIDE (OUTPUT) ==================
with right:
    st.subheader("📊 Predictions")

    st.markdown(f"""
    <div style="
        background:#f8f9fa;
        padding:20px;
        border-radius:12px;
        text-align:center;
        margin-bottom:15px;
        box-shadow:0 2px 5px rgba(0,0,0,0.1);
    ">
        <h5>🌱 Renewable Output</h5>
        <h1 style="color:#1565C0;">{pred_r:.2f}%</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background:#f8f9fa;
        padding:20px;
        border-radius:12px;
        text-align:center;
        box-shadow:0 2px 5px rgba(0,0,0,0.1);
    ">
        <h5>⚡ Energy Intensity</h5>
        <h1 style="color:#2E7D32;">{pred_e:.3f}</h1>
        <small>MJ/$2021 PPP GDP</small>
    </div>
    """, unsafe_allow_html=True)

    # Progress comparison
    st.markdown("---")
    st.markdown("#### 📉 Historical Comparison")

    hist_min_r, hist_max_r = min(pipe_r["historical_values"]), max(pipe_r["historical_values"])
    hist_min_e, hist_max_e = min(pipe_e["historical_values"]), max(pipe_e["historical_values"])

    pct_r = (pred_r - hist_min_r) / (hist_max_r - hist_min_r) * 100
    pct_e = (pred_e - hist_min_e) / (hist_max_e - hist_min_e) * 100

    st.progress(int(max(0, min(100, pct_r))))
    st.caption(f"Renewable Range: {hist_min_r:.1f}% – {hist_max_r:.1f}%")

    st.progress(int(max(0, min(100, pct_e))))
    st.caption(f"Energy Intensity Range: {hist_min_e:.2f} – {hist_max_e:.2f}")

# ── Footer ─────────────────────────────────────────────
st.markdown("---")
st.caption("ML Model: Ridge Regression | Data: World Bank (1990–2021)")

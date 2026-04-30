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
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Energy Forecast",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load pipelines ──────────────────────────────────────────────────────────
@st.cache_resource
def load_pipelines():
    p1 = joblib.load("renewable_output_pipeline.pkl")
    p2 = joblib.load("energy_intensity_pipeline.pkl")
    return p1, p2

pipe_r, pipe_e = load_pipelines()

# ── Helper: predict single year from manual inputs ─────────────────────────
def predict(pipe, input_dict):
    row = pd.DataFrame([input_dict]).reindex(columns=pipe["features"])
    row_imp = pipe["imputer"].transform(row)
    row_sc  = pipe["scaler"].transform(row_imp)
    return float(pipe["model"].predict(row_sc)[0])

# ── Helper: auto-forecast a year using trend extrapolation ─────────────────
def auto_forecast(pipe, year):
    row = {}
    for col, trend in pipe["feature_trends"].items():
        row[col] = trend["slope"] * year + trend["intercept"]
    return predict(pipe, row)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://flagcdn.com/w80/pk.png", width=60)
st.sidebar.title("🔋 Pakistan Energy\nForecast Tool")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Forecast Mode",
    ["🤖 Auto-Forecast (trend-based)", "🎛️ Manual Input (scenario)"],
    help="Auto uses linear feature trend extrapolation. Manual lets you set each parameter."
)
st.sidebar.markdown("---")

# ── Main header ────────────────────────────────────────────────────────────
st.title("🔋 Pakistan Energy Forecasting Dashboard")
st.markdown(
    "Forecasting **(a) Renewable Electricity Output (%)** and **(b) Energy Intensity (MJ/\\$2021 PPP GDP)** "
    "using Ridge Regression trained on World Bank data (1990–2021)."
)
st.markdown(f"**Model accuracy:** Renewable Output LOO-R² = `{pipe_r['loo_r2']:.4f}` &nbsp;|&nbsp; "
            f"Energy Intensity LOO-R² = `{pipe_e['loo_r2']:.4f}`")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
#  AUTO-FORECAST MODE
# ══════════════════════════════════════════════════════════════════════════
if mode == "🤖 Auto-Forecast (trend-based)":

    st.subheader("📈 Auto-Forecast: 2022 – 2040")
    st.info("Features are extrapolated using their historical linear trends. "
            "Confidence bands widen with forecast horizon to reflect growing uncertainty.")

    col1, col2 = st.columns([1, 2])
    with col1:
        start_yr = st.slider("Forecast start year", 2022, 2030, 2022)
        end_yr   = st.slider("Forecast end year",   2025, 2040, 2035)
        show_ci  = st.checkbox("Show 95% confidence band", value=True)

    years_fc = list(range(start_yr, end_yr + 1))

    # Generate forecasts
    fc_r = [auto_forecast(pipe_r, y) for y in years_fc]
    fc_e = [auto_forecast(pipe_e, y) for y in years_fc]

    # Build result table
    df_fc = pd.DataFrame({
        "Year": years_fc,
        "Renewable Output (%)": np.round(fc_r, 3),
        "Energy Intensity (MJ/$2021 PPP GDP)": np.round(fc_e, 3),
    })

    with col2:
        st.dataframe(df_fc.set_index("Year"), use_container_width=True)

    # ── Plots ──────────────────────────────────────────────────────────────
    hist_yrs_r = pipe_r["historical_years"]
    hist_val_r = pipe_r["historical_values"]
    hist_yrs_e = pipe_e["historical_years"]
    hist_val_e = pipe_e["historical_values"]
    std_r = pipe_r["residual_std"]
    std_e = pipe_e["residual_std"]
    fy = np.array(years_fc)
    widening = np.linspace(1.0, 2.5, len(fy))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hist_y, hist_v, fv, std, color, unit, label in zip(
        axes,
        [hist_yrs_r, hist_yrs_e], [hist_val_r, hist_val_e],
        [np.array(fc_r), np.array(fc_e)], [std_r, std_e],
        ["#1565C0", "#2E7D32"],
        ["% of Total Electricity", "MJ / $2021 PPP GDP"],
        ["Renewable Electricity Output", "Energy Intensity"]
    ):
        ax.plot(hist_y, hist_v, "o-", color=color, lw=2.2, ms=5, label="Historical")
        ax.plot(fy, fv, "D-", color="#E65100", lw=2.5, ms=6, label=f"Forecast {start_yr}–{end_yr}")
        if show_ci:
            ax.fill_between(fy,
                fv - 1.96 * std * widening,
                fv + 1.96 * std * widening,
                color="#E65100", alpha=0.15, label="95% CI")
        ax.axvline(pipe_r["train_max_year"] + 0.5, color="black", ls=":", lw=1.2, alpha=0.5)
        ax.set_title(f"Pakistan: {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel(unit)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Download CSV
    csv = df_fc.to_csv(index=False)
    st.download_button("⬇️ Download Forecast CSV", csv, "pakistan_energy_forecast.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
#  MANUAL / SCENARIO MODE
# ══════════════════════════════════════════════════════════════════════════
else:
    st.subheader("🎛️ Manual Scenario Prediction")
    st.info("Set each indicator manually to run a custom scenario. "
            "Leave sliders at their default (median) values for a baseline prediction.")

    target_year = st.sidebar.slider("Target Year", 2022, 2040, 2027)

    # ================= TARGET (a) =================
    st.markdown("#### Target (a): Renewable Electricity Output")

    c1, c2, c3 = st.columns(3)
    with c1:
        hydro    = st.slider("Hydroelectric production (%)", 5.0, 55.0, 25.0, 0.5)
        non_hydro= st.slider("Non-hydro renewable prod. (%)", 0.0, 20.0, 3.0, 0.5)
    with c2:
        fossil_r = st.slider("Fossil fuel consumption (%)", 40.0, 90.0, 62.0, 0.5)
        coal_r   = st.slider("Coal electricity prod. (%)", 0.0, 40.0, 20.0, 0.5)
    with c3:
        gas_r    = st.slider("Natural gas electricity (%)", 10.0, 55.0, 30.0, 0.5)
        nuclear_r= st.slider("Nuclear electricity (%)", 0.0, 10.0, 4.0, 0.5)
        imports_r= st.slider("Net energy imports (%)", 0.0, 35.0, 15.0, 0.5)
        rents_r  = st.slider("Natural resource rents (% GDP)", 0.5, 12.0, 5.0, 0.5)

    inp_r = {
        "Year": target_year,
        "Electricity production from hydroelectric sources (% of total)": hydro,
        "Electricity production from renewable sources, excluding hydroelectric (% of total)": non_hydro,
        "Fossil fuel energy consumption (% of total)": fossil_r,
        "Electricity production from coal sources (% of total)": coal_r,
        "Electricity production from natural gas sources (% of total)": gas_r,
        "Electricity production from nuclear sources (% of total)": nuclear_r,
        "Energy imports, net (% of energy use)": imports_r,
        "Total natural resources rents (% of GDP)": rents_r,
    }

    pred_r = predict(pipe_r, inp_r)

    # ================= OUTPUT (MIDDLE) =================
    st.markdown("---")
    st.subheader(f"📊 Predictions for Year {target_year}")

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(f"""
        <div style="
            background-color:#f8f9fa;
            padding:20px;
            border-radius:12px;
            text-align:center;
            box-shadow:0 2px 5px rgba(0,0,0,0.1);
        ">
            <h4>🌱 Renewable Electricity Output</h4>
            <h2 style="color:#1565C0;">{pred_r:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            padding:20px;
            border-radius:12px;
            text-align:center;
            box-shadow:0 2px 5px rgba(0,0,0,0.1);
        ">
            <h4>⚡ Energy Intensity</h4>
            <p style="margin-top:20px;">Adjust inputs below to calculate</p>
        </div>
        """, unsafe_allow_html=True)

    # ================= TARGET (b) =================
    st.markdown("---")
    st.markdown("#### Target (b): Energy Intensity")

    d1, d2, d3 = st.columns(3)
    with d1:
      gdp_eu = st.slider("GDP per energy unit (PPP $/kg oil eq.)", 3.0, 15.0, 8.5, 0.1, key="gdp_eu")
       energy_u = st.slider("Energy use (kg oil eq./capita)", 300.0, 800.0, 490.0, 5.0, key="energy_u")
    with d2:
       fossil_e = st.slider("Fossil fuel consumption (%)", 40.0, 90.0, 62.0, 0.5, key="fossil_e")
       elec_c = st.slider("Electric power consumption (kWh/capita)", 200.0, 700.0, 470.0, 5.0, key="elec_c")

    with d3:
       coal_e = st.slider("Coal electricity prod. (%)", 0.0, 40.0, 20.0, 0.5, key="coal_e")
      renew_c = st.slider("Renewable energy consumption (%)", 15.0, 55.0, 42.0, 0.5, key="renew_c")
       imports_e = st.slider("Net energy imports (%)", 0.0, 35.0, 15.0, 0.5, key="imports_e")
      tnd_loss = st.slider("T&D losses (%)", 10.0, 30.0, 18.0, 0.5, key="tnd_loss")
    inp_e = {
        "Year": target_year,
        "GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)": gdp_eu,
        "Energy use (kg of oil equivalent per capita)": energy_u,
        "Fossil fuel energy consumption (% of total)": fossil_e,
        "Electric power consumption (kWh per capita)": elec_c,
        "Electricity production from coal sources (% of total)": coal_e,
        "Renewable energy consumption (% of total final energy consumption)": renew_c,
        "Energy imports, net (% of energy use)": imports_e,
        "Electric power transmission and distribution losses (% of output)": tnd_loss,
    }

    pred_e = predict(pipe_e, inp_e)

    # ===== UPDATE SECOND CARD =====
    with m2:
        st.markdown(f"""
        <div style="
            background-color:#f8f9fa;
            padding:20px;
            border-radius:12px;
            text-align:center;
            box-shadow:0 2px 5px rgba(0,0,0,0.1);
        ">
            <h4>⚡ Energy Intensity</h4>
            <h2 style="color:#2E7D32;">{pred_e:.3f} MJ/$2021 PPP GDP</h2>
        </div>
        """, unsafe_allow_html=True)

    # ================= COMPARISON =================
    st.markdown("#### How do these compare to historical range?")

    g1, g2 = st.columns(2)

    hist_min_r, hist_max_r = min(pipe_r["historical_values"]), max(pipe_r["historical_values"])
    hist_min_e, hist_max_e = min(pipe_e["historical_values"]), max(pipe_e["historical_values"])

    with g1:
        pct_r = (pred_r - hist_min_r) / (hist_max_r - hist_min_r) * 100
        st.progress(min(max(int(pct_r), 0), 100))
        st.caption(f"Renewable: {hist_min_r:.1f}% – {hist_max_r:.1f}% | Pred: {pred_r:.2f}%")

    with g2:
        pct_e = (pred_e - hist_min_e) / (hist_max_e - hist_min_e) * 100
        st.progress(min(max(int(pct_e), 0), 100))
        st.caption(f"Energy Intensity: {hist_min_e:.2f} – {hist_max_e:.2f} | Pred: {pred_e:.3f}")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: Ridge Regression (alpha=1.0) | Validation: Leave-One-Out CV | "
    "Data: World Bank Energy & Mining — Pakistan (1990–2021) | "
    "Assignment #2 — ML Project"
)

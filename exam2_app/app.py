"""
Midterm 2 - Team 5 Smart Grid Streamlit dashboard.
Run from this folder:  streamlit run app.py
Loads the serialized artifacts from ../models (never trains here).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "exam2_team5_smartgrid.csv"
MODELS = ROOT / "models"
TARGET = "power_demand_mw"

st.set_page_config(page_title="Team 5 Smart Grid - Load Shedding", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODELS / "exam2_metrics.json").read_text(encoding="utf-8"))
    return (joblib.load(MODELS / "best_model.joblib"), joblib.load(MODELS / "ann_model.joblib"),
            joblib.load(MODELS / "scaler.joblib"), meta)


df = load_data()
best_model, ann_model, scaler, meta = load_artifacts()
features = meta["features"]
p75, p90 = meta["train_demand_p75"], meta["train_demand_p90"]

with st.sidebar:
    st.header("Loaded artifacts")
    st.success("models/best_model.joblib")
    st.success("models/ann_model.joblib")
    st.success("models/scaler.joblib")
    st.caption(f"Winner: {meta['winner']}")

st.title("Team 5 Smart Grid: Power Demand & Load Shedding")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Mean demand", f"{df[TARGET].mean():.1f} MW")
k3.metric("Load shedding threshold (P90)", f"{p90:.1f} MW")
best_row = next(r for r in meta["metrics"] if r["Model"] == meta["winner"])
k4.metric("Winner test RMSE", f"{best_row['RMSE']:.2f} MW")

tab1, tab2, tab3, tab4 = st.tabs(["1. Data Explorer", "2. Preprocessing & Correlation",
                                  "3. Model Benchmark", "4. What-If Inference"])

with tab1:
    st.subheader("Data sample and summary statistics")
    st.dataframe(df.head(10), width="stretch")
    st.dataframe(df.describe().T.round(2), width="stretch")
    col = st.selectbox("Feature distribution", features + [TARGET], key="eda_col")
    st.plotly_chart(px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}"), width="stretch")

with tab2:
    st.subheader("StandardScaler (fit on the 80% training split only)")
    st.dataframe(pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=features).round(3),
                 width="stretch")
    st.caption(f"Missing values in the dataset: {int(df.isna().sum().sum())}")
    corr = df.corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Pearson correlation matrix"), width="stretch")

with tab3:
    st.subheader("Holdout benchmark (same 80/20 split, random_state=42)")
    st.dataframe(pd.DataFrame(meta["metrics"]).set_index("Model").round(4), width="stretch")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=meta["ann_loss_curve"], name="Training loss"))
    fig.update_layout(title="ANN (64, 32) loss convergence (loss_curve_)", xaxis_title="Epoch",
                      yaxis_title="Loss", yaxis_type="log")
    st.plotly_chart(fig, width="stretch")

with tab4:
    st.subheader("What-if: adjust active / reactive load")
    model_name = st.radio("Model", [f"Winner ({meta['winner']})", "ANN (64, 32)"], horizontal=True, key="wi_model")
    c1, c2 = st.columns(2)
    vals = {}
    for i, f in enumerate(features):
        lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
        step = 0.01 if f == "frequency_hz" else 0.1
        vals[f] = (c1 if i % 2 == 0 else c2).slider(f, lo, hi, med, step=step, key=f"wi_{f}")
    x_new = pd.DataFrame([vals])[features]
    model = best_model if model_name.startswith("Winner") else ann_model
    pred = float(model.predict(scaler.transform(x_new))[0])
    st.metric("Predicted power demand", f"{pred:.1f} MW")
    st.progress(float(np.clip(pred / df[TARGET].max(), 0, 1)))
    if pred >= p90:
        st.error(f"HIGH RISK: {pred:.1f} MW is above the P90 threshold ({p90:.1f} MW). Trigger load shedding.")
    elif pred >= p75:
        st.warning(f"WATCH: {pred:.1f} MW is above P75 ({p75:.1f} MW). Prepare reserve capacity.")
    else:
        st.success(f"NORMAL: {pred:.1f} MW is below P75 ({p75:.1f} MW). Normal operation.")

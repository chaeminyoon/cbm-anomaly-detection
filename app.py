"""
Ship Shaft CBM Anomaly Detection Dashboard - Main Page
Production-grade Streamlit application
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import io

# UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config
from utils.model_utils import (
    resolve_run_id,
    load_model, load_training_data, prepare_scaler,
    get_feature_importance, calculate_statistics
)

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Compact, professional styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }
    .page-header {
        border-bottom: 3px solid #0F4C81;
        padding-bottom: 0.55rem;
        margin-bottom: 0.9rem;
    }
    .page-header .kicker {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0;
    }
    .page-header h1 {
        color: #0B3C61;
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0.1rem 0 0 0;
        padding: 0;
    }
    [data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.55rem 0.85rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; }
    h2 { font-size: 1.12rem !important; padding-top: 0.4rem !important; }
    h3 { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

CHART_MARGIN = dict(l=10, r=10, t=32, b=10)
NORMAL_COLOR = '#2E6F9E'
ANOMALY_COLOR = '#C4453C'

# Load model and data
with st.spinner("Loading model and data..."):
    model = load_model()
    df, feature_cols = load_training_data()

    if model is None or df is None:
        st.error("Failed to load model or data. Please check configuration.")
        st.stop()

    scaler = prepare_scaler(df, feature_cols)

    X = scaler.transform(df[feature_cols])
    predictions = model.predict(X)
    anomaly_scores = model.score_samples(X)

    stats = calculate_statistics(df, predictions, anomaly_scores)

# Sidebar
with st.sidebar:
    st.markdown("### System Status")
    st.success("Model loaded")
    run_id = resolve_run_id()
    st.caption(f"MLflow run `{run_id[:12]}`" if run_id else "Run ID: not resolved")
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Anomaly Rate", f"{stats['anomaly_rate']:.2f}%")
    st.metric("Windows Monitored", f"{stats['total_samples']:,}")

# Header
st.markdown(
    '<div class="page-header">'
    '<div class="kicker">Condition-Based Maintenance &middot; Propulsion Shaft</div>'
    '<h1>Ship Shaft CBM Monitoring</h1>'
    '</div>',
    unsafe_allow_html=True)

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Monitored Windows", f"{stats['total_samples']:,}")
col2.metric("Normal", f"{stats['normal_count']:,}")
col3.metric("Anomalies", f"{stats['anomaly_count']:,}")
col4.metric("Anomaly Rate", f"{stats['anomaly_rate']:.2f}%")
col5.metric("Mean Anomaly Score", f"{stats['avg_anomaly_score']:.4f}")

# ---- Charts grid (2 x 2) ----
row1_left, row1_right = st.columns(2)

with row1_left:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=anomaly_scores[predictions == 1], name='Normal',
        marker_color=NORMAL_COLOR, opacity=0.8, nbinsx=50))
    fig.add_trace(go.Histogram(
        x=anomaly_scores[predictions == -1], name='Anomaly',
        marker_color=ANOMALY_COLOR, opacity=0.8, nbinsx=50))
    fig.add_vline(x=config.ANOMALY_THRESHOLD, line_dash='dash', line_color=ANOMALY_COLOR,
                  line_width=1.5)
    fig.update_layout(
        title=dict(text='Anomaly Score Distribution (dashed: alert threshold)', font_size=13),
        barmode='overlay', height=300, template='plotly_white', margin=CHART_MARGIN,
        xaxis_title=None, yaxis_title='windows',
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1, font_size=10))
    st.plotly_chart(fig, use_container_width=True)

with row1_right:
    importance_df = get_feature_importance(model, scaler, feature_cols, df, predictions)
    top_10 = importance_df.head(10)
    fig = px.bar(
        top_10.iloc[::-1], x='importance', y='feature', orientation='h',
        color='importance', color_continuous_scale='Blues',
        labels={'importance': 'mean |z| over anomalies', 'feature': ''})
    fig.update_layout(
        title=dict(text='Top 10 Anomaly-Contributing Sensors', font_size=13),
        height=300, template='plotly_white', margin=CHART_MARGIN,
        showlegend=False, coloraxis_showscale=False, yaxis=dict(tickfont_size=10))
    st.plotly_chart(fig, use_container_width=True)

row2_left, row2_right = st.columns(2)

with row2_left:
    idx = np.arange(len(anomaly_scores))
    normal_mask = predictions == 1
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=idx[normal_mask], y=anomaly_scores[normal_mask], mode='markers',
        name='Normal', marker=dict(color=NORMAL_COLOR, size=3, opacity=0.45)))
    fig.add_trace(go.Scattergl(
        x=idx[~normal_mask], y=anomaly_scores[~normal_mask], mode='markers',
        name='Anomaly', marker=dict(color=ANOMALY_COLOR, size=5, opacity=0.85)))
    fig.add_hline(y=config.ANOMALY_THRESHOLD, line_dash='dash', line_color=ANOMALY_COLOR,
                  line_width=1.5)
    fig.update_layout(
        title=dict(text='Anomaly Score Timeline', font_size=13),
        height=300, template='plotly_white', margin=CHART_MARGIN,
        xaxis_title='window index', yaxis_title='score',
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1, font_size=10))
    st.plotly_chart(fig, use_container_width=True)

with row2_right:
    window = 100
    rolling_rate = pd.Series((predictions == -1).astype(float)).rolling(
        window, min_periods=20).mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=rolling_rate, mode='lines', name='rolling anomaly rate',
        line=dict(color=NORMAL_COLOR, width=2)))
    fig.add_hline(y=config.CONTAMINATION_RATE * 100, line_dash='dash',
                  line_color='#94A3B8', line_width=1.5,
                  annotation_text='expected rate', annotation_font_size=10)
    fig.update_layout(
        title=dict(text=f'Rolling Anomaly Rate (window = {window})', font_size=13),
        height=300, template='plotly_white', margin=CHART_MARGIN,
        xaxis_title='window index', yaxis_title='%')
    st.plotly_chart(fig, use_container_width=True)

# ---- Model / data summary ----
st.markdown("## System Information")
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.dataframe(pd.DataFrame({
        'Property': ['Model', 'MLflow run', 'Contamination', 'Score threshold', 'Features'],
        'Value': ['Isolation Forest', (run_id or '-')[:16],
                  f"{config.CONTAMINATION_RATE:.0%}", str(config.ANOMALY_THRESHOLD),
                  str(len(feature_cols))],
    }), hide_index=True, use_container_width=True)
with info_col2:
    st.dataframe(pd.DataFrame({
        'Property': ['Data source', 'Records', 'Shape', 'Loaded at'],
        'Value': [config.DATA_PATH, f"{stats['total_samples']:,}", str(df.shape),
                  datetime.now().strftime('%Y-%m-%d %H:%M')],
    }), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("CBM Anomaly Detection Dashboard v1.1 | Streamlit + MLflow | "
           "[MLflow UI](http://localhost:5000) | "
           "[GitHub](https://github.com/chaeminyoon/cbm-anomaly-detection)")

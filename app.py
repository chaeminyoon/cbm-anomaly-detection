"""
Ship Shaft CBM Anomaly Detection Dashboard - Main Page
Production-grade Streamlit application for AWS deployment
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import io

# UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config
from utils.model_utils import (
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

# Custom CSS for production-grade styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #d32f2f;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=CBM+Logo", use_container_width=True)
    st.markdown("---")
    st.markdown("### Navigation")
    st.info("""
    **Main Pages:**
    - 🏠 Home (current)
    - 🔍 Real-time Detection
    - 📊 Batch Analysis
    - 📈 Statistics & Trends
    - ⚙️ Settings
    """)

    st.markdown("---")
    st.markdown("### System Status")
    st.success("✅ Model Loaded")
    st.info(f"🔧 Run ID: {config.MLFLOW_RUN_ID[:8]}...")

    st.markdown("---")
    st.markdown("### Quick Stats")

# Main content
st.markdown('<div class="main-header">🚢 Ship Shaft CBM Monitoring Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Real-time Condition-Based Maintenance Anomaly Detection System**")
st.markdown("---")

# Load model and data
with st.spinner("Loading model and data..."):
    model = load_model()
    df, feature_cols = load_training_data()

    if model is None or df is None:
        st.error("❌ Failed to load model or data. Please check configuration.")
        st.stop()

    scaler = prepare_scaler(df, feature_cols)

    # Get predictions on training data
    X = scaler.transform(df[feature_cols])
    predictions = model.predict(X)
    anomaly_scores = model.score_samples(X)

    # Calculate statistics
    stats = calculate_statistics(df, predictions, anomaly_scores)

# Key Metrics Dashboard
st.markdown("## 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Samples",
        value=f"{stats['total_samples']:,}",
        delta=None
    )

with col2:
    st.metric(
        label="Normal Data",
        value=f"{stats['normal_count']:,}",
        delta=f"{100 - stats['anomaly_rate']:.1f}%"
    )

with col3:
    st.metric(
        label="Anomalies Detected",
        value=f"{stats['anomaly_count']:,}",
        delta=f"-{stats['anomaly_rate']:.2f}%",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Avg Anomaly Score",
        value=f"{stats['avg_anomaly_score']:.4f}",
        delta=None
    )

st.markdown("---")

# Visualizations
st.markdown("## 📈 Data Visualization")

tab1, tab2, tab3 = st.tabs(["Distribution", "Feature Importance", "Recent Activity"])

with tab1:
    st.markdown("### Anomaly Score Distribution")

    # Create histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=anomaly_scores[predictions == 1],
        name='Normal',
        marker_color='green',
        opacity=0.7,
        nbinsx=50
    ))

    fig.add_trace(go.Histogram(
        x=anomaly_scores[predictions == -1],
        name='Anomaly',
        marker_color='red',
        opacity=0.7,
        nbinsx=50
    ))

    fig.update_layout(
        barmode='overlay',
        xaxis_title='Anomaly Score',
        yaxis_title='Frequency',
        height=400,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Top 10 Important Features")

    # Get feature importance
    importance_df = get_feature_importance(model, scaler, feature_cols, df, predictions)
    top_10 = importance_df.head(10)

    fig = px.bar(
        top_10.iloc[::-1],
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Reds',
        labels={'importance': 'Avg Z-Score (Anomaly Contribution)', 'feature': 'Sensor/Feature'}
    )

    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Full Feature Importance Table"):
        st.dataframe(importance_df, use_container_width=True)

with tab3:
    st.markdown("### Recent Anomaly Activity (Simulated)")

    # Simulate recent activity with timestamps
    recent_anomalies = df[predictions == -1].tail(10).copy()
    recent_anomalies['timestamp'] = pd.date_range(
        end=datetime.now(),
        periods=len(recent_anomalies),
        freq='1H'
    )
    recent_anomalies['anomaly_score'] = anomaly_scores[predictions == -1][-10:]

    if len(recent_anomalies) > 0:
        st.dataframe(
            recent_anomalies[['timestamp', 'anomaly_score']].style.format({
                'anomaly_score': '{:.6f}'
            }),
            use_container_width=True
        )
    else:
        st.info("No recent anomalies detected.")

st.markdown("---")

# System Information
st.markdown("## ⚙️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model Information")
    st.json({
        "Model Type": "Isolation Forest",
        "MLflow Run ID": config.MLFLOW_RUN_ID,
        "Contamination Rate": f"{config.CONTAMINATION_RATE * 100}%",
        "Threshold": config.ANOMALY_THRESHOLD,
        "Features": len(feature_cols)
    })

with col2:
    st.markdown("### Data Information")
    st.json({
        "Data Source": config.DATA_PATH,
        "Total Records": stats['total_samples'],
        "Data Shape": str(df.shape),
        "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>CBM Anomaly Detection Dashboard v1.0 | Powered by Streamlit & MLflow</p>
    <p>🔗 <a href='http://localhost:5000' target='_blank'>MLflow UI</a> |
    📚 <a href='#'>Documentation</a> |
    🐛 <a href='#'>Report Issue</a></p>
</div>
""", unsafe_allow_html=True)
